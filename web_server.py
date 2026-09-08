"""
web_server.py
=============
Lightweight, non-blocking WebSocket & Static File server for the AMR Fleet
web dashboard.

Architecture
------------
* FastAPI + Uvicorn (ASGI, fully async)
* One background asyncio.Task runs the AMR simulation loop at 10 Hz
* A separate asyncio.Task broadcasts serialised fleet state to every
  connected WebSocket client every 100 ms (decoupled from the sim rate)
* WebSocket commands from the frontend are dispatched synchronously into
  the running AMRSwarm state without blocking the event loop (all swarm
  methods are pure-Python / CPU-bound and fast)

No Godot / UDP code — that stack has been removed entirely.

Usage
-----
    pip install fastapi uvicorn[standard]
    python web_server.py                  # http://localhost:8000

    # Options:
    python web_server.py --host 0.0.0.0 --port 8080 --tasks 30

Endpoints
---------
    GET  /          -> serves frontend/index.html (or ./index.html)
    GET  /static/*  -> serves frontend/ static assets
    GET  /api/status -> liveness + metrics JSON
    WS   /ws        -> bidirectional control channel

WS Command schema (client -> server)
-------------------------------------
    {"action": "add_obstacle",  "x": int, "y": int}
    {"action": "kill_robot",    "robot_id": int}   # index into agents list
    {"action": "spawn_robot"}
    {"action": "add_task",      "pickup": [x, y], "drop": [x, y]}

WS Broadcast schema (server -> all clients, every 100 ms)
----------------------------------------------------------
    {
      "warehouse": {
        "width": 20, "height": 20,
        "shelves": [[c, r], ...],
        "chargers": [],
        "pickups":  [[c, r], ...],
        "drops":    [[c, r], ...]
      },
      "obstacles": [[x, y], ...],
      "robots": [
        {
          "id": "AMR_01",
          "x": 4.2, "y": 8.1,
          "battery": 87.5,
          "state": "NAVIGATING",
          "path": [[5, 8], [6, 8]],
          "task": "T012"
        },
        ...
      ],
      "metrics": {
        "collisions":       0,
        "completed_tasks":  14,
        "time_saved_pct":   27.2
      }
    }
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import random
import sys
import time as _time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# FastAPI / Uvicorn
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse, JSONResponse
    import uvicorn
except ImportError as exc:  # pragma: no cover
    sys.exit(
        f"[web_server] Missing dependency: {exc}\n"
        "  Install with:  pip install fastapi uvicorn[standard]"
    )

# ---------------------------------------------------------------------------
# Project modules
# ---------------------------------------------------------------------------
from core.grid_map import GridMap, WALKWAY as W, SHELF as S, PICKUP as P, DROP as D
from core.space_time_astar import ReservationTable as STATable

from agent.amr_agent import AMRAgent, AgentState
from benchmark.runner import (
    build_warehouse_grid,
    generate_tasks,
    _walkable_cells,
)
from tasks.auction_manager import (
    Task, Bid, BidFormula, _AuctionState, AuctionParticipantState,
)
import time as _t_module

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("web_server")

# ---------------------------------------------------------------------------
# Simulation constants (can be overridden via CLI args)
# ---------------------------------------------------------------------------
SIM_GRID_COLS  = 20
SIM_GRID_ROWS  = 20
SIM_NUM_AGENTS = 4
SIM_SEED       = 7
SIM_TICK_HZ    = 10        # simulation ticks per second
BROADCAST_HZ   = 10        # state broadcasts per second (every 100 ms)


# ---------------------------------------------------------------------------
# AMR Swarm — mutable shared state, touched only from the sim task
# ---------------------------------------------------------------------------

from core.grid_map import GridMap, WALKWAY as W, SHELF as S, PICKUP as P, DROP as D

def build_web_grid() -> GridMap:
    raw = []
    for r in range(24):
        if r in (4, 8, 12, 16):
            row = [S] * 24
            row[0] = W; row[11] = W; row[23] = W
            raw.append(row)
        else:
            raw.append([W] * 24)
    # Pickups
    for p in [(0,0), (2,0), (4,0), (6,0)]:
        raw[p[1]][p[0]] = P
    # Drops
    for d in [(14,20), (16,20), (18,20), (20,20)]:
        raw[d[1]][d[0]] = D
    return GridMap(raw)

class AMRSwarm:
    """
    Encapsulates the entire running AMR simulation.

    All public methods that mutate state are designed to be called from the
    asyncio event loop (single-threaded) — no locking required.
    """

    def __init__(self, num_tasks: int = 50, seed: int = SIM_SEED) -> None:
        self._seed       = seed
        self._rng        = random.Random(seed)
        self.grid        = build_web_grid()
        self._tasks_raw  = []
        self._cells      = _walkable_cells(self.grid)

        # Shared space-time reservation table
        self._sta        = STATable()

        # Task queue and completed log
        self._task_queue: List[Task] = []
        self._completed_tasks: int = 0
        self.obstacles: Set[Tuple[int, int]] = set()

        # Metrics
        self.total_collisions: int = 0
        self.tick_count:       int = 0

        # In-process P2P intent buses
        self._intent_bus: Dict[str, list] = {}
        self._hazard_bus: list = []
        self._lease_bus:  list = []

        # Pre-compute static warehouse layout for the broadcast payload
        self._warehouse_snapshot = self._build_warehouse_snapshot()
        self._pickups = [tuple(p) for p in self._warehouse_snapshot["pickups"]]
        self._destinations = [tuple(p) for p in self._warehouse_snapshot["drops"]] + [tuple(p) for p in self._warehouse_snapshot["shelves"]]
        self._shelves = [tuple(p) for p in self._warehouse_snapshot["shelves"]]

        self.shelf_inventory = {s: self._rng.randint(3, 6) for s in self._shelves}
        self.total_restocked = 0
        self.total_dispatched = 0

        # Fleet
        self.agents: List[AMRAgent] = []
        self._spawn_agents(SIM_NUM_AGENTS)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _spawn_agents(self, n: int) -> None:
        """Spawn *n* new AMR agents at random walkable positions."""
        occupied = {ag.position for ag in self.agents}
        free = [c for c in self._cells if c not in occupied]
        starts = self._rng.sample(free, min(n, len(free)))
        for i, pos in enumerate(starts):
            aid = f"AMR_{len(self.agents) + i + 1:02d}"
            ag  = AMRAgent(
                robot_id=aid,
                start_pos=pos,
                grid=self.grid,
                sta_table=self._sta,
                battery=self._rng.uniform(75.0, 100.0),
                on_intent_broadcast=self._on_intent,
                on_hazard_broadcast=self._hazard_bus.append,
                on_lease_broadcast=self._lease_bus.append,
                on_task_complete=self._on_task_complete,
            )
            self.agents.append(ag)
            logger.info("Spawned %s at %s", aid, pos)

    def _on_task_complete(self, task: Task) -> None:
        self._completed.append(task)
        if task.task_type == "INBOUND_RESTOCK":
            self.shelf_inventory[task.drop_pos] = min(10, self.shelf_inventory.get(task.drop_pos, 0) + 1)
            self.total_restocked += 1
        elif task.task_type == "OUTBOUND_FULFILLMENT":
            self.shelf_inventory[task.pickup_pos] = max(0, self.shelf_inventory.get(task.pickup_pos, 0) - 1)
            self.total_dispatched += 1

    def _on_intent(self, sender: str, wps: list) -> None:
        self._intent_bus[sender] = wps

    def _build_warehouse_snapshot(self) -> dict:
        """
        Extract static cell lists from the grid for the broadcast payload.
        Recomputed whenever the grid mutates (e.g. obstacle added).
        """
        shelves, pickups, drops = [], [], []
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                ct = self.grid.cell_type(c, r)
                if ct == S:
                    shelves.append([c, r])
                elif ct == P:
                    pickups.append([c, r])
                elif ct == D:
                    drops.append([c, r])
        return {
            "width":    self.grid.cols,
            "height":   self.grid.rows,
            "shelves":  shelves,
            "chargers": [],          # extend if charger cells are added later
            "pickups":  pickups,
            "drops":    drops,
        }

    # ------------------------------------------------------------------
    # Single simulation tick (called by the background async task)
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Advance the simulation by one 100 ms tick."""
        t = self.tick_count

        # Auto-replenishing task generator
        idle = [ag for ag in self.agents if ag.state == AgentState.IDLE]
        
        active_pickups = {t.pickup_pos for t in self._task_queue}
        active_drops = {t.drop_pos for t in self._task_queue}
        for ag in self.agents:
            if ag._current_task:
                active_pickups.add(ag._current_task.pickup_pos)
                active_drops.add(ag._current_task.drop_pos)
                
        if len(self._task_queue) < 4 and len(idle) > 0:
            if self._rng.random() < 0.5:
                # INBOUND_RESTOCK
                shelves_with_capacity = [s for s, c in self.shelf_inventory.items() if c < 10 and s not in active_drops and s not in active_pickups]
                free_pickups = [p for p in self._pickups if p not in active_pickups and p not in active_drops]
                if shelves_with_capacity and free_pickups:
                    pickup = self._rng.choice(free_pickups)
                    drop = self._rng.choice(shelves_with_capacity)
                    tid = f"IN_{t:05d}"
                    task = Task(tid, pickup, drop, round(self._rng.uniform(0.5, 1.0), 2), t, "INBOUND_RESTOCK")
                    self._task_queue.append(task)
                    active_pickups.add(pickup)
                    active_drops.add(drop)
                    logger.info("Auto-generated task %s: pickup=%s drop=%s", tid, pickup, drop)
            else:
                # OUTBOUND_FULFILLMENT
                shelves_with_stock = [s for s, c in self.shelf_inventory.items() if c > 0 and s not in active_pickups and s not in active_drops]
                drops = [tuple(p) for p in self._warehouse_snapshot["drops"] if tuple(p) not in active_drops and tuple(p) not in active_pickups]
                if shelves_with_stock and drops:
                    pickup = self._rng.choice(shelves_with_stock)
                    drop = self._rng.choice(drops)
                    tid = f"OUT_{t:05d}"
                    task = Task(tid, pickup, drop, round(self._rng.uniform(0.5, 1.0), 2), t, "OUTBOUND_FULFILLMENT")
                    self._task_queue.append(task)
                    active_pickups.add(pickup)
                    active_drops.add(drop)
                    logger.info("Auto-generated task %s: pickup=%s drop=%s", tid, pickup, drop)

        # Auction newly issued tasks
        available = [tk for tk in self._task_queue if tk.issued_tick <= t]
        for task in available:
            winner = self._run_auction(task)
            if winner:
                self._task_queue.remove(task)

        # Tick all agents
        for ag in self.agents:
            ag.tick(delta_time=0.1)

        # Distribute P2P messages
        for ag in self.agents:
            for sid, wps in self._intent_bus.items():
                if sid != ag.robot_id:
                    ag.on_peer_intent(sid, wps)
            for hm in self._hazard_bus:
                if hm.robot_id != ag.robot_id:
                    ag.on_peer_hazard(hm)
            for lm in self._lease_bus:
                if lm.robot_id != ag.robot_id:
                    ag.on_peer_lease(lm)
        self._hazard_bus.clear()
        self._lease_bus.clear()

        # Collision detection
        pos_map: Dict[Tuple, str] = {}
        for ag in self.agents:
            if ag.position in pos_map:
                self.total_collisions += 1
                logger.debug(
                    "COLLISION @ %s between %s and %s (tick %d)",
                    ag.position, ag.robot_id, pos_map[ag.position], t,
                )
            else:
                pos_map[ag.position] = ag.robot_id

        self.tick_count += 1

    def _run_auction(self, task: Task) -> Optional[str]:
        """Run one Contract-Net auction. Returns winner robot_id or None."""
        idle = [ag for ag in self.agents if ag.state == AgentState.IDLE]
        if not idle:
            return None
        bids = [
            Bid(
                task_id=task.task_id,
                robot_id=ag.robot_id,
                bid_value=BidFormula.compute(
                    ag.robot_id, ag.position, task.pickup_pos, ag.battery
                ),
                battery=ag.battery,
                position=ag.position,
            )
            for ag in idle
        ]
        for ag in idle:
            my_bid = next(b for b in bids if b.robot_id == ag.robot_id)
            ag._auction._auctions[task.task_id] = _AuctionState(
                task=task,
                my_bid=my_bid,
                bids={b.robot_id: b for b in bids},
                close_time=_t_module.time() + 0.15,
            )
            ag._auction._state = AuctionParticipantState.BIDDING
        winner = None
        for ag in idle:
            winner = ag.resolve_auction(task.task_id)
        return winner

    # ------------------------------------------------------------------
    # WebSocket command handlers (called from the WS handler coroutine)
    # ------------------------------------------------------------------

    def add_obstacle(self, x: int, y: int) -> None:
        """Block a walkway cell as a dynamic obstacle."""
        if self.grid.in_bounds(x, y) and self.grid.is_passable(x, y):
            self.grid.block_cell(x, y)
            self.obstacles.add((x, y))
            # Rebuild warehouse snapshot so next broadcast reflects the change
            self._warehouse_snapshot = self._build_warehouse_snapshot()
            logger.info("Obstacle added at (%d, %d)", x, y)
        else:
            logger.warning(
                "add_obstacle: (%d, %d) is out-of-bounds or already blocked", x, y
            )

    def kill_robot(self, robot_id) -> None:
        """
        Remove a robot from the fleet.

        Accepts either:
        * int  — 0-based index into self.agents
        * str  — matching robot_id string (e.g. "AMR_02")
        """
        target: Optional[AMRAgent] = None

        if isinstance(robot_id, int):
            if 0 <= robot_id < len(self.agents):
                target = self.agents[robot_id]
        else:
            rid_str = str(robot_id)
            target = next(
                (a for a in self.agents if a.robot_id == rid_str), None
            )

        if target is None:
            logger.warning("kill_robot: robot %r not found", robot_id)
            return

        self.agents.remove(target)
        logger.info("Robot %s removed from fleet", target.robot_id)

    def spawn_robot(self) -> None:
        """Spawn one additional robot at a random free walkable position."""
        occupied = {ag.position for ag in self.agents}
        free = [c for c in self._cells if c not in occupied]
        if not free:
            logger.warning("spawn_robot: no free cells available")
            return
        pos = self._rng.choice(free)
        aid = f"AMR_{len(self.agents) + 1:02d}"
        ag  = AMRAgent(
            robot_id=aid,
            start_pos=pos,
            grid=self.grid,
            sta_table=self._sta,
            battery=self._rng.uniform(75.0, 100.0),
            on_intent_broadcast=self._on_intent,
            on_hazard_broadcast=self._hazard_bus.append,
            on_lease_broadcast=self._lease_bus.append,
            on_task_complete=self._completed.append,
        )
        self.agents.append(ag)
        logger.info("Spawned new robot %s at %s", aid, pos)

    def add_task(
        self,
        pickup: Tuple[int, int],
        drop:   Tuple[int, int],
    ) -> None:
        """Inject a new task into the queue, effective immediately."""
        tid  = f"WEB_{self.tick_count:05d}"
        task = Task(
            task_id=tid,
            pickup_pos=tuple(pickup),   # type: ignore[arg-type]
            drop_pos=tuple(drop),       # type: ignore[arg-type]
            urgency=1.0,
            issued_tick=self.tick_count,  # available this tick
        )
        self._task_queue.append(task)
        logger.info("Task %s added: pickup=%s drop=%s", tid, pickup, drop)

    # ------------------------------------------------------------------
    # Broadcast payload builder
    # ------------------------------------------------------------------

    def build_state_payload(self) -> dict:
        """Serialise the full fleet state for broadcast to web clients."""
        completed_count = len(self._completed)

        # Rough time-saved percentage vs naive sequential baseline (40 ticks/task)
        time_saved_pct = 0.0
        if self.tick_count > 0 and completed_count > 0:
            actual_per_task = self.tick_count / completed_count
            time_saved_pct  = max(0.0, round((1.0 - actual_per_task / 40.0) * 100.0, 1))

        robots = []
        for ag in self.agents:
            tel = ag.telemetry
            path = [[c, r] for c, r in tel.get("laser_trail", [])]
            task_label = (
                ag._current_task.task_id
                if ag._current_task is not None
                else None
            )
            robots.append({
                "id":      tel["robot_id"],
                "x":       round(tel["x"], 2),
                "y":       round(tel["y"], 2),
                "battery": round(tel["battery"], 1),
                "state":   tel["state"],
                "path":    path,
                "task":    task_label,
                "has_cargo": tel.get("has_cargo", False),
                "task_type": tel.get("task_type", "IDLE"),
            })

        return {
            "warehouse": self._warehouse_snapshot,
            "obstacles": [[x, y] for x, y in self.obstacles],
            "robots":    robots,
            "inventory": [{"pos": list(k), "count": v} for k, v in self.shelf_inventory.items()],
            "metrics": {
                "collisions":      self.total_collisions,
                "completed_tasks": completed_count,
                "time_saved_pct":  time_saved_pct,
                "total_restocked": self.total_restocked,
                "total_dispatched": self.total_dispatched,
            },
        }


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="Fleet-X AMR Dashboard", version="1.0.0")

# Mutable runtime state — set during lifespan startup
swarm: Optional[AMRSwarm] = None
_ws_clients: Set[WebSocket] = set()


# ------------------------------------------------------------------
# Background asyncio tasks
# ------------------------------------------------------------------

async def _sim_loop() -> None:
    """
    Run the AMR swarm simulation at SIM_TICK_HZ (10 Hz, 100 ms/tick).

    Uses asyncio.sleep() between ticks so the event loop stays responsive
    and WS command handling is never blocked.
    """
    assert swarm is not None
    interval = 1.0 / SIM_TICK_HZ
    logger.info("Simulation loop started at %d Hz", SIM_TICK_HZ)
    while True:
        t0 = _time.monotonic()
        swarm.tick()                              # ~1-5 ms for 4 agents
        elapsed = _time.monotonic() - t0
        await asyncio.sleep(max(0.0, interval - elapsed))


async def _broadcast_loop() -> None:
    """
    Serialise and push fleet state to every connected WebSocket client
    every 100 ms — completely decoupled from the simulation tick.
    """
    assert swarm is not None
    interval = 1.0 / BROADCAST_HZ
    logger.info("Broadcast loop started at %d Hz", BROADCAST_HZ)
    while True:
        t0 = _time.monotonic()

        if _ws_clients:
            payload_str = json.dumps(swarm.build_state_payload())
            dead: List[WebSocket] = []
            for ws in list(_ws_clients):
                try:
                    await ws.send_text(payload_str)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                _ws_clients.discard(ws)

        elapsed = _time.monotonic() - t0
        await asyncio.sleep(max(0.0, interval - elapsed))


# ------------------------------------------------------------------
# Application lifespan (startup / shutdown)
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background tasks on startup, cancel them on shutdown."""
    global swarm
    num_tasks = getattr(app.state, "num_tasks", 50)
    seed      = getattr(app.state, "seed",      SIM_SEED)

    logger.info("Initialising AMR swarm (tasks=%d, seed=%d)…", num_tasks, seed)
    swarm = AMRSwarm(num_tasks=num_tasks, seed=seed)
    logger.info(
        "Swarm ready: %d agents, %d tasks queued",
        len(swarm.agents), len(swarm._task_queue),
    )

    sim_task   = asyncio.create_task(_sim_loop(),       name="sim_loop")
    bcast_task = asyncio.create_task(_broadcast_loop(), name="broadcast_loop")

    yield  # ← application is live here

    logger.info("Shutting down background tasks…")
    sim_task.cancel()
    bcast_task.cancel()
    await asyncio.gather(sim_task, bcast_task, return_exceptions=True)

app.router.lifespan_context = lifespan


# ------------------------------------------------------------------
# WebSocket endpoint  /ws
# ------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    _ws_clients.add(ws)
    logger.info("WS client connected: %s", ws.client)

    try:
        while True:
            raw = await ws.receive_text()
            await _handle_ws_command(raw, ws)
    except WebSocketDisconnect:
        logger.info("WS client disconnected: %s", ws.client)
    except Exception as exc:
        logger.warning("WS error (%s): %s", ws.client, exc)
    finally:
        _ws_clients.discard(ws)


async def _handle_ws_command(raw: str, ws: WebSocket) -> None:
    """Dispatch a single JSON command received from the frontend."""
    assert swarm is not None
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError as exc:
        await ws.send_text(json.dumps({"error": f"Invalid JSON: {exc}"}))
        return

    action = msg.get("action", "")

    try:
        if action == "add_obstacle":
            x = int(msg["x"])
            y = int(msg["y"])
            swarm.add_obstacle(x, y)
            await ws.send_text(
                json.dumps({"ok": True, "action": action, "x": x, "y": y})
            )

        elif action == "kill_robot":
            robot_id = msg["robot_id"]
            swarm.kill_robot(robot_id)
            await ws.send_text(json.dumps({"ok": True, "action": action}))

        elif action == "spawn_robot":
            swarm.spawn_robot()
            await ws.send_text(json.dumps({"ok": True, "action": action}))

        elif action == "clear_obstacles":
            swarm.obstacles.clear()
            # Inform resolver
            swarm._resolver._static_obstacles = set(swarm.obstacles)
            await ws.send_text(json.dumps({"ok": True, "action": action}))

        elif action == "add_task":
            pickup = msg["pickup"]   # [x, y]
            drop   = msg["drop"]     # [x, y]
            swarm.add_task(pickup, drop)
            await ws.send_text(json.dumps({"ok": True, "action": action}))

        elif action == "trigger_restock":
            active_pickups = {t.pickup_pos for t in swarm._task_queue}
            active_drops = {t.drop_pos for t in swarm._task_queue}
            for ag in swarm.agents:
                if ag._current_task:
                    active_pickups.add(ag._current_task.pickup_pos)
                    active_drops.add(ag._current_task.drop_pos)
            
            for _ in range(3):
                t = swarm.tick_count
                shelves_with_capacity = [s for s, c in swarm.shelf_inventory.items() if c < 10 and s not in active_drops and s not in active_pickups]
                free_pickups = [p for p in swarm._pickups if p not in active_pickups and p not in active_drops]
                if shelves_with_capacity and free_pickups:
                    pickup = swarm._rng.choice(free_pickups)
                    drop = swarm._rng.choice(shelves_with_capacity)
                    tid = f"IN_{t:05d}_{swarm._rng.randint(0,999)}"
                    task = Task(tid, pickup, drop, 1.0, t, "INBOUND_RESTOCK")
                    swarm._task_queue.append(task)
                    active_pickups.add(pickup)
                    active_drops.add(drop)
            await ws.send_text(json.dumps({"ok": True, "action": action}))
            
        elif action == "trigger_order":
            active_pickups = {t.pickup_pos for t in swarm._task_queue}
            active_drops = {t.drop_pos for t in swarm._task_queue}
            for ag in swarm.agents:
                if ag._current_task:
                    active_pickups.add(ag._current_task.pickup_pos)
                    active_drops.add(ag._current_task.drop_pos)
            
            t = swarm.tick_count
            shelves_with_stock = [s for s, c in swarm.shelf_inventory.items() if c > 0 and s not in active_pickups and s not in active_drops]
            drops = [tuple(p) for p in swarm._warehouse_snapshot["drops"] if tuple(p) not in active_drops and tuple(p) not in active_pickups]
            if shelves_with_stock and drops:
                pickup = swarm._rng.choice(shelves_with_stock)
                drop = swarm._rng.choice(drops)
                tid = f"OUT_{t:05d}_{swarm._rng.randint(0,999)}"
                task = Task(tid, pickup, drop, 1.0, t, "OUTBOUND_FULFILLMENT")
                swarm._task_queue.append(task)
            await ws.send_text(json.dumps({"ok": True, "action": action}))

        else:
            await ws.send_text(
                json.dumps({"error": f"Unknown action: {action!r}"})
            )

    except (KeyError, ValueError, TypeError) as exc:
        await ws.send_text(
            json.dumps({"error": f"Bad payload for {action!r}: {exc}"})
        )


# ------------------------------------------------------------------
# REST endpoint — quick health-check / metrics snapshot
# ------------------------------------------------------------------

@app.get("/api/status")
async def api_status() -> JSONResponse:
    """Liveness probe + metrics snapshot (no WS required)."""
    if swarm is None:
        return JSONResponse({"status": "initialising"}, status_code=503)
    return JSONResponse({
        "status":          "ok",
        "tick":            swarm.tick_count,
        "agents":          len(swarm.agents),
        "queued_tasks":    len(swarm._task_queue),
        "completed_tasks": swarm._completed_tasks,
        "collisions":      swarm.total_collisions,
        "ws_clients":      len(_ws_clients),
    })


# ------------------------------------------------------------------
# Static file serving — tries frontend/ first, then ./ as fallback
# ------------------------------------------------------------------

def _mount_static(application: FastAPI) -> None:
    """Mount index.html root + /static asset tree."""
    frontend_dir = Path(__file__).parent / "frontend"
    fallback_dir = Path(__file__).parent
    static_root  = frontend_dir if frontend_dir.is_dir() else fallback_dir

    # Root route serves index.html
    index_candidates = [static_root / "index.html", fallback_dir / "index.html"]
    index_path = next((p for p in index_candidates if p.is_file()), None)

    if index_path:
        @application.get("/", include_in_schema=False)
        async def serve_index() -> FileResponse:
            return FileResponse(str(index_path))
    else:
        @application.get("/", include_in_schema=False)
        async def serve_index_placeholder() -> JSONResponse:
            return JSONResponse(
                {
                    "info": (
                        "No index.html found. "
                        "Place your dashboard at frontend/index.html"
                    )
                }
            )

    # Mount everything else under /static
    if static_root.is_dir():
        application.mount(
            "/static",
            StaticFiles(directory=str(static_root), html=True),
            name="static",
        )
        logger.info("Static files served from: %s", static_root)


_mount_static(app)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fleet-X AMR Dashboard — WebSocket + Static Server"
    )
    parser.add_argument(
        "--host",  default=os.getenv("WEB_HOST", "0.0.0.0"),
        help="Bind host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",  type=int, default=int(os.getenv("WEB_PORT", "8000")),
        help="Bind port (default: 8000)",
    )
    parser.add_argument(
        "--tasks", type=int, default=50,
        help="Number of simulation tasks (default: 50)",
    )
    parser.add_argument(
        "--seed",  type=int, default=SIM_SEED,
        help=f"RNG seed (default: {SIM_SEED})",
    )
    parser.add_argument(
        "--reload", action="store_true",
        help="Enable Uvicorn hot-reload (dev mode)",
    )
    args = parser.parse_args()

    # Stash sim params in app.state so lifespan can read them
    app.state.num_tasks = args.tasks
    app.state.seed      = args.seed

    logger.info(
        "Starting Fleet-X server on http://%s:%d  (tasks=%d, seed=%d)",
        args.host, args.port, args.tasks, args.seed,
    )

    uvicorn.run(
        "web_server:app",   # module:attribute string — required for --reload
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
