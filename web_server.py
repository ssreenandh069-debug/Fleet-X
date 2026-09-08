"""
web_server.py — Fleet-X AMR Swarm & Digital Twin Server
=========================================================
Industrial-grade Autonomous Mobile Robot (AMR) warehouse swarm controller.
Features:
- Infinite, auto-replenishing mission generator (Inbound Restock & Outbound Orders)
- Multi-tier dynamic collision & deadlock resolution
- Strict shelf avoidance & dynamic obstacle routing
- Full WebSocket Digital Twin telemetry broadcast (10 Hz)
- Dynamic fleet sizing: spawn, kill, recall, restock, order, barrier commands
"""

from __future__ import annotations

import argparse
import asyncio
import heapq
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

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse, JSONResponse
    import uvicorn
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\n  pip install fastapi uvicorn[standard]")

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
# Simulation constants
# ---------------------------------------------------------------------------
SIM_NUM_AGENTS = 4
SIM_SEED       = 7
SIM_TICK_HZ    = 10       # simulation ticks per second
BROADCAST_HZ   = 10       # telemetry broadcasts per second

# ---------------------------------------------------------------------------
# Grid definition — matches frontend index.html layout exactly
#   GRID_W=28, GRID_H=20
# ---------------------------------------------------------------------------
GRID_W, GRID_H = 28, 20

# Cell types
WALKWAY, SHELF, PICKUP, DROP, CHARGER = 0, 1, 2, 3, 4

# Charger bays along top row
CHARGER_CELLS: List[Tuple[int,int]] = [
    (2, 0), (5, 0), (8, 0), (14, 0), (20, 0), (25, 0)
]

def build_grid() -> List[List[int]]:
    """Build 28x20 grid matching frontend layout."""
    grid = [[WALKWAY] * GRID_W for _ in range(GRID_H)]

    # Shelf pods — matches SHELF_PODS in index.html
    shelf_pods = [
        (2, 6,  3, 4),   (10,14, 3, 4),   (17,21, 3, 4),
        (2, 6,  7, 8),   (10,14, 7, 8),   (17,21, 7, 8),
        (2, 6,  11,12),  (10,14, 11,12),  (17,21, 11,12),
        (4, 9,  15,16),  (13,18, 15,16),
    ]
    for (c0, c1, r0, r1) in shelf_pods:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if 0 <= r < GRID_H and 0 <= c < GRID_W:
                    grid[r][c] = SHELF

    # Pickup bays — matches PICKUP_BAYS in index.html
    for (col, row) in [(1,18),(2,18),(13,18),(14,18)]:
        grid[row][col] = PICKUP

    # Drop bays — matches DROP_BAYS in index.html
    for (col, row) in [(27,4),(27,8),(27,12),(27,16)]:
        grid[row][col] = DROP

    # Chargers
    for (col, row) in CHARGER_CELLS:
        grid[row][col] = CHARGER

    return grid


GRID = build_grid()
WALKABLE: Set[Tuple[int,int]] = {
    (c, r)
    for r in range(GRID_H)
    for c in range(GRID_W)
    if GRID[r][c] != SHELF
}
PICKUP_BAYS:  List[Tuple[int,int]] = [(c, r) for r in range(GRID_H) for c in range(GRID_W) if GRID[r][c] == PICKUP]
DROP_BAYS:    List[Tuple[int,int]] = [(c, r) for r in range(GRID_H) for c in range(GRID_W) if GRID[r][c] == DROP]
SHELF_CELLS:  List[Tuple[int,int]] = [(c, r) for r in range(GRID_H) for c in range(GRID_W) if GRID[r][c] == SHELF]

# Staging access points for each shelf cell
SHELF_ACCESS: Dict[Tuple[int,int], Tuple[int,int]] = {}
for _sc in SHELF_CELLS:
    for _dc, _dr in [(0,-1),(0,1),(-1,0),(1,0)]:
        _nb = (_sc[0]+_dc, _sc[1]+_dr)
        if _nb in WALKABLE:
            SHELF_ACCESS[_sc] = _nb
            break
ACCESSIBLE_SHELVES = [s for s in SHELF_CELLS if s in SHELF_ACCESS]


# ---------------------------------------------------------------------------
# Simple Fast A* pathfinder
# ---------------------------------------------------------------------------
def astar(
    start: Tuple[int,int],
    goal:  Tuple[int,int],
    extra_blocked: Optional[Set[Tuple[int,int]]] = None,
) -> Optional[List[Tuple[int,int]]]:
    """Return shortest walkable path from start to goal (inclusive) or None."""
    if start == goal:
        return [start]
    blocked = extra_blocked or set()

    def h(pos: Tuple[int,int]) -> int:
        return abs(pos[0]-goal[0]) + abs(pos[1]-goal[1])

    heap = [(h(start), 0, start)]
    came_from: Dict[Tuple[int,int], Tuple[int,int]] = {}
    g_score:   Dict[Tuple[int,int], int] = {start: 0}
    visited:   Set[Tuple[int,int]] = set()

    while heap:
        f, g, pos = heapq.heappop(heap)
        if pos in visited:
            continue
        visited.add(pos)

        if pos == goal:
            path = []
            node: Optional[Tuple[int,int]] = goal
            while node is not None:
                path.append(node)
                node = came_from.get(node)
            path.reverse()
            return path

        for dc, dr in [(0,-1),(0,1),(-1,0),(1,0)]:
            nb = (pos[0]+dc, pos[1]+dr)
            if nb not in WALKABLE or nb in blocked or nb in visited:
                continue
            ng = g + 1
            if ng < g_score.get(nb, 10**9):
                g_score[nb] = ng
                came_from[nb] = pos
                heapq.heappush(heap, (ng + h(nb), ng, nb))

    return None


# ---------------------------------------------------------------------------
# AMR Agent
# ---------------------------------------------------------------------------
class SimpleAMR:
    DWELL_LOAD   = 8    # ticks (0.8s at 10Hz)
    DWELL_UNLOAD = 8    # ticks (0.8s at 10Hz)

    def __init__(self, robot_id: str, start: Tuple[int,int], battery: float) -> None:
        self.robot_id    = robot_id
        self.pos         = start
        self.battery     = battery
        self.state       = "IDLE"   # IDLE, NAVIGATING_PICKUP, LOADING, NAVIGATING_DROP, UNLOADING, RETURNING, DOCKING
        self.path:       List[Tuple[int,int]] = []
        self.task:       Optional[dict] = None
        self.has_cargo   = False
        self.dwell_ticks = 0
        self.wait_ticks  = 0
        self.charger_cell: Optional[Tuple[int,int]] = None

    @property
    def telemetry(self) -> dict:
        return {
            "id":        self.robot_id,
            "x":         float(self.pos[0]),
            "y":         float(self.pos[1]),
            "battery":   round(self.battery, 1),
            "state":     self.state,
            "path":      [[c, r] for c, r in self.path],
            "task":      self.task["id"] if self.task else None,
            "has_cargo": self.has_cargo,
            "task_type": self.task["type"] if self.task else "IDLE",
        }


# ---------------------------------------------------------------------------
# Swarm Controller
# ---------------------------------------------------------------------------
class SimpleSwarm:

    def __init__(self, seed: int = SIM_SEED, num_agents: int = SIM_NUM_AGENTS) -> None:
        self._rng       = random.Random(seed)
        self.agents:    List[SimpleAMR] = []
        self.obstacles: Set[Tuple[int,int]] = set()

        self.shelf_inventory: Dict[Tuple[int,int], int] = {
            s: self._rng.randint(3, 7) for s in ACCESSIBLE_SHELVES
        }

        self.task_queue:       List[dict] = []
        self.tick_count:       int = 0
        self.total_restocked:  int = 0
        self.total_dispatched: int = 0
        self.total_collisions: int = 0
        self._task_serial:     int = 0
        self._robot_serial:    int = 0

        # Spawn initial fleet with sequential IDs
        free = list(WALKABLE - set(PICKUP_BAYS) - set(DROP_BAYS) - set(CHARGER_CELLS))
        self._rng.shuffle(free)
        for i in range(min(num_agents, len(free))):
            self._robot_serial += 1
            pos = free[i]
            aid = f"AMR_{self._robot_serial:02d}"
            bat = self._rng.uniform(80, 100)
            self.agents.append(SimpleAMR(aid, pos, bat))
            logger.info("Spawned %s at %s", aid, pos)

    # ------------------------------------------------------------------ #
    #  Task Generation                                                   #
    # ------------------------------------------------------------------ #
    def _gen_task_id(self, prefix: str) -> str:
        self._task_serial += 1
        return f"{prefix}_{self._task_serial:05d}"

    def _occupied_pickups(self) -> Set[Tuple[int,int]]:
        s: Set[Tuple[int,int]] = set()
        for t in self.task_queue:
            s.add(t["pickup"])
        for a in self.agents:
            if a.task and a.state in ("NAVIGATING_PICKUP", "LOADING"):
                s.add(a.task["pickup"])
        return s

    def _occupied_drops(self) -> Set[Tuple[int,int]]:
        s: Set[Tuple[int,int]] = set()
        for t in self.task_queue:
            s.add(t["drop"])
        for a in self.agents:
            if a.task and a.state in ("NAVIGATING_DROP", "UNLOADING"):
                s.add(a.task["drop"])
        return s

    def _try_gen_inbound(self) -> bool:
        ocp = self._occupied_pickups()
        ocd = self._occupied_drops()
        shelves = [s for s in ACCESSIBLE_SHELVES
                   if self.shelf_inventory.get(s, 0) < 10
                   and SHELF_ACCESS[s] not in ocd]
        free_p = [p for p in PICKUP_BAYS if p not in ocp]
        if not shelves or not free_p:
            return False
        shelf = self._rng.choice(shelves)
        pickup = self._rng.choice(free_p)
        drop   = SHELF_ACCESS[shelf]
        self.task_queue.append({
            "id":     self._gen_task_id("IN"),
            "type":   "INBOUND_RESTOCK",
            "pickup": pickup,
            "drop":   drop,
            "shelf":  shelf,
        })
        return True

    def _try_gen_outbound(self) -> bool:
        ocp = self._occupied_pickups()
        ocd = self._occupied_drops()
        shelves = [s for s in ACCESSIBLE_SHELVES
                   if self.shelf_inventory.get(s, 0) > 0
                   and SHELF_ACCESS[s] not in ocp]
        free_d = [d for d in DROP_BAYS if d not in ocd]
        if not shelves or not free_d:
            return False
        shelf  = self._rng.choice(shelves)
        pickup = SHELF_ACCESS[shelf]
        drop   = self._rng.choice(free_d)
        self.task_queue.append({
            "id":     self._gen_task_id("OUT"),
            "type":   "OUTBOUND_FULFILLMENT",
            "pickup": pickup,
            "drop":   drop,
            "shelf":  shelf,
        })
        return True

    def _replenish_queue(self) -> None:
        target_tasks = max(len(self.agents) * 2, 8)
        while len(self.task_queue) < target_tasks:
            added = False
            if self._rng.random() < 0.5:
                added = self._try_gen_inbound() or self._try_gen_outbound()
            else:
                added = self._try_gen_outbound() or self._try_gen_inbound()
            if not added:
                break

    def add_inbound_task(self) -> None:
        for _ in range(3):
            self._try_gen_inbound()

    def add_outbound_task(self) -> None:
        for _ in range(3):
            self._try_gen_outbound()

    # ------------------------------------------------------------------ #
    #  Tick & Dynamic Collision / Deadlock Avoidance                      #
    # ------------------------------------------------------------------ #
    def tick(self) -> None:
        self._replenish_queue()

        # Current robot positions
        positions = {a.pos for a in self.agents}

        # 1. Assign tasks to IDLE agents
        for agent in self.agents:
            if agent.state == "IDLE":
                # Check battery
                if agent.battery < 25.0:
                    # Low battery -> Head to charger
                    free_chargers = [
                        ch for ch in CHARGER_CELLS
                        if ch not in positions
                    ]
                    if free_chargers:
                        target_ch = min(free_chargers, key=lambda c: abs(c[0]-agent.pos[0])+abs(c[1]-agent.pos[1]))
                        blocked = (positions - {agent.pos}) | self.obstacles
                        p = astar(agent.pos, target_ch, blocked)
                        if p:
                            agent.path = p[1:]
                            agent.charger_cell = target_ch
                            agent.state = "RETURNING"
                            continue

                if self.task_queue:
                    task = self.task_queue.pop(0)
                    blocked = (positions - {agent.pos}) | self.obstacles
                    path = astar(agent.pos, task["pickup"], blocked)
                    if path:
                        agent.task  = task
                        agent.path  = path[1:]
                        agent.state = "NAVIGATING_PICKUP"
                        agent.wait_ticks = 0
                    else:
                        # Re-queue task
                        self.task_queue.append(task)

        # 2. Process Dwell & State transitions
        for agent in self.agents:
            if agent.state == "LOADING":
                agent.dwell_ticks -= 1
                if agent.dwell_ticks <= 0:
                    agent.has_cargo = True
                    blocked = (positions - {agent.pos}) | self.obstacles
                    path = astar(agent.pos, agent.task["drop"], blocked)
                    if path:
                        agent.path  = path[1:]
                        agent.state = "NAVIGATING_DROP"
                        agent.wait_ticks = 0
                    else:
                        # Retry next tick
                        agent.dwell_ticks = 1

            elif agent.state == "UNLOADING":
                agent.dwell_ticks -= 1
                if agent.dwell_ticks <= 0:
                    agent.has_cargo = False
                    if agent.task:
                        shelf = agent.task["shelf"]
                        if agent.task["type"] == "INBOUND_RESTOCK":
                            self.shelf_inventory[shelf] = min(10, self.shelf_inventory.get(shelf, 0) + 1)
                            self.total_restocked += 1
                        elif agent.task["type"] == "OUTBOUND_FULFILLMENT":
                            self.shelf_inventory[shelf] = max(0, self.shelf_inventory.get(shelf, 0) - 1)
                            self.total_dispatched += 1
                    agent.task  = None
                    agent.state = "IDLE"
                    agent.wait_ticks = 0

            elif agent.state == "DOCKING":
                agent.battery = min(100.0, agent.battery + 0.8)
                if agent.battery >= 99.0:
                    agent.state = "IDLE"
                    agent.charger_cell = None

        # 3. Dynamic Conflict-Free Step Resolution
        # Gather desired steps
        desired: Dict[str, Tuple[int,int]] = {}
        for agent in self.agents:
            if agent.state in ("NAVIGATING_PICKUP", "NAVIGATING_DROP", "RETURNING") and agent.path:
                desired[agent.robot_id] = agent.path[0]
            else:
                desired[agent.robot_id] = agent.pos

        # Check for destination conflicts and head-on swaps
        curr_map = {a.robot_id: a.pos for a in self.agents}
        final_pos: Dict[str, Tuple[int,int]] = {}

        # Sort agents by priority (cargo holders move first, then higher wait ticks, then ID)
        sorted_agents = sorted(
            self.agents,
            key=lambda a: (a.has_cargo, a.wait_ticks, a.robot_id),
            reverse=True,
        )

        reserved_next: Set[Tuple[int,int]] = set()

        for agent in sorted_agents:
            aid = agent.robot_id
            target = desired[aid]

            # Head-on collision check: If another agent wants our current cell AND we want its current cell
            head_on = False
            for other_id, other_pos in curr_map.items():
                if other_id != aid and target == other_pos and desired.get(other_id) == curr_map[aid]:
                    head_on = True
                    break

            # Valid step if: target is free from other reservations and not an obstacle and not a head-on swap
            if target != curr_map[aid]:
                other_curr = {p for i, p in curr_map.items() if i != aid}
                if (target not in reserved_next and 
                    target not in self.obstacles and 
                    target in WALKABLE and
                    not head_on and
                    target not in other_curr):
                    # Step accepted
                    final_pos[aid] = target
                    reserved_next.add(target)
                    agent.pos = agent.path.pop(0)
                    agent.wait_ticks = 0
                else:
                    # Must wait this tick
                    final_pos[aid] = curr_map[aid]
                    reserved_next.add(curr_map[aid])
                    agent.wait_ticks += 1
            else:
                final_pos[aid] = curr_map[aid]
                reserved_next.add(curr_map[aid])

        # 4. Multi-Tier Deadlock Resolution
        for agent in self.agents:
            if agent.wait_ticks >= 2 and agent.state in ("NAVIGATING_PICKUP", "NAVIGATING_DROP", "RETURNING"):
                # Tier 1: Re-plan A* around the blocker
                goal = None
                if agent.task:
                    goal = agent.task["pickup"] if agent.state == "NAVIGATING_PICKUP" else agent.task["drop"]
                elif agent.state == "RETURNING" and agent.charger_cell:
                    goal = agent.charger_cell
                elif agent.path:
                    goal = agent.path[-1]

                if goal:
                    other_pos = {a.pos for a in self.agents if a is not agent}
                    blocked = other_pos | self.obstacles
                    new_p = astar(agent.pos, goal, blocked)
                    if new_p and len(new_p) > 1:
                        agent.path = new_p[1:]
                        agent.wait_ticks = 0
                        continue

            if agent.wait_ticks >= 4 and agent.state in ("NAVIGATING_PICKUP", "NAVIGATING_DROP"):
                # Tier 2: Cooperative Sidestep into adjacent free cell to clear lane
                all_occ = {a.pos for a in self.agents} | self.obstacles
                free_neighbors = [
                    (agent.pos[0]+dc, agent.pos[1]+dr)
                    for dc, dr in [(0,1), (0,-1), (1,0), (-1,0)]
                    if (agent.pos[0]+dc, agent.pos[1]+dr) in WALKABLE
                    and (agent.pos[0]+dc, agent.pos[1]+dr) not in all_occ
                ]
                if free_neighbors:
                    sidestep = self._rng.choice(free_neighbors)
                    agent.pos = sidestep
                    agent.wait_ticks = 0
                    if agent.task:
                        targ = agent.task["pickup"] if agent.state == "NAVIGATING_PICKUP" else agent.task["drop"]
                        np = astar(agent.pos, targ, {a.pos for a in self.agents if a is not agent} | self.obstacles)
                        agent.path = np[1:] if np else []
                    continue

            if agent.wait_ticks >= 8 and agent.state == "NAVIGATING_PICKUP":
                # Tier 3: Release task back to queue to eliminate bottleneck
                if agent.task:
                    self.task_queue.append(agent.task)
                    agent.task = None
                agent.path = []
                agent.state = "IDLE"
                agent.wait_ticks = 0

            # Check arrival at destination
            if not agent.path:
                if agent.state == "NAVIGATING_PICKUP":
                    agent.dwell_ticks = SimpleAMR.DWELL_LOAD
                    agent.state = "LOADING"
                elif agent.state == "NAVIGATING_DROP":
                    agent.dwell_ticks = SimpleAMR.DWELL_UNLOAD
                    agent.state = "UNLOADING"
                elif agent.state == "RETURNING":
                    agent.state = "DOCKING"

            # Battery management
            if agent.state == "DOCKING":
                pass
            elif agent.state == "IDLE":
                agent.battery = min(100.0, agent.battery + 0.1)
            else:
                agent.battery = max(10.0, agent.battery - 0.015)

        # Collision verification
        seen: Dict[Tuple[int,int], str] = {}
        for a in self.agents:
            if a.pos in seen:
                self.total_collisions += 1
            else:
                seen[a.pos] = a.robot_id

        self.tick_count += 1

    # ------------------------------------------------------------------ #
    #  Swarm Fleet Commands                                              #
    # ------------------------------------------------------------------ #
    def spawn_robot(self) -> Optional[SimpleAMR]:
        occupied = {a.pos for a in self.agents}
        free = [c for c in WALKABLE if c not in occupied and c not in self.obstacles]
        if not free:
            logger.warning("spawn_robot: no free walkable cells")
            return None
        pos = self._rng.choice(free)
        self._robot_serial += 1
        aid = f"AMR_{self._robot_serial:02d}"
        ag = SimpleAMR(aid, pos, self._rng.uniform(85, 100))
        self.agents.append(ag)
        logger.info("Spawned %s at %s (Total swarm: %d)", aid, pos, len(self.agents))
        return ag

    def kill_robot(self, robot_id) -> None:
        rid = str(robot_id).strip()
        num_str = "".join(c for c in rid if c.isdigit())
        num_val = int(num_str) if num_str else None
        before = len(self.agents)
        self.agents = [
            a for a in self.agents
            if a.robot_id != rid
            and a.robot_id != f"AMR_{rid}"
            and (num_val is None or a.robot_id != f"AMR_{num_val:02d}")
        ]
        logger.info("kill_robot: %s (fleet: %d -> %d)", robot_id, before, len(self.agents))

    def recall_all(self) -> None:
        chargers = list(CHARGER_CELLS)
        self.task_queue.clear()
        for idx, a in enumerate(self.agents):
            if a.task:
                a.has_cargo = False
                a.task = None
            ch = chargers[idx % len(chargers)]
            a.state = "RETURNING"
            a.charger_cell = ch
            blocked = {o.pos for o in self.agents if o is not a} | self.obstacles
            p = astar(a.pos, ch, blocked)
            a.path = p[1:] if p else []
        logger.info("Recall ordered for %d agents", len(self.agents))

    def add_obstacle(self, x: int, y: int) -> None:
        self.obstacles.add((x, y))
        for a in self.agents:
            if (x, y) in a.path:
                a.path = []
                if a.task:
                    targ = a.task["pickup"] if a.state == "NAVIGATING_PICKUP" else a.task["drop"]
                    blocked = {o.pos for o in self.agents if o is not a} | self.obstacles
                    np = astar(a.pos, targ, blocked)
                    if np:
                        a.path = np[1:]
                    else:
                        a.state = "IDLE"
                        a.has_cargo = False
                        a.task = None

    def remove_obstacle(self, x: int, y: int) -> None:
        self.obstacles.discard((x, y))

    def clear_obstacles(self) -> None:
        self.obstacles.clear()

    # ------------------------------------------------------------------ #
    #  Payload                                                           #
    # ------------------------------------------------------------------ #
    def build_state_payload(self) -> dict:
        completed = self.total_restocked + self.total_dispatched
        time_saved = 0.0
        if self.tick_count > 0 and completed > 0:
            actual_per = self.tick_count / completed
            time_saved = max(0.0, round((1.0 - actual_per / 40.0) * 100.0, 1))

        robots = [a.telemetry for a in self.agents]

        inventory = [
            {"pos": list(k), "count": v}
            for k, v in self.shelf_inventory.items()
        ]

        warehouse = {
            "width":    GRID_W,
            "height":   GRID_H,
            "shelves":  [[c, r] for (c, r) in SHELF_CELLS],
            "pickups":  [[c, r] for (c, r) in PICKUP_BAYS],
            "drops":    [[c, r] for (c, r) in DROP_BAYS],
            "chargers": [[c, r] for (c, r) in CHARGER_CELLS],
        }

        return {
            "tick":      self.tick_count,
            "warehouse": warehouse,
            "obstacles": [[x, y] for (x, y) in self.obstacles],
            "robots":    robots,
            "inventory": inventory,
            "metrics": {
                "collisions":      self.total_collisions,
                "completed_tasks": completed,
                "total_restocked": self.total_restocked,
                "total_dispatched": self.total_dispatched,
                "time_saved_pct":  time_saved,
            },
        }


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(title="Fleet-X AMR Swarm Controller", version="3.0.0")

swarm: Optional[SimpleSwarm] = None
_ws_clients: Set[WebSocket] = set()


# ---------------------------------------------------------------------------
# Simulation & Broadcast Background Loops
# ---------------------------------------------------------------------------
async def _sim_loop() -> None:
    assert swarm is not None
    interval = 1.0 / SIM_TICK_HZ
    logger.info("Simulation loop active at %d Hz", SIM_TICK_HZ)
    while True:
        t0 = _time.monotonic()
        try:
            swarm.tick()
        except Exception as exc:
            logger.error("sim_loop error: %s", exc, exc_info=True)
        elapsed = _time.monotonic() - t0
        await asyncio.sleep(max(0.0, interval - elapsed))


async def _broadcast_loop() -> None:
    assert swarm is not None
    interval = 1.0 / BROADCAST_HZ
    logger.info("Broadcast loop active at %d Hz", BROADCAST_HZ)
    while True:
        t0 = _time.monotonic()
        if _ws_clients:
            try:
                payload = json.dumps(swarm.build_state_payload())
                dead: List[WebSocket] = []
                for ws in list(_ws_clients):
                    try:
                        await ws.send_text(payload)
                    except Exception:
                        dead.append(ws)
                for ws in dead:
                    _ws_clients.discard(ws)
            except Exception as exc:
                logger.error("broadcast_loop error: %s", exc)
        elapsed = _time.monotonic() - t0
        await asyncio.sleep(max(0.0, interval - elapsed))


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global swarm
    seed = getattr(app.state, "seed", SIM_SEED)
    logger.info("Initialising SimpleSwarm (seed=%d)…", seed)
    swarm = SimpleSwarm(seed=seed)
    logger.info("Swarm ready: %d agents", len(swarm.agents))

    sim_task   = asyncio.create_task(_sim_loop(),       name="sim_loop")
    bcast_task = asyncio.create_task(_broadcast_loop(), name="broadcast_loop")
    yield
    logger.info("Shutting down Fleet-X Swarm…")
    sim_task.cancel()
    bcast_task.cancel()
    await asyncio.gather(sim_task, bcast_task, return_exceptions=True)


app.router.lifespan_context = lifespan


# ---------------------------------------------------------------------------
# WebSocket Endpoint
# ---------------------------------------------------------------------------
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
        pass
    except Exception as exc:
        logger.warning("WS error: %s", exc)
    finally:
        _ws_clients.discard(ws)
        logger.info("WS client disconnected: %s", ws.client)


async def _handle_ws_command(raw: str, ws: WebSocket) -> None:
    assert swarm is not None
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError as exc:
        await ws.send_text(json.dumps({"error": f"Invalid JSON: {exc}"}))
        return

    action = msg.get("action", "")
    try:
        if action == "spawn_robot":
            swarm.spawn_robot()
            await ws.send_text(json.dumps({"ok": True, "action": action}))

        elif action == "kill_robot":
            swarm.kill_robot(msg["robot_id"])
            await ws.send_text(json.dumps({"ok": True, "action": action}))

        elif action == "recall_all":
            swarm.recall_all()
            await ws.send_text(json.dumps({"ok": True, "action": action}))

        elif action == "add_obstacle":
            swarm.add_obstacle(int(msg["x"]), int(msg["y"]))
            await ws.send_text(json.dumps({"ok": True, "action": action}))

        elif action == "remove_obstacle":
            swarm.remove_obstacle(int(msg["x"]), int(msg["y"]))
            await ws.send_text(json.dumps({"ok": True, "action": action}))

        elif action == "clear_obstacles":
            swarm.clear_obstacles()
            await ws.send_text(json.dumps({"ok": True, "action": action}))

        elif action == "trigger_restock":
            swarm.add_inbound_task()
            await ws.send_text(json.dumps({"ok": True, "action": action}))

        elif action == "trigger_order":
            swarm.add_outbound_task()
            await ws.send_text(json.dumps({"ok": True, "action": action}))

        elif action == "add_task":
            pickup = tuple(msg["pickup"])
            drop   = tuple(msg["drop"])
            swarm.task_queue.append({
                "id":     swarm._gen_task_id("WEB"),
                "type":   "OUTBOUND_FULFILLMENT",
                "pickup": pickup,
                "drop":   drop,
                "shelf":  drop,
            })
            await ws.send_text(json.dumps({"ok": True, "action": action}))

        else:
            await ws.send_text(json.dumps({"error": f"Unknown action: {action!r}"}))

    except (KeyError, ValueError, TypeError) as exc:
        await ws.send_text(json.dumps({"error": f"Bad payload for {action!r}: {exc}"}))


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/status")
async def api_status() -> JSONResponse:
    if swarm is None:
        return JSONResponse({"status": "initialising"}, status_code=503)
    return JSONResponse({
        "status":          "ok",
        "tick":            swarm.tick_count,
        "agents":          len(swarm.agents),
        "queued_tasks":    len(swarm.task_queue),
        "completed_tasks": swarm.total_restocked + swarm.total_dispatched,
        "collisions":      swarm.total_collisions,
        "ws_clients":      len(_ws_clients),
    })


# ---------------------------------------------------------------------------
# Static File Serving
# ---------------------------------------------------------------------------
def _mount_static(application: FastAPI) -> None:
    frontend_dir = Path(__file__).parent / "frontend"
    fallback_dir = Path(__file__).parent
    static_root  = frontend_dir if frontend_dir.is_dir() else fallback_dir

    index_path = next(
        (p for p in [static_root / "index.html", fallback_dir / "index.html"] if p.is_file()),
        None,
    )

    if index_path:
        @application.get("/", include_in_schema=False)
        async def serve_index() -> FileResponse:
            return FileResponse(str(index_path))
    else:
        @application.get("/", include_in_schema=False)
        async def serve_index_placeholder() -> JSONResponse:
            return JSONResponse({"info": "No index.html found"})

    if static_root.is_dir():
        application.mount("/static", StaticFiles(directory=str(static_root), html=True), name="static")
        logger.info("Static files served from: %s", static_root)


_mount_static(app)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Fleet-X AMR Dashboard")
    parser.add_argument("--host",   default=os.getenv("WEB_HOST", "0.0.0.0"))
    parser.add_argument("--port",   type=int, default=int(os.getenv("WEB_PORT", "8000")))
    parser.add_argument("--seed",   type=int, default=SIM_SEED)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    app.state.seed = args.seed
    logger.info("Starting Fleet-X Swarm Server on http://%s:%d", args.host, args.port)

    uvicorn.run(
        "web_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
        ws_ping_interval=None,   # Disable keepalive pings to avoid timeout disconnects
        ws_ping_timeout=None,
    )


if __name__ == "__main__":
    main()
