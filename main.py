#!/usr/bin/env python3
"""
main.py
=======
Live backend for ``frontend/index.html`` — stdlib only, no extra deps.

Serves (default ``127.0.0.1:8000``):
    GET /               -> frontend/index.html (static)
    GET /api/snapshot   -> latest fleet snapshot (JSON, CORS-enabled)
    GET /ws             -> WebSocket stream of snapshots (~5 Hz, JSON text frames)

Snapshot shape (what the HTML canvas expects):
    {
      "world": {
        "tick": int,
        "warehouse": {"width": int, "height": int,
                      "shelves": [[x, y], ...],
                      "stations": {name: [x, y]}},
        "obstacles": [[x, y], ...]
      },
      "robots": {
        "r1": {"position": [x, y], "path": [[x, y], ...],
               "status": "NAVIGATING", "heading": 1.57 (radians),
               "battery": 88.4, "task_id": "T003" | None,
               "peers": ["r2", ...], "waiting_reason": ""}, ...
      },
      "tasks": [{"id": "T003", "pickup": [x, y], "destination": [x, y]}, ...],
      "lifecycle": "running" | "completed",
      "metrics": {"tasks_completed": int, "tasks_total": int,
                  "all_completed": bool, "elapsed_simulation_seconds": float,
                  "collisions": int, "obstacle_collisions": int,
                  "deadlocks_resolved": int, "reroutes": int, "conflicts": int,
                  "completed_tasks": {task_id: tick}},
      "events": [{"tick": int, "type": str, "robot_id": str,
                  "task_id": str, "reason": str}, ...]   # latest ~50
    }

Usage:
    python main.py serve [--host 127.0.0.1 --port 8000 --tasks 50 ...]
    python main.py serve --tasks 20 --ticks 1000   # quick demo
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import random
import socket
import struct
import threading
import time as _time
from pathlib import Path
from typing import Dict, List, Optional

from core.grid_map import SHELF, PICKUP, DROP
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

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
ROOT = Path(__file__).resolve().parent
FRONTEND_INDEX = ROOT / "frontend" / "index.html"


# ---------------------------------------------------------------------------
# Fleet simulation (background thread, drives snapshots + WS messages)
# ---------------------------------------------------------------------------
class FleetSim:
    """Runs the AMR fleet tick loop and keeps the latest frontend snapshot."""

    def __init__(
        self,
        num_tasks: int = 50,
        max_ticks: int = 2000,
        num_agents: int = 4,
        seed: int = 7,
        tick_interval: float = 0.1,
    ) -> None:
        self.num_tasks = num_tasks
        self.max_ticks = max_ticks
        self.num_agents = num_agents
        self.seed = seed
        self.tick_interval = tick_interval

        rng = random.Random(seed)
        self.grid = build_warehouse_grid()
        self.all_tasks: List[Task] = generate_tasks(self.grid, n=num_tasks, seed=seed)
        self.task_queue: List[Task] = list(self.all_tasks)
        cells = _walkable_cells(self.grid)
        starts = rng.sample(cells, num_agents)

        self.sta = STATable()
        self.intent_bus: Dict[str, list] = {}
        self.hazard_bus: list = []
        self.lease_bus: list = []
        self.completed: List[Task] = []
        self.completed_tick: Dict[str, int] = {}

        # Display ids r1..rN (match frontend ROBOT_COLORS) -> internal agent.
        self.agents: List[AMRAgent] = []
        self.disp_ids: List[str] = [f"r{i + 1}" for i in range(num_agents)]
        self._agent_by_disp: Dict[str, AMRAgent] = {}
        for i, pos in enumerate(starts):
            internal_id = f"AMR_{i + 1:02d}"
            ag = AMRAgent(
                robot_id=internal_id,
                start_pos=pos,
                grid=self.grid,
                sta_table=self.sta,
                battery=rng.uniform(75.0, 100.0),
                on_intent_broadcast=self._on_intent,
                on_hazard_broadcast=self.hazard_bus.append,
                on_lease_broadcast=self.lease_bus.append,
                on_task_complete=self._on_task_complete,
            )
            self.agents.append(ag)
            self._agent_by_disp[self.disp_ids[i]] = ag

        self.tick = 0
        self.collisions = 0
        self.reroutes = 0
        self.conflicts = 0
        self.deadlocks_resolved = 0
        self.events: List[dict] = []
        self.lifecycle = "running"
        self.sim_start = _time.time()
        self._lock = threading.Lock()
        self._snapshot: Optional[dict] = None
        self._stop = threading.Event()

    # -- callbacks ------------------------------------------------------
    def _on_intent(self, sender: str, wps: list) -> None:
        self.intent_bus[sender] = wps

    def _on_task_complete(self, task: Task) -> None:
        self.completed.append(task)
        self.completed_tick[task.task_id] = self.tick
        self._push_event("TASK_COMPLETED", task_id=task.task_id)

    def _push_event(self, etype: str, robot_id: str = "",
                    task_id: str = "", reason: str = "") -> None:
        ev: dict = {"tick": self.tick, "type": etype}
        if robot_id:
            ev["robot_id"] = robot_id
        if task_id:
            ev["task_id"] = task_id
        if reason:
            ev["reason"] = reason
        self.events.append(ev)
        if len(self.events) > 100:
            self.events = self.events[-100:]

    # -- auction --------------------------------------------------------
    def _run_auction(self, task: Task) -> Optional[str]:
        idle = [ag for ag in self.agents if ag.state == AgentState.IDLE]
        if not idle:
            return None
        bids = [
            Bid(task_id=task.task_id, robot_id=ag.robot_id,
                bid_value=BidFormula.compute(
                    ag.robot_id, ag.position, task.pickup_pos, ag.battery),
                battery=ag.battery, position=ag.position)
            for ag in idle
        ]
        for ag in idle:
            my_bid = next(b for b in bids if b.robot_id == ag.robot_id)
            ag._auction._auctions[task.task_id] = _AuctionState(
                task=task, my_bid=my_bid,
                bids={b.robot_id: b for b in bids},
                close_time=_time.time() + 0.15)
            ag._auction._state = AuctionParticipantState.BIDDING
        winner = None
        for ag in idle:
            winner = ag.resolve_auction(task.task_id)
        return winner

    # -- one sim tick ---------------------------------------------------
    def step(self) -> None:
        available = [t for t in self.task_queue if t.issued_tick <= self.tick]
        for task in available:
            winner = self._run_auction(task)
            if winner:
                self.task_queue.remove(task)
                disp = self._disp_of(winner)
                self._push_event("TASK_ASSIGNED", robot_id=disp,
                                 task_id=task.task_id)

        snapshot = {ag.robot_id: ag.position for ag in self.agents}
        for ag in self.agents:
            ag.set_peer_positions(snapshot)
        for ag in self.agents:
            prev = ag.state
            ag.tick(delta_time=0.1)
            if ag.state != prev and ag.state in (
                    AgentState.YIELDING, AgentState.REVERSING,
                    AgentState.REROUTING, AgentState.EMERGENCY_STOP):
                self.conflicts += 1
                if ag.state == AgentState.REVERSING:
                    self.deadlocks_resolved += 1
                if ag.state == AgentState.REROUTING:
                    self.reroutes += 1
                self._push_event("CONFLICT_" + ag.state.value,
                                 robot_id=self._disp_of(ag.robot_id))

        for ag in self.agents:
            for sid, wps in self.intent_bus.items():
                if sid != ag.robot_id:
                    ag.on_peer_intent(sid, wps)
            for hm in self.hazard_bus:
                if hm.robot_id != ag.robot_id:
                    ag.on_peer_hazard(hm)
            for lm in self.lease_bus:
                if lm.robot_id != ag.robot_id:
                    ag.on_peer_lease(lm)
        self.hazard_bus.clear()
        self.lease_bus.clear()

        pos_map: Dict[tuple, str] = {}
        for ag in self.agents:
            if ag.position in pos_map:
                self.collisions += 1
                self._push_event("COLLISION",
                                 robot_id=self._disp_of(ag.robot_id),
                                 reason=f"at {ag.position} with "
                                        f"{self._disp_of(pos_map[ag.position])}")
            else:
                pos_map[ag.position] = ag.robot_id

        self.tick += 1
        if len(self.completed) >= self.num_tasks or self.tick >= self.max_ticks:
            self.lifecycle = "completed"
        with self._lock:
            self._snapshot = self.build_snapshot()

    def _disp_of(self, internal_id: str) -> str:
        for disp, ag in self._agent_by_disp.items():
            if ag.robot_id == internal_id:
                return disp
        return internal_id

    # -- snapshot for frontend ------------------------------------------
    def build_snapshot(self) -> dict:
        shelves, stations = [], {}
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                ct = self.grid.grid[r][c]
                if ct == SHELF and (c, r) not in getattr(
                        self.grid, "_runtime_blocked", set()):
                    shelves.append([c, r])
                elif ct == PICKUP:
                    stations.setdefault(f"pickup_{c}_{r}", [c, r])
                elif ct == DROP:
                    stations.setdefault(f"drop_{c}_{r}", [c, r])
        obstacles = [list(p) for p in sorted(
            getattr(self.grid, "_runtime_blocked", set()))]

        robots: dict = {}
        for disp in self.disp_ids:
            ag = self._agent_by_disp[disp]
            try:
                raw_path = ag._ctx.planned_path[ag._ctx.path_index:]
                path = [[int(x), int(y)] for x, y, *_ in raw_path][:60]
            except Exception:
                path = []
            try:
                task_id = (ag._current_task.task_id
                           if ag._current_task is not None else None)
            except Exception:
                task_id = None
            try:
                deg = float(ag.telemetry.get("heading", 0.0))
            except Exception:
                deg = 0.0
            peers = [d for d in self.disp_ids if d != disp]
            waiting_reason = ""
            if ag.state == AgentState.YIELDING:
                waiting_reason = "yielding to higher-priority peer"
            elif ag.state == AgentState.REVERSING:
                waiting_reason = "deadlock reversal"
            elif ag.state == AgentState.EMERGENCY_STOP:
                waiting_reason = "emergency stop"
            robots[disp] = {
                "position": [int(ag.position[0]), int(ag.position[1])],
                "path": path,
                "status": ag.state.value,
                "heading": math.radians(deg),
                "battery": round(float(ag.battery), 1),
                "task_id": task_id,
                "peers": peers,
                "waiting_reason": waiting_reason,
            }

        tasks = [{"id": t.task_id, "pickup": [int(t.pickup_pos[0]),
                                              int(t.pickup_pos[1])],
                  "destination": [int(t.drop_pos[0]), int(t.drop_pos[1])]}
                 for t in self.all_tasks]
        return {
            "world": {
                "tick": self.tick,
                "warehouse": {
                    "width": self.grid.cols, "height": self.grid.rows,
                    "shelves": shelves, "stations": stations},
                "obstacles": obstacles,
            },
            "robots": robots,
            "tasks": tasks,
            "lifecycle": self.lifecycle,
            "metrics": {
                "tasks_completed": len(self.completed),
                "tasks_total": self.num_tasks,
                "all_completed": len(self.completed) >= self.num_tasks,
                "elapsed_simulation_seconds": round(self.tick * 0.1, 1),
                "collisions": self.collisions,
                "obstacle_collisions": 0,
                "deadlocks_resolved": self.deadlocks_resolved,
                "reroutes": self.reroutes,
                "conflicts": self.conflicts,
                "completed_tasks": dict(self.completed_tick),
            },
            "events": list(self.events[-50:]),
        }

    def get_snapshot(self) -> dict:
        with self._lock:
            if self._snapshot is None:
                return self.build_snapshot()
            return self._snapshot

    # -- background loop -------------------------------------------------
    def run_loop(self) -> None:
        with self._lock:
            self._snapshot = self.build_snapshot()
        while not self._stop.is_set():
            if self.lifecycle != "completed":
                self.step()
            _time.sleep(self.tick_interval)

    def stop(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# Minimal HTTP + WebSocket server (stdlib only)
# ---------------------------------------------------------------------------
def _ws_accept_key(key: str) -> str:
    digest = hashlib.sha1((key + WS_GUID).encode()).digest()
    return base64.b64encode(digest).decode()


def _ws_encode(payload: bytes, opcode: int = 0x1) -> bytes:
    b1 = 0x80 | (opcode & 0x0F)
    n = len(payload)
    if n < 126:
        return struct.pack("!BB", b1, n) + payload
    if n < (1 << 16):
        return struct.pack("!BBH", b1, 126, n) + payload
    return struct.pack("!BBQ", b1, 127, n) + payload


def _http_response(status: str, body: bytes, ctype: str = "text/plain",
                   extra: str = "") -> bytes:
    head = (f"HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Access-Control-Allow-Origin: *\r\nConnection: close\r\n"
            f"{extra}\r\n").encode()
    return head + body


def _handle_ws(conn: socket.socket, sim: FleetSim, ws_interval: float) -> None:
    conn.settimeout(0.3)
    try:
        while True:
            snap = sim.get_snapshot()
            try:
                conn.sendall(_ws_encode(
                    json.dumps(snap, separators=(",", ":")).encode()))
            except OSError:
                break
            # Drain any client frames (close/ping) without blocking long.
            try:
                hdr = conn.recv(2)
                if hdr:
                    if len(hdr) < 2:
                        break
                    opcode = hdr[0] & 0x0F
                    masked = (hdr[1] & 0x80) != 0
                    length = hdr[1] & 0x7F
                    if length == 126:
                        ext = conn.recv(2)
                        length = struct.unpack("!H", ext)[0]
                    elif length == 127:
                        ext = conn.recv(8)
                        length = struct.unpack("!Q", ext)[0]
                    if masked:
                        conn.recv(4)  # mask
                    while length > 0:
                        chunk = conn.recv(min(length, 4096))
                        if not chunk:
                            break
                        length -= len(chunk)
                    if opcode == 0x8:  # close
                        break
                    if opcode == 0x9:  # ping -> pong
                        try:
                            conn.sendall(_ws_encode(b"", opcode=0xA))
                        except OSError:
                            break
            except socket.timeout:
                pass
            _time.sleep(ws_interval)
    finally:
        try:
            conn.close()
        except OSError:
            pass


def _handle_conn(conn: socket.socket, sim: FleetSim, ws_interval: float) -> None:
    try:
        conn.settimeout(5.0)
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
            if len(data) > 65536:
                break
        if not data:
            conn.close()
            return
        header, _, _ = data.partition(b"\r\n\r\n")
        lines = header.decode("latin-1").split("\r\n")
        method, path = lines[0].split(" ", 2)[:2]
        headers = {}
        for ln in lines[1:]:
            if ":" in ln:
                k, v = ln.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        path_only = path.split("?", 1)[0]

        if method == "OPTIONS":
            conn.sendall(_http_response(
                "204 No Content", b"",
                extra="Access-Control-Allow-Methods: GET, OPTIONS\r\n"
                      "Access-Control-Allow-Headers: *\r\n"))
            conn.close()
            return

        if path_only == "/ws":
            key = headers.get("sec-websocket-key", "")
            upgrade = headers.get("upgrade", "").lower()
            if "websocket" not in upgrade or not key:
                conn.sendall(_http_response("400 Bad Request",
                                            b"Expected WebSocket upgrade"))
                conn.close()
                return
            accept = _ws_accept_key(key)
            conn.sendall(
                ("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                 "Connection: Upgrade\r\n"
                 f"Sec-WebSocket-Accept: {accept}\r\n\r\n").encode())
            _handle_ws(conn, sim, ws_interval)
            return

        if path_only in ("/", "/index.html"):
            body = (FRONTEND_INDEX.read_bytes() if FRONTEND_INDEX.exists()
                    else b"<h1>Fleet-X live backend</h1>")
            conn.sendall(_http_response("200 OK", body, "text/html"))
            conn.close()
            return

        if path_only == "/api/snapshot":
            body = json.dumps(sim.get_snapshot(),
                              separators=(",", ":")).encode()
            conn.sendall(_http_response(
                "200 OK", body, "application/json",
                extra="Cache-Control: no-store\r\n"))
            conn.close()
            return

        conn.sendall(_http_response("404 Not Found", b"not found"))
        conn.close()
    except Exception:
        try:
            conn.close()
        except OSError:
            pass


def serve(sim: FleetSim, host: str, port: int, ws_interval: float) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(50)
    print(f"Fleet-X live backend on http://{host}:{port}  "
          f"(WS /ws @ ~{1.0 / ws_interval:.1f} Hz, snapshot /api/snapshot)")
    try:
        while True:
            conn, _ = srv.accept()
            threading.Thread(target=_handle_conn,
                             args=(conn, sim, ws_interval),
                             daemon=True).start()
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Fleet-X live backend for frontend")
    ap.add_argument("command", nargs="?", default="serve",
                    choices=["serve"], help="serve live snapshots + WS")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tasks", type=int, default=50)
    ap.add_argument("--ticks", type=int, default=2000)
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tick-interval", type=float, default=0.1)
    ap.add_argument("--ws-interval", type=float, default=0.2)
    args = ap.parse_args()

    sim = FleetSim(num_tasks=args.tasks, max_ticks=args.ticks,
                   num_agents=args.agents, seed=args.seed,
                   tick_interval=args.tick_interval)
    threading.Thread(target=sim.run_loop, daemon=True).start()
    try:
        serve(sim, args.host, args.port, args.ws_interval)
    finally:
        sim.stop()


if __name__ == "__main__":
    main()
