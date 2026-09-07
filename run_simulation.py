#!/usr/bin/env python3
"""
run_simulation.py
==================
Real-time 4-AMR simulation driver with Godot UDP telemetry and BEL benchmark.

Usage
-----
    python run_simulation.py               # 4-AMR sim (50 tasks) + benchmark
    python run_simulation.py --sim-only    # Skip benchmark
    python run_simulation.py --bench-only  # Skip real-time sim
    python run_simulation.py --tasks 20    # Fewer tasks for quick demo
    python run_simulation.py --ticks 500   # Shorter sim run

Output
------
  * Real-time per-tick status table (10 Hz, compressed to 100ms/line)
  * GodotBridge: JSON datagrams to 127.0.0.1:4242 (silently skipped if
    Godot isn't running)
  * BEL benchmark table with success-criteria assertions
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
import time as _time
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# All modules
# ---------------------------------------------------------------------------
from core.grid_map import GridMap, WALKWAY as W, SHELF as S
from core.space_time_astar import ReservationTable as STATable

from agent.amr_agent import AMRAgent, AgentState
from telemetry.godot_bridge import GodotBridge
from benchmark.runner import (
    BenchmarkRunner,
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
    level=logging.WARNING,
    format="%(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("sim")

# ---------------------------------------------------------------------------
# Colour helpers (ANSI — skipped on Windows if not supported)
# ---------------------------------------------------------------------------
try:
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(  # type: ignore[attr-defined]
        ctypes.windll.kernel32.GetStdHandle(-11), 7  # type: ignore[attr-defined]
    )
    _COLOUR = True
except Exception:
    _COLOUR = False

_C = {
    "idle":   "\033[90m",
    "nav":    "\033[32m",
    "bid":    "\033[33m",
    "plan":   "\033[36m",
    "stop":   "\033[31m",
    "yield":  "\033[35m",
    "rev":    "\033[34m",
    "reroute":"\033[33m",
    "reset":  "\033[0m",
}

def _coloured(state: str, text: str) -> str:
    if not _COLOUR:
        return text
    key = state.lower().replace("_", "").replace("emergency", "stop")[:6]
    c   = _C.get(key, "")
    return f"{c}{text}{_C['reset']}"


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
SIM_GRID_COLS = 20
SIM_GRID_ROWS = 20
SIM_NUM_AGENTS = 4
SIM_SEED       = 7

STATE_ABBREV = {
    "IDLE":           "IDLE  ",
    "BIDDING":        "BID   ",
    "PLANNING":       "PLAN  ",
    "NAVIGATING":     "NAV   ",
    "YIELDING":       "YIELD ",
    "REVERSING":      "REV   ",
    "REROUTING":      "REROUTE",
    "EMERGENCY_STOP": "E-STOP",
}


# ---------------------------------------------------------------------------
# Auction helper (shared with benchmark/runner.py logic)
# ---------------------------------------------------------------------------
def _run_auction(agents: List[AMRAgent], task: Task) -> Optional[str]:
    """Run one auction round across all idle agents. Returns winner id."""
    idle = [ag for ag in agents if ag.state == AgentState.IDLE]
    if not idle:
        return None
    bids = [
        Bid(
            task_id=task.task_id,
            robot_id=ag.robot_id,
            bid_value=BidFormula.compute(ag.robot_id, ag.position, task.pickup_pos, ag.battery),
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


# ---------------------------------------------------------------------------
# Real-time 4-AMR simulation
# ---------------------------------------------------------------------------
def run_realtime_sim(
    num_tasks: int = 50,
    max_ticks: int = 2000,
    godot_enabled: bool = True,
) -> Dict:
    """
    Run the 4-AMR real-time simulation and return summary metrics.
    """
    rng   = random.Random(SIM_SEED)
    grid  = build_warehouse_grid()
    tasks = generate_tasks(grid, n=num_tasks, seed=SIM_SEED)
    cells = _walkable_cells(grid)
    starts = rng.sample(cells, SIM_NUM_AGENTS)

    # Shared reservation table
    sta = STATable()

    # Intent bus
    intent_bus: Dict[str, list]  = {}
    hazard_bus: list             = []
    lease_bus:  list             = []
    completed:  List[Task]       = []

    def _on_intent(sender: str, wps: list) -> None:
        intent_bus[sender] = wps

    agents: List[AMRAgent] = []
    for i, pos in enumerate(starts):
        aid = f"AMR_{i+1:02d}"
        ag  = AMRAgent(
            robot_id=aid,
            start_pos=pos,
            grid=grid,
            sta_table=sta,
            battery=rng.uniform(75.0, 100.0),
            on_intent_broadcast=_on_intent,
            on_hazard_broadcast=hazard_bus.append,
            on_lease_broadcast=lease_bus.append,
            on_task_complete=completed.append,
        )
        agents.append(ag)

    task_queue  = list(tasks)
    total_collision = 0
    idle_ticks  = 0

    # ── Header ──────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  Real-Time 4-AMR Simulation   (Ctrl+C to stop early)")
    print("=" * 72)
    hdr = f"  {'Tick':>5}  " + "  ".join(f"{'AMR_'+str(i+1):^22}" for i in range(SIM_NUM_AGENTS))
    print(hdr)
    sub = f"  {'':>5}  " + "  ".join(f"{'Pos':^8}{'State':^8}{'Bat':^6}" for _ in range(SIM_NUM_AGENTS))
    print(sub)
    print("  " + "-" * 68)

    with GodotBridge(enabled=godot_enabled) as bridge:
        sim_start = _time.time()

        for tick in range(max_ticks):
            # Auction for newly available tasks
            available = [t for t in task_queue if t.issued_tick <= tick]
            for task in available:
                winner = _run_auction(agents, task)
                if winner:
                    task_queue.remove(task)

            # Tick all agents
            for ag in agents:
                ag.tick(delta_time=0.1)

            # Distribute P2P messages
            for ag in agents:
                for sid, wps in intent_bus.items():
                    if sid != ag.robot_id:
                        ag.on_peer_intent(sid, wps)
                for hm in hazard_bus:
                    if hm.robot_id != ag.robot_id:
                        ag.on_peer_hazard(hm)
                for lm in lease_bus:
                    if lm.robot_id != ag.robot_id:
                        ag.on_peer_lease(lm)
            hazard_bus.clear()
            lease_bus.clear()

            # Telemetry flush to Godot
            bridge.flush(agents, tick)

            # Collision detection
            pos_map: Dict[Tuple, str] = {}
            for ag in agents:
                if ag.position in pos_map:
                    total_collision += 1
                    print(f"COLLISION at {ag.position} between {ag.robot_id} and {pos_map[ag.position]} at tick {tick}")
                else:
                    pos_map[ag.position] = ag.robot_id
            idle_ticks += sum(1 for ag in agents if ag.state == AgentState.IDLE)

            # Status print every 20 ticks
            if tick % 20 == 0:
                cols = []
                for ag in agents:
                    abbr = STATE_ABBREV.get(ag.state.value, ag.state.value[:6])
                    cell = _coloured(ag.state.value, f"({ag.position[0]:2d},{ag.position[1]:2d}) {abbr} {ag.battery:4.1f}%")
                    cols.append(f"{cell:^22}")
                row = f"  {tick:>5}  " + "  ".join(cols)
                print(row)

            if len(completed) >= num_tasks:
                break

        sim_elapsed = _time.time() - sim_start

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"  Simulation complete in {tick+1} ticks  ({sim_elapsed:.2f}s real time)")
    print(f"  Tasks completed : {len(completed)}/{num_tasks}")
    print(f"  Collisions      : {total_collision}")
    print(f"  Idle ticks      : {idle_ticks}")
    print(f"  Godot datagrams : {bridge._sent}")
    print("=" * 72)

    return {
        "ticks":       tick + 1,
        "completed":   len(completed),
        "collisions":  total_collision,
        "idle_ticks":  idle_ticks,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="BEL AMR Fleet Simulation")
    parser.add_argument("--sim-only",   action="store_true", help="Skip benchmark")
    parser.add_argument("--bench-only", action="store_true", help="Skip real-time sim")
    parser.add_argument("--tasks",  type=int, default=50,   help="Tasks for real-time sim")
    parser.add_argument("--ticks",  type=int, default=2000, help="Max ticks for sim")
    parser.add_argument("--no-godot",   action="store_true", help="Disable UDP telemetry")
    parser.add_argument("--bench-tasks",type=int, default=100, help="Tasks for benchmark")
    args = parser.parse_args()

    # ── Real-time simulation ──────────────────────────────────────────
    if not args.bench_only:
        run_realtime_sim(
            num_tasks=args.tasks,
            max_ticks=args.ticks,
            godot_enabled=not args.no_godot,
        )

    # ── BEL Benchmark ────────────────────────────────────────────────
    if not args.sim_only:
        print()
        runner = BenchmarkRunner(num_tasks=args.bench_tasks, verbose=False)
        runner.run()


if __name__ == "__main__":
    main()
