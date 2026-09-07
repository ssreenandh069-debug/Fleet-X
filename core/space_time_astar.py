"""
core/space_time_astar.py
========================
Spatio-Temporal A* (Space-Time A*) path planner for Autonomous Mobile Robots.

Search space
------------
Nodes are 3-tuples (x, y, t) where:
    x : column index (East-positive)
    y : row index    (South-positive)
    t : discrete time step (integer >= 0)

Actions (5 total)
-----------------
    NORTH  : (x, y-1, t+1)  cost = grid.move_cost(...)
    SOUTH  : (x, y+1, t+1)  cost = grid.move_cost(...)
    EAST   : (x+1, y, t+1)  cost = grid.move_cost(...)
    WEST   : (x-1, y, t+1)  cost = grid.move_cost(...)
    WAIT   : (x, y,   t+1)  cost = WAIT_COST_FACTOR * 1.0  (default 1.2)

Collision avoidance via ReservationTable
-----------------------------------------
    Vertex collision  : Reject if (x, y, t) is reserved by another agent.
    Edge-swap collision: Reject move (x1,y1)->(x2,y2) at time t if the
                         reservation table contains the edge (x2,y2)->(x1,y1)
                         at the same t (i.e., two robots swapping cells).

Heuristic
---------
2D Manhattan distance in space ignoring time (admissible and consistent for
unit-cost 4-connected grids; remains admissible under highway penalties since
penalties only *increase* actual costs).

Termination
-----------
The search terminates when a node (goal_x, goal_y, t) is popped from the
open list for *any* t.  This yields the minimum-cost space-time path.

A configurable ``max_time`` cap prevents unbounded search in deadlock
scenarios (default: 200).  If no path is found within the cap, the planner
returns an empty list.

Sub-15 ms design notes
-----------------------
* Open list: binary min-heap via heapq.
* Closed set: flat dict (x, y, t) -> g_cost for O(1) lookup.
* Path reconstruction: parent dict; no auxiliary data structures.
* No external dependencies (heapq, dataclasses, typing, math only).

Usage example
-------------
    from core.grid_map import GridMap
    from core.space_time_astar import ReservationTable, SpaceTimeAstar

    grid = GridMap([[0]*10 for _ in range(10)])
    table = ReservationTable()
    planner = SpaceTimeAstar(grid, table, agent_id="robot_0")
    path = planner.plan(start=(0, 0), goal=(9, 9), start_time=0)
    # path is a list of (x, y, t) tuples, or [] if no path found.
    planner.commit_path(path)  # reserve cells for this robot
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from core.grid_map import GridMap

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------
WAIT_COST_FACTOR: float = 1.2   # cost multiplier for the WAIT action
DEFAULT_MAX_TIME: int   = 200   # hard cap on time horizon

# Type aliases
Position  = Tuple[int, int]          # (col, row) = (x, y)
STNode    = Tuple[int, int, int]     # (x, y, t)
EdgeKey   = Tuple[int, int, int, int, int]  # (x1, y1, x2, y2, t)


# ---------------------------------------------------------------------------
# ReservationTable
# ---------------------------------------------------------------------------
class ReservationTable:
    """
    Shared conflict-avoidance database for all agents in the fleet.

    It records:
    * **Vertex reservations** : which agent occupies (x, y) at time t.
    * **Edge reservations**   : which agent traverses (x1,y1)->(x2,y2) at t.

    The table is *mutable* and is updated when an agent commits its path via
    ``SpaceTimeAstar.commit_path()``.

    Thread-safety
    -------------
    This implementation is NOT thread-safe.  The fleet coordinator must
    serialize writes (e.g., via a threading.Lock) if called from multiple
    threads.
    """

    def __init__(self) -> None:
        # (x, y, t) -> agent_id
        self._vertex: Dict[STNode, str] = {}
        # (x1, y1, x2, y2, t) -> agent_id
        self._edge: Dict[EdgeKey, str] = {}

    # ------------------------------------------------------------------
    # Vertex API
    # ------------------------------------------------------------------

    def reserve_vertex(self, x: int, y: int, t: int, agent_id: str) -> None:
        """Mark (x, y) at time t as occupied by *agent_id*."""
        self._vertex[(x, y, t)] = agent_id

    def is_vertex_free(
        self, x: int, y: int, t: int, querying_agent: str
    ) -> bool:
        """
        Return True if (x, y, t) is either unreserved or reserved by the
        *querying_agent* itself.
        """
        owner = self._vertex.get((x, y, t))
        return owner is None or owner == querying_agent

    # ------------------------------------------------------------------
    # Edge API
    # ------------------------------------------------------------------

    def reserve_edge(
        self,
        x1: int, y1: int,
        x2: int, y2: int,
        t: int,
        agent_id: str,
    ) -> None:
        """
        Record that *agent_id* traverses (x1,y1)->(x2,y2) during [t, t+1].
        """
        self._edge[(x1, y1, x2, y2, t)] = agent_id

    def has_swap_conflict(
        self,
        x1: int, y1: int,
        x2: int, y2: int,
        t: int,
        querying_agent: str,
    ) -> bool:
        """
        Return True if another agent is traversing (x2,y2)->(x1,y1) at t
        (i.e., an edge-swap / head-on collision would occur).
        """
        owner = self._edge.get((x2, y2, x1, y1, t))
        return owner is not None and owner != querying_agent

    # ------------------------------------------------------------------
    # Bulk release
    # ------------------------------------------------------------------

    def release_agent(self, agent_id: str) -> None:
        """
        Remove all vertex and edge reservations belonging to *agent_id*.
        Called when an agent's plan is re-computed.
        """
        self._vertex = {
            k: v for k, v in self._vertex.items() if v != agent_id
        }
        self._edge = {
            k: v for k, v in self._edge.items() if v != agent_id
        }

    def purge(self, current_tick: int) -> int:
        """
        Remove all reservations for t < current_tick.
        Returns the total number of removed entries.
        """
        stale_v = [k for k in self._vertex if k[2] < current_tick]
        for k in stale_v:
            del self._vertex[k]

        stale_e = [k for k in self._edge if k[4] < current_tick]
        for k in stale_e:
            del self._edge[k]

        return len(stale_v) + len(stale_e)

    def __repr__(self) -> str:
        return (
            f"ReservationTable("
            f"vertices={len(self._vertex)}, "
            f"edges={len(self._edge)})"
        )


# ---------------------------------------------------------------------------
# Internal A* node wrapper (heap-compatible)
# ---------------------------------------------------------------------------
@dataclass(order=True)
class _HeapNode:
    """
    Priority-queue entry for the A* open list.

    ``f_cost`` is used for heap ordering; the remaining fields are stored
    as payload.  Using ``order=True`` on the dataclass lets Python compare
    ``_HeapNode`` objects by field order (f_cost first, then g_cost, etc.),
    providing a deterministic tie-breaking rule without a custom comparator.
    """
    f_cost: float          # f = g + h  (primary sort key)
    g_cost: float          # g: actual cost from start
    x: int     = field(compare=False)
    y: int     = field(compare=False)
    t: int     = field(compare=False)


# ---------------------------------------------------------------------------
# SpaceTimeAstar
# ---------------------------------------------------------------------------
class SpaceTimeAstar:
    """
    Spatio-Temporal A* planner for a single AMR agent.

    Parameters
    ----------
    grid_map : GridMap
        The warehouse occupancy grid.
    reservation_table : ReservationTable
        Shared reservation table updated by all agents.
    agent_id : str
        Unique identifier for this agent (used for collision-check ownership).
    wait_cost : float, optional
        Cost multiplier for the WAIT action.  Default: ``WAIT_COST_FACTOR``.
    max_time : int, optional
        Maximum time horizon.  Planning fails if goal is not reached by this
        time step.  Default: ``DEFAULT_MAX_TIME``.
    """

    def __init__(
        self,
        grid_map: GridMap,
        reservation_table: ReservationTable,
        agent_id: str,
        wait_cost: float = WAIT_COST_FACTOR,
        max_time: int = DEFAULT_MAX_TIME,
    ) -> None:
        self._grid      = grid_map
        self._table     = reservation_table
        self._agent_id  = agent_id
        self._wait_cost = wait_cost
        self._max_time  = max_time

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def plan(
        self,
        start: Position,
        goal:  Position,
        start_time: int = 0,
    ) -> List[STNode]:
        """
        Compute the minimum-cost space-time path from *start* to *goal*.

        Parameters
        ----------
        start : (x, y)
            Starting position (column, row).
        goal : (x, y)
            Goal position (column, row).
        start_time : int
            The time step at which the agent begins moving.  Allows planning
            to begin in the middle of a schedule.

        Returns
        -------
        list of (x, y, t)
            Ordered space-time path from start to goal, inclusive of both
            endpoints.  Returns ``[]`` if no path exists within ``max_time``.

        Raises
        ------
        ValueError
            If start or goal positions are out of bounds or impassable.
        """
        sx, sy = start
        gx, gy = goal

        # Sanity checks
        if not self._grid.is_passable(sx, sy):
            raise ValueError(
                f"SpaceTimeAstar: start {start} is out of bounds or impassable."
            )
        if not self._grid.is_passable(gx, gy):
            raise ValueError(
                f"SpaceTimeAstar: goal {goal} is out of bounds or impassable."
            )

        # ------------------------------------------------------------------
        # A* data structures
        # ------------------------------------------------------------------
        # Open list: min-heap of _HeapNode
        open_heap: List[_HeapNode] = []

        # Closed dict: (x, y, t) -> g_cost  (best known g at this ST node)
        closed: Dict[STNode, float] = {}

        # Parent dict for path reconstruction
        parent: Dict[STNode, Optional[STNode]] = {}

        # Seed the search
        h0 = self._heuristic(sx, sy, gx, gy)
        start_node = _HeapNode(f_cost=h0, g_cost=0.0, x=sx, y=sy, t=start_time)
        heapq.heappush(open_heap, start_node)
        start_st: STNode = (sx, sy, start_time)
        closed[start_st] = 0.0
        parent[start_st] = None

        # ------------------------------------------------------------------
        # Main A* loop
        # ------------------------------------------------------------------
        while open_heap:
            current = heapq.heappop(open_heap)
            cx, cy, ct = current.x, current.y, current.t
            cg = current.g_cost
            cst: STNode = (cx, cy, ct)

            # Skip if we have already found a cheaper path to this ST node
            if cg > closed.get(cst, math.inf):
                continue

            # Goal check — reached goal position at any time
            if cx == gx and cy == gy:
                # To prevent collisions when parked, we must ensure the goal is completely
                # free for the foreseeable future (i.e. up to max_time).
                can_park = True
                for dt in range(1, self._max_time - ct + 1):
                    if not self._table.is_vertex_free(cx, cy, ct + dt, self._agent_id):
                        can_park = False
                        break
                
                if can_park:
                    return self._reconstruct_path(parent, cst)
                # If we cannot park safely, we must wait or route around (A* will explore this)

            # Time cap: do not expand beyond max_time
            if ct >= self._max_time:
                continue

            nt = ct + 1  # next time step

            # Expand the 5 actions
            for nx, ny, step_cost in self._expand(cx, cy):
                # ---- Vertex collision check ----
                if not self._table.is_vertex_free(nx, ny, nt, self._agent_id):
                    continue

                # ---- Edge-swap collision check ----
                if self._table.has_swap_conflict(
                    cx, cy, nx, ny, ct, self._agent_id
                ):
                    continue

                ng = cg + step_cost
                nst: STNode = (nx, ny, nt)

                if ng < closed.get(nst, math.inf):
                    closed[nst] = ng
                    parent[nst] = cst
                    nh = self._heuristic(nx, ny, gx, gy)
                    heapq.heappush(
                        open_heap,
                        _HeapNode(f_cost=ng + nh, g_cost=ng, x=nx, y=ny, t=nt),
                    )

        # No path found within time horizon
        return []

    def commit_path(self, path: List[STNode]) -> None:
        """
        Register all vertex and edge reservations for the given path in the
        shared ReservationTable.

        Call this after a successful ``plan()`` to prevent other agents from
        conflicting with this path during their own planning.

        Parameters
        ----------
        path : list of (x, y, t)
            The space-time path returned by ``plan()``.
        """
        if not path:
            return

        # First release any previous reservations for this agent
        self._table.release_agent(self._agent_id)

        for i, (x, y, t) in enumerate(path):
            self._table.reserve_vertex(x, y, t, self._agent_id)
            if i + 1 < len(path):
                nx, ny, _ = path[i + 1]
                self._table.reserve_edge(x, y, nx, ny, t, self._agent_id)
                
        # Reserve the final goal node indefinitely (up to max_time)
        # to act as a solid obstacle while the agent is parked.
        goal_x, goal_y, goal_t = path[-1]
        for dt in range(1, self._max_time + 1):
            self._table.reserve_vertex(goal_x, goal_y, goal_t + dt, self._agent_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _heuristic(self, x: int, y: int, gx: int, gy: int) -> float:
        """
        Admissible, consistent heuristic: Manhattan distance in 2D space.

        Under highway penalties, actual move costs are >= 1.0, so Manhattan
        distance remains a lower bound.  Time dimension is excluded because
        t increases by 1 at every expansion regardless of spatial progress,
        hence adding t would make the heuristic inadmissible.
        """
        return float(abs(x - gx) + abs(y - gy))

    def _expand(
        self, x: int, y: int
    ) -> List[Tuple[int, int, float]]:
        """
        Generate (nx, ny, cost) triples for all valid successor actions
        from position (x, y).

        Includes the WAIT action (nx == x, ny == y) with cost
        ``_wait_cost``.
        """
        successors: List[Tuple[int, int, float]] = []

        # Spatial moves: NORTH / SOUTH / EAST / WEST
        for nx, ny, cost in self._grid.passable_neighbours(x, y):
            successors.append((nx, ny, cost))

        # WAIT action: stay in place at t+1
        # The current cell is already confirmed passable by the planner seed.
        successors.append((x, y, self._wait_cost))

        return successors

    @staticmethod
    def _reconstruct_path(
        parent: Dict[STNode, Optional[STNode]],
        goal_st: STNode,
    ) -> List[STNode]:
        """
        Walk the parent pointers from *goal_st* back to the start and return
        the path in chronological order.
        """
        path: List[STNode] = []
        node: Optional[STNode] = goal_st
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path


# ===========================================================================
# Standalone test suite
# ===========================================================================

def _make_open_grid(rows: int = 8, cols: int = 8) -> GridMap:
    """Helper: fully open (all walkways) grid of given size."""
    return GridMap([[0] * cols for _ in range(rows)])


def test_space_time_astar() -> None:
    """
    Three self-contained test cases for SpaceTimeAstar.

    Case A — Free path (0,0) -> (5,5)
    -----------------------------------
    Verifies that the planner finds a valid path on an open grid with no
    obstacles or other agents.

    Case B — Temporal obstacle at (2,2) only at t=2
    -------------------------------------------------
    The cell (2,2) is permanently walkable, but another agent has reserved
    vertex (2,2) at t=2.  The planner must either wait at t=1 (arriving at
    (2,2) at t=3 instead) or detour around the reserved cell.

    Case C — Two-robot crossing intersection
    -----------------------------------------
    Robot A travels East along row 3; Robot B travels South along col 3.
    Both would converge on cell (3,3) at the same time.  Robot B (planned
    second) must wait or detour to resolve the conflict.
    """
    print("=" * 60)
    print("Space-Time A*  --  Test Suite")
    print("=" * 60)

    # ---------------------------------------------------------------
    # Case A: Free path (0,0) -> (5,5)
    # ---------------------------------------------------------------
    print("\n--- Case A: Free path (0,0) -> (5,5) ---")
    grid_a = _make_open_grid(8, 8)
    table_a = ReservationTable()
    planner_a = SpaceTimeAstar(grid_a, table_a, agent_id="robot_A")

    path_a = planner_a.plan(start=(0, 0), goal=(5, 5))

    assert path_a, "Case A FAILED: No path found on open grid."
    assert path_a[0] == (0, 0, 0), f"Case A FAILED: Bad start node {path_a[0]}"
    assert path_a[-1][:2] == (5, 5), f"Case A FAILED: Bad goal node {path_a[-1]}"

    # Verify continuity: each step moves to an adjacent cell or waits
    for i in range(len(path_a) - 1):
        x0, y0, t0 = path_a[i]
        x1, y1, t1 = path_a[i + 1]
        assert t1 == t0 + 1, f"Case A: time not monotone at step {i}"
        assert abs(x1 - x0) + abs(y1 - y0) <= 1, (
            f"Case A: non-adjacent move at step {i}: {path_a[i]} -> {path_a[i+1]}"
        )

    # Minimum path length for Manhattan distance = 10 steps
    min_steps = abs(5 - 0) + abs(5 - 0)
    assert len(path_a) - 1 >= min_steps, (
        f"Case A: path too short ({len(path_a)-1} steps, min {min_steps})"
    )

    print(f"  Path length : {len(path_a)-1} steps  (min Manhattan = {min_steps})")
    print(f"  Path        : {path_a}")
    print("  Case A PASSED [OK]")

    # ---------------------------------------------------------------
    # Case B: Temporal obstacle at (2,2) at t=2 only
    # ---------------------------------------------------------------
    print("\n--- Case B: Temporal obstacle at (2,2) ONLY at t=2 ---")
    grid_b = _make_open_grid(8, 8)
    table_b = ReservationTable()

    # Simulate an external agent blocking (2,2) at t=2
    table_b.reserve_vertex(2, 2, 2, agent_id="blocker")

    planner_b = SpaceTimeAstar(grid_b, table_b, agent_id="robot_B")
    path_b = planner_b.plan(start=(0, 0), goal=(5, 5))

    assert path_b, "Case B FAILED: No path found despite only a single temporal block."
    assert path_b[-1][:2] == (5, 5), f"Case B FAILED: Goal not reached: {path_b[-1]}"

    # Verify: robot does NOT occupy (2,2) at t=2
    blocked = any(
        x == 2 and y == 2 and t == 2
        for x, y, t in path_b
    )
    assert not blocked, (
        f"Case B FAILED: Robot passes through reserved (2,2,t=2).\n  Path: {path_b}"
    )

    # Verify path continuity
    for i in range(len(path_b) - 1):
        x0, y0, t0 = path_b[i]
        x1, y1, t1 = path_b[i + 1]
        assert t1 == t0 + 1
        assert abs(x1 - x0) + abs(y1 - y0) <= 1

    print(f"  Path length : {len(path_b)-1} steps")
    print(f"  Path        : {path_b}")
    print("  Case B PASSED [OK]")

    # ---------------------------------------------------------------
    # Case C: Two robots crossing a 1-lane intersection
    # ---------------------------------------------------------------
    print("\n--- Case C: Two robots crossing a 1-lane intersection ---")
    #
    #  10-column x 8-row grid.
    #  Robot A: (0,3) -> (9,3)  — travelling East along row 3
    #  Robot B: (3,0) -> (3,7)  — travelling South along col 3
    #  Both would arrive at intersection (3,3) at t=3.
    #
    grid_c = _make_open_grid(rows=8, cols=10)
    table_c = ReservationTable()

    # Plan Robot A first (higher priority)
    planner_a_c = SpaceTimeAstar(grid_c, table_c, agent_id="robot_A_c")
    path_a_c = planner_a_c.plan(start=(0, 3), goal=(9, 3))

    assert path_a_c, "Case C FAILED: Robot A could not find a path."
    planner_a_c.commit_path(path_a_c)  # Lock in Robot A's path

    # Plan Robot B (must avoid Robot A's reservations)
    planner_b_c = SpaceTimeAstar(grid_c, table_c, agent_id="robot_B_c")
    path_b_c = planner_b_c.plan(start=(3, 0), goal=(3, 7))

    assert path_b_c, "Case C FAILED: Robot B could not find a path."

    # Verify: no vertex collisions between A and B
    st_set_a: Set[STNode] = set(path_a_c)
    for node in path_b_c:
        assert node not in st_set_a, (
            f"Case C FAILED: Vertex collision at {node}.\n"
            f"  Robot A path: {path_a_c}\n"
            f"  Robot B path: {path_b_c}"
        )

    # Verify: no edge-swap collisions between A and B
    edges_a: Set[Tuple[STNode, STNode]] = {
        (path_a_c[i], path_a_c[i + 1])
        for i in range(len(path_a_c) - 1)
    }
    for i in range(len(path_b_c) - 1):
        reverse_edge = (path_b_c[i + 1], path_b_c[i])
        assert reverse_edge not in edges_a, (
            f"Case C FAILED: Edge-swap collision at step {i}.\n"
            f"  Robot B move: {path_b_c[i]} -> {path_b_c[i+1]}\n"
            f"  Robot A path: {path_a_c}"
        )

    print(f"  Robot A path: {path_a_c}")
    print(f"  Robot B path: {path_b_c}")
    print("  Case C PASSED [OK]")

    print("\n" + "=" * 60)
    print("ALL 3 TEST CASES PASSED")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Benchmark: measure planning time for Cases A-C
# ---------------------------------------------------------------------------

def benchmark_planner() -> None:
    """
    Quick wall-clock benchmark to verify sub-15 ms planning times.
    Uses ``time.perf_counter`` (high-resolution timer).
    """
    import time

    REPEATS = 50
    results: Dict[str, float] = {}

    cases = {
        "Case A (free 8x8, (0,0)->(5,5))": lambda: (
            SpaceTimeAstar(
                _make_open_grid(8, 8), ReservationTable(), "bm_A"
            ).plan((0, 0), (5, 5))
        ),
        "Case B (temporal block, (0,0)->(5,5))": lambda: (
            _run_case_b_once()
        ),
        "Case C (two-robot crossing)": lambda: (
            _run_case_c_once()
        ),
    }

    print("\n--- Benchmark Results ---")
    for label, fn in cases.items():
        times: List[float] = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

        avg_ms = sum(times) / len(times)
        max_ms = max(times)
        results[label] = avg_ms
        status = "OK" if max_ms < 15.0 else "WARN (>15ms peak)"
        print(
            f"  {label}\n"
            f"    avg={avg_ms:.3f} ms  max={max_ms:.3f} ms  [{status}]"
        )

    return results


def _run_case_b_once() -> List[STNode]:
    grid = _make_open_grid(8, 8)
    table = ReservationTable()
    table.reserve_vertex(2, 2, 2, "blocker")
    return SpaceTimeAstar(grid, table, "bm_B").plan((0, 0), (5, 5))


def _run_case_c_once() -> List[STNode]:
    grid = _make_open_grid(rows=8, cols=10)
    table = ReservationTable()
    pa = SpaceTimeAstar(grid, table, "bm_A_c")
    path_a = pa.plan((0, 3), (9, 3))
    pa.commit_path(path_a)
    return SpaceTimeAstar(grid, table, "bm_B_c").plan((3, 0), (3, 7))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_space_time_astar()
    benchmark_planner()
