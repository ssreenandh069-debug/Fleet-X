"""
coordination/conflict_resolver.py
===================================
Dynamic conflict-resolution engine and narrow-aisle deadlock solver for
Autonomous Mobile Robots (AMRs).

System overview
---------------

    ┌─────────────────────────────────────────────────────────────────┐
    │                    ConflictResolver                              │
    │                                                                  │
    │  ┌──────────────┐   on_peer_packet()   ┌──────────────────────┐│
    │  │  P2PNode     │ ──────────────────── │ PriorityEngine       ││
    │  │  (Cycle 2)   │                      │ compute_priority()   ││
    │  └──────────────┘                      └──────────────────────┘│
    │          │                                       │              │
    │          ▼                                       ▼              │
    │  ┌──────────────────┐              ┌──────────────────────────┐│
    │  │ LocalReservation │              │ Resolution Strategy       ││
    │  │ Table (Cycle 3)  │              │ YIELD / RETAIN / DEADLOCK ││
    │  └──────────────────┘              └──────────────────────────┘│
    │          │                                       │              │
    │          └──────────────┬────────────────────────┘             │
    │                         ▼                                       │
    │               SpaceTimeAstar.plan()  (Cycle 1)                  │
    └─────────────────────────────────────────────────────────────────┘

State machine
-------------

    IDLE  ──────────► NAVIGATING ──────────► GOAL_REACHED
                           │
                           ├─ conflict? ──► YIELDING
                           │                   │ (replan done)
                           │◄──────────────────┘
                           │
                           ├─ stuck 4 ticks in 1-lane? ──► REVERSING
                           │                                    │
                           │◄───────────────────────────────────┘

Priority formula (deterministic, both sides compute the same winner)
--------------------------------------------------------------------

    P = (urgency × 40.0)
      + ((100.0 − battery) × 0.3)
      + (1.0 / (dist_to_goal + 1) × 30.0)
      + tie_breaker

    tie_breaker = int(md5(robot_id.encode())[:4], 16) / 0xFFFF_FFFF
                  (normalised to [0, 1])

Zero external dependencies — only ``hashlib``, ``math``, ``logging``,
``dataclasses``, ``enum``, and ``typing`` from the standard library.

Integration
-----------
    from coordination.conflict_resolver import ConflictResolver, RobotContext
    from coordination.reservation_table import LocalReservationTable
    from core.grid_map import GridMap
    from core.space_time_astar import ReservationTable, SpaceTimeAstar

    grid        = GridMap(...)
    local_rt    = LocalReservationTable()
    sta_table   = ReservationTable()
    ctx         = RobotContext(robot_id="AMR_01", ...)
    resolver    = ConflictResolver(grid, local_rt, sta_table, ctx)

    # Feed incoming peer packets (from P2PNode.on_packet callback):
    resolver.on_peer_packet(intent_packet)

    # Call once per tick from the robot's main control loop:
    resolver.tick(current_pos=(x, y), current_tick=t)
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

from coordination.reservation_table import LocalReservationTable, PeerClaim
from core.grid_map import GridMap
from core.space_time_astar import ReservationTable, SpaceTimeAstar

# Optional: IntentPacket from Cycle 2 (imported lazily to avoid hard dep)
try:
    from p2p.protocol import IntentPacket
    _HAS_P2P = True
except ImportError:
    _HAS_P2P = False
    IntentPacket = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------
STUCK_TICK_THRESHOLD:  int   = 4      # ticks without spatial progress → stuck
MIN_JUNCTION_NEIGHBORS: int  = 2      # minimum open neighbors to qualify as junction
DEADLOCK_SCAN_RADIUS:  int   = 10     # cells to search for a reversal junction
URGENCY_WEIGHT:        float = 40.0
BATTERY_WEIGHT:        float = 0.3
DIST_WEIGHT:           float = 30.0

# Type aliases
Position = Tuple[int, int]
STNode   = Tuple[int, int, int]


# ---------------------------------------------------------------------------
# Robot navigation state
# ---------------------------------------------------------------------------

class RobotState(Enum):
    IDLE         = auto()   # No active goal.
    NAVIGATING   = auto()   # Following planned path.
    YIELDING     = auto()   # Waiting 1 tick or replanning after yield decision.
    REVERSING    = auto()   # Backing to junction to resolve deadlock.
    GOAL_REACHED = auto()   # At destination.


# ---------------------------------------------------------------------------
# RobotContext — mutable snapshot of this robot's own state
# ---------------------------------------------------------------------------

@dataclass
class RobotContext:
    """
    Mutable snapshot of the robot's operational state.

    Parameters
    ----------
    robot_id : str
        Globally unique robot identifier.
    urgency : float
        Task urgency in [0, 1].  1.0 = highest urgency.
    battery : float
        Remaining battery percentage in [0, 100].
    goal : Position | None
        Current target cell (column, row).
    start : Position | None
        Current position cell (column, row) at planning time.
    planned_path : list of STNode
        Current STA*-computed space-time path (updated on replan).
    path_index : int
        Index into ``planned_path`` of the *next* node to execute.
    state : RobotState
        Current FSM state.
    """
    robot_id:     str
    urgency:      float                  = 1.0
    battery:      float                  = 100.0
    goal:         Optional[Position]     = None
    start:        Optional[Position]     = None
    planned_path: List[STNode]           = field(default_factory=list)
    path_index:   int                    = 0
    state:        RobotState             = RobotState.IDLE

    # Stuck detection internals
    _last_positions: List[Position]      = field(default_factory=list, repr=False)
    _reversal_target: Optional[Position] = field(default=None, repr=False)


# ===========================================================================
# Priority Engine
# ===========================================================================

class PriorityEngine:
    """
    Deterministic, decentralised priority calculator.

    Both ends of a conflict call ``compute_priority()`` with their own
    parameters.  The formula is identical on every robot, guaranteeing that
    both independently select the **same** winner without any negotiation
    round-trip.

    Formula
    -------
    P = (urgency × 40.0)
      + ((100.0 − battery) × 0.3)
      + (1.0 / (dist_to_goal + 1) × 30.0)
      + tie_breaker

    tie_breaker = lower 32 bits of MD5(robot_id) / 0xFFFF_FFFF  ∈ (0, 1)
    """

    @staticmethod
    def _tie_breaker(robot_id: str) -> float:
        """
        Deterministic, robot-id-derived tie-breaker in the interval (0, 1).

        Uses the first 4 bytes of MD5(robot_id.encode('utf-8')) as a
        big-endian uint32, then normalises by 0xFFFF_FFFF.  Every robot
        in the fleet that calls this function with the same *robot_id*
        will obtain the identical floating-point value.
        """
        digest = hashlib.md5(robot_id.encode("utf-8")).digest()
        uint32 = int.from_bytes(digest[:4], byteorder="big")
        return uint32 / 0xFFFF_FFFF

    @classmethod
    def compute_priority(
        cls,
        robot_id:    str,
        urgency:     float,
        battery:     float,
        dist_to_goal: float,
    ) -> float:
        """
        Compute the priority score for a robot.

        Parameters
        ----------
        robot_id : str
            Unique identifier (used for deterministic tie-breaking).
        urgency : float
            Task urgency [0.0, 1.0].  1.0 = highest urgency.
        battery : float
            Remaining battery [0.0, 100.0].
        dist_to_goal : float
            Manhattan or Euclidean distance to the current goal in cells.

        Returns
        -------
        float
            Priority score.  Higher → this robot should retain its path.
        """
        tb = cls._tie_breaker(robot_id)
        return (
            urgency * URGENCY_WEIGHT
            + (100.0 - battery) * BATTERY_WEIGHT
            + (1.0 / (dist_to_goal + 1.0)) * DIST_WEIGHT
            + tb
        )

    @classmethod
    def compare(
        cls,
        robot_a: str, priority_a: float,
        robot_b: str, priority_b: float,
    ) -> int:
        """
        Compare two priority scores, with the deterministic tie-breaker
        as a final resort.

        Returns
        -------
        int
            +1 if A wins, -1 if B wins, 0 if identical (should not happen
            unless both IDs are the same).
        """
        if math.isclose(priority_a, priority_b, rel_tol=1e-9):
            # Fall back to lexicographic comparison on tie-breaker values
            tba = cls._tie_breaker(robot_a)
            tbb = cls._tie_breaker(robot_b)
            if tba > tbb:
                return 1
            if tba < tbb:
                return -1
            return 0
        return 1 if priority_a > priority_b else -1


# ===========================================================================
# Junction Finder (for deadlock reversal)
# ===========================================================================

class JunctionFinder:
    """
    Finds the nearest walkable junction to a given position in a grid.

    A *junction* is a cell that has >= ``MIN_JUNCTION_NEIGHBORS`` passable
    cardinal neighbours, meaning there is enough space for a robot to pull
    aside and let another pass.
    """

    def __init__(self, grid: GridMap) -> None:
        self._grid = grid

    def is_junction(self, x: int, y: int) -> bool:
        """
        Return True if (x, y) qualifies as a junction.

        A corridor cell has exactly 2 passable neighbours (prev/next in lane).
        A junction has >= 3 passable neighbours (branching point or open area).
        We use the constant MIN_JUNCTION_NEIGHBORS (default 2) as the *extra*
        neighbour threshold above the corridor baseline of 2, so the required
        count is MIN_JUNCTION_NEIGHBORS + 2.
        """
        neighbours = self._grid.passable_neighbours(x, y)
        # A dead-end has 1 neighbour, corridor has 2, junction has >=3
        return len(neighbours) >= 3

    def nearest_junction(
        self,
        from_pos: Position,
        scan_radius: int = DEADLOCK_SCAN_RADIUS,
    ) -> Optional[Position]:
        """
        BFS from *from_pos* to find the closest junction within *scan_radius*
        cells.

        Returns
        -------
        Position or None
            The nearest junction (col, row), or None if none found.
        """
        from collections import deque

        visited: Set[Position] = {from_pos}
        queue: deque = deque()
        queue.append((from_pos, 0))

        while queue:
            pos, depth = queue.popleft()
            if depth > scan_radius:
                continue
            x, y = pos
            if self.is_junction(x, y):
                return pos
            for nx, ny, _ in self._grid.passable_neighbours(x, y):
                npos: Position = (nx, ny)
                if npos not in visited:
                    visited.add(npos)
                    queue.append((npos, depth + 1))

        return None


# ===========================================================================
# ConflictResolver — main engine
# ===========================================================================

class ConflictResolver:
    """
    Per-robot conflict resolution engine.

    Wires together:
    * ``LocalReservationTable`` — peer trajectory cache (Cycle 3).
    * ``PriorityEngine``        — deterministic priority scorer.
    * ``JunctionFinder``        — deadlock reversal BFS.
    * ``SpaceTimeAstar``        — path planner (Cycle 1).
    * ``P2PNode.on_packet``     — packet ingestion hook (Cycle 2).

    Parameters
    ----------
    grid : GridMap
        Warehouse occupancy grid (shared, read-only).
    local_rt : LocalReservationTable
        This robot's local peer-claim cache.
    sta_table : ReservationTable
        STA*-internal table (shared with the planner).
    ctx : RobotContext
        Mutable state of *this* robot.
    max_time : int
        Max time horizon passed to STA* replanning.  Default: 200.
    """

    def __init__(
        self,
        grid:      GridMap,
        local_rt:  LocalReservationTable,
        sta_table: ReservationTable,
        ctx:       RobotContext,
        max_time:  int = 200,
    ) -> None:
        self._grid      = grid
        self._local_rt  = local_rt
        self._sta_table = sta_table
        self._ctx       = ctx
        self._max_time  = 500

        self._priority_engine  = PriorityEngine()
        self._junction_finder  = JunctionFinder(grid)

        # Cache own priority at planning time (recomputed on replan)
        self._my_priority: float = 0.0

        # Live peer positions snapshot (robot_id -> (x, y)), refreshed by
        # the fleet driver at the start of every tick.  This is the
        # drift-proof net: STA reservations assume agents stay on schedule,
        # but execution can run ahead/behind plan (WAITs skipped, early
        # arrivals, goal-detection lag).  A parked peer physically occupies
        # its cell even when no reservation covers that exact tick.
        self._peer_positions: Dict[str, Position] = {}

    def update_peer_positions(self, mapping: Dict[str, Position]) -> None:
        """Refresh the live peer-position snapshot (self excluded by caller)."""
        self._peer_positions = dict(mapping)

    # ------------------------------------------------------------------
    # Public: set a new navigation goal and trigger initial plan
    # ------------------------------------------------------------------

    def set_goal(
        self,
        start:       Position,
        goal:        Position,
        current_tick: int,
        urgency:     float = 1.0,
        battery:     float = 100.0,
    ) -> bool:
        """
        Assign a new goal to this robot and compute the initial STA* path.

        Parameters
        ----------
        start : (x, y)
            Current position.
        goal : (x, y)
            Target position.
        current_tick : int
            Current discrete time step.
        urgency : float
            Task urgency [0, 1].
        battery : float
            Remaining battery %.

        Returns
        -------
        bool
            True if a path was found; False if planning failed.
        """
        self._ctx.urgency = urgency
        self._ctx.battery = battery
        self._ctx.goal    = goal
        self._ctx.start   = start
        self._ctx.state   = RobotState.NAVIGATING
        self._ctx._last_positions.clear()
        self._ctx.path_index = 0

        return self._replan(start, goal, current_tick)

    # ------------------------------------------------------------------
    # Public: P2P packet hook (attach to P2PNode.on_packet)
    # ------------------------------------------------------------------

    def on_peer_packet(self, packet: "IntentPacket") -> None:  # type: ignore[name-defined]
        """
        Process an incoming ``IntentPacket`` from a neighbouring robot.

        Called directly from ``P2PNode``'s ``on_packet`` callback.  Ingests
        the peer's trajectory into the local reservation table and triggers
        conflict resolution if the path overlaps with ours.

        Parameters
        ----------
        packet : IntentPacket
            A fully verified (HMAC passed, anti-replay passed) packet.
        """
        if not packet.trajectory_points:
            return

        # Check if the peer's trajectory is completely stationary (IDLE or STUCK)
        pts = packet.trajectory_points
        is_stationary = all(x == pts[0][0] and y == pts[0][1] for x, y, _ in pts)

        if is_stationary:
            # An IDLE/STUCK agent cannot yield; treat it as an immovable obstacle
            peer_priority = float('inf')
        else:
            # Compute peer's priority using the deterministic formula
            peer_dist = self._peer_dist_estimate(packet)
            peer_priority = PriorityEngine.compute_priority(
                robot_id=packet.robot_id,
                urgency=1.0,        # conservative: assume maximum urgency
                battery=100.0,      # conservative: assume full battery
                dist_to_goal=peer_dist,
            )

        # Record in local table
        self._local_rt.ingest_trajectory(
            peer_id=packet.robot_id,
            priority=peer_priority,
            waypoints=packet.trajectory_points,
        )

        # Run conflict check only if we have an active plan
        if (
            self._ctx.state in (RobotState.NAVIGATING, RobotState.YIELDING)
            and self._ctx.planned_path
        ):
            self._resolve_conflicts(packet.robot_id, peer_priority)

    # ------------------------------------------------------------------
    # Public: per-tick step (call from robot control loop)
    # ------------------------------------------------------------------

    def tick(
        self,
        current_pos:  Position,
        current_tick: int,
    ) -> Optional[Position]:
        """
        Advance the engine by one time step.

        * Purges stale reservation table entries.
        * Detects stuck condition → deadlock if in a 1-lane corridor.
        * Returns the next position the robot should move to, or None if
          the robot should stay put (IDLE / GOAL_REACHED / stuck).

        Parameters
        ----------
        current_pos : (x, y)
            Robot's confirmed current position this tick.
        current_tick : int
            Current discrete time step.

        Returns
        -------
        (x, y) or None
            Target position for this tick.
        """
        # Purge stale entries
        removed = self._local_rt.purge(current_tick)
        if removed:
            logger.debug("[%s] Purged %d stale entries.", self._ctx.robot_id, removed)

        # Update stuck detector
        self._update_stuck_history(current_pos)

        # FSM dispatch
        if self._ctx.state == RobotState.IDLE:
            return None

        if self._ctx.state == RobotState.GOAL_REACHED:
            return None

        if self._ctx.state == RobotState.NAVIGATING:
            return self._tick_navigating(current_pos, current_tick)

        if self._ctx.state == RobotState.YIELDING:
            return self._tick_yielding(current_pos, current_tick)

        if self._ctx.state == RobotState.REVERSING:
            return self._tick_reversing(current_pos, current_tick)

        return None

    # ------------------------------------------------------------------
    # Private: FSM tick handlers
    # ------------------------------------------------------------------

    def _tick_navigating(
        self, current_pos: Position, current_tick: int
    ) -> Optional[Position]:
        ctx = self._ctx

        # Goal check
        if current_pos == ctx.goal:
            ctx.state = RobotState.GOAL_REACHED
            logger.info("[%s] GOAL_REACHED at %s t=%d", ctx.robot_id, current_pos, current_tick)
            return None

        # If we have no path at all (initial plan failed), treat as stuck.
        if not ctx.planned_path:
            if self._in_single_lane(current_pos):
                return self._enter_reversing(current_pos, current_tick)
            # Try replanning directly
            if ctx.goal:
                self._replan(current_pos, ctx.goal, current_tick)
            return current_pos

        # Stuck detection: 0 movement over STUCK_TICK_THRESHOLD ticks
        if self._is_stuck():
            logger.warning(
                "[%s] STUCK detected at %s over %d ticks — checking for deadlock.",
                ctx.robot_id, current_pos, STUCK_TICK_THRESHOLD,
            )
            if self._in_single_lane(current_pos):
                return self._enter_reversing(current_pos, current_tick)

        # Advance along planned path
        return self._advance_path(current_pos, current_tick)

    def _tick_yielding(
        self, current_pos: Position, current_tick: int
    ) -> Optional[Position]:
        # After one wait tick, replan and switch back to NAVIGATING
        ctx = self._ctx
        if ctx.goal is None:
            ctx.state = RobotState.IDLE
            return None

        logger.debug("[%s] YIELDING: replanning from %s.", ctx.robot_id, current_pos)
        success = self._replan(current_pos, ctx.goal, current_tick)
        if success:
            ctx.state = RobotState.NAVIGATING
            return self._advance_path(current_pos, current_tick)
        # Still can't plan — wait another tick
        return current_pos

    def _tick_reversing(
        self, current_pos: Position, current_tick: int
    ) -> Optional[Position]:
        ctx = self._ctx
        target = ctx._reversal_target

        if target is None or current_pos == target:
            # Reached junction; hold and wait
            logger.info(
                "[%s] REVERSING: reached junction %s — holding.",
                ctx.robot_id, current_pos,
            )
            # After a brief hold, replan forward
            if ctx.goal is not None:
                success = self._replan(current_pos, ctx.goal, current_tick)
                if success:
                    ctx.state = RobotState.NAVIGATING
                    ctx._reversal_target = None
                    return self._advance_path(current_pos, current_tick)
            return current_pos

        # Continue reversing toward junction
        return self._advance_path(current_pos, current_tick)

    # ------------------------------------------------------------------
    # Private: conflict resolution
    # ------------------------------------------------------------------

    def _resolve_conflicts(self, peer_id: str, peer_priority: float) -> None:
        """
        Check my planned path against the peer's claims.  Yield if they
        have higher priority; retain if I do.
        """
        ctx = self._ctx
        if ctx.goal is None or not ctx.planned_path:
            return

        # Compute my current priority
        dist = self._my_dist_to_goal()
        self._my_priority = PriorityEngine.compute_priority(
            robot_id=ctx.robot_id,
            urgency=ctx.urgency,
            battery=ctx.battery,
            dist_to_goal=dist,
        )

        # Find conflicting waypoints
        remaining = ctx.planned_path[ctx.path_index:]
        conflicts = self._local_rt.find_conflicts(ctx.robot_id, remaining)

        if not conflicts:
            return  # No overlap

        first_conflict_key, first_claim = conflicts[0]
        fx, fy, ft = first_conflict_key

        result = PriorityEngine.compare(
            ctx.robot_id, self._my_priority,
            peer_id, peer_priority,
        )

        if result > 0:
            # I win — retain path, do nothing
            logger.debug(
                "[%s] Priority WIN vs %s (%.3f > %.3f) — retain path.",
                ctx.robot_id, peer_id, self._my_priority, peer_priority,
            )
            return

        # I lose — mark conflicting node blocked, trigger yield
        logger.info(
            "[%s] Priority YIELD to %s (%.3f < %.3f) — blocking (%d,%d,t=%d).",
            ctx.robot_id, peer_id, self._my_priority, peer_priority,
            fx, fy, ft,
        )
        for (x, y, t), claim in conflicts:
            self._local_rt.mark_blocked(x, y, t, claim.peer_id, claim.priority)

        # Transition: NAVIGATING → YIELDING (next tick triggers 1-step wait)
        if ctx.state == RobotState.NAVIGATING:
            ctx.state = RobotState.YIELDING

    # ------------------------------------------------------------------
    # Private: planning helpers
    # ------------------------------------------------------------------

    def _replan(
        self,
        start:        Position,
        goal:         Position,
        current_tick: int,
    ) -> bool:
        """
        Run STA* with all current peer claims blocked in the STA* table.

        Returns True on success.
        """
        # Refresh the STA* table with peer claims
        self._sta_table.release_agent(self._ctx.robot_id)
        self._local_rt.export_to_sta_table(
            self._sta_table, exclude_peer=self._ctx.robot_id
        )

        planner = SpaceTimeAstar(
            grid_map=self._grid,
            reservation_table=self._sta_table,
            agent_id=self._ctx.robot_id,
            max_time=self._max_time,
        )
        path = planner.plan(start=start, goal=goal, start_time=current_tick)

        if path:
            planner.commit_path(path)
            self._ctx.planned_path = path
            self._ctx.path_index   = 0
            logger.debug(
                "[%s] Replanned: %d steps from %s to %s (tick=%d).",
                self._ctx.robot_id, len(path) - 1, start, goal, current_tick,
            )
            return True

        logger.warning(
            "[%s] Replan FAILED: no path from %s to %s at tick=%d.",
            self._ctx.robot_id, start, goal, current_tick,
        )
        return False

    def _advance_path(
        self, current_pos: Position, current_tick: int
    ) -> Optional[Position]:
        """
        Return the next position from the planned path.  If the path index
        is exhausted, return current_pos (forces a wait).
        """
        ctx = self._ctx
        path = ctx.planned_path

        # Sync path_index to current_tick
        for i, (x, y, t) in enumerate(path):
            if t == current_tick and (x, y) == current_pos:
                ctx.path_index = i
                break

        next_idx = ctx.path_index + 1
        if next_idx < len(path):
            nx, ny, _ = path[next_idx]
            # Execution-time safety gate: never step into a cell reserved
            # by another agent at the actual arrival tick.  Own reservations
            # are ignored by the table, so our own committed path always
            # passes.  Without this, agents follow stale plans open-loop
            # (e.g. driving into a peer that parked on its goal after we
            # planned).
            arrive_t = current_tick + 1
            if not self._sta_table.is_vertex_free(nx, ny, arrive_t, ctx.robot_id):
                if ctx.state == RobotState.NAVIGATING:
                    ctx.state = RobotState.YIELDING
                return current_pos
            # Live-occupancy gate: never step into a cell physically
            # occupied by a peer at the start of this tick, even if the
            # reservation table has a hole there due to execution drift
            # (e.g. peer arrived early/parked on its goal).
            if (nx, ny) in self._peer_positions.values():
                if ctx.state == RobotState.NAVIGATING:
                    ctx.state = RobotState.YIELDING
                return current_pos
            # Position-driven advance: stay robust to waits/drift instead
            # of relying solely on exact (t, pos) sync above.
            ctx.path_index = next_idx
            return (nx, ny)

        # End of path
        if current_pos == ctx.goal:
            ctx.state = RobotState.GOAL_REACHED
        return current_pos

    # ------------------------------------------------------------------
    # Private: deadlock detection
    # ------------------------------------------------------------------

    def _update_stuck_history(self, current_pos: Position) -> None:
        hist = self._ctx._last_positions
        hist.append(current_pos)
        if len(hist) > STUCK_TICK_THRESHOLD + 1:
            hist.pop(0)

    def _is_stuck(self) -> bool:
        hist = self._ctx._last_positions
        if len(hist) < STUCK_TICK_THRESHOLD:
            return False
        # All positions in the last STUCK_TICK_THRESHOLD ticks identical
        return len(set(hist[-STUCK_TICK_THRESHOLD:])) == 1

    def _in_single_lane(self, pos: Position) -> bool:
        """
        Return True if *pos* is a corridor cell or dead-end.
        Both cases require reversal:
        - Dead-end (1 neighbour): the robot is stuck at a corridor terminus.
        - Corridor (2 neighbours): only forward and backward along the lane.
        Open-area cells (>=3 neighbours) are junctions and don't need reversal.
        """
        neighbours = self._grid.passable_neighbours(*pos)
        return len(neighbours) <= 2

    def _enter_reversing(
        self, current_pos: Position, current_tick: int
    ) -> Optional[Position]:
        """
        Initiate REVERSING state: find nearest junction and replan toward it.

        The reversal replan uses an **isolated** (empty) ReservationTable so
        that the higher-priority peer's reservations do not block the yielder
        from backing out.  The yielder's reversal path is short (to the
        nearest junction) and does not need to know about the peer's full
        forward route.
        """
        ctx = self._ctx
        junction = self._junction_finder.nearest_junction(current_pos)
        if junction is None:
            logger.warning(
                "[%s] No junction found within %d cells — forced WAIT.",
                ctx.robot_id, DEADLOCK_SCAN_RADIUS,
            )
            return current_pos

        ctx._reversal_target = junction
        ctx.state = RobotState.REVERSING
        logger.info(
            "[%s] REVERSING: backing to junction %s from %s.",
            ctx.robot_id, junction, current_pos,
        )

        # Use an isolated STA* table for the short reversal segment so that
        # the blocker's corridor reservations don't prevent the yielder from
        # reversing.  The yielder's reversal path is committed back to the
        # shared table so the winner can see it.
        isolated_table = ReservationTable()
        planner = SpaceTimeAstar(
            grid_map=self._grid,
            reservation_table=isolated_table,
            agent_id=ctx.robot_id,
            max_time=self._max_time,
        )
        path = planner.plan(start=current_pos, goal=junction, start_time=current_tick)
        if path:
            # Commit reversal path to the shared table
            planner.commit_path(path)
            ctx.planned_path = path
            ctx.path_index   = 0
            return self._advance_path(current_pos, current_tick)

        logger.warning(
            "[%s] REVERSING replan to junction %s FAILED — forced WAIT.",
            ctx.robot_id, junction,
        )
        return current_pos

    # ------------------------------------------------------------------
    # Private: utility
    # ------------------------------------------------------------------

    def _my_dist_to_goal(self) -> float:
        ctx = self._ctx
        if ctx.goal is None or ctx.planned_path:
            if ctx.planned_path:
                gx, gy = ctx.goal  # type: ignore[misc]
                last_x, last_y, _ = ctx.planned_path[-1]
                return float(abs(last_x - gx) + abs(last_y - gy))
        if ctx.goal and ctx.start:
            gx, gy = ctx.goal
            sx, sy = ctx.start
            return float(abs(sx - gx) + abs(sy - gy))
        return 0.0

    def _peer_dist_estimate(self, packet: "IntentPacket") -> float:  # type: ignore[name-defined]
        """
        Estimate peer's distance-to-goal from its trajectory:
        the last waypoint's spatial position relative to the first.
        """
        pts = packet.trajectory_points
        if len(pts) < 2:
            return 0.0
        x0, y0, _ = pts[0]
        xl, yl, _ = pts[-1]
        return float(abs(xl - x0) + abs(yl - y0))

    def __repr__(self) -> str:
        return (
            f"ConflictResolver("
            f"robot={self._ctx.robot_id!r}, "
            f"state={self._ctx.state.name}, "
            f"priority={self._my_priority:.3f})"
        )


# ===========================================================================
# Deadlock resolution test
# ===========================================================================

def test_deadlock_resolution() -> None:
    """
    Simulate the classic 1-lane head-on standoff:

        Layout (6-wide × 5-tall, row 2 is a single-cell-wide corridor):

            Col:  0  1  2  3  4  5
         Row 0:  .  .  .  .  .  .
         Row 1:  #  #  J  #  #  #   J = junction  (col 2 has 3 neighbors)
         Row 2:  .  .  .  .  .  .   <-- 1-lane corridor (row 2)
         Row 3:  #  #  J  #  #  #   J = junction  (col 2 has 3 neighbors)
         Row 4:  .  .  .  .  .  .

        AMR-1 starts at (0, 2) → goal (5, 2)   [travelling East]
        AMR-2 starts at (5, 2) → goal (0, 2)   [travelling West]

    Both robots share a LocalReservationTable.  AMR-1 is planned first and
    commits its path.  AMR-2 detects the conflict, computes priorities, yields
    (REVERSING to junction col-2), and then replans after AMR-1 passes.

    Verification:
        * Both robots reach their goals.
        * No vertex collision occurs at any tick.
        * The lower-priority robot enters REVERSING state at least once.
    """
    from core.grid_map import GridMap, WALKWAY, SHELF
    from core.space_time_astar import ReservationTable as STATable, SpaceTimeAstar

    print("=" * 65)
    print("Deadlock Resolution Test  —  1-lane Head-On Standoff")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Build the 1-lane corridor grid
    # ------------------------------------------------------------------
    #
    #   6 cols × 5 rows
    #   Row 0: all walkable (open)
    #   Row 1: shelves except col 2 (junction)
    #   Row 2: all walkable (the single-lane corridor)
    #   Row 3: shelves except col 2 (junction)
    #   Row 4: all walkable (open)
    #
    W, S = WALKWAY, SHELF
    raw_grid = [
        [W, W, W, W, W, W],   # row 0
        [S, S, W, S, S, S],   # row 1  (col 2 is junction)
        [W, W, W, W, W, W],   # row 2  (1-lane corridor)
        [S, S, W, S, S, S],   # row 3  (col 2 is junction)
        [W, W, W, W, W, W],   # row 4
    ]
    grid = GridMap(raw_grid)
    print(f"\nGrid layout:\n{grid.ascii_render()}\n")

    # Verify junction finder correctly classifies col-2 in corridor as junction
    jf = JunctionFinder(grid)
    assert jf.is_junction(2, 2), "col-2 row-2 should be a junction (3 neighbours)"
    assert not jf.is_junction(0, 2), "col-0 row-2 is a dead-end (1 neighbor)"
    assert not jf.is_junction(3, 2), "col-3 row-2 is a corridor (2 neighbors)"

    # ------------------------------------------------------------------
    # Shared infrastructure
    # ------------------------------------------------------------------
    shared_sta_table = STATable()

    local_rt_1 = LocalReservationTable()
    local_rt_2 = LocalReservationTable()

    ctx_1 = RobotContext(robot_id="AMR_1", urgency=1.0, battery=80.0)
    ctx_2 = RobotContext(robot_id="AMR_2", urgency=1.0, battery=80.0)

    resolver_1 = ConflictResolver(grid, local_rt_1, shared_sta_table, ctx_1)
    resolver_2 = ConflictResolver(grid, local_rt_2, shared_sta_table, ctx_2)

    # ------------------------------------------------------------------
    # Compute initial priorities (deterministic)
    # ------------------------------------------------------------------
    dist_1 = GridMap.manhattan_distance(0, 2, 5, 2)  # = 5
    dist_2 = GridMap.manhattan_distance(5, 2, 0, 2)  # = 5

    p1 = PriorityEngine.compute_priority("AMR_1", urgency=1.0, battery=80.0, dist_to_goal=dist_1)
    p2 = PriorityEngine.compute_priority("AMR_2", urgency=1.0, battery=80.0, dist_to_goal=dist_2)

    winner_id   = "AMR_1" if p1 > p2 else "AMR_2"
    yielder_id  = "AMR_2" if p1 > p2 else "AMR_1"
    print(f"  Priority AMR_1 = {p1:.6f}")
    print(f"  Priority AMR_2 = {p2:.6f}")
    print(f"  Winner (higher priority) : {winner_id}")
    print(f"  Yielder (lower priority) : {yielder_id}\n")

    # Both sides must agree — no negotiation
    assert p1 != p2, "Tie-breaker must make priorities strictly different."

    # ------------------------------------------------------------------
    # AMR_1 plans and commits its path first (tick=0)
    # ------------------------------------------------------------------
    ok1 = resolver_1.set_goal(start=(0, 2), goal=(5, 2), current_tick=0,
                               urgency=1.0, battery=80.0)
    assert ok1, "AMR_1 initial planning failed."
    print(f"  AMR_1 initial path: {ctx_1.planned_path}")

    # ------------------------------------------------------------------
    # AMR_2 plans with AMR_1's path already committed in shared STA table
    # ------------------------------------------------------------------
    ok2 = resolver_2.set_goal(start=(5, 2), goal=(0, 2), current_tick=0,
                               urgency=1.0, battery=80.0)
    # AMR_2 may or may not find a path immediately; if not, it will replan
    print(f"  AMR_2 initial path: {ctx_2.planned_path}  (ok={ok2})")

    # ------------------------------------------------------------------
    # AMR_2 ingests AMR_1's trajectory as a peer packet
    # ------------------------------------------------------------------
    # Simulate what P2PNode.on_packet callback would deliver
    resolver_2._local_rt.ingest_trajectory(
        peer_id="AMR_1",
        priority=p1,
        waypoints=ctx_1.planned_path,
    )
    # Trigger conflict resolution on AMR_2
    conflicts = resolver_2._local_rt.find_conflicts("AMR_2", ctx_2.planned_path)
    print(f"\n  AMR_2 detects {len(conflicts)} conflict(s) with AMR_1")

    if conflicts:
        # Manually trigger resolution as P2P on_peer_packet would
        resolver_2._my_priority = p2
        resolver_2._resolve_conflicts("AMR_1", p1)
        print(f"  AMR_2 state after resolution: {ctx_2.state.name}")

    # ------------------------------------------------------------------
    # Simulate discrete-tick execution
    # ------------------------------------------------------------------
    print("\n  --- Tick-by-tick simulation ---")
    print(f"  {'Tick':<5}  {'AMR_1 pos':<12}  {'AMR_1 state':<14}  "
          f"{'AMR_2 pos':<12}  {'AMR_2 state'}")
    print(f"  {'-'*75}")

    pos_1: Position = (0, 2)
    pos_2: Position = (5, 2)

    MAX_TICKS = 40
    occupied_positions: Dict[int, Dict[str, Position]] = {}  # tick → {robot_id: pos}
    reversing_observed = False

    for tick in range(MAX_TICKS):
        occupied_positions[tick] = {"AMR_1": pos_1, "AMR_2": pos_2}

        # Vertex collision check
        assert pos_1 != pos_2, (
            f"COLLISION at tick={tick}: both robots at {pos_1}"
        )

        state_1_name = ctx_1.state.name
        state_2_name = ctx_2.state.name

        if ctx_2.state == RobotState.REVERSING:
            reversing_observed = True

        print(
            f"  {tick:<5}  {str(pos_1):<12}  {state_1_name:<14}  "
            f"{str(pos_2):<12}  {state_2_name}"
        )

        if ctx_1.state == RobotState.GOAL_REACHED and ctx_2.state == RobotState.GOAL_REACHED:
            print(f"\n  Both robots reached their goals at tick={tick}!")
            break

        # AMR_1 tick
        if ctx_1.state not in (RobotState.GOAL_REACHED, RobotState.IDLE):
            next_1 = resolver_1.tick(current_pos=pos_1, current_tick=tick)
            if next_1 is not None:
                pos_1 = next_1

        # AMR_2 tick
        if ctx_2.state not in (RobotState.GOAL_REACHED, RobotState.IDLE):
            next_2 = resolver_2.tick(current_pos=pos_2, current_tick=tick)
            if next_2 is not None:
                pos_2 = next_2
    else:
        # Allow loop to complete for long deadlock scenarios
        pass

    # ------------------------------------------------------------------
    # Final assertions
    # ------------------------------------------------------------------
    print("\n  --- Assertions ---")

    assert ctx_1.state == RobotState.GOAL_REACHED, (
        f"AMR_1 did not reach goal.  Final state: {ctx_1.state.name}, pos: {pos_1}"
    )
    assert ctx_2.state == RobotState.GOAL_REACHED, (
        f"AMR_2 did not reach goal.  Final state: {ctx_2.state.name}, pos: {pos_2}"
    )
    print("  AMR_1 reached goal (5,2)  [OK]")
    print("  AMR_2 reached goal (0,2)  [OK]")

    # No vertex collisions
    for t, positions in occupied_positions.items():
        p1t = positions["AMR_1"]
        p2t = positions["AMR_2"]
        assert p1t != p2t, f"Vertex collision at tick={t}: {p1t}"
    print("  No vertex collisions detected  [OK]")

    # Lower-priority robot must have entered REVERSING
    assert reversing_observed, (
        f"Yielder ({yielder_id}) never entered REVERSING state — "
        f"deadlock resolution did not trigger."
    )
    print(f"  {yielder_id} correctly entered REVERSING state  [OK]")

    # Priorities are deterministic — both sides agree on the same winner
    p1_check = PriorityEngine.compute_priority("AMR_1", 1.0, 80.0, 5.0)
    p2_check = PriorityEngine.compute_priority("AMR_2", 1.0, 80.0, 5.0)
    assert math.isclose(p1, p1_check) and math.isclose(p2, p2_check), (
        "Priority not deterministic — MD5 tie-breaker is non-repeatable!"
    )
    print("  Priorities are deterministic (MD5 tie-breaker consistent)  [OK]")

    print("\n" + "=" * 65)
    print("DEADLOCK RESOLUTION TEST PASSED")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    test_deadlock_resolution()
