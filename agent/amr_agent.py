"""
agent/amr_agent.py
==================
Master AMR Agent — unifies all six preceding modules into a single,
tick-driven entity that runs at 10 Hz on edge hardware.

Full dependency graph
---------------------
    AMRAgent
    ├── core.grid_map          GridMap          (shared, read/write)
    ├── core.space_time_astar  SpaceTimeAstar   (replanning)
    │                          ReservationTable (shared with fleet)
    ├── p2p.spatial_mesh       P2PNode          (mesh comms — optional)
    ├── p2p.protocol           IntentPacket     (trajectory broadcast)
    ├── coordination.conflict_resolver
    │                          ConflictResolver (priority + deadlock)
    │                          RobotContext     (FSM state holder)
    ├── tasks.auction_manager  AuctionManager   (Contract-Net)
    ├── tasks.target_lease     TargetLeaseTable (spatial locking)
    └── safety.hazard_manager  HazardManager    (kinematic + costmap)

Agent FSM
---------
    IDLE ──► BIDDING ──► PLANNING ──► NAVIGATING
                │                         │
                │            conflict ────► YIELDING ──► NAVIGATING
                │                         │
                │         stuck+lane ─────► REVERSING ──► NAVIGATING
                │                         │
                │         hazard ──────────► REROUTING ──► NAVIGATING
                │                         │
                │         obstacle <0.8m ──► EMERGENCY_STOP
                │
                └──► IDLE (task complete / no task)

Tick cycle (10 Hz, 100 ms)
---------------------------
    1. Update simulated battery drain.
    2. Process all queued inbound P2P messages.
    3. Run KinematicSafetyBumper on latest sensor reading.
    4. Run GhostNodeEviction check.
    5. Dispatch FSM action for current state.
    6. Advance position along planned path (if NAVIGATING).
    7. Broadcast IntentPacket to spatial-mesh peers.
    8. Update GodotBridge telemetry snapshot.

Zero external dependencies — pure standard library.
"""

from __future__ import annotations

import logging
import math
import time as _time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Cycle 1
# ---------------------------------------------------------------------------
from core.grid_map import GridMap
from core.space_time_astar import (
    ReservationTable as STATable,
    SpaceTimeAstar,
)

# ---------------------------------------------------------------------------
# Cycle 2  (optional — graceful when running without network)
# ---------------------------------------------------------------------------
try:
    from p2p.protocol import IntentPacket, sign_packet, serialise
    from p2p.spatial_mesh import P2PNode, SpatialHasher, PeerRegistry, PeerInfo
    _HAS_P2P = True
except ImportError:
    _HAS_P2P = False
    P2PNode = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Cycle 3
# ---------------------------------------------------------------------------
from coordination.reservation_table import LocalReservationTable
from coordination.conflict_resolver import (
    ConflictResolver,
    RobotContext,
    RobotState,
    PriorityEngine,
)

# ---------------------------------------------------------------------------
# Cycle 4
# ---------------------------------------------------------------------------
from tasks.auction_manager import (
    AuctionManager,
    AuctionParticipantState,
    Task,
    TaskAnnouncement,
    Bid,
    BidFormula,
)
from tasks.target_lease import TargetLeaseTable, LeaseAcquiredMessage

# ---------------------------------------------------------------------------
# Cycle 5
# ---------------------------------------------------------------------------
from safety.hazard_manager import (
    HazardManager,
    HazardAlertMessage,
    SpeedZone,
    SpeedCommand,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TICK_HZ:            float = 10.0          # target frequency
TICK_PERIOD_S:      float = 1.0 / TICK_HZ # 100 ms
BATTERY_DRAIN_PER_TICK: float = 0.002     # % per tick (0 → 100 in ~8 min)
BROADCAST_EVERY_N_TICKS: int  = 3         # intent broadcast rate (≈3 Hz)
SENSOR_RANGE_CELLS: float = 6.0           # simulated LIDAR range (cells)

Position = Tuple[int, int]
STNode   = Tuple[int, int, int]


# ---------------------------------------------------------------------------
# AMR agent-level FSM states (superset of RobotState)
# ---------------------------------------------------------------------------
class AgentState(Enum):
    IDLE            = "IDLE"
    BIDDING         = "BIDDING"
    PLANNING        = "PLANNING"
    NAVIGATING      = "NAVIGATING"
    YIELDING        = "YIELDING"
    REVERSING       = "REVERSING"
    REROUTING       = "REROUTING"
    EMERGENCY_STOP  = "EMERGENCY_STOP"


# ---------------------------------------------------------------------------
# AMRAgent
# ---------------------------------------------------------------------------
class AMRAgent:
    """
    Master AMR Agent encapsulating all six preceding modules.

    Parameters
    ----------
    robot_id : str
        Globally unique identifier (e.g. "AMR_001").
    start_pos : (col, row)
        Initial grid position.
    grid : GridMap
        Warehouse grid (shared with other agents).
    sta_table : STATable
        STA* reservation table (shared with other agents for collision
        avoidance during planning).
    battery : float
        Initial battery level [0, 100].
    urgency : float
        Default task urgency [0, 1].
    on_intent_broadcast : callable | None
        Called with a signed ``IntentPacket`` after each trajectory update.
        Use this to inject into P2PNode or the benchmark peer exchange.
    on_hazard_broadcast : callable | None
        Called with a ``HazardAlertMessage`` on new obstacle detection.
    on_lease_broadcast : callable | None
        Called with a ``LeaseAcquiredMessage`` when the agent wins an auction.
    on_task_complete : callable | None
        Called with the completed ``Task`` when goal is reached.
    """

    def __init__(
        self,
        robot_id:             str,
        start_pos:            Position,
        grid:                 GridMap,
        sta_table:            STATable,
        battery:              float   = 100.0,
        urgency:              float   = 1.0,
        on_intent_broadcast:  Optional[Callable] = None,
        on_hazard_broadcast:  Optional[Callable] = None,
        on_lease_broadcast:   Optional[Callable] = None,
        on_task_complete:     Optional[Callable] = None,
    ) -> None:
        self.robot_id   = robot_id
        self._pos:  Position = start_pos
        self._grid  = grid
        self._sta   = sta_table

        # ── Cycle 3 ──────────────────────────────────────────────────
        self._local_rt  = LocalReservationTable()
        self._ctx       = RobotContext(
            robot_id=robot_id,
            urgency=urgency,
            battery=battery,
        )
        self._resolver  = ConflictResolver(
            grid=grid,
            local_rt=self._local_rt,
            sta_table=sta_table,
            ctx=self._ctx,
        )

        # ── Cycle 4 ──────────────────────────────────────────────────
        self._lease_table = TargetLeaseTable()
        self._auction     = AuctionManager(
            robot_id=robot_id,
            position=start_pos,
            battery=battery,
            lease_table=self._lease_table,
            resolver=self._resolver,
            on_lease_broadcast=on_lease_broadcast,
            on_task_assigned=self._on_task_assigned,
            current_tick_fn=lambda: self._tick_count,
        )

        # ── Cycle 5 ──────────────────────────────────────────────────
        self._hazard = HazardManager(
            robot_id=robot_id,
            grid=grid,
            local_rt=self._local_rt,
            sta_table=sta_table,
            resolver=self._resolver,
            lease_table=self._lease_table,
            on_hazard_broadcast=on_hazard_broadcast,
        )

        # ── External callbacks ────────────────────────────────────────
        self._on_intent_broadcast = on_intent_broadcast
        self._on_task_complete    = on_task_complete

        # ── FSM ──────────────────────────────────────────────────────
        self._state: AgentState = AgentState.IDLE
        self._speed_frac: float = 1.0   # 0.0 = stopped, 1.0 = full speed
        self._heading:    float = 0.0   # degrees East = 0, North = 90 etc.

        # ── Tick bookkeeping ─────────────────────────────────────────
        self._tick_count:      int   = 0
        self._last_sensor_dist: float = math.inf
        self._path_cursor:     int   = 0
        self._current_task:    Optional[Task] = None

        # ── Telemetry snapshot (read by GodotBridge) ─────────────────
        self.telemetry: Dict = {
            "robot_id": robot_id,
            "x": float(start_pos[0]),
            "y": float(start_pos[1]),
            "heading": 0.0,
            "state": AgentState.IDLE.value,
            "battery": battery,
            "laser_trail": [],
            "speed_frac": 1.0,
        }

        # Metrics
        self.tasks_completed:  int   = 0
        self.idle_ticks:       int   = 0
        self.collision_count:  int   = 0    # managed externally by BenchmarkRunner
        self.total_ticks:      int   = 0

    # ------------------------------------------------------------------
    # Public: main 10 Hz tick
    # ------------------------------------------------------------------
    def tick(
        self,
        delta_time:    float,
        sensor_dist_m: float = math.inf,
        obstacle_pos:  Optional[Position] = None,
    ) -> None:
        """
        Advance the agent by one simulation tick (~100 ms).

        Parameters
        ----------
        delta_time : float
            Elapsed real time since last tick (seconds).
        sensor_dist_m : float
            Distance to nearest dynamic obstacle (metres).
            Pass ``math.inf`` when the sensor sees no obstacles.
        obstacle_pos : (col, row) | None
            Precise grid position of detected obstacle (from depth camera).
        """
        self._tick_count += 1
        self.total_ticks += 1

        # 1. Battery drain
        self._ctx.battery = max(0.0, self._ctx.battery - BATTERY_DRAIN_PER_TICK)
        self._auction.update_battery(self._ctx.battery)

        # 2. Kinematic safety bumper
        cmd: SpeedCommand = self._hazard.on_sensor_reading(
            distance_m=sensor_dist_m,
            obstacle_pos=obstacle_pos,
            current_tick=self._tick_count,
        )
        self._speed_frac = cmd.speed_frac
        if cmd.zone == SpeedZone.EMERGENCY_STOP:
            self._transition(AgentState.EMERGENCY_STOP)
        elif self._state == AgentState.EMERGENCY_STOP and cmd.zone != SpeedZone.EMERGENCY_STOP:
            self._transition(AgentState.NAVIGATING if self._ctx.planned_path else AgentState.IDLE)

        # 3. Ghost node check + hazard replan
        hazard_path = self._hazard.tick(
            current_pos=self._pos,
            current_tick=self._tick_count,
            goal=self._ctx.goal,
        )
        if hazard_path:
            self._ctx.planned_path = hazard_path
            self._ctx.path_index   = 0
            self._transition(AgentState.REROUTING)

        # 3b. Purge stale reservations from the shared table
        self._sta.purge(self._tick_count)

        # 4. FSM dispatch
        self._fsm_step()

        # 5. Advance position
        active_states = (
            AgentState.NAVIGATING, AgentState.YIELDING, 
            AgentState.REVERSING, AgentState.REROUTING
        )
        if self._state in active_states and self._speed_frac > 0.0:
            self._advance_position()

        # 6. Count idle ticks
        if self._state == AgentState.IDLE:
            self.idle_ticks += 1

        # 7. Intent broadcast (every N ticks)
        if (
            self._tick_count % BROADCAST_EVERY_N_TICKS == 0
            and self._on_intent_broadcast is not None
        ):
            if self._ctx.planned_path and self._ctx.path_index < len(self._ctx.planned_path):
                remaining = self._ctx.planned_path[self._ctx.path_index:]
            else:
                # If idle/finished, project current position into the future to block it!
                remaining = [
                    (self._pos[0], self._pos[1], self._tick_count + dt) 
                    for dt in range(20)
                ]
            self._on_intent_broadcast(self.robot_id, remaining)

        # 8. Update telemetry
        self._update_telemetry()

    # ------------------------------------------------------------------
    # External message ingestion
    # ------------------------------------------------------------------
    def on_peer_intent(self, peer_id: str, waypoints: List[STNode]) -> None:
        """
        Ingest a peer's broadcast trajectory.
        Feeds ConflictResolver and GhostNodeEviction heartbeat.
        """
        if not waypoints:
            return
        first_pos: Position = (waypoints[0][0], waypoints[0][1])
        self._hazard.ghost_eviction.heartbeat(peer_id, first_pos)
        # A fully stationary broadcast means the peer is IDLE/parked and
        # cannot yield — treat it as an immovable obstacle so we always
        # yield (mirrors ConflictResolver.on_peer_packet).
        is_stationary = all(
            x == waypoints[0][0] and y == waypoints[0][1]
            for x, y, _ in waypoints
        )
        if is_stationary:
            peer_priority = float("inf")
        else:
            # Estimate peer priority conservatively
            peer_priority = PriorityEngine.compute_priority(
                robot_id=peer_id, urgency=1.0, battery=100.0, dist_to_goal=len(waypoints)
            )
        self._local_rt.ingest_trajectory(
            peer_id=peer_id, priority=peer_priority, waypoints=waypoints
        )
        # Runtime conflict check (was previously dead code — intents were
        # ingested but never acted on, so NAVIGATING agents drove open-loop
        # into parked peers, e.g. the (10,6) tick-56 crash).
        if (
            self._ctx.goal is not None
            and self._ctx.planned_path
            and self._ctx.state in (RobotState.NAVIGATING, RobotState.YIELDING)
        ):
            self._resolver._resolve_conflicts(peer_id, peer_priority)
            # Propagate resolver state to the agent FSM immediately so the
            # yield takes effect on the next tick's _do_navigating/_do_yielding.
            if (
                self._ctx.state == RobotState.YIELDING
                and self._state in (AgentState.NAVIGATING, AgentState.REROUTING)
            ):
                self._transition(AgentState.YIELDING)
            elif (
                self._ctx.state == RobotState.REVERSING
                and self._state != AgentState.REVERSING
            ):
                self._transition(AgentState.REVERSING)

    def set_peer_positions(self, mapping: Dict[str, Position]) -> None:
        """Refresh the live peer-position snapshot (start-of-tick)."""
        self._resolver.update_peer_positions(
            {k: v for k, v in mapping.items() if k != self.robot_id}
        )

    def on_peer_hazard(self, msg: HazardAlertMessage) -> None:
        """Ingest a peer's hazard alert."""
        self._hazard.on_peer_hazard_alert(
            msg=msg,
            current_pos=self._pos,
            current_tick=self._tick_count,
            goal=self._ctx.goal,
        )

    def on_peer_lease(self, msg: LeaseAcquiredMessage) -> None:
        """Ingest a peer's lease acquisition."""
        self._auction.on_lease_acquired(msg)

    def on_task_announcement(self, task: Task) -> bool:
        """
        Evaluate an incoming task.  Returns True if a bid was submitted.
        """
        if self._state != AgentState.IDLE:
            return False
        bid_val = BidFormula.compute(
            self.robot_id, self._pos, task.pickup_pos, self._ctx.battery
        )
        my_bid = Bid(
            task_id=task.task_id,
            robot_id=self.robot_id,
            bid_value=bid_val,
            battery=self._ctx.battery,
            position=self._pos,
        )
        # Register auction + own bid
        from tasks.auction_manager import _AuctionState
        import time as t_
        self._auction._auctions[task.task_id] = _AuctionState(
            task=task,
            my_bid=my_bid,
            bids={self.robot_id: my_bid},
            close_time=t_.time() + 0.15,
        )
        self._auction._state = AuctionParticipantState.BIDDING
        self._transition(AgentState.BIDDING)
        return True

    def on_peer_bid(self, bid: Bid) -> None:
        """Receive a competitor's bid for an open auction."""
        auction = self._auction._auctions.get(bid.task_id)
        if auction:
            auction.bids[bid.robot_id] = bid

    def resolve_auction(self, task_id: str) -> Optional[str]:
        """Resolve an open auction synchronously. Returns winner robot_id."""
        winner = self._auction.resolve_sync(task_id, self._tick_count)
        if winner == self.robot_id:
            self._transition(AgentState.PLANNING)
        else:
            # Loser — explicitly reset auction state and return to IDLE
            self._auction._state = AuctionParticipantState.IDLE
            self._transition(AgentState.IDLE)
        return winner

    # ------------------------------------------------------------------
    # Internal FSM
    # ------------------------------------------------------------------
    def _fsm_step(self) -> None:
        state = self._state
        if state == AgentState.EMERGENCY_STOP:
            return
        if state == AgentState.IDLE:
            return
        if state == AgentState.BIDDING:
            return
        if state == AgentState.PLANNING:
            self._do_planning()
        elif state in (AgentState.NAVIGATING, AgentState.REROUTING):
            self._do_navigating()
        elif state == AgentState.YIELDING:
            self._do_yielding()
        elif state == AgentState.REVERSING:
            self._do_reversing()

    def _do_planning(self) -> None:
        """Run STA* for the current task goal via ConflictResolver."""
        task = self._current_task
        if task is None:
            self._transition(AgentState.IDLE)
            return

        ok = self._resolver.set_goal(
            start=self._pos,
            goal=task.pickup_pos,
            current_tick=self._tick_count,
            urgency=task.urgency,
            battery=self._ctx.battery,
        )
        if ok:
            self._transition(AgentState.NAVIGATING)
        else:
            logger.warning("[%s] Planning failed -- back to IDLE.", self.robot_id)
            self._current_task = None
            self._transition(AgentState.IDLE)
            self._protect_idle_pos()

    def _protect_idle_pos(self) -> None:
        """Reserve current position in the shared STA table to prevent collisions before the next broadcast."""
        for dt in range(20):
            self._sta.reserve_vertex(self._pos[0], self._pos[1], self._tick_count + dt, self.robot_id)

    def _do_navigating(self) -> None:
        """Check for goal reached or resolver state transitions."""
        # Position-based goal detection (most reliable)
        if self._ctx.goal and self._pos == self._ctx.goal:
            self._on_goal_reached()
            return
        c_state = self._ctx.state
        if c_state == RobotState.GOAL_REACHED:
            self._on_goal_reached()
        elif c_state == RobotState.YIELDING:
            self._transition(AgentState.YIELDING)
        elif c_state == RobotState.REVERSING:
            self._transition(AgentState.REVERSING)

    def _do_yielding(self) -> None:
        c_state = self._ctx.state
        if c_state == RobotState.NAVIGATING:
            self._transition(AgentState.NAVIGATING)
        elif c_state == RobotState.GOAL_REACHED:
            self._on_goal_reached()

    def _do_reversing(self) -> None:
        c_state = self._ctx.state
        if c_state == RobotState.NAVIGATING:
            self._transition(AgentState.NAVIGATING)

    def _on_goal_reached(self) -> None:
        task = self._current_task
        if task is not None:
            self.tasks_completed += 1
            self._lease_table.release(self.robot_id, task.pickup_pos)
            if self._on_task_complete:
                self._on_task_complete(task)
            logger.info("[%s] Task %s COMPLETE.", self.robot_id, task.task_id)
        self._current_task = None
        self._ctx.state    = RobotState.IDLE
        self._transition(AgentState.IDLE)
        self._protect_idle_pos()

    def _on_task_assigned(self, task: Task) -> None:
        self._current_task = task
        self._transition(AgentState.PLANNING)

    # ------------------------------------------------------------------
    # Position advance
    # ------------------------------------------------------------------
    def _advance_position(self) -> None:
        """Call ConflictResolver to determine next position based on FSM."""
        next_pos = self._resolver.tick(self._pos, self._tick_count)
        if next_pos and next_pos != self._pos:
            dx = next_pos[0] - self._pos[0]
            dy = next_pos[1] - self._pos[1]
            self._heading = math.degrees(math.atan2(-dy, dx)) % 360
            self._pos = next_pos
            self._auction.update_position(self._pos)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _transition(self, new_state: AgentState) -> None:
        if new_state != self._state:
            logger.debug("[%s] %s -> %s", self.robot_id, self._state.value, new_state.value)
            self._state = new_state

    def _update_telemetry(self) -> None:
        trail = [(x, y) for x, y, _ in self._ctx.planned_path[self._ctx.path_index:]][:10]
        self.telemetry.update({
            "robot_id":   self.robot_id,
            "x":          float(self._pos[0]),
            "y":          float(self._pos[1]),
            "heading":    round(self._heading, 1),
            "state":      self._state.value,
            "battery":    round(self._ctx.battery, 1),
            "laser_trail": trail,
            "speed_frac": self._speed_frac,
        })

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------
    @property
    def position(self) -> Position:
        return self._pos

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def battery(self) -> float:
        return self._ctx.battery

    @property
    def tick_count(self) -> int:
        return self._tick_count

    def __repr__(self) -> str:
        return (
            f"AMRAgent(id={self.robot_id!r}, pos={self._pos}, "
            f"state={self._state.value}, battery={self._ctx.battery:.1f}%)"
        )
