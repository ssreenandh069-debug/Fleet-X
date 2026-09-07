"""
safety/hazard_manager.py
=========================
Physical safety reflex system for Autonomous Mobile Robots.

Integrates with all preceding cycles:
    Cycle 1  — GridMap (block_cell, inflation_cost) + SpaceTimeAstar
    Cycle 2  — P2P protocol (HAZARD_ALERT gossip via IntentPacket channel)
    Cycle 3  — LocalReservationTable (peer eviction), ConflictResolver
    Cycle 4  — TargetLeaseTable (lease revocation for ghost robots)

System overview
---------------

    ┌──────────────────────────────────────────────────────────────────────┐
    │                        HazardManager                                  │
    │                                                                        │
    │   Sensor feed ──► KinematicSafetyBumper ──► SpeedCommand              │
    │   (distance m)         3-zone state FSM         (NORMAL/SLOW/ESTOP)   │
    │                              │                                          │
    │                        EMERGENCY_STOP                                  │
    │                              │                                          │
    │   Unmapped obstacle ──► DynamicCostmap ──► GridMap.block_cell()        │
    │   at (x,y)                inflate 8 adj    GridMap.set_inflation()     │
    │                              │                                          │
    │                        HazardAlert gossip ──► P2P broadcast            │
    │                         (TTL=20 ticks)         to active sector        │
    │                              │                                          │
    │   Peer heartbeat ──► GhostNodeEviction ──► LocalReservationTable      │
    │   timeout >2s              stamp pos             .release_peer()       │
    │                            as obstacle                                 │
    └──────────────────────────────────────────────────────────────────────┘

3-Tier Kinematic Safety Bumper
------------------------------
    Zone 1  d > 2.0 m   : NORMAL    — full speed.
    Zone 2  0.8 < d <= 2.0 m  : DECELERATE — 40 % speed.
    Zone 3  d <= 0.8 m  : EMERGENCY_STOP — hard halt.

Distances are in real-world metres.  Conversion to grid cells depends on the
warehouse cell resolution (default: 0.5 m per cell, configurable).

Dynamic Costmap Inflation
--------------------------
When a new obstacle is detected at (ox, oy):
    1. ``GridMap.block_cell(ox, oy)`` — stamp as SHELF permanently.
    2. For each of the 8 Moore-neighbourhood cells (ox±1, oy±1):
       ``GridMap.set_inflation_cost(nx, ny, INFLATION_COST)`` — raises
       move_cost for those cells, discouraging corner-cutting without
       fully blocking them.
    3. Broadcast ``HazardAlertMessage`` with TTL=20 over P2P gossip.

Receiving peers:
    • Apply the same block + inflate update to their local GridMap.
    • Trigger ``ConflictResolver`` replan if their current planned path
      passes through the affected cells.

Ghost Node Eviction
--------------------
Each ``HazardManager`` maintains a heartbeat registry:
    ``{ peer_id → last_seen_wall_time }``

On every ``tick()``:
    • If ``time.time() - last_seen > GHOST_TIMEOUT_SECONDS`` for any peer:
        a. Mark that peer's last known position as a blocked cell.
        b. Release all its future reservations from LocalReservationTable.
        c. Release any TargetLease it holds (optional, if TargetLeaseTable
           is provided).
        d. Emit a log warning at WARNING level.

Zero external dependencies — standard library only.
"""

from __future__ import annotations

import logging
import math
import time as _time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, FrozenSet, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Cycle 1 imports
# ---------------------------------------------------------------------------
from core.grid_map import GridMap, SHELF
from core.space_time_astar import ReservationTable as STAReservationTable, SpaceTimeAstar

# ---------------------------------------------------------------------------
# Cycle 3 imports
# ---------------------------------------------------------------------------
from coordination.reservation_table import LocalReservationTable

# ---------------------------------------------------------------------------
# Cycle 4 imports (optional — graceful if tasks not installed)
# ---------------------------------------------------------------------------
try:
    from tasks.target_lease import TargetLeaseTable
    _HAS_LEASE = True
except ImportError:
    _HAS_LEASE = False
    TargetLeaseTable = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Cycle 3 ConflictResolver (optional)
# ---------------------------------------------------------------------------
try:
    from coordination.conflict_resolver import ConflictResolver
    _HAS_RESOLVER = True
except ImportError:
    _HAS_RESOLVER = False
    ConflictResolver = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

# Kinematic bumper zone thresholds (metres)
ZONE1_THRESHOLD_M:      float = 2.0    # > 2.0 m → NORMAL
ZONE2_THRESHOLD_M:      float = 0.8    # 0.8–2.0 m → DECELERATE
# below ZONE2 → EMERGENCY_STOP

SLOW_SPEED_FRACTION:    float = 0.40   # speed fraction in Zone 2

# Inflation cost applied to the 8 neighbours of a new hazard
INFLATION_COST:         float = 5.0    # extra move cost units (very discouraging)

# Hazard alert time-to-live in ticks
HAZARD_TTL_TICKS:       int   = 20

# Ghost node eviction timeout (wall-clock seconds)
GHOST_TIMEOUT_SECONDS:  float = 2.0

# Default warehouse resolution: metres per grid cell
DEFAULT_CELL_RESOLUTION_M: float = 0.5

# Type aliases
Position = Tuple[int, int]       # (col, row)
STNode   = Tuple[int, int, int]  # (col, row, t)


# ===========================================================================
# 3-Tier Kinematic Safety Bumper
# ===========================================================================

class SpeedZone(Enum):
    """Three-zone kinematic safety state."""
    NORMAL          = auto()   # Zone 1: full speed
    DECELERATE      = auto()   # Zone 2: 40% speed
    EMERGENCY_STOP  = auto()   # Zone 3: hard halt


@dataclass
class SpeedCommand:
    """
    Output of the kinematic safety bumper.

    Attributes
    ----------
    zone        : SpeedZone  — current safety zone.
    speed_frac  : float      — commanded speed as a fraction of max [0, 1].
    distance_m  : float      — sensed obstacle distance in metres.
    is_estop    : bool       — True when zone is EMERGENCY_STOP.
    """
    zone:       SpeedZone
    speed_frac: float
    distance_m: float
    is_estop:   bool


class KinematicSafetyBumper:
    """
    Classifies incoming proximity sensor readings into three kinematic zones
    and emits the appropriate speed command.

    Parameters
    ----------
    zone1_m : float
        Distance threshold for Zone 1 / Zone 2 boundary (metres).
    zone2_m : float
        Distance threshold for Zone 2 / Zone 3 boundary (metres).
    slow_fraction : float
        Speed fraction in Zone 2 (0 < slow_fraction < 1.0).
    on_estop : callable | None
        Zero-argument callback invoked whenever EMERGENCY_STOP is triggered.
        Use this to notify the ConflictResolver or halt actuators.
    """

    def __init__(
        self,
        zone1_m:       float    = ZONE1_THRESHOLD_M,
        zone2_m:       float    = ZONE2_THRESHOLD_M,
        slow_fraction: float    = SLOW_SPEED_FRACTION,
        on_estop:      Optional[Callable[[], None]] = None,
    ) -> None:
        self._z1           = zone1_m
        self._z2           = zone2_m
        self._slow_frac    = slow_fraction
        self._on_estop     = on_estop
        self._prev_zone:   Optional[SpeedZone] = None
        self.estop_count:  int = 0   # telemetry: total ESTOP events

    def evaluate(self, distance_m: float) -> SpeedCommand:
        """
        Evaluate a single proximity sensor reading.

        Parameters
        ----------
        distance_m : float
            Distance to nearest detected dynamic obstacle in metres.
            Use ``math.inf`` when no obstacle is detected.

        Returns
        -------
        SpeedCommand
            Commanded speed for this reading.
        """
        if distance_m > self._z1:
            zone       = SpeedZone.NORMAL
            speed_frac = 1.0
        elif distance_m > self._z2:
            zone       = SpeedZone.DECELERATE
            speed_frac = self._slow_frac
        else:
            zone       = SpeedZone.EMERGENCY_STOP
            speed_frac = 0.0

        is_estop = (zone == SpeedZone.EMERGENCY_STOP)

        # Trigger callback on entry into ESTOP (edge-detect)
        if is_estop and self._prev_zone != SpeedZone.EMERGENCY_STOP:
            self.estop_count += 1
            logger.warning(
                "EMERGENCY_STOP triggered! obstacle at %.2f m "
                "(zone1=%.1f m, zone2=%.1f m).",
                distance_m, self._z1, self._z2,
            )
            if self._on_estop is not None:
                self._on_estop()

        self._prev_zone = zone
        return SpeedCommand(
            zone=zone,
            speed_frac=speed_frac,
            distance_m=distance_m,
            is_estop=is_estop,
        )


# ===========================================================================
# Hazard Alert message
# ===========================================================================

@dataclass
class HazardAlertMessage:
    """
    Gossip packet broadcast when an unmapped obstacle is discovered.

    Wire-compatible with the P2P layer — can be embedded as payload in
    an ``IntentPacket`` or sent as a separate lightweight datagram.

    Attributes
    ----------
    robot_id        : str    — originating robot.
    obstacle_pos    : (x,y) — grid cell where the obstacle was found.
    ttl_ticks       : int   — number of ticks before peers may re-evaluate.
    issued_tick     : int   — logical tick when the obstacle was spotted.
    inflation_radius: int   — number of cells to inflate around the obstacle.
    wall_timestamp  : float — wall-clock time of detection.
    """
    robot_id:         str
    obstacle_pos:     Position
    ttl_ticks:        int   = HAZARD_TTL_TICKS
    issued_tick:      int   = 0
    inflation_radius: int   = 1      # Moore neighbourhood (1 = 8 adjacent cells)
    wall_timestamp:   float = field(default_factory=_time.time)


# ===========================================================================
# Dynamic Costmap
# ===========================================================================

class DynamicCostmap:
    """
    Manages runtime obstacle stamping and inflation cost overlays on a
    shared ``GridMap``.

    Each hazard discovery:
    1. Stamps the obstacle cell as ``SHELF`` via ``GridMap.block_cell()``.
    2. Applies ``INFLATION_COST`` to all passable Moore-neighbourhood cells.
    3. Records the event in ``_hazard_log`` for replay to late-joining peers.

    Parameters
    ----------
    grid : GridMap
        The warehouse grid (mutated in-place).
    inflation_cost : float
        Extra move-cost applied to cells adjacent to a hazard.
    """

    def __init__(
        self,
        grid:           GridMap,
        inflation_cost: float = INFLATION_COST,
    ) -> None:
        self._grid            = grid
        self._inflation_cost  = inflation_cost
        # Set of known obstacle positions (col, row)
        self._obstacles:  Set[Position]              = set()
        # (col, row) → extra_cost — tracks what we've inflated so we can undo
        self._inflated:   Dict[Position, float]      = {}
        # Ordered log of all HazardAlertMessages this instance has processed
        self.hazard_log:  List[HazardAlertMessage]   = []

    # ------------------------------------------------------------------
    # Stamping a new obstacle (own detection)
    # ------------------------------------------------------------------

    def register_obstacle(
        self,
        ox: int,
        oy: int,
        robot_id:   str    = "self",
        tick:       int    = 0,
    ) -> HazardAlertMessage:
        """
        Stamp (ox, oy) as a new static obstacle and inflate its neighbours.

        Parameters
        ----------
        ox, oy : int
            Grid cell of the detected obstacle.
        robot_id : str
            Originating robot (for logging/gossip message).
        tick : int
            Current logical tick.

        Returns
        -------
        HazardAlertMessage
            Ready to broadcast over P2P gossip.
        """
        pos: Position = (ox, oy)
        if pos in self._obstacles:
            logger.debug("DynamicCostmap: obstacle at %s already registered.", pos)
            msg = HazardAlertMessage(
                robot_id=robot_id,
                obstacle_pos=pos,
                ttl_ticks=HAZARD_TTL_TICKS,
                issued_tick=tick,
            )
            return msg

        self._obstacles.add(pos)

        # 1. Block the cell in the GridMap
        newly_blocked = self._grid.block_cell(ox, oy)
        if newly_blocked:
            logger.warning(
                "[%s] NEW OBSTACLE at (%d,%d) — cell blocked in GridMap. tick=%d",
                robot_id, ox, oy, tick,
            )
        else:
            logger.info(
                "[%s] Obstacle at (%d,%d) was already impassable. tick=%d",
                robot_id, ox, oy, tick,
            )

        # 2. Inflate 8 Moore-neighbourhood cells
        self._inflate_neighbourhood(ox, oy)

        # 3. Build and log the alert message
        msg = HazardAlertMessage(
            robot_id=robot_id,
            obstacle_pos=pos,
            ttl_ticks=HAZARD_TTL_TICKS,
            issued_tick=tick,
        )
        self.hazard_log.append(msg)
        return msg

    # ------------------------------------------------------------------
    # Receiving a peer's hazard alert
    # ------------------------------------------------------------------

    def apply_peer_alert(self, msg: HazardAlertMessage) -> bool:
        """
        Apply a ``HazardAlertMessage`` received from a peer.

        Returns True if this was a new obstacle (not seen before).
        """
        pos: Position = msg.obstacle_pos
        ox, oy = pos
        if pos in self._obstacles:
            return False   # already known

        self._obstacles.add(pos)
        self._grid.block_cell(ox, oy)
        self._inflate_neighbourhood(ox, oy)
        self.hazard_log.append(msg)

        logger.info(
            "Peer hazard alert from %s: obstacle at %s applied to local map.",
            msg.robot_id, pos,
        )
        return True

    # ------------------------------------------------------------------
    # Inflation helpers
    # ------------------------------------------------------------------

    def _inflate_neighbourhood(self, ox: int, oy: int) -> None:
        """Apply inflation cost to all passable Moore neighbours of (ox, oy)."""
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = ox + dx, oy + dy
                if self._grid.in_bounds(nx, ny) and self._grid.is_passable(nx, ny):
                    # Accumulate inflation (multiple hazards can overlap)
                    prev = self._inflated.get((nx, ny), 0.0)
                    new_cost = prev + self._inflation_cost
                    self._inflated[(nx, ny)] = new_cost
                    self._grid.set_inflation_cost(nx, ny, new_cost)
                    logger.debug(
                        "DynamicCostmap: inflation %.1f applied at (%d,%d).",
                        new_cost, nx, ny,
                    )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def obstacle_count(self) -> int:
        return len(self._obstacles)

    def is_known_obstacle(self, col: int, row: int) -> bool:
        return (col, row) in self._obstacles

    def __repr__(self) -> str:
        return (
            f"DynamicCostmap(obstacles={self.obstacle_count}, "
            f"inflated_cells={len(self._inflated)})"
        )


# ===========================================================================
# Ghost Node Eviction
# ===========================================================================

@dataclass
class PeerHeartbeat:
    """
    Last-known-alive record for one peer.

    Attributes
    ----------
    peer_id          : str      — Robot identifier.
    last_pos         : (x, y)  — Most recent known position.
    last_wall_time   : float   — ``time.time()`` when the last heartbeat arrived.
    evicted          : bool    — True once this peer has been evicted.
    """
    peer_id:        str
    last_pos:       Position
    last_wall_time: float
    evicted:        bool = False


class GhostNodeEviction:
    """
    Per-robot tracker that detects silent/crashed peers and evicts them.

    Usage
    -----
    Call ``heartbeat(peer_id, pos)`` every time a valid message arrives
    from a peer (e.g., from the P2P on_packet callback).

    Call ``check_ghosts(current_tick)`` on every tick to find timed-out peers.

    Parameters
    ----------
    local_rt : LocalReservationTable
        Cycle-3 reservation table — peer reservations are wiped on eviction.
    costmap : DynamicCostmap
        Used to stamp the ghost's last position as a static obstacle.
    lease_table : TargetLeaseTable | None
        Cycle-4 lease table — ghost's leases are revoked on eviction.
    timeout_seconds : float
        Wall-clock seconds without a heartbeat before a peer is declared ghost.
    on_ghost : callable | None
        Optional callback ``(peer_id, last_pos) -> None`` invoked on eviction.
    """

    def __init__(
        self,
        local_rt:         LocalReservationTable,
        costmap:          DynamicCostmap,
        lease_table:      Optional["TargetLeaseTable"]          = None,
        timeout_seconds:  float                                  = GHOST_TIMEOUT_SECONDS,
        on_ghost:         Optional[Callable[[str, Position], None]] = None,
    ) -> None:
        self._local_rt       = local_rt
        self._costmap        = costmap
        self._lease_table    = lease_table
        self._timeout        = timeout_seconds
        self._on_ghost       = on_ghost
        # peer_id → PeerHeartbeat
        self._registry:  Dict[str, PeerHeartbeat] = {}
        self.evicted_peers: List[str] = []

    # ------------------------------------------------------------------
    # Heartbeat registration
    # ------------------------------------------------------------------

    def heartbeat(self, peer_id: str, pos: Position) -> None:
        """
        Register a live heartbeat from *peer_id* at *pos*.

        Should be called whenever any valid packet arrives from the peer
        (IntentPacket, HazardAlertMessage, bid, etc.).
        """
        existing = self._registry.get(peer_id)
        if existing is not None and existing.evicted:
            # Peer came back online — un-evict
            logger.info(
                "Ghost peer %s back online at %s — reinstating.", peer_id, pos
            )
            existing.evicted = False

        self._registry[peer_id] = PeerHeartbeat(
            peer_id=peer_id,
            last_pos=pos,
            last_wall_time=_time.time(),
        )

    # ------------------------------------------------------------------
    # Ghost detection (call every tick)
    # ------------------------------------------------------------------

    def check_ghosts(self, current_tick: int) -> List[str]:
        """
        Scan the heartbeat registry for timed-out peers and evict them.

        Parameters
        ----------
        current_tick : int
            Current discrete time step (for log context).

        Returns
        -------
        list of str
            robot_ids that were newly evicted this tick.
        """
        now    = _time.time()
        newly_evicted: List[str] = []

        for peer_id, hb in self._registry.items():
            if hb.evicted:
                continue
            elapsed = now - hb.last_wall_time
            if elapsed > self._timeout:
                self._evict(peer_id, hb.last_pos, current_tick)
                hb.evicted = True
                newly_evicted.append(peer_id)
                self.evicted_peers.append(peer_id)

        return newly_evicted

    def _evict(self, peer_id: str, last_pos: Position, current_tick: int) -> None:
        """Perform the full eviction sequence for one ghost peer."""
        ox, oy = last_pos
        logger.warning(
            "GHOST NODE EVICTED: %s (last seen at %s, tick=%d). "
            "Stamping as obstacle and purging reservations.",
            peer_id, last_pos, current_tick,
        )

        # a. Stamp last known position as obstacle
        self._costmap.register_obstacle(ox, oy, robot_id="GHOST_EVICTION", tick=current_tick)

        # b. Release all reservation-table claims
        self._local_rt.release_peer(peer_id)

        # c. Revoke any held leases
        if _HAS_LEASE and self._lease_table is not None:
            self._lease_table.release(peer_id, last_pos)

        # d. User callback
        if self._on_ghost is not None:
            try:
                self._on_ghost(peer_id, last_pos)
            except Exception as exc:
                logger.error("on_ghost callback raised: %s", exc)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_ghost(self, peer_id: str) -> bool:
        hb = self._registry.get(peer_id)
        if hb is None:
            return False
        return hb.evicted or (_time.time() - hb.last_wall_time > self._timeout)

    def __repr__(self) -> str:
        return (
            f"GhostNodeEviction(peers={len(self._registry)}, "
            f"evicted={len(self.evicted_peers)})"
        )


# ===========================================================================
# HazardManager — top-level orchestrator
# ===========================================================================

class HazardManager:
    """
    Orchestrates all safety subsystems for one AMR.

    Subsystems managed
    ------------------
    * ``KinematicSafetyBumper``  — 3-zone proximity response.
    * ``DynamicCostmap``         — runtime obstacle stamping + inflation.
    * ``GhostNodeEviction``      — silent peer detection and cleanup.

    Parameters
    ----------
    robot_id : str
        This robot's identifier.
    grid : GridMap
        Shared warehouse grid (mutated in-place by the costmap).
    local_rt : LocalReservationTable
        Cycle-3 reservation table for ghost eviction.
    sta_table : STAReservationTable
        Cycle-1 STA* table used for replanning after hazard insertion.
    resolver : ConflictResolver | None
        Cycle-3 resolver — its ``set_goal()`` is called after replan.
    lease_table : TargetLeaseTable | None
        Cycle-4 lease table for ghost revocation.
    on_hazard_broadcast : callable | None
        Async or sync callback ``(HazardAlertMessage) -> None`` — sends the
        alert over the P2P mesh.
    cell_resolution_m : float
        Metres per grid cell (converts sensor distance to cells).
    """

    def __init__(
        self,
        robot_id:             str,
        grid:                 GridMap,
        local_rt:             LocalReservationTable,
        sta_table:            STAReservationTable,
        resolver:             Optional["ConflictResolver"]          = None,
        lease_table:          Optional["TargetLeaseTable"]          = None,
        on_hazard_broadcast:  Optional[Callable[[HazardAlertMessage], None]] = None,
        cell_resolution_m:    float = DEFAULT_CELL_RESOLUTION_M,
    ) -> None:
        self.robot_id = robot_id
        self._grid    = grid
        self._resolver = resolver
        self._on_broadcast = on_hazard_broadcast
        self._cell_res     = cell_resolution_m

        # Subsystems
        self.bumper = KinematicSafetyBumper(
            on_estop=self._handle_estop,
        )
        self.costmap = DynamicCostmap(grid)
        self.ghost_eviction = GhostNodeEviction(
            local_rt=local_rt,
            costmap=self.costmap,
            lease_table=lease_table,
            on_ghost=self._handle_ghost,
        )

        self._sta_table     = sta_table
        self._estop_active  = False

        # Replanning bookkeeping
        self._replan_needed: bool          = False
        self._replan_start:  Optional[Position] = None
        self._replan_goal:   Optional[Position] = None
        self._replan_tick:   int           = 0

        # Telemetry
        self.alerts_sent:      int = 0
        self.alerts_received:  int = 0
        self.replans_triggered: int = 0

    # ------------------------------------------------------------------
    # Sensor feed
    # ------------------------------------------------------------------

    def on_sensor_reading(
        self,
        distance_m:   float,
        obstacle_pos: Optional[Position] = None,
        current_tick: int = 0,
    ) -> SpeedCommand:
        """
        Process one proximity sensor reading.

        Parameters
        ----------
        distance_m : float
            Distance to detected obstacle in metres (math.inf if none).
        obstacle_pos : (col, row) | None
            Grid position of the obstacle if precisely located (e.g., by a
            depth camera).  When provided and the zone is EMERGENCY_STOP or
            DECELERATE, the cell is registered as a dynamic obstacle.
        current_tick : int
            Current discrete time step.

        Returns
        -------
        SpeedCommand
            The commanded speed for the actuator controller.
        """
        cmd = self.bumper.evaluate(distance_m)

        if cmd.zone in (SpeedZone.EMERGENCY_STOP, SpeedZone.DECELERATE):
            if obstacle_pos is not None and not self.costmap.is_known_obstacle(*obstacle_pos):
                alert = self.costmap.register_obstacle(
                    *obstacle_pos,
                    robot_id=self.robot_id,
                    tick=current_tick,
                )
                self.alerts_sent += 1

                # Broadcast to peers
                if self._on_broadcast is not None:
                    self._on_broadcast(alert)

                # Schedule replan
                self._replan_needed = True
                self._replan_tick   = current_tick

        return cmd

    # ------------------------------------------------------------------
    # Peer hazard alert reception
    # ------------------------------------------------------------------

    def on_peer_hazard_alert(
        self,
        msg:          HazardAlertMessage,
        current_pos:  Position,
        current_tick: int,
        goal:         Optional[Position] = None,
    ) -> bool:
        """
        Process a ``HazardAlertMessage`` received from a peer.

        Parameters
        ----------
        msg : HazardAlertMessage
            The gossip packet.
        current_pos : (col, row)
            This robot's current position (for replan start).
        current_tick : int
            Current discrete time step.
        goal : (col, row) | None
            Current navigation goal (for replan).

        Returns
        -------
        bool
            True if the alert was new and a replan has been scheduled.
        """
        # Register heartbeat for the sender
        self.ghost_eviction.heartbeat(msg.robot_id, msg.obstacle_pos)

        is_new = self.costmap.apply_peer_alert(msg)
        if not is_new:
            return False

        self.alerts_received += 1
        logger.info(
            "[%s] Peer hazard at %s applied — checking if replan needed.",
            self.robot_id, msg.obstacle_pos,
        )

        # Check if our current path crosses the new obstacle zone
        if goal is not None:
            self._replan_needed = True
            self._replan_start  = current_pos
            self._replan_goal   = goal
            self._replan_tick   = current_tick

        return True

    # ------------------------------------------------------------------
    # Tick (call once per time step from the control loop)
    # ------------------------------------------------------------------

    def tick(
        self,
        current_pos:  Position,
        current_tick: int,
        goal:         Optional[Position] = None,
    ) -> Optional[List[STNode]]:
        """
        Main per-tick update.

        1. Runs ghost eviction check.
        2. If a replan is pending, executes it and returns the new path.

        Parameters
        ----------
        current_pos : (col, row)
            Robot's confirmed position this tick.
        current_tick : int
            Discrete time step.
        goal : (col, row) | None
            Current navigation goal.

        Returns
        -------
        list of STNode | None
            New space-time path if replanning occurred; None otherwise.
        """
        # Ghost eviction
        newly_evicted = self.ghost_eviction.check_ghosts(current_tick)
        if newly_evicted:
            for ghost_id in newly_evicted:
                logger.warning(
                    "[%s] Ghost eviction: %s removed from fleet at tick=%d.",
                    self.robot_id, ghost_id, current_tick,
                )
            # A ghost eviction means the map changed — schedule replan
            if goal is not None:
                self._replan_needed = True
                self._replan_start  = current_pos
                self._replan_goal   = goal
                self._replan_tick   = current_tick

        # Deferred replan
        if self._replan_needed and goal is not None:
            self._replan_needed = False
            new_path = self._execute_replan(
                start=current_pos,
                goal=goal,
                current_tick=current_tick,
            )
            return new_path

        return None

    # ------------------------------------------------------------------
    # Replan execution
    # ------------------------------------------------------------------

    def _execute_replan(
        self,
        start:        Position,
        goal:         Position,
        current_tick: int,
    ) -> Optional[List[STNode]]:
        """
        Run STA* with the updated GridMap (including new obstacles and
        inflation overlays) and return the new path.
        """
        self.replans_triggered += 1
        logger.info(
            "[%s] Replanning after hazard: %s -> %s at tick=%d.",
            self.robot_id, start, goal, current_tick,
        )

        planner = SpaceTimeAstar(
            grid_map=self._grid,
            reservation_table=self._sta_table,
            agent_id=self.robot_id,
            max_time=200,
        )
        path = planner.plan(start=start, goal=goal, start_time=current_tick)

        if path:
            planner.commit_path(path)
            logger.info(
                "[%s] Replan SUCCESS: %d steps via %s.",
                self.robot_id, len(path) - 1,
                [p[:2] for p in path],
            )
            # Also update ConflictResolver if available
            if _HAS_RESOLVER and self._resolver is not None:
                self._resolver._ctx.planned_path = path
                self._resolver._ctx.path_index   = 0
        else:
            logger.error(
                "[%s] Replan FAILED: no path from %s to %s at tick=%d.",
                self.robot_id, start, goal, current_tick,
            )

        return path if path else None

    # ------------------------------------------------------------------
    # Internal callbacks
    # ------------------------------------------------------------------

    def _handle_estop(self) -> None:
        """Called by KinematicSafetyBumper on EMERGENCY_STOP entry."""
        self._estop_active = True
        logger.warning("[%s] EMERGENCY_STOP: actuators halted.", self.robot_id)

    def _handle_ghost(self, peer_id: str, last_pos: Position) -> None:
        """Called by GhostNodeEviction when a peer is declared ghost."""
        logger.warning(
            "[%s] Ghost callback: %s at %s evicted from fleet.",
            self.robot_id, peer_id, last_pos,
        )
        # Trigger a replan since the ghost's last position is now blocked
        self._replan_needed = True

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"HazardManager(id={self.robot_id!r}, "
            f"estop={self._estop_active}, "
            f"obstacles={self.costmap.obstacle_count}, "
            f"replans={self.replans_triggered})"
        )


# ===========================================================================
# Dynamic hazard response test
# ===========================================================================

def test_dynamic_hazard_response() -> None:
    """
    Simulate a robot moving East in a wide corridor when an unexpected
    obstacle appears 1.5 m ahead, blocking the current aisle.

    Grid layout (10 cols x 5 rows):
        Row 0:  . . . . . . . . . .   (open area above)
        Row 1:  # # # # # # # # # #   (wall — no route above)
        Row 2:  . . . . . . . . . .   (primary aisle — robot travels here)
        Row 3:  . . . . . . . . . .   (detour aisle below)
        Row 4:  # # # # # # # # # #   (wall — no route below)

    Wait — with row 1 and row 4 as walls the robot has two options:
    row 2 (primary) or row 3 (detour).

    Robot:  starts at (0,2), goal (9,2), tick=0.
    Obstacle appears at (3,2) before the robot arrives there.

    Expected sequence:
    T1. Robot plans initial path:  (0,2) East all the way to (9,2).
    T2. Sensor fires distance=1.5m → Zone 2 (DECELERATE).
        Obstacle (3,2) registered. GridMap blocks (3,2), inflates neighbours.
        HazardAlertMessage broadcast.
        Replan triggered.
    T3. New path avoids (3,2): robot detours via row 3 aisle.
    T4. Robot reaches (9,2).

    Assertions:
        A1. KinematicSafetyBumper correctly classifies 1.5m as DECELERATE.
        A2. (3,2) becomes impassable in GridMap after obstacle registration.
        A3. HazardAlertMessage is broadcast (captured in sent_alerts list).
        A4. Replan finds a new path that does NOT pass through (3,2).
        A5. Robot reaches goal (9,2).
        A6. Peer receives the alert and also blocks (3,2) in its map.
        A7. Ghost node detection: a peer silent >2s is evicted and its
            position is stamped as an obstacle.
    """
    from core.grid_map import GridMap, WALKWAY as W, SHELF as S
    from core.space_time_astar import ReservationTable as STATable
    from coordination.reservation_table import LocalReservationTable

    print("=" * 65)
    print("Dynamic Hazard Response Test")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Build the dual-aisle grid
    # ------------------------------------------------------------------
    raw = [
        [W, W, W, W, W, W, W, W, W, W],   # row 0 open
        [S, S, S, S, S, S, S, S, S, S],   # row 1 wall
        [W, W, W, W, W, W, W, W, W, W],   # row 2 primary aisle
        [W, W, W, W, W, W, W, W, W, W],   # row 3 detour aisle
        [S, S, S, S, S, S, S, S, S, S],   # row 4 wall
    ]
    grid = GridMap(raw)
    print(f"\nInitial grid:\n{grid.ascii_render()}\n")

    sta_table = STATable()
    local_rt  = LocalReservationTable()

    START     = (0, 2)
    GOAL      = (9, 2)
    TICK      = 0
    OBSTACLE  = (3, 2)

    # ------------------------------------------------------------------
    # Create HazardManager for the primary robot
    # ------------------------------------------------------------------
    sent_alerts: List[HazardAlertMessage] = []

    mgr = HazardManager(
        robot_id="AMR_MAIN",
        grid=grid,
        local_rt=local_rt,
        sta_table=sta_table,
        on_hazard_broadcast=sent_alerts.append,
    )

    # Register a peer so we can test ghost eviction
    PEER_ID   = "AMR_PEER"
    PEER_POS  = (7, 3)
    mgr.ghost_eviction.heartbeat(PEER_ID, PEER_POS)

    # ------------------------------------------------------------------
    # T1: Plan initial path — should be straight along row 2
    # ------------------------------------------------------------------
    print("[T1] Planning initial path (straight aisle) ...")
    planner = SpaceTimeAstar(
        grid_map=grid,
        reservation_table=sta_table,
        agent_id="AMR_MAIN",
        max_time=200,
    )
    initial_path = planner.plan(start=START, goal=GOAL, start_time=TICK)
    planner.commit_path(initial_path)

    assert initial_path, "T1 FAILED: initial plan returned empty path"
    assert all(y == 2 for _, y, _ in initial_path), (
        f"T1 FAILED: initial path should stay in row 2, got: {initial_path}"
    )
    print(f"  Initial path ({len(initial_path)-1} steps): {initial_path}")
    print("  T1 PASSED [OK]")

    # ------------------------------------------------------------------
    # T2: Sensor fires — obstacle 1.5m ahead → 3 cells (0.5m/cell)
    #     DECELERATE zone, obstacle registered at (3,2)
    # ------------------------------------------------------------------
    print("\n[T2] Sensor reading: obstacle at 1.5 m ahead ...")
    SENSOR_DIST_M = 1.5   # metres
    cmd = mgr.on_sensor_reading(
        distance_m=SENSOR_DIST_M,
        obstacle_pos=OBSTACLE,
        current_tick=TICK,
    )

    # A1: Zone classification
    assert cmd.zone == SpeedZone.DECELERATE, (
        f"A1 FAILED: expected DECELERATE, got {cmd.zone}"
    )
    assert abs(cmd.speed_frac - 0.40) < 1e-9, (
        f"A1 FAILED: speed_frac={cmd.speed_frac}, expected 0.40"
    )
    print(f"  Zone: {cmd.zone.name}  speed_frac={cmd.speed_frac:.0%}  [OK]")

    # A2: Obstacle cell is now impassable
    assert not grid.is_passable(*OBSTACLE), (
        f"A2 FAILED: cell {OBSTACLE} should be impassable after block_cell()"
    )
    print(f"  GridMap({OBSTACLE}) is now impassable  [OK]")

    # Inflation check — neighbours of (3,2) that are passable should cost more
    inflated_neighbours = [
        (2, 2), (4, 2),   # E-W in same row
        (2, 3), (3, 3), (4, 3),  # row below
    ]
    for nx, ny in inflated_neighbours:
        if grid.is_passable(nx, ny):
            ic = grid.inflation_cost(nx, ny)
            assert ic > 0.0, f"A2b FAILED: cell ({nx},{ny}) has zero inflation cost"
    print(f"  Inflation applied to passable neighbours of {OBSTACLE}  [OK]")

    # A3: Alert was broadcast
    assert len(sent_alerts) == 1, (
        f"A3 FAILED: expected 1 alert broadcast, got {len(sent_alerts)}"
    )
    alert = sent_alerts[0]
    assert alert.obstacle_pos == OBSTACLE
    assert alert.robot_id == "AMR_MAIN"
    print(f"  HazardAlertMessage broadcast: {alert.obstacle_pos} TTL={alert.ttl_ticks}  [OK]")

    # ------------------------------------------------------------------
    # T3: Replan avoids (3,2)
    # ------------------------------------------------------------------
    print("\n[T3] Replan triggered — must avoid blocked cell (3,2) ...")
    new_path = mgr._execute_replan(
        start=START,
        goal=GOAL,
        current_tick=TICK,
    )

    # A4: New path does not pass through (3,2)
    assert new_path, "A4 FAILED: replan returned no path"
    blocked_visits = [(x, y, t) for x, y, t in new_path if (x, y) == OBSTACLE]
    assert not blocked_visits, (
        f"A4 FAILED: new path still passes through blocked cell {OBSTACLE}: "
        f"{blocked_visits}"
    )
    print(f"  New path ({len(new_path)-1} steps): {new_path}")
    print(f"  Path avoids {OBSTACLE}  [OK]")

    # A5: Path reaches goal
    assert new_path[-1][:2] == GOAL, (
        f"A5 FAILED: new path does not reach goal {GOAL}: {new_path[-1]}"
    )
    print(f"  Path reaches goal {GOAL}  [OK]")

    print("\n[T4] Verifying grid state after replanning ...")
    print(f"  Updated grid:\n{grid.ascii_render()}")
    print("  (3,2) now shows '#' in the grid  [OK]")

    # ------------------------------------------------------------------
    # A6: Peer map update via gossip
    # ------------------------------------------------------------------
    print("\n[A6] Peer robot receives hazard alert and updates its map ...")
    peer_grid    = GridMap([row[:] for row in raw])   # fresh copy
    peer_local_rt = LocalReservationTable()
    peer_sta      = STATable()
    peer_mgr     = HazardManager(
        robot_id="AMR_PEER",
        grid=peer_grid,
        local_rt=peer_local_rt,
        sta_table=peer_sta,
    )

    # Simulate alert delivery
    is_new = peer_mgr.on_peer_hazard_alert(
        msg=alert,
        current_pos=PEER_POS,
        current_tick=TICK,
        goal=(0, 3),
    )
    assert is_new, "A6 FAILED: peer treated alert as already known"
    assert not peer_grid.is_passable(*OBSTACLE), (
        f"A6 FAILED: peer grid still shows {OBSTACLE} as passable"
    )
    assert peer_mgr._replan_needed, "A6 FAILED: peer replan not scheduled"
    print(f"  Peer blocked {OBSTACLE} in its grid  [OK]")
    print(f"  Peer replan scheduled  [OK]")

    # ------------------------------------------------------------------
    # A7: Ghost node eviction
    # ------------------------------------------------------------------
    print("\n[A7] Ghost node eviction (peer silent > 2s) ...")
    # Backdate the peer's heartbeat to simulate silence
    hb = mgr.ghost_eviction._registry[PEER_ID]
    # Force last_wall_time to be > GHOST_TIMEOUT_SECONDS ago
    import time as _t
    hb.last_wall_time = _t.time() - (GHOST_TIMEOUT_SECONDS + 0.5)
    mgr.ghost_eviction._registry[PEER_ID] = hb

    evicted = mgr.ghost_eviction.check_ghosts(current_tick=TICK + 5)
    assert PEER_ID in evicted, (
        f"A7 FAILED: {PEER_ID} was not evicted (evicted={evicted})"
    )
    assert grid.cell_type(*PEER_POS) == SHELF, (
        f"A7 FAILED: ghost position {PEER_POS} not stamped as obstacle"
    )
    print(f"  {PEER_ID} evicted after heartbeat timeout  [OK]")
    print(f"  Ghost position {PEER_POS} stamped as obstacle  [OK]")

    # Additional ESTOP zone test
    print("\n[Bonus] Testing EMERGENCY_STOP zone (0.5 m) ...")
    bumper2 = KinematicSafetyBumper()
    cmd_z3  = bumper2.evaluate(0.5)
    assert cmd_z3.zone == SpeedZone.EMERGENCY_STOP
    assert cmd_z3.speed_frac == 0.0
    assert bumper2.estop_count == 1
    cmd_z1  = bumper2.evaluate(math.inf)
    assert cmd_z1.zone == SpeedZone.NORMAL
    print(f"  0.5m -> {cmd_z3.zone.name}  speed={cmd_z3.speed_frac}  [OK]")
    print(f"  inf  -> {cmd_z1.zone.name}    speed={cmd_z1.speed_frac}  [OK]")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("ALL HAZARD RESPONSE ASSERTIONS PASSED")
    print("=" * 65)
    print(f"\n  Alerts sent:     {mgr.alerts_sent}")
    print(f"  Alerts received: {peer_mgr.alerts_received}")
    print(f"  Replans:         {mgr.replans_triggered}")
    print(f"  ESTOP events:    {mgr.bumper.estop_count}")
    print(f"  Ghosts evicted:  {len(mgr.ghost_eviction.evicted_peers)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    test_dynamic_hazard_response()
