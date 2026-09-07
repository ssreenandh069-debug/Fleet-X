"""
tasks/auction_manager.py
=========================
Decentralised Contract-Net Protocol (Market-Based Auction) for AMR task
allocation — no central server required.

Protocol overview
-----------------

    Edge terminal discovers new order
             │
             ▼
    ┌────────────────────────────────────────────────────────────────┐
    │              Task_Announcement broadcast                        │
    │  (flood to all neighbours in 9-sector spatial mesh)            │
    └────────────────────────────────────────────────────────────────┘
             │  (received by all eligible IDLE AMRs in proximity)
             ▼
    ┌─────────────────────────────────┐
    │  Each eligible AMR computes:     │
    │    Bid = dist_to_pickup × 1.5    │
    │        + (100 − battery) × 2.0   │
    │  Lower bid = more suitable robot │
    └─────────────────────────────────┘
             │
             ▼  150 ms auction window
    ┌──────────────────────────────────────────────────────────────┐
    │  Each robot independently resolves the winner:               │
    │    • Collects all bids received during the window.           │
    │    • Selects the robot with the LOWEST bid.                  │
    │    • Tie-break: lexicographically smallest robot_id.         │
    │    • Decision is DETERMINISTIC — all robots see the same     │
    │      bid set ⟹ all agree on the same winner.                │
    └──────────────────────────────────────────────────────────────┘
             │
             ▼  Winner only
    ┌──────────────────────────────────────────────────────────────┐
    │  TARGET_LEASE_ACQUIRED(target=pickup_pos, holder=robot_id,   │
    │                         expiry_tick)                          │
    │  Broadcast via P2P layer.                                     │
    │  All peers call TargetLeaseTable.record_peer_lease().         │
    └──────────────────────────────────────────────────────────────┘
             │
             ▼
    ConflictResolver.set_goal(pickup_pos, drop_pos)
    → SpaceTimeAstar.plan()  (Cycle 1)

State machine per AuctionParticipant
--------------------------------------

    IDLE ──► BIDDING ──► WINNER (commit to task, lock lease)
               │
               └────────► LOSER  (discard, remain IDLE)

Scalability
-----------
* Only robots within the task's 9-sector neighbourhood receive the
  ``Task_Announcement``.  O(k) not O(N).
* No bid messages are sent to a central broker — each robot decides
  locally using the same bid-resolution algorithm.

Zero external dependencies — ``asyncio``, ``dataclasses``, ``enum``,
``hashlib``, ``heapq``, ``logging``, ``math``, ``typing`` from the
standard library.

Integration
-----------
    from tasks.auction_manager import AuctionManager, Task
    from tasks.target_lease    import TargetLeaseTable
    from coordination.conflict_resolver import ConflictResolver, RobotContext
    from core.grid_map import GridMap

    mgr = AuctionManager(
        robot_id    = "AMR_003",
        position    = (12, 8),
        battery     = 87.5,
        lease_table = TargetLeaseTable(),
        resolver    = my_resolver,
        grid        = grid,
    )
    await mgr.start()

    # When a Task_Announcement arrives from the P2P layer:
    await mgr.on_task_announcement(task, announcer_id="edge_terminal_01")

    # When a Bid arrives from a peer:
    await mgr.on_bid_received(bid)

    # When a LeaseAcquiredMessage arrives from a peer:
    mgr.on_lease_acquired(lease_msg)
"""

from __future__ import annotations

import asyncio
import hashlib
import heapq
import logging
import math
import time as _time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, FrozenSet, List, Optional, Set, Tuple

from tasks.target_lease import (
    DEFAULT_LEASE_DURATION_TICKS,
    DEFAULT_LEASE_WALL_SECONDS,
    LeaseAcquiredMessage,
    LeaseReleasedMessage,
    LeaseRenewMessage,
    TargetLeaseTable,
)

# Cycle 3 integration (optional — graceful if coordination not installed)
try:
    from coordination.conflict_resolver import ConflictResolver, RobotState
    _HAS_RESOLVER = True
except ImportError:
    _HAS_RESOLVER = False
    ConflictResolver = None  # type: ignore[assignment, misc]
    RobotState       = None  # type: ignore[assignment, misc]

# Cycle 1 integration
try:
    from core.grid_map import GridMap
    _HAS_GRID = True
except ImportError:
    _HAS_GRID = False
    GridMap = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------
AUCTION_WINDOW_MS:     float = 150.0   # ms — how long to collect bids
BID_DIST_WEIGHT:       float = 1.5     # distance-to-pickup multiplier
BID_BATTERY_WEIGHT:    float = 2.0     # (100 − battery) multiplier
PROXIMITY_THRESHOLD:   float = 50.0    # Manhattan cells — only bid if within this range

# Type aliases
Position = Tuple[int, int]


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    """
    A transport order issued by an edge terminal.

    Attributes
    ----------
    task_id   : str          Globally unique order identifier.
    pickup_pos: (col, row)   Shelf / source position.
    drop_pos  : (col, row)   Delivery / destination position.
    urgency   : float        Task priority in [0, 1].  1.0 = highest.
    issued_tick: int         Discrete time step when the order was created.
    """
    task_id:    str
    pickup_pos: Position
    drop_pos:   Position
    urgency:    float = 1.0
    issued_tick: int  = 0


# ---------------------------------------------------------------------------
# Bid dataclass
# ---------------------------------------------------------------------------

@dataclass
class Bid:
    """
    A bid submitted by one AMR for a specific task.

    Lower bid_value ⟹ more eligible robot.

    Attributes
    ----------
    task_id    : str    Identifies which task this bid is for.
    robot_id   : str    Bidder's identifier.
    bid_value  : float  Computed bid score (lower = better).
    battery    : float  Bidder's current battery level.
    position   : (x,y) Bidder's position at the moment of bidding.
    timestamp  : float  Wall-clock time when the bid was created.
    """
    task_id:   str
    robot_id:  str
    bid_value: float
    battery:   float
    position:  Position
    timestamp: float = field(default_factory=_time.time)


# ---------------------------------------------------------------------------
# TaskAnnouncement dataclass
# ---------------------------------------------------------------------------

@dataclass
class TaskAnnouncement:
    """
    Broadcast by an edge terminal (or any proxy node) to announce a new order.
    """
    task:         Task
    announcer_id: str
    timestamp:    float = field(default_factory=_time.time)


# ---------------------------------------------------------------------------
# AuctionState — per-auction bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class _AuctionState:
    """Internal state for one running auction."""
    task:          Task
    my_bid:        Optional[Bid]              # None if this robot didn't bid
    bids:          Dict[str, Bid]             # robot_id → Bid (all received)
    close_time:    float                      # wall-clock when auction closes
    resolved:      bool = False
    winner_id:     Optional[str] = None


# ---------------------------------------------------------------------------
# AuctionParticipant state
# ---------------------------------------------------------------------------

class AuctionParticipantState(Enum):
    IDLE    = auto()   # No task assigned; eligible to bid.
    BIDDING = auto()   # Participating in a live auction.
    WORKING = auto()   # Committed to a task; not available for new bids.


# ---------------------------------------------------------------------------
# BidFormula
# ---------------------------------------------------------------------------

class BidFormula:
    """
    Stateless, deterministic bid calculator.

    Formula
    -------
        Bid = (manhattan_dist(pos, pickup) × BID_DIST_WEIGHT)
            + ((100.0 − battery) × BID_BATTERY_WEIGHT)

    Lower bid ⟹ closer and/or more charged ⟹ better candidate.

    Tie-breaking
    ------------
    If two bids are within floating-point epsilon, the robot with the
    **lexicographically smaller robot_id** wins.  All robots apply this
    same rule deterministically without communication.
    """

    @staticmethod
    def compute(
        robot_id:   str,
        position:   Position,
        pickup_pos: Position,
        battery:    float,
    ) -> float:
        """
        Return the bid value for this robot and pickup position.

        Parameters
        ----------
        robot_id : str
            Robot identifier (unused in formula, used for tie-breaking by caller).
        position : (col, row)
            Robot's current position.
        pickup_pos : (col, row)
            The task's pickup location.
        battery : float
            Battery remaining [0, 100].

        Returns
        -------
        float
            Bid score.  Lower is better.
        """
        dist = abs(position[0] - pickup_pos[0]) + abs(position[1] - pickup_pos[1])
        if dist == 0:
            return 0.0  # Automatic win if already on the exact spot
            
        battery_penalty = 100.0 - max(0.0, min(100.0, battery))
        return (dist * BID_DIST_WEIGHT) + (battery_penalty * BID_BATTERY_WEIGHT)

    @staticmethod
    def resolve_winner(bids: List[Bid]) -> Optional[Bid]:
        """
        From a list of bids for the same task, select the winner.

        Resolution rules (applied in order):
        1. Lowest ``bid_value``.
        2. Tie-break: lexicographically smallest ``robot_id``.

        Returns None if *bids* is empty.
        """
        if not bids:
            return None
        return min(bids, key=lambda b: (b.bid_value, b.robot_id))


# ---------------------------------------------------------------------------
# AuctionManager
# ---------------------------------------------------------------------------

class AuctionManager:
    """
    Per-robot auction engine implementing the Contract-Net Protocol.

    Responsibilities
    ----------------
    * Receive ``TaskAnnouncement`` from the P2P layer.
    * Compute and *locally record* this robot's bid (if eligible).
    * Receive ``Bid`` messages from peers.
    * After ``AUCTION_WINDOW_MS`` resolve the winner from all collected bids.
    * If winner: acquire lease, trigger ``ConflictResolver.set_goal()``.
    * If loser: discard, remain IDLE.

    Parameters
    ----------
    robot_id : str
        This robot's unique identifier.
    position : (col, row)
        Current position (should be updated via ``update_position()``
        on every tick).
    battery : float
        Current battery level [0, 100].  Update via ``update_battery()``.
    lease_table : TargetLeaseTable
        This robot's local lease table.
    resolver : ConflictResolver | None
        Cycle-3 conflict resolver (optional; set goal when winner).
    grid : GridMap | None
        Cycle-1 grid (used for proximity filtering).
    on_bid_broadcast : callable | None
        Async callback ``(Bid) → None`` to send our bid over the P2P layer.
    on_lease_broadcast : callable | None
        Async callback ``(LeaseAcquiredMessage) → None`` to broadcast
        a won lease over the P2P layer.
    on_task_assigned : callable | None
        Sync callback ``(Task) → None`` invoked when this robot wins.
    current_tick_fn : callable | None
        Zero-argument callable returning the current tick (``int``).
        Defaults to ``lambda: 0`` if not provided.
    """

    def __init__(
        self,
        robot_id:           str,
        position:           Position,
        battery:            float,
        lease_table:        TargetLeaseTable,
        resolver:           Optional["ConflictResolver"] = None,
        grid:               Optional["GridMap"]           = None,
        on_bid_broadcast:   Optional[Callable]            = None,
        on_lease_broadcast: Optional[Callable]            = None,
        on_task_assigned:   Optional[Callable]            = None,
        current_tick_fn:    Optional[Callable[[], int]]   = None,
    ) -> None:
        self.robot_id           = robot_id
        self._position:   Position  = position
        self._battery:    float     = battery
        self._lease_table           = lease_table
        self._resolver              = resolver
        self._grid                  = grid
        self._on_bid_broadcast      = on_bid_broadcast
        self._on_lease_broadcast    = on_lease_broadcast
        self._on_task_assigned      = on_task_assigned
        self._current_tick_fn       = current_tick_fn or (lambda: 0)

        self._state: AuctionParticipantState = AuctionParticipantState.IDLE

        # Active auctions: task_id → _AuctionState
        self._auctions: Dict[str, _AuctionState] = {}

        # Tasks won by this robot (history)
        self.won_tasks:  List[Task] = []
        # Tasks lost (for metrics)
        self.lost_tasks: List[Task] = []

        # Pending asyncio close-timers: task_id → asyncio.TimerHandle
        self._timers: Dict[str, asyncio.TimerHandle] = {}

    # ------------------------------------------------------------------
    # Position / battery updates (called from control loop)
    # ------------------------------------------------------------------

    def update_position(self, new_pos: Position) -> None:
        self._position = new_pos

    def update_battery(self, battery: float) -> None:
        self._battery = max(0.0, min(100.0, battery))

    @property
    def state(self) -> AuctionParticipantState:
        return self._state

    # ------------------------------------------------------------------
    # Task announcement handler (P2P receive hook)
    # ------------------------------------------------------------------

    async def on_task_announcement(
        self,
        announcement: TaskAnnouncement,
    ) -> Optional[Bid]:
        """
        Called when a ``TaskAnnouncement`` is received from the P2P layer.

        If this robot is IDLE and within proximity, it:
        1. Computes its bid.
        2. Records the auction locally.
        3. Broadcasts the bid (via ``on_bid_broadcast``).
        4. Schedules the auction close timer.

        Parameters
        ----------
        announcement : TaskAnnouncement
            The announced task.

        Returns
        -------
        Bid or None
            The bid submitted, or None if ineligible.
        """
        task = announcement.task
        current_tick = self._current_tick_fn()

        # -- Eligibility checks --
        # 1. Not already working on a task
        if self._state == AuctionParticipantState.WORKING:
            logger.debug("[%s] Skipping task %s — already working.", self.robot_id, task.task_id)
            return None

        # 2. Pickup must not already be leased by another robot
        if self._lease_table.is_locked(task.pickup_pos, current_tick):
            holder = self._lease_table.holder(task.pickup_pos)
            logger.debug(
                "[%s] Skipping task %s — pickup %s leased by %s.",
                self.robot_id, task.task_id, task.pickup_pos, holder,
            )
            return None

        # 3. Proximity check (don't bid if obviously too far)
        dist = abs(self._position[0] - task.pickup_pos[0]) + abs(self._position[1] - task.pickup_pos[1])
        if dist > PROXIMITY_THRESHOLD:
            logger.debug(
                "[%s] Skipping task %s — too far (%d > %d cells).",
                self.robot_id, task.task_id, dist, PROXIMITY_THRESHOLD,
            )
            return None

        # -- Compute bid --
        bid_value = BidFormula.compute(
            robot_id=self.robot_id,
            position=self._position,
            pickup_pos=task.pickup_pos,
            battery=self._battery,
        )
        my_bid = Bid(
            task_id=task.task_id,
            robot_id=self.robot_id,
            bid_value=bid_value,
            battery=self._battery,
            position=self._position,
        )

        # -- Register auction --
        if task.task_id not in self._auctions:
            close_time = _time.time() + (AUCTION_WINDOW_MS / 1000.0)
            auction = _AuctionState(
                task=task,
                my_bid=my_bid,
                bids={self.robot_id: my_bid},   # include our own bid
                close_time=close_time,
            )
            self._auctions[task.task_id] = auction
            self._state = AuctionParticipantState.BIDDING

            # Schedule the resolution callback
            try:
                loop = asyncio.get_running_loop()
                handle = loop.call_later(
                    AUCTION_WINDOW_MS / 1000.0,
                    lambda tid=task.task_id: asyncio.ensure_future(
                        self._resolve_auction(tid)
                    ),
                )
                self._timers[task.task_id] = handle
            except RuntimeError:
                # No running event loop (synchronous test context)
                pass

            logger.info(
                "[%s] BID submitted for task %s: value=%.3f (dist=%d, battery=%.1f%%)",
                self.robot_id, task.task_id, bid_value, dist, self._battery,
            )

            # -- Broadcast bid --
            if self._on_bid_broadcast is not None:
                await self._on_bid_broadcast(my_bid)

            return my_bid

        return None

    # ------------------------------------------------------------------
    # Peer bid received (P2P receive hook)
    # ------------------------------------------------------------------

    async def on_bid_received(self, bid: Bid) -> None:
        """
        Record a bid received from a peer robot.

        If this robot is not participating in the same auction, the bid is
        still stored so it can determine who won (needed for lease tracking).

        Parameters
        ----------
        bid : Bid
            Peer's bid for the named task.
        """
        if bid.robot_id == self.robot_id:
            return  # ignore own echo

        auction = self._auctions.get(bid.task_id)
        if auction is None:
            # We aren't in this auction — create a passive observer entry
            # so we can still determine the winner and track the lease
            passive = _AuctionState(
                task=Task(
                    task_id=bid.task_id,
                    pickup_pos=bid.position,   # best guess from bid
                    drop_pos=bid.position,
                ),
                my_bid=None,
                bids={bid.robot_id: bid},
                close_time=_time.time() + (AUCTION_WINDOW_MS / 1000.0),
            )
            self._auctions[bid.task_id] = passive
        else:
            auction.bids[bid.robot_id] = bid
            logger.debug(
                "[%s] Bid from %s received for task %s: %.3f",
                self.robot_id, bid.robot_id, bid.task_id, bid.bid_value,
            )

    # ------------------------------------------------------------------
    # Peer lease notification handler
    # ------------------------------------------------------------------

    def on_lease_acquired(self, msg: LeaseAcquiredMessage) -> None:
        """
        Called when a ``LeaseAcquiredMessage`` arrives from a peer.
        Records the lease in the local table; marks the target as unavailable.
        """
        self._lease_table.record_peer_lease(msg)
        logger.info(
            "[%s] Peer lease recorded: %s holds %s until tick=%d.",
            self.robot_id, msg.robot_id, msg.target, msg.lease_expiry_tick,
        )

    def on_lease_released(self, msg: LeaseReleasedMessage) -> None:
        """Called when a ``LeaseReleasedMessage`` arrives from a peer."""
        self._lease_table.release_peer_lease(msg)

    def on_lease_renewed(self, msg: LeaseRenewMessage) -> None:
        """Called when a ``LeaseRenewMessage`` arrives from a peer."""
        self._lease_table.renew_peer_lease(msg)

    # ------------------------------------------------------------------
    # Auction resolution (internal)
    # ------------------------------------------------------------------

    async def _resolve_auction(self, task_id: str) -> None:
        """
        Called by the close-timer to resolve the winner of an auction.

        Implements the deterministic local resolution rule:
          winner = robot with lowest bid_value (tie-break: smallest robot_id)
        """
        auction = self._auctions.get(task_id)
        if auction is None or auction.resolved:
            return

        auction.resolved = True
        all_bids = list(auction.bids.values())
        winner_bid = BidFormula.resolve_winner(all_bids)

        if winner_bid is None:
            logger.warning("[%s] Auction %s closed with zero bids.", self.robot_id, task_id)
            if self._state == AuctionParticipantState.BIDDING:
                self._state = AuctionParticipantState.IDLE
            return

        auction.winner_id = winner_bid.robot_id
        current_tick = self._current_tick_fn()
        task = auction.task

        logger.info(
            "[%s] Auction %s resolved → WINNER = %s (bid=%.3f)",
            self.robot_id, task_id, winner_bid.robot_id, winner_bid.bid_value,
        )

        if winner_bid.robot_id == self.robot_id:
            await self._claim_task(task, current_tick)
        else:
            # I lost — remain IDLE if I was the only one bidding on this task
            if self._state == AuctionParticipantState.BIDDING:
                self._state = AuctionParticipantState.IDLE
            self.lost_tasks.append(task)
            logger.info("[%s] Lost auction %s to %s.", self.robot_id, task_id, winner_bid.robot_id)

    async def _claim_task(self, task: Task, current_tick: int) -> None:
        """Execute all winner actions: lease, goal assignment, callback."""
        # 1. Acquire lease on pickup position
        lease_ok = self._lease_table.acquire(
            robot_id=self.robot_id,
            task_id=task.task_id,
            target=task.pickup_pos,
            current_tick=current_tick,
            lease_duration=DEFAULT_LEASE_DURATION_TICKS,
            wall_duration=DEFAULT_LEASE_WALL_SECONDS,
        )
        if not lease_ok:
            logger.warning(
                "[%s] Won auction %s but lease acquire FAILED (race condition) — yielding.",
                self.robot_id, task.task_id,
            )
            self._state = AuctionParticipantState.IDLE
            return

        # 2. Broadcast lease to peers
        rec = self._lease_table.get_record(task.pickup_pos)
        lease_msg = LeaseAcquiredMessage(
            robot_id=self.robot_id,
            task_id=task.task_id,
            target=task.pickup_pos,
            lease_expiry_tick=rec.lease_expiry_tick,  # type: ignore[union-attr]
            wall_expiry=rec.wall_expiry,               # type: ignore[union-attr]
        )
        if self._on_lease_broadcast is not None:
            await self._on_lease_broadcast(lease_msg)

        # 3. Transition state to WORKING
        self._state = AuctionParticipantState.WORKING
        self.won_tasks.append(task)

        logger.info(
            "[%s] Task %s CLAIMED — navigating to pickup %s then drop %s.",
            self.robot_id, task.task_id, task.pickup_pos, task.drop_pos,
        )

        # 4. Assign goal to ConflictResolver
        if _HAS_RESOLVER and self._resolver is not None:
            self._resolver.set_goal(
                start=self._position,
                goal=task.pickup_pos,
                current_tick=current_tick,
                urgency=task.urgency,
                battery=self._battery,
            )

        # 5. User-provided callback
        if self._on_task_assigned is not None:
            self._on_task_assigned(task)

    # ------------------------------------------------------------------
    # Synchronous resolution (for testing without an event loop)
    # ------------------------------------------------------------------

    def resolve_sync(self, task_id: str, current_tick: int = 0) -> Optional[str]:
        """
        Synchronously resolve an auction without asyncio.

        Used in ``test_distributed_auction()`` to avoid event-loop
        complexity in the test harness.

        Returns the winner's robot_id, or None if no bids.
        """
        auction = self._auctions.get(task_id)
        if auction is None:
            return None

        all_bids   = list(auction.bids.values())
        winner_bid = BidFormula.resolve_winner(all_bids)
        if winner_bid is None:
            return None

        auction.resolved  = True
        auction.winner_id = winner_bid.robot_id

        task = auction.task
        if winner_bid.robot_id == self.robot_id:
            # Claim synchronously
            lease_ok = self._lease_table.acquire(
                robot_id=self.robot_id,
                task_id=task.task_id,
                target=task.pickup_pos,
                current_tick=current_tick,
                lease_duration=DEFAULT_LEASE_DURATION_TICKS,
                wall_duration=DEFAULT_LEASE_WALL_SECONDS,
            )
            if lease_ok:
                self._state = AuctionParticipantState.WORKING
                self.won_tasks.append(task)

                if _HAS_RESOLVER and self._resolver is not None:
                    self._resolver.set_goal(
                        start=self._position,
                        goal=task.pickup_pos,
                        current_tick=current_tick,
                        urgency=task.urgency,
                        battery=self._battery,
                    )
                if self._on_task_assigned is not None:
                    self._on_task_assigned(task)
        else:
            if self._state == AuctionParticipantState.BIDDING:
                self._state = AuctionParticipantState.IDLE
            self.lost_tasks.append(task)

        return winner_bid.robot_id

    # ------------------------------------------------------------------
    # Lease helpers
    # ------------------------------------------------------------------

    def try_claim_target(
        self,
        target:       Position,
        task_id:      str,
        current_tick: int,
    ) -> bool:
        """
        Attempt to acquire a lease on *target*.

        Returns False immediately if the target is already locked.
        Used by external code that needs to claim a position outside the
        normal auction flow (e.g., charger locking).
        """
        if self._lease_table.is_locked(target, current_tick):
            return False
        return self._lease_table.acquire(
            robot_id=self.robot_id,
            task_id=task_id,
            target=target,
            current_tick=current_tick,
        )

    def release_target(self, target: Position) -> bool:
        """Release the lease on *target* held by this robot."""
        return self._lease_table.release(self.robot_id, target)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AuctionManager(id={self.robot_id!r}, state={self._state.name}, "
            f"won={len(self.won_tasks)}, lost={len(self.lost_tasks)})"
        )


# ===========================================================================
# Distributed auction stress test
# ===========================================================================

def test_distributed_auction() -> None:
    """
    Validate the decentralised Contract-Net Protocol with 3 AMRs.

    Scenario
    --------
    A new transport order appears:
        Task(task_id="T001", pickup_pos=(10, 5), drop_pos=(20, 5), urgency=0.8)

    Three idle AMRs at different distances and battery levels:

        AMR_A : pos=(7,  5), battery=90%  → dist=3  → bid= 3×1.5 + 10×2.0 = 24.5
        AMR_B : pos=(15, 5), battery=60%  → dist=5  → bid= 5×1.5 + 40×2.0 = 87.5
        AMR_C : pos=(10, 8), battery=95%  → dist=3  → bid= 3×1.5 + 5×2.0  = 14.5  ← WINNER

    AMR_C wins because it has the lowest bid (same distance as AMR_A but
    much higher battery → lower ``(100 - battery)`` penalty).

    Verifications
    -------------
    V1: AMR_C wins the auction (lowest bid).
    V2: AMR_C holds the lease on pickup_pos (10, 5).
    V3: AMR_A trying to independently claim (10, 5) is rejected.
    V4: Bid values are deterministic — all 3 managers agree on the same winner.
    V5: After simulating lease expiry, the position becomes claimable again.
    """
    print("=" * 65)
    print("Distributed Auction Test  —  3 AMRs, 1 Task")
    print("=" * 65)

    TASK = Task(
        task_id="T001",
        pickup_pos=(10, 5),
        drop_pos=(20, 5),
        urgency=0.8,
        issued_tick=0,
    )
    CURRENT_TICK = 0

    # ------------------------------------------------------------------
    # Three AMR configurations
    # ------------------------------------------------------------------
    robots = [
        {"robot_id": "AMR_A", "position": (7,  5), "battery": 90.0},
        {"robot_id": "AMR_B", "position": (15, 5), "battery": 60.0},
        {"robot_id": "AMR_C", "position": (10, 8), "battery": 95.0},
    ]

    # Per-robot auction managers with individual lease tables
    managers: Dict[str, AuctionManager] = {}
    lease_tables: Dict[str, TargetLeaseTable] = {}

    for cfg in robots:
        lt = TargetLeaseTable()
        lease_tables[cfg["robot_id"]] = lt
        mgr = AuctionManager(
            robot_id=cfg["robot_id"],
            position=cfg["position"],
            battery=cfg["battery"],
            lease_table=lt,
        )
        managers[cfg["robot_id"]] = mgr

    # ------------------------------------------------------------------
    # Step 1: Compute bids and register auction on all three managers
    # ------------------------------------------------------------------
    print("\n[Step 1] Computing bids ...")
    print(f"  Task: pickup={TASK.pickup_pos}  drop={TASK.drop_pos}  urgency={TASK.urgency}")
    print()

    expected_winner = "AMR_C"
    announcement = TaskAnnouncement(task=TASK, announcer_id="edge_terminal_01")

    all_bids: List[Bid] = []
    for cfg in robots:
        rid     = cfg["robot_id"]
        pos     = cfg["position"]
        battery = cfg["battery"]
        bid_val = BidFormula.compute(rid, pos, TASK.pickup_pos, battery)
        dist    = abs(pos[0] - TASK.pickup_pos[0]) + abs(pos[1] - TASK.pickup_pos[1])
        bid     = Bid(
            task_id=TASK.task_id,
            robot_id=rid,
            bid_value=bid_val,
            battery=battery,
            position=pos,
        )
        all_bids.append(bid)
        print(
            f"  {rid}: pos={pos}  battery={battery}%  dist={dist}"
            f"  -> bid = {dist}x{BID_DIST_WEIGHT} + {100-battery}x{BID_BATTERY_WEIGHT}"
            f" = {bid_val:.2f}"
        )

    # ------------------------------------------------------------------
    # Step 2: Simulate the bids being received by all managers
    #         (in a real system these travel over P2P UDP)
    # ------------------------------------------------------------------
    print(f"\n[Step 2] Distributing bids to all managers ...")
    for mgr in managers.values():
        # Register the auction on this manager
        mgr._auctions[TASK.task_id] = _AuctionState(
            task=TASK,
            my_bid=next((b for b in all_bids if b.robot_id == mgr.robot_id), None),
            bids={b.robot_id: b for b in all_bids},
            close_time=_time.time() + 0.15,
        )
        if mgr.robot_id != "AMR_B":     # AMR_B has no path (far), still IDLE
            mgr._state = AuctionParticipantState.BIDDING

    # ------------------------------------------------------------------
    # Step 3: All managers resolve synchronously — must agree on same winner
    # ------------------------------------------------------------------
    print(f"\n[Step 3] Resolving auction (all managers must agree) ...")
    winners: Dict[str, str] = {}
    for rid, mgr in managers.items():
        w = mgr.resolve_sync(TASK.task_id, current_tick=CURRENT_TICK)
        winners[rid] = w or "NONE"
        print(f"  {rid} resolved winner = {winners[rid]}")

    # V4: All agree
    unique_winners = set(winners.values())
    assert len(unique_winners) == 1, (
        f"V4 FAILED: Managers disagree on winner: {winners}"
    )
    print(f"  All managers agree on winner: {unique_winners.pop()}  [OK]")

    # V1: Correct winner
    resolved_winner = winners["AMR_A"]
    assert resolved_winner == expected_winner, (
        f"V1 FAILED: Expected {expected_winner}, got {resolved_winner}"
    )
    print(f"\n[V1] Correct winner is {expected_winner} (lowest bid)  [OK]")

    # ------------------------------------------------------------------
    # Step 4: Winner (AMR_C) holds the lease
    # ------------------------------------------------------------------
    print(f"\n[Step 4] Checking lease on pickup_pos {TASK.pickup_pos} ...")
    winner_lt = lease_tables["AMR_C"]

    # V2: AMR_C has the lease
    assert winner_lt.is_locked(TASK.pickup_pos, CURRENT_TICK), (
        f"V2 FAILED: AMR_C does not hold the lease on {TASK.pickup_pos}"
    )
    assert winner_lt.holder(TASK.pickup_pos) == "AMR_C", (
        f"V2 FAILED: Lease holder is {winner_lt.holder(TASK.pickup_pos)}, expected AMR_C"
    )
    print(f"  AMR_C holds lease on {TASK.pickup_pos}  [OK]")

    # ------------------------------------------------------------------
    # Step 5: Simulate AMR_A trying to claim the same shelf
    # ------------------------------------------------------------------
    print(f"\n[Step 5] AMR_A attempting to claim {TASK.pickup_pos} (should fail) ...")

    # AMR_A must first receive the lease broadcast from AMR_C
    # (simulates P2P LeaseAcquiredMessage delivery)
    winner_record = winner_lt.get_record(TASK.pickup_pos)
    lease_msg = LeaseAcquiredMessage(
        robot_id="AMR_C",
        task_id=TASK.task_id,
        target=TASK.pickup_pos,
        lease_expiry_tick=winner_record.lease_expiry_tick,
        wall_expiry=winner_record.wall_expiry,
    )
    managers["AMR_A"].on_lease_acquired(lease_msg)

    # Now AMR_A tries to claim directly
    a_lt     = lease_tables["AMR_A"]
    a_mgr    = managers["AMR_A"]
    claim_ok = a_mgr.try_claim_target(TASK.pickup_pos, "T001_spurious", CURRENT_TICK)

    # V3: Rejected
    assert not claim_ok, (
        f"V3 FAILED: AMR_A was able to claim {TASK.pickup_pos} despite AMR_C's lease!"
    )
    print(f"  AMR_A's claim on {TASK.pickup_pos} correctly REJECTED  [OK]")

    # Also check the lease table itself rejects the raw acquire
    acquire_ok = a_lt.acquire(
        robot_id="AMR_A",
        task_id="spurious",
        target=TASK.pickup_pos,
        current_tick=CURRENT_TICK,
    )
    assert not acquire_ok, "V3b FAILED: raw acquire by AMR_A should have been rejected."
    print(f"  Raw lease acquire by AMR_A also REJECTED  [OK]")

    # ------------------------------------------------------------------
    # Step 6: Lease expiry — position becomes claimable again
    # ------------------------------------------------------------------
    print(f"\n[Step 6] Simulating lease expiry ...")
    expiry_tick = winner_record.lease_expiry_tick
    purged      = winner_lt.purge(current_tick=expiry_tick)
    assert purged == 1, f"V5 FAILED: Expected 1 purged lease, got {purged}"
    assert not winner_lt.is_locked(TASK.pickup_pos, expiry_tick), (
        f"V5 FAILED: Lease on {TASK.pickup_pos} still active after expiry tick."
    )
    print(f"  Lease expired at tick={expiry_tick} -> {TASK.pickup_pos} is now free  [OK]")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n  --- Auction Summary ---")
    header = f"  {'Robot':<8}  {'State':<10}  {'Won':<4}  {'Lost':<5}  {'Bid Value':>10}"
    print(header)
    print("  " + "-" * 48)
    for cfg in robots:
        rid     = cfg["robot_id"]
        mgr     = managers[rid]
        bid_val = BidFormula.compute(rid, cfg["position"], TASK.pickup_pos, cfg["battery"])
        print(
            f"  {rid:<8}  {mgr.state.name:<10}  {len(mgr.won_tasks):<4}  "
            f"{len(mgr.lost_tasks):<5}  {bid_val:>10.3f}"
        )

    print("\n" + "=" * 65)
    print("ALL AUCTION VERIFICATIONS PASSED")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    test_distributed_auction()
