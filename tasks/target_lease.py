"""
tasks/target_lease.py
======================
Spatial Lease system — prevents two AMRs from simultaneously navigating to
the same pickup shelf, drop station, or charger.

Problem solved
--------------
Without a central server, multiple AMRs may independently decide to collect
the same item from the same shelf.  A lease system lets the winner
*broadcast* its intent so peers can immediately remove that destination from
their own candidate sets.

Lease lifecycle
---------------
                                lease_expiry_tick
                                        │
  AMR acquires ──► broadcast ──► peers  │ mark blocked ──► lease expires
  TARGET_LEASE                   update │              ──► position unblocked
  _ACQUIRED                      local  │
                                  table │

    • Winning AMR broadcasts  ``LeaseAcquiredMessage``.
    • All peers store ``(target_coord) → LeaseRecord`` in their local
      ``TargetLeaseTable``.
    • On every tick the ``TargetLeaseTable.purge(current_tick)`` call removes
      expired entries automatically.
    • If the winner crashes or is pre-empted, the lease simply expires at
      ``lease_expiry_tick`` with no explicit release required — **zero central
      coordination**.
    • If the winner completes its task early, it broadcasts a
      ``LeaseReleasedMessage`` so peers can act sooner.

Integration with GridMap (Cycle 1)
-----------------------------------
``TargetLeaseTable.locked_positions(current_tick)`` returns the set of
currently leased positions.  These are **excluded from the planner's
goal candidates** by the AuctionManager before it calls
``ConflictResolver.set_goal()``.  The positions are NOT injected into the
occupancy grid itself (shelves remain reachable for traversal; they are only
blocked as *destinations*).

Wire messages (plain dataclasses, serialised in ``auction_manager.py``)
-----------------------------------------------------------------------
``LeaseAcquiredMessage``  — winner announces ownership.
``LeaseReleasedMessage``  — winner voluntarily releases early.
``LeaseRenewMessage``     — winner extends expiry before it lapses.

Zero external dependencies — ``dataclasses``, ``typing``, ``time``, and
``logging`` from the standard library.
"""

from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Position = Tuple[int, int]   # (col, row) / (x, y)

# Default lease duration in ticks — enough for a mid-size warehouse crossing
DEFAULT_LEASE_DURATION_TICKS: int   = 60
# Wall-clock backstop: even if ticks stall, expire after this many seconds
DEFAULT_LEASE_WALL_SECONDS:   float = 30.0


# ---------------------------------------------------------------------------
# LeaseRecord — one entry in the table
# ---------------------------------------------------------------------------

@dataclass
class LeaseRecord:
    """
    A single spatial lease held by one AMR over one target position.

    Attributes
    ----------
    target:             (col, row) of the locked destination.
    holder_id:          Robot that acquired the lease.
    lease_expiry_tick:  The table purges this record when
                        ``current_tick >= lease_expiry_tick``.
    wall_expiry:        Absolute ``time.time()`` value beyond which the lease
                        is considered stale regardless of tick (crash guard).
    acquired_wall:      Wall-clock timestamp when the lease was acquired.
    """
    target:            Position
    holder_id:         str
    lease_expiry_tick: int
    wall_expiry:       float
    acquired_wall:     float


# ---------------------------------------------------------------------------
# Wire-level message dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LeaseAcquiredMessage:
    """
    Broadcast by the auction winner to announce exclusive ownership of a
    target position.

    Fields map 1-to-1 to ``LeaseRecord`` so peers can reconstruct the record
    from the message alone.
    """
    robot_id:          str
    task_id:           str
    target:            Position          # (col, row) of the pickup/drop cell
    lease_expiry_tick: int
    wall_expiry:       float


@dataclass
class LeaseReleasedMessage:
    """
    Optional early-release broadcast — lets peers reclaim the destination
    sooner than the expiry tick.
    """
    robot_id:  str
    task_id:   str
    target:    Position


@dataclass
class LeaseRenewMessage:
    """
    Renewal broadcast — extends expiry before the current lease lapses.
    The holder must send this if its journey takes longer than originally
    estimated.
    """
    robot_id:          str
    task_id:           str
    target:            Position
    new_expiry_tick:   int
    new_wall_expiry:   float


# ---------------------------------------------------------------------------
# TargetLeaseTable
# ---------------------------------------------------------------------------

class TargetLeaseTable:
    """
    Per-robot registry of active spatial leases across the fleet.

    Each robot maintains one ``TargetLeaseTable`` instance.  It is updated
    by:
    * ``acquire(...)``  — when *this* robot wins an auction.
    * ``record_peer_lease(msg)`` — when a ``LeaseAcquiredMessage`` arrives
      from a peer (via the P2P layer).
    * ``renew_peer_lease(msg)`` — when a ``LeaseRenewMessage`` arrives.
    * ``release(...)``  / ``release_peer(msg)`` — early release.
    * ``purge(current_tick)`` — called every tick to expire stale records.

    Design invariants
    -----------------
    * At most **one** active lease per target position at any time.
    * A robot may hold multiple leases simultaneously (pickup + drop).
    * Leases are identified by ``target`` position, not ``task_id``, because
      position conflicts (not task IDs) are what need to be prevented.
    """

    def __init__(self) -> None:
        # target_pos → LeaseRecord
        self._leases: Dict[Position, LeaseRecord] = {}

    # ------------------------------------------------------------------
    # Acquire / release (own robot)
    # ------------------------------------------------------------------

    def acquire(
        self,
        robot_id:          str,
        task_id:           str,
        target:            Position,
        current_tick:      int,
        lease_duration:    int   = DEFAULT_LEASE_DURATION_TICKS,
        wall_duration:     float = DEFAULT_LEASE_WALL_SECONDS,
    ) -> bool:
        """
        Attempt to acquire a lease on *target* for *robot_id*.

        Parameters
        ----------
        robot_id : str
            The robot claiming the lease.
        task_id : str
            Associated task identifier (for logging / release matching).
        target : (col, row)
            Destination cell to lock.
        current_tick : int
            Current discrete time step.
        lease_duration : int
            Duration in ticks.
        wall_duration : float
            Duration in wall-clock seconds (crash guard).

        Returns
        -------
        bool
            ``True`` if the lease was granted.
            ``False`` if the position is already leased by another robot.
        """
        existing = self._leases.get(target)
        if existing is not None and existing.holder_id != robot_id:
            logger.warning(
                "[%s] acquire DENIED for %s — already held by %s (expires t=%d).",
                robot_id, target, existing.holder_id, existing.lease_expiry_tick,
            )
            return False

        now = _time.time()
        record = LeaseRecord(
            target=target,
            holder_id=robot_id,
            lease_expiry_tick=current_tick + lease_duration,
            wall_expiry=now + wall_duration,
            acquired_wall=now,
        )
        self._leases[target] = record
        logger.info(
            "[%s] LEASE ACQUIRED on %s (task=%s, expires_tick=%d).",
            robot_id, target, task_id, record.lease_expiry_tick,
        )
        return True

    def release(self, robot_id: str, target: Position) -> bool:
        """
        Voluntarily release the lease on *target* held by *robot_id*.

        Returns True if the release succeeded, False if no matching lease.
        """
        existing = self._leases.get(target)
        if existing is None:
            return False
        if existing.holder_id != robot_id:
            logger.warning(
                "[%s] release DENIED for %s — held by %s.",
                robot_id, target, existing.holder_id,
            )
            return False
        del self._leases[target]
        logger.info("[%s] LEASE RELEASED on %s.", robot_id, target)
        return True

    def renew(
        self,
        robot_id:       str,
        target:         Position,
        current_tick:   int,
        extra_ticks:    int   = DEFAULT_LEASE_DURATION_TICKS,
        extra_wall:     float = DEFAULT_LEASE_WALL_SECONDS,
    ) -> bool:
        """
        Extend the expiry of an existing lease held by *robot_id*.

        Returns True on success, False if no matching lease found.
        """
        existing = self._leases.get(target)
        if existing is None or existing.holder_id != robot_id:
            return False
        now = _time.time()
        renewed = LeaseRecord(
            target=target,
            holder_id=robot_id,
            lease_expiry_tick=current_tick + extra_ticks,
            wall_expiry=now + extra_wall,
            acquired_wall=existing.acquired_wall,
        )
        self._leases[target] = renewed
        logger.debug(
            "[%s] LEASE RENEWED on %s (new_expiry_tick=%d).",
            robot_id, target, renewed.lease_expiry_tick,
        )
        return True

    # ------------------------------------------------------------------
    # Peer message ingestion
    # ------------------------------------------------------------------

    def record_peer_lease(self, msg: LeaseAcquiredMessage) -> bool:
        """
        Ingest a ``LeaseAcquiredMessage`` from a peer.

        Overwrites any earlier record for the same target (last-writer-wins
        with the implicit assumption that auction protocol prevents two
        simultaneous winners — validated in ``AuctionManager``).

        Returns True if the lease was new, False if it updated an existing
        record for the same holder.
        """
        existing = self._leases.get(msg.target)
        is_new   = existing is None or existing.holder_id != msg.robot_id

        self._leases[msg.target] = LeaseRecord(
            target=msg.target,
            holder_id=msg.robot_id,
            lease_expiry_tick=msg.lease_expiry_tick,
            wall_expiry=msg.wall_expiry,
            acquired_wall=_time.time(),
        )
        if is_new:
            logger.info(
                "Peer LEASE recorded: %s → %s (expires_tick=%d).",
                msg.robot_id, msg.target, msg.lease_expiry_tick,
            )
        return is_new

    def release_peer_lease(self, msg: LeaseReleasedMessage) -> bool:
        """Ingest a ``LeaseReleasedMessage`` from a peer."""
        existing = self._leases.get(msg.target)
        if existing is not None and existing.holder_id == msg.robot_id:
            del self._leases[msg.target]
            logger.info(
                "Peer LEASE released by %s on %s.", msg.robot_id, msg.target
            )
            return True
        return False

    def renew_peer_lease(self, msg: LeaseRenewMessage) -> bool:
        """Ingest a ``LeaseRenewMessage`` from a peer."""
        existing = self._leases.get(msg.target)
        if existing is not None and existing.holder_id == msg.robot_id:
            self._leases[msg.target] = LeaseRecord(
                target=msg.target,
                holder_id=msg.robot_id,
                lease_expiry_tick=msg.new_expiry_tick,
                wall_expiry=msg.new_wall_expiry,
                acquired_wall=existing.acquired_wall,
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Tick-based purge
    # ------------------------------------------------------------------

    def purge(self, current_tick: int) -> int:
        """
        Remove all leases where ``lease_expiry_tick <= current_tick`` OR
        ``wall_expiry <= time.time()``.

        Returns the number of leases purged.
        """
        now    = _time.time()
        before = len(self._leases)
        self._leases = {
            pos: rec
            for pos, rec in self._leases.items()
            if rec.lease_expiry_tick > current_tick and rec.wall_expiry > now
        }
        purged = before - len(self._leases)
        if purged:
            logger.debug("TargetLeaseTable: purged %d expired leases at tick=%d.", purged, current_tick)
        return purged

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_locked(self, target: Position, current_tick: int) -> bool:
        """
        Return True if *target* is currently leased by any robot (that is
        not expired).
        """
        rec = self._leases.get(target)
        if rec is None:
            return False
        now = _time.time()
        return rec.lease_expiry_tick > current_tick and rec.wall_expiry > now

    def holder(self, target: Position) -> Optional[str]:
        """Return the robot_id holding a lease on *target*, or None."""
        rec = self._leases.get(target)
        return rec.holder_id if rec is not None else None

    def locked_positions(self, current_tick: int) -> FrozenSet[Position]:
        """
        Return all currently locked positions.

        Used by ``AuctionManager`` to filter out already-claimed destinations
        before computing bids.
        """
        now = _time.time()
        return frozenset(
            pos
            for pos, rec in self._leases.items()
            if rec.lease_expiry_tick > current_tick and rec.wall_expiry > now
        )

    def get_record(self, target: Position) -> Optional[LeaseRecord]:
        """Return the raw LeaseRecord for a target, or None."""
        return self._leases.get(target)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def active_count(self) -> int:
        return len(self._leases)

    def __repr__(self) -> str:
        return f"TargetLeaseTable(active_leases={self.active_count})"
