"""
coordination/reservation_table.py
==================================
Local, time-decaying reservation cache for each AMR's conflict-resolution
engine.

Relationship to ``core.space_time_astar.ReservationTable``
----------------------------------------------------------
``core.space_time_astar.ReservationTable`` is the *planner-internal* table
that the STA* search uses to avoid vertex/edge collisions during path
computation.

This class (``LocalReservationTable``) is the *coordination-layer* cache
that lives one abstraction above:
  * It stores **peer claims** received via P2P broadcasts, not just binary
    occupied/free flags.
  * Each entry carries ``(peer_id, priority, wall_timestamp)`` so the
    conflict resolver can make priority-based decisions.
  * Entries are automatically purged when their logical time step ``t`` falls
    below the engine's current tick — avoiding unbounded memory growth.
  * It produces a read-only view that can be *injected* into a fresh
    ``core.space_time_astar.ReservationTable`` so the STA* planner
    automatically routes around all known peer claims.

Data layout
-----------
    vertex_claims : dict[(x, y, t)] → PeerClaim
    edge_claims   : dict[(x1,y1,x2,y2,t)] → PeerClaim

Zero external dependencies (standard library only).
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# PeerClaim
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PeerClaim:
    """
    A single reservation entry made by a remote peer.

    Attributes
    ----------
    peer_id : str
        Robot that owns this claim.
    priority : float
        Computed priority score at the moment the packet was received.
        Higher is more urgent.
    wall_timestamp : float
        ``time.time()`` value when the claim was recorded locally.
        Used to expire stale claims if the peer goes silent.
    """
    peer_id:        str
    priority:       float
    wall_timestamp: float


# Composite key types (mirror core.space_time_astar)
VertexKey = Tuple[int, int, int]                 # (x, y, t)
EdgeKey   = Tuple[int, int, int, int, int]       # (x1, y1, x2, y2, t)


# ---------------------------------------------------------------------------
# LocalReservationTable
# ---------------------------------------------------------------------------

class LocalReservationTable:
    """
    Per-robot, time-decaying cache of peer trajectory claims.

    Parameters
    ----------
    stale_wall_seconds : float
        Entries older than this many wall-clock seconds are considered stale
        and are removed on the next ``purge()`` call, even if ``t`` is still
        in the future.  Default: 5.0 s.  Should be ≥ 1 replanning cycle.
    """

    def __init__(self, stale_wall_seconds: float = 5.0) -> None:
        self._stale: float = stale_wall_seconds

        # Primary stores
        self._vertex: Dict[VertexKey, PeerClaim] = {}
        self._edge:   Dict[EdgeKey,   PeerClaim] = {}

    # ------------------------------------------------------------------
    # Ingestion from peer broadcasts
    # ------------------------------------------------------------------

    def ingest_trajectory(
        self,
        peer_id:   str,
        priority:  float,
        waypoints: List[Tuple[int, int, int]],   # (x, y, t)
    ) -> None:
        """
        Record vertex *and* edge claims for every step in a peer's trajectory.

        Called by the ConflictResolver whenever a valid ``IntentPacket``
        arrives from a neighbour.

        Parameters
        ----------
        peer_id : str
            Originating robot identifier.
        priority : float
            Peer's pre-computed priority score (from the packet or computed
            locally using the deterministic priority engine).
        waypoints : list of (x, y, t)
            The full space-time path as returned by STA*.
        """
        now = _time.time()
        claim = PeerClaim(peer_id=peer_id, priority=priority, wall_timestamp=now)

        for step_idx, (x, y, t) in enumerate(waypoints):
            # Vertex claim
            vk: VertexKey = (x, y, t)
            # Keep higher-priority claim if a key already exists
            existing = self._vertex.get(vk)
            if existing is None or priority > existing.priority:
                self._vertex[vk] = claim

            # Edge claim: (prev) → (current)
            if step_idx > 0:
                px, py, pt = waypoints[step_idx - 1]
                ek: EdgeKey = (px, py, x, y, pt)
                existing_e = self._edge.get(ek)
                if existing_e is None or priority > existing_e.priority:
                    self._edge[ek] = claim

    def release_peer(self, peer_id: str) -> None:
        """Remove all claims belonging to *peer_id* (e.g., after task complete)."""
        self._vertex = {k: v for k, v in self._vertex.items() if v.peer_id != peer_id}
        self._edge   = {k: v for k, v in self._edge.items()   if v.peer_id != peer_id}

    # ------------------------------------------------------------------
    # Time-decay purge
    # ------------------------------------------------------------------

    def purge(self, current_tick: int) -> int:
        """
        Remove all entries whose logical time step ``t < current_tick`` OR
        whose wall-clock age exceeds ``stale_wall_seconds``.

        Parameters
        ----------
        current_tick : int
            The engine's current discrete time step.

        Returns
        -------
        int
            Number of entries removed.
        """
        now   = _time.time()
        cutoff = now - self._stale
        before = len(self._vertex) + len(self._edge)

        self._vertex = {
            (x, y, t): v
            for (x, y, t), v in self._vertex.items()
            if t >= current_tick and v.wall_timestamp >= cutoff
        }
        self._edge = {
            (x1, y1, x2, y2, t): v
            for (x1, y1, x2, y2, t), v in self._edge.items()
            if t >= current_tick and v.wall_timestamp >= cutoff
        }

        after = len(self._vertex) + len(self._edge)
        return before - after

    # ------------------------------------------------------------------
    # Conflict queries
    # ------------------------------------------------------------------

    def get_vertex_claim(
        self, x: int, y: int, t: int
    ) -> Optional[PeerClaim]:
        """Return the PeerClaim at (x, y, t), or None if unclaimed."""
        return self._vertex.get((x, y, t))

    def get_edge_claim(
        self, x1: int, y1: int, x2: int, y2: int, t: int
    ) -> Optional[PeerClaim]:
        """Return the PeerClaim for edge (x1,y1)→(x2,y2) at t, or None."""
        return self._edge.get((x1, y1, x2, y2, t))

    def find_conflicts(
        self,
        my_id:       str,
        my_waypoints: List[Tuple[int, int, int]],
    ) -> List[Tuple[VertexKey, PeerClaim]]:
        """
        Return all (vertex_key, peer_claim) pairs where a peer's claim
        overlaps with *my_waypoints*.

        Only vertex conflicts are returned here; edge-swap conflicts are
        handled separately in the ConflictResolver via ``has_swap_conflict()``.

        Parameters
        ----------
        my_id : str
            This robot's own identifier (claims owned by self are skipped).
        my_waypoints : list of (x, y, t)
            This robot's planned space-time path.

        Returns
        -------
        list of (VertexKey, PeerClaim)
            Sorted by time step ascending.
        """
        conflicts: List[Tuple[VertexKey, PeerClaim]] = []
        for x, y, t in my_waypoints:
            claim = self._vertex.get((x, y, t))
            if claim is not None and claim.peer_id != my_id:
                conflicts.append(((x, y, t), claim))
        conflicts.sort(key=lambda item: item[0][2])  # sort by t
        return conflicts

    def has_swap_conflict(
        self,
        x1: int, y1: int,
        x2: int, y2: int,
        t: int,
        my_id: str,
    ) -> bool:
        """
        Return True if a peer has claimed the *reverse* edge (x2,y2)→(x1,y1)
        at time t (head-on / swap collision).
        """
        claim = self._edge.get((x2, y2, x1, y1, t))
        return claim is not None and claim.peer_id != my_id

    # ------------------------------------------------------------------
    # Export to STA* ReservationTable
    # ------------------------------------------------------------------

    def export_to_sta_table(
        self,
        sta_table: "core_ReservationTable",  # type: ignore[name-defined]
        exclude_peer: Optional[str] = None,
    ) -> None:
        """
        Inject all current claims into a ``core.space_time_astar.ReservationTable``
        so the STA* planner automatically avoids them.

        Parameters
        ----------
        sta_table
            A ``core.space_time_astar.ReservationTable`` instance.
        exclude_peer : str, optional
            If set, skip claims belonging to this peer (useful to exclude
            self-reservations already committed by ``SpaceTimeAstar.commit_path``).
        """
        for (x, y, t), claim in self._vertex.items():
            if exclude_peer is None or claim.peer_id != exclude_peer:
                sta_table.reserve_vertex(x, y, t, claim.peer_id)

        for (x1, y1, x2, y2, t), claim in self._edge.items():
            if exclude_peer is None or claim.peer_id != exclude_peer:
                sta_table.reserve_edge(x1, y1, x2, y2, t, claim.peer_id)

    # ------------------------------------------------------------------
    # Blocked-cell helpers for rapid 1-step blocking
    # ------------------------------------------------------------------

    def mark_blocked(
        self,
        x: int, y: int, t: int,
        peer_id: str,
        priority: float,
    ) -> None:
        """
        Directly block a single vertex (x, y, t) without a full trajectory
        ingest.  Used by the ConflictResolver to enforce immediate yields.
        """
        self._vertex[(x, y, t)] = PeerClaim(
            peer_id=peer_id,
            priority=priority,
            wall_timestamp=_time.time(),
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def vertex_count(self) -> int:
        return len(self._vertex)

    @property
    def edge_count(self) -> int:
        return len(self._edge)

    def __repr__(self) -> str:
        return (
            f"LocalReservationTable("
            f"vertices={self.vertex_count}, "
            f"edges={self.edge_count}, "
            f"stale_s={self._stale})"
        )
