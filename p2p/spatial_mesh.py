"""
p2p/spatial_mesh.py
====================
Decentralized, localized Peer-to-Peer communication layer for a fleet of
200+ Autonomous Mobile Robots (AMRs).

Architecture overview
---------------------

    ┌─────────────────────────────────────────────────────────────┐
    │                     Global Grid (e.g. 100×80)               │
    │                                                             │
    │   ┌──────┬──────┬──────┬──────┐                            │
    │   │  A1  │  B1  │  C1  │  D1  │   Sector size: 10×10      │
    │   ├──────┼──────┼──────┼──────┤                            │
    │   │  A2  │  B2  │  C2  │  D2  │   Active neighbourhood:   │
    │   ├──────┼──────┼──────┼──────┤   robot's sector + 8      │
    │   │  A3  │  B3  │  C3  │  D3  │   surrounding sectors     │
    │   └──────┴──────┴──────┴──────┘                            │
    └─────────────────────────────────────────────────────────────┘

Scalability
-----------
Instead of flooding all N robots, each robot broadcasts its IntentPacket
*only* to peers within its 9-sector neighbourhood (O(k) where k is the
local density ≪ N).  This replaces an O(N²) all-pairs broadcast with a
localised O(k) scatter.

PeerRegistry
------------
Maintains a mapping of ``robot_id → (host, port, position)`` and answers
``peers_in_sectors(sector_set)`` queries in O(k) time using an inverted
index ``sector_label → {robot_id, ...}``.

P2PNode
-------
Each robot runs exactly one ``P2PNode`` that:
  1. Binds a non-blocking UDP socket on port ``BASE_PORT + robot_id_int``.
  2. Listens for incoming packets in an asyncio UDP protocol handler.
  3. On ``broadcast_intent(packet)``:
       a. Signs the packet.
       b. Queries SpatialHasher for its 9 active sectors.
       c. Looks up peer ports from the PeerRegistry.
       d. Sends the signed datagram to each relevant peer only.
  4. On packet receipt:
       a. Deserialises.
       b. Verifies HMAC.
       c. Checks AntiReplayGuard.
       d. Updates local SpatialHasher knowledge.
       e. Appends to ``received_packets`` for upstream consumption.

Running the nodes
-----------------
``P2PNode`` is built on ``asyncio`` DatagramProtocol / DatagramTransport.
Use ``asyncio.run(run_mesh(...))`` in the test or ``asyncio.get_event_loop()``
on edge hardware.

Zero external dependencies — ``asyncio``, ``socket``, ``logging``,
``threading``, ``dataclasses``, and ``typing`` from the standard library.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Optional, Set, Tuple

from p2p.protocol import (
    AntiReplayGuard,
    IntentPacket,
    MAX_CLOCK_SKEW_SECONDS,
    deserialise,
    serialise,
    sign_packet,
    verify_packet,
)

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SECTOR_SIZE:  int = 10          # cells per sector edge
BASE_PORT:    int = 5000        # port = BASE_PORT + numeric suffix of robot id
MAX_DGRAM:    int = 65_507      # safe max UDP payload (bytes)


# ---------------------------------------------------------------------------
# SpatialHasher
# ---------------------------------------------------------------------------

class SpatialHasher:
    """
    Partition the warehouse grid into ``SECTOR_SIZE × SECTOR_SIZE`` sectors
    and answer neighbourhood queries.

    Sector naming convention
    ------------------------
    Column sectors map to letter labels (A, B, …, Z, AA, AB, …).
    Row sectors map to 1-based integer labels.
    Combined: ``{col_label}{row_number}`` — e.g. ``"A1"``, ``"B3"``, ``"AA2"``.

    Parameters
    ----------
    grid_cols : int
        Total number of columns in the global grid.
    grid_rows : int
        Total number of rows in the global grid.
    sector_size : int, optional
        Edge length of each square sector in cells.  Default: 10.
    """

    def __init__(
        self,
        grid_cols: int,
        grid_rows: int,
        sector_size: int = SECTOR_SIZE,
    ) -> None:
        if grid_cols <= 0 or grid_rows <= 0:
            raise ValueError("SpatialHasher: grid dimensions must be positive.")
        self.grid_cols   = grid_cols
        self.grid_rows   = grid_rows
        self.sector_size = sector_size

        # Number of sectors along each axis (ceiling division)
        self.num_col_sectors: int = (grid_cols + sector_size - 1) // sector_size
        self.num_row_sectors: int = (grid_rows + sector_size - 1) // sector_size

    # ------------------------------------------------------------------
    # Sector label encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _col_label(col_idx: int) -> str:
        """
        Convert a 0-based column sector index to an alphabetic label.

        Examples:  0 → 'A',  25 → 'Z',  26 → 'AA',  27 → 'AB', …
        """
        label = ""
        n = col_idx
        while True:
            label = chr(ord("A") + n % 26) + label
            n = n // 26 - 1
            if n < 0:
                break
        return label

    def sector_label(self, col_sector: int, row_sector: int) -> str:
        """
        Return the human-readable sector label for the given sector indices.

        Parameters
        ----------
        col_sector : int
            0-based column sector index.
        row_sector : int
            0-based row sector index.

        Returns
        -------
        str
            Label such as ``"A1"``, ``"B3"``, ``"AA12"``.
        """
        return f"{self._col_label(col_sector)}{row_sector + 1}"

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def get_sector(self, x: int, y: int) -> str:
        """
        Return the sector label for cell position (x, y) where x is the
        column and y is the row.

        Clamps out-of-bounds positions to the nearest valid sector.
        """
        col_s = min(x // self.sector_size, self.num_col_sectors - 1)
        row_s = min(y // self.sector_size, self.num_row_sectors - 1)
        col_s = max(col_s, 0)
        row_s = max(row_s, 0)
        return self.sector_label(col_s, row_s)

    def get_sector_indices(self, x: int, y: int) -> Tuple[int, int]:
        """Return (col_sector_idx, row_sector_idx) for position (x, y)."""
        col_s = max(0, min(x // self.sector_size, self.num_col_sectors - 1))
        row_s = max(0, min(y // self.sector_size, self.num_row_sectors - 1))
        return col_s, row_s

    def get_active_neighborhood(self, current_pos: Tuple[int, int]) -> FrozenSet[str]:
        """
        Return the 9-sector Moore neighbourhood (current sector + 8 adjacent)
        for the robot at *current_pos*.

        Border sectors clamp gracefully — a corner robot's neighbourhood
        contains only the valid 4 or 6 sectors that exist.

        Parameters
        ----------
        current_pos : (x, y)
            Current (column, row) of the robot.

        Returns
        -------
        frozenset of str
            Sector labels (1 ≤ |result| ≤ 9).
        """
        cx, cy = current_pos
        cs_col, cs_row = self.get_sector_indices(cx, cy)
        labels: Set[str] = set()
        for dc in (-1, 0, 1):
            for dr in (-1, 0, 1):
                nc = cs_col + dc
                nr = cs_row + dr
                if 0 <= nc < self.num_col_sectors and 0 <= nr < self.num_row_sectors:
                    labels.add(self.sector_label(nc, nr))
        return frozenset(labels)

    def sector_bounds(self, label: str) -> Tuple[int, int, int, int]:
        """
        Return the (col_min, row_min, col_max, row_max) cell bounds for
        the given sector label.

        Primarily useful for visualisation / debugging.
        """
        # Reverse label → indices by iterating all sectors
        for cs in range(self.num_col_sectors):
            for rs in range(self.num_row_sectors):
                if self.sector_label(cs, rs) == label:
                    col_min = cs * self.sector_size
                    row_min = rs * self.sector_size
                    col_max = min(col_min + self.sector_size - 1, self.grid_cols - 1)
                    row_max = min(row_min + self.sector_size - 1, self.grid_rows - 1)
                    return col_min, row_min, col_max, row_max
        raise KeyError(f"sector_bounds: unknown sector label '{label}'")

    def __repr__(self) -> str:
        return (
            f"SpatialHasher(grid={self.grid_cols}x{self.grid_rows}, "
            f"sector={self.sector_size}x{self.sector_size}, "
            f"sectors={self.num_col_sectors}x{self.num_row_sectors})"
        )


# ---------------------------------------------------------------------------
# PeerInfo & PeerRegistry
# ---------------------------------------------------------------------------

@dataclass
class PeerInfo:
    """Lightweight descriptor of a known peer in the fleet."""
    robot_id: str
    host:     str
    port:     int
    position: Tuple[int, int]   # last known (x, y) — updated on receipt


class PeerRegistry:
    """
    Shared fleet directory: maps robot_id to PeerInfo and maintains an
    inverted index of sector → {robot_ids}.

    This class is used by each ``P2PNode`` to resolve *who to unicast* after
    the spatial neighbourhood filter reduces the candidate set.

    Thread-safety
    -------------
    Not thread-safe.  Use within a single asyncio event loop only.
    """

    def __init__(self, hasher: SpatialHasher) -> None:
        self._hasher:     SpatialHasher           = hasher
        self._peers:      Dict[str, PeerInfo]     = {}
        # Inverted index: sector_label -> set of robot_ids in that sector
        self._sector_idx: Dict[str, Set[str]]     = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, info: PeerInfo) -> None:
        """
        Register (or update) a peer's PeerInfo.  Automatically updates the
        sector inverted index.
        """
        # Remove old sector entry if position changed
        old = self._peers.get(info.robot_id)
        if old is not None:
            old_sector = self._hasher.get_sector(*old.position)
            if old_sector in self._sector_idx:
                self._sector_idx[old_sector].discard(info.robot_id)

        self._peers[info.robot_id] = info
        new_sector = self._hasher.get_sector(*info.position)
        self._sector_idx.setdefault(new_sector, set()).add(info.robot_id)

    def update_position(self, robot_id: str, new_pos: Tuple[int, int]) -> None:
        """Move a peer's position and update the sector index."""
        info = self._peers.get(robot_id)
        if info is None:
            return
        self.register(PeerInfo(info.robot_id, info.host, info.port, new_pos))

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def peers_in_sectors(
        self,
        sectors: FrozenSet[str],
        exclude_id: Optional[str] = None,
    ) -> List[PeerInfo]:
        """
        Return all PeerInfo objects whose *current sector* is within
        *sectors*, excluding the querying robot itself.

        Complexity: O(k) where k is the number of peers in the local region.
        """
        result: List[PeerInfo] = []
        seen: Set[str] = set()
        for s in sectors:
            for rid in self._sector_idx.get(s, set()):
                if rid not in seen and rid != exclude_id:
                    seen.add(rid)
                    info = self._peers.get(rid)
                    if info is not None:
                        result.append(info)
        return result

    def get(self, robot_id: str) -> Optional[PeerInfo]:
        """Return PeerInfo for *robot_id*, or None if not registered."""
        return self._peers.get(robot_id)

    def all_peers(self) -> List[PeerInfo]:
        """Return all registered peers."""
        return list(self._peers.values())

    def __len__(self) -> int:
        return len(self._peers)

    def __repr__(self) -> str:
        return f"PeerRegistry(peers={len(self._peers)})"


# ---------------------------------------------------------------------------
# Asyncio UDP Protocol handler
# ---------------------------------------------------------------------------

class _UDPProtocol(asyncio.DatagramProtocol):
    """
    asyncio DatagramProtocol attached to each P2PNode's socket.

    Received datagrams are queued into ``recv_queue`` for the node's
    processing loop.
    """

    def __init__(self, recv_queue: asyncio.Queue) -> None:  # type: ignore[type-arg]
        self._queue = recv_queue
        self.transport: Optional[asyncio.DatagramTransport] = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:  # type: ignore[override]
        self.transport = transport  # type: ignore[assignment]

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        self._queue.put_nowait((data, addr))

    def error_received(self, exc: Exception) -> None:
        logger.warning("UDP error received: %s", exc)

    def connection_lost(self, exc: Optional[Exception]) -> None:
        logger.debug("UDP connection lost: %s", exc)


# ---------------------------------------------------------------------------
# P2PNode
# ---------------------------------------------------------------------------

class P2PNode:
    """
    Asynchronous P2P fleet node for a single AMR.

    Each node:
    * Binds to ``(host, BASE_PORT + robot_numeric_id)``.
    * Sends signed ``IntentPacket`` datagrams only to peers inside the
      9-sector neighbourhood (O(k) broadcast cost).
    * Receives and validates incoming datagrams:
        - HMAC verification (via ``verify_packet``).
        - Anti-replay (via ``AntiReplayGuard``).
        - Clock-skew check (secondary time-based replay guard).

    Parameters
    ----------
    robot_id : str
        Unique robot identifier.  Must contain or be a parseable integer
        suffix when combined with ``BASE_PORT``.
    host : str
        Interface to bind (``"127.0.0.1"`` for loopback testing).
    port : int
        UDP port to bind.
    shared_key : bytes
        Fleet-wide pre-shared secret (>= 16 bytes).
    hasher : SpatialHasher
        Shared spatial hasher for neighbourhood computation.
    registry : PeerRegistry
        Shared peer directory.
    position : (x, y)
        Initial position of this robot.
    on_packet : callable, optional
        Callback ``(IntentPacket) -> None`` invoked for every valid
        received packet.  Useful for feeding the ReservationTable.
    """

    def __init__(
        self,
        robot_id:   str,
        host:       str,
        port:       int,
        shared_key: bytes,
        hasher:     SpatialHasher,
        registry:   PeerRegistry,
        position:   Tuple[int, int],
        on_packet:  Optional[Callable[[IntentPacket], None]] = None,
    ) -> None:
        self.robot_id   = robot_id
        self.host       = host
        self.port       = port
        self._key       = shared_key
        self._hasher    = hasher
        self._registry  = registry
        self._position  = position
        self._on_packet = on_packet

        self._seq:           int                     = 0
        self._replay_guard:  AntiReplayGuard         = AntiReplayGuard()
        self._recv_queue:    asyncio.Queue           = asyncio.Queue()  # type: ignore[type-arg]

        # Collected packets accessible after running
        self.received_packets: List[IntentPacket]   = []
        self.rejected_spoofed: int                  = 0
        self.rejected_replayed: int                 = 0

        self._transport:  Optional[asyncio.DatagramTransport] = None
        self._protocol:   Optional[_UDPProtocol]              = None
        self._recv_task:  Optional[asyncio.Task]              = None  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Bind the UDP socket and start the background receive loop.

        Must be called inside an active asyncio event loop.
        """
        loop = asyncio.get_running_loop()
        self._transport, self._protocol = await loop.create_datagram_endpoint(
            lambda: _UDPProtocol(self._recv_queue),
            local_addr=(self.host, self.port),
            family=socket.AF_INET,
        )
        self._recv_task = asyncio.create_task(
            self._receive_loop(), name=f"recv-{self.robot_id}"
        )
        logger.info("[%s] P2PNode started on %s:%d", self.robot_id, self.host, self.port)

    async def stop(self) -> None:
        """Gracefully shut down the node: cancel receive loop, close socket."""
        if self._recv_task is not None:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        if self._transport is not None:
            self._transport.close()
        logger.info("[%s] P2PNode stopped.", self.robot_id)

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    def update_position(self, new_pos: Tuple[int, int]) -> None:
        """Update this node's position and refresh the PeerRegistry entry."""
        self._position = new_pos
        self._registry.update_position(self.robot_id, new_pos)

    async def broadcast_intent(
        self,
        trajectory: List[Tuple[int, int, int]],
    ) -> int:
        """
        Sign and broadcast an ``IntentPacket`` to all peers in the 9-sector
        neighbourhood.

        Parameters
        ----------
        trajectory : list of (x, y, t)
            Space-time path from ``core.space_time_astar``.

        Returns
        -------
        int
            Number of peers the packet was sent to.
        """
        self._seq += 1
        packet = IntentPacket.create(
            robot_id=self.robot_id,
            seq=self._seq,
            trajectory=trajectory,
        )
        signed = sign_packet(packet, self._key)
        data   = serialise(signed)

        if len(data) > MAX_DGRAM:
            logger.error(
                "[%s] Packet too large (%d bytes) — truncate trajectory.",
                self.robot_id, len(data),
            )
            return 0

        # Neighbourhood filter — O(k) not O(N)
        active_sectors = self._hasher.get_active_neighborhood(self._position)
        neighbours     = self._registry.peers_in_sectors(
            active_sectors, exclude_id=self.robot_id
        )

        if self._transport is None:
            logger.warning("[%s] Transport not ready; packet not sent.", self.robot_id)
            return 0

        for peer in neighbours:
            try:
                self._transport.sendto(data, (peer.host, peer.port))
            except Exception as exc:
                logger.warning(
                    "[%s] Failed to send to %s:%d — %s",
                    self.robot_id, peer.host, peer.port, exc,
                )
        return len(neighbours)

    async def send_raw(self, data: bytes, host: str, port: int) -> None:
        """
        Send a raw datagram to a specific endpoint.

        Used in the test harness to inject spoofed / replayed packets
        directly without going through ``broadcast_intent``.
        """
        if self._transport is not None:
            self._transport.sendto(data, (host, port))

    # ------------------------------------------------------------------
    # Receiving
    # ------------------------------------------------------------------

    async def _receive_loop(self) -> None:
        """
        Background coroutine: dequeue received datagrams, validate them,
        and dispatch to ``on_packet`` callback.
        """
        while True:
            data, addr = await self._recv_queue.get()
            await self._handle_datagram(data, addr)

    async def _handle_datagram(
        self, data: bytes, addr: Tuple[str, int]
    ) -> None:
        """
        Full validation pipeline for a received datagram:

        1. Deserialise.
        2. Clock-skew guard.
        3. HMAC verification.
        4. Anti-replay sequence check.
        5. Dispatch to ``on_packet``.
        """
        # 1. Deserialise
        try:
            packet = deserialise(data)
        except ValueError as exc:
            logger.warning("[%s] Malformed datagram from %s: %s", self.robot_id, addr, exc)
            self.rejected_spoofed += 1
            return

        # 2. Clock-skew guard
        age = abs(time.time() - packet.timestamp)
        if age > MAX_CLOCK_SKEW_SECONDS:
            logger.warning(
                "[%s] Packet from %s rejected: clock skew %.1f s > %.1f s limit.",
                self.robot_id, packet.robot_id, age, MAX_CLOCK_SKEW_SECONDS,
            )
            self.rejected_replayed += 1
            return

        # 3. HMAC verification
        if not verify_packet(packet, self._key):
            logger.warning(
                "[%s] HMAC verification FAILED for packet from %s (seq=%d).",
                self.robot_id, packet.robot_id, packet.monotonic_seq,
            )
            self.rejected_spoofed += 1
            return

        # 4. Anti-replay sequence check
        if not self._replay_guard.accept(packet.robot_id, packet.monotonic_seq):
            logger.warning(
                "[%s] REPLAY rejected: %s seq=%d (last_seen=%s).",
                self.robot_id,
                packet.robot_id,
                packet.monotonic_seq,
                self._replay_guard.last_seen(packet.robot_id),
            )
            self.rejected_replayed += 1
            return

        # 5. Valid packet — update registry and dispatch
        first_pos: Optional[Tuple[int, int]] = (
            (packet.trajectory_points[0][0], packet.trajectory_points[0][1])
            if packet.trajectory_points
            else None
        )
        if first_pos is not None:
            peer_info = self._registry.get(packet.robot_id)
            if peer_info is None:
                logger.debug(
                    "[%s] Auto-registering new peer %s at %s",
                    self.robot_id, packet.robot_id, first_pos,
                )
                self._registry.register(
                    PeerInfo(packet.robot_id, addr[0], addr[1], first_pos)
                )
            else:
                self._registry.update_position(packet.robot_id, first_pos)

        self.received_packets.append(packet)
        logger.debug(
            "[%s] Accepted packet from %s seq=%d (%d waypoints).",
            self.robot_id,
            packet.robot_id,
            packet.monotonic_seq,
            len(packet.trajectory_points),
        )
        if self._on_packet is not None:
            try:
                self._on_packet(packet)
            except Exception as exc:
                logger.error("[%s] on_packet callback raised: %s", self.robot_id, exc)

    def __repr__(self) -> str:
        return (
            f"P2PNode(id={self.robot_id!r}, port={self.port}, "
            f"pos={self._position}, seq={self._seq})"
        )


# ===========================================================================
# Stress test
# ===========================================================================

async def _async_test_p2p_mesh() -> None:
    """
    Async implementation of the P2P mesh stress test.

    Five virtual P2P nodes on localhost ports 5000–5004.

    Steps
    -----
    1. Build a 50×50 open grid, SpatialHasher, and shared PeerRegistry.
    2. Start 5 P2PNode instances and register them all with the registry.
    3. Node-0 broadcasts a legitimate IntentPacket; verify it is received
       by all neighbours (Nodes 1–4 are all within the same neighbourhood
       for this small grid).
    4. Inject a spoofed packet (wrong HMAC key) into Node-0's socket;
       verify all receivers increment ``rejected_spoofed``.
    5. Inject a replayed packet (same seq as a previously accepted packet)
       into Node-0's socket; verify all receivers increment
       ``rejected_replayed``.
    """
    logging.basicConfig(level=logging.WARNING)
    print("=" * 65)
    print("P2P Spatial Mesh  —  Stress Test")
    print("=" * 65)

    HOST       = "127.0.0.1"
    NUM_NODES  = 5
    GRID_SIZE  = 50
    FLEET_KEY  = b"fleet_shared_secret_key_32bytes!"  # 32-byte key
    WRONG_KEY  = b"attacker_wrong_key_for_spoof!!!!!"

    hasher   = SpatialHasher(grid_cols=GRID_SIZE, grid_rows=GRID_SIZE)
    registry = PeerRegistry(hasher)

    # Positions: spread across sector A1 (col 0-9, row 0-9) so all are
    # in the same neighbourhood.
    positions = [(i * 2, i * 2) for i in range(NUM_NODES)]
    ports     = [BASE_PORT + i for i in range(NUM_NODES)]

    # Create and pre-register all peers
    nodes: List[P2PNode] = []
    for i in range(NUM_NODES):
        node = P2PNode(
            robot_id=f"AMR_{i:03d}",
            host=HOST,
            port=ports[i],
            shared_key=FLEET_KEY,
            hasher=hasher,
            registry=registry,
            position=positions[i],
        )
        nodes.append(node)
        registry.register(
            PeerInfo(
                robot_id=f"AMR_{i:03d}",
                host=HOST,
                port=ports[i],
                position=positions[i],
            )
        )

    # Start all nodes
    for node in nodes:
        await node.start()

    await asyncio.sleep(0.05)  # allow sockets to bind

    # ------------------------------------------------------------------
    # Step 3: Legitimate broadcast from Node-0
    # ------------------------------------------------------------------
    print("\n[Step 3] Node-0 broadcasts a legitimate IntentPacket ...")
    sample_trajectory = [(0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 3)]
    sent_to = await nodes[0].broadcast_intent(sample_trajectory)
    print(f"  Sent to {sent_to} peers (expected {NUM_NODES - 1})")

    await asyncio.sleep(0.15)  # allow datagrams to arrive

    # Count receivers that got at least 1 valid packet
    receivers_ok = sum(
        1 for n in nodes[1:] if len(n.received_packets) >= 1
    )
    assert receivers_ok == NUM_NODES - 1, (
        f"Step 3 FAILED: only {receivers_ok}/{NUM_NODES-1} nodes received "
        f"the legitimate packet."
    )
    print(f"  {receivers_ok}/{NUM_NODES-1} nodes received legitimate packet  [OK]")

    # ------------------------------------------------------------------
    # Step 4: Inject a spoofed packet (wrong HMAC key)
    # ------------------------------------------------------------------
    print("\n[Step 4] Injecting spoofed packet (wrong HMAC key) ...")

    spoofed_pkt   = IntentPacket.create("SPOOFER", seq=999, trajectory=[(9, 9, 0)])
    spoofed_signed = sign_packet(spoofed_pkt, WRONG_KEY)   # signed with wrong key
    spoofed_data   = serialise(spoofed_signed)

    # Directly send to Node-1 through Node-0's transport (simulates
    # a hostile packet appearing on the wire)
    for target in nodes[1:]:
        await nodes[0].send_raw(spoofed_data, HOST, target.port)

    await asyncio.sleep(0.15)

    spoofed_rejections = sum(n.rejected_spoofed for n in nodes[1:])
    assert spoofed_rejections >= NUM_NODES - 1, (
        f"Step 4 FAILED: only {spoofed_rejections} spoofed packets rejected "
        f"(expected >= {NUM_NODES-1})."
    )
    print(f"  {spoofed_rejections} spoofed packets rejected by HMAC check  [OK]")

    # ------------------------------------------------------------------
    # Step 5: Inject a replayed packet (valid HMAC, old seq)
    # ------------------------------------------------------------------
    print("\n[Step 5] Injecting replayed packet (old seq=1) ...")

    # Build a genuine-looking packet from AMR_000 with seq=1 (already
    # accepted in Step 3 — replay guard will fire)
    replay_pkt    = IntentPacket(
        robot_id="AMR_000",
        monotonic_seq=1,          # previously accepted seq
        timestamp=time.time(),    # fresh timestamp to pass clock-skew
        trajectory_points=[(0, 0, 0)],
    )
    replay_signed = sign_packet(replay_pkt, FLEET_KEY)   # valid HMAC
    replay_data   = serialise(replay_signed)

    # First delivery — all nodes record seq=1 for AMR_000
    # (Step 3 already sent seq=1; receivers already have last_seen=1)
    for target in nodes[1:]:
        await nodes[0].send_raw(replay_data, HOST, target.port)

    await asyncio.sleep(0.15)

    replay_rejections = sum(n.rejected_replayed for n in nodes[1:])
    assert replay_rejections >= NUM_NODES - 1, (
        f"Step 5 FAILED: only {replay_rejections} replayed packets rejected "
        f"(expected >= {NUM_NODES-1})."
    )
    print(f"  {replay_rejections} replayed packets rejected by anti-replay guard  [OK]")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    for node in nodes:
        await node.stop()

    print("\n" + "=" * 65)
    print("ALL P2P MESH STRESS TESTS PASSED")
    print("=" * 65)

    # Summary table
    print("\n  Node              recv_ok  rej_spoof  rej_replay  port")
    print("  " + "-" * 60)
    for n in nodes:
        print(
            f"  {n.robot_id:<16}  {len(n.received_packets):<7}  "
            f"{n.rejected_spoofed:<9}  {n.rejected_replayed:<10}  {n.port}"
        )


def test_p2p_mesh() -> None:
    """
    Entry point: run the async stress test synchronously.

    Call this from a CLI or test runner::

        from p2p.spatial_mesh import test_p2p_mesh
        test_p2p_mesh()
    """
    asyncio.run(_async_test_p2p_mesh())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_p2p_mesh()
