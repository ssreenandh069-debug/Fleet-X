"""
p2p/protocol.py
===============
Wire-protocol definitions and cryptographic primitives for the AMR P2P mesh.

Packet structure
----------------
Every ``IntentPacket`` carries exactly five fields over the wire:

    robot_id          : str   — globally unique robot identifier.
    monotonic_seq     : int   — strictly increasing per-robot counter;
                                 anti-replay gate rejects seq <= last_seen.
    timestamp         : float — Unix epoch (time.monotonic() offset at boot
                                 is mapped to wall time via a constant delta).
    trajectory_points : list  — ordered space-time waypoints [(x, y, t), ...]
                                 produced by core.space_time_astar.
    hmac_signature    : bytes — 16-byte truncated HMAC-SHA256 over the
                                 canonical payload (all four fields above).

Security model
--------------
* Shared-secret HMAC-SHA256:  all peers share a pre-distributed fleet key
  (loaded from an environment variable or a key file in production).
* Signature is computed over a deterministic JSON encoding of the four
  non-signature fields (UTF-8, sorted keys, no extra whitespace).
* Truncation to 16 bytes is a deliberate trade-off: enough for collision
  resistance (2^64 guessing difficulty) while keeping per-packet overhead
  small over constrained edge links.
* Anti-replay: per-sender ``monotonic_seq`` is checked by the *receiver*.
  The first packet from a new sender is always accepted; all subsequent
  packets must carry a strictly greater sequence number.

Wire format
-----------
Packets are serialised to UTF-8 JSON for portability and debuggability.
The ``hmac_signature`` is hex-encoded inside JSON (no binary in the stream).
Maximum intended packet size is < 2 KB for a 20-waypoint trajectory.

Zero external dependencies — only ``hashlib``, ``hmac``, ``json``, ``time``,
``dataclasses``, and ``typing`` from the standard library.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HMAC_TRUNCATE_BYTES: int = 16          # 128-bit truncated digest
_HMAC_ALGORITHM:     str = "sha256"

# Maximum clock skew tolerated between fleet nodes (seconds).
# Packets older than this relative to the receiver's clock are discarded
# even if the HMAC is valid, as a secondary time-based replay guard.
MAX_CLOCK_SKEW_SECONDS: float = 30.0

# Type alias — matches STNode from core.space_time_astar
TrajectoryPoint = Tuple[int, int, int]    # (x, y, t)


# ---------------------------------------------------------------------------
# IntentPacket
# ---------------------------------------------------------------------------

@dataclass
class IntentPacket:
    """
    A single broadcast message advertising a robot's planned trajectory.

    Attributes
    ----------
    robot_id : str
        Unique robot identifier (e.g., "AMR_007").
    monotonic_seq : int
        Strictly monotonically increasing per-robot counter.  Starts at 1.
    timestamp : float
        Wall-clock time (time.time()) at which this packet was created.
    trajectory_points : List[TrajectoryPoint]
        Ordered list of (x, y, t) space-time waypoints — output of
        ``core.space_time_astar.SpaceTimeAstar.plan()``.
    hmac_signature : bytes
        16-byte truncated HMAC-SHA256 of the canonical payload.
        Set to ``b""`` before signing; filled by ``sign_packet()``.
    """

    robot_id:           str
    monotonic_seq:      int
    timestamp:          float
    trajectory_points:  List[TrajectoryPoint]
    hmac_signature:     bytes = field(default=b"", compare=False, repr=False)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        robot_id: str,
        seq: int,
        trajectory: List[TrajectoryPoint],
    ) -> "IntentPacket":
        """
        Create an unsigned packet with the current wall-clock timestamp.

        Parameters
        ----------
        robot_id : str
            Robot identifier.
        seq : int
            Monotonic sequence number (must be > previous seq for this robot).
        trajectory : list of (x, y, t)
            Space-time path from the planner.

        Returns
        -------
        IntentPacket
            Unsigned packet (``hmac_signature == b""``).
            Call ``sign_packet(packet, key)`` before transmitting.
        """
        return cls(
            robot_id=robot_id,
            monotonic_seq=seq,
            timestamp=time.time(),
            trajectory_points=list(trajectory),
        )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self, include_signature: bool = True) -> Dict[str, Any]:
        """
        Convert to a plain dict suitable for JSON serialisation.

        The ``trajectory_points`` list contains 3-tuples; JSON does not
        distinguish tuples from lists, so we encode them as lists.

        Parameters
        ----------
        include_signature : bool
            If *False*, the ``hmac_signature`` key is omitted.  This is used
            internally to compute the HMAC payload.
        """
        d: Dict[str, Any] = {
            "robot_id":         self.robot_id,
            "monotonic_seq":    self.monotonic_seq,
            "timestamp":        self.timestamp,
            "trajectory_points": [list(p) for p in self.trajectory_points],
        }
        if include_signature:
            d["hmac_signature"] = self.hmac_signature.hex()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IntentPacket":
        """
        Reconstruct an ``IntentPacket`` from a plain dict (e.g., parsed JSON).

        Raises
        ------
        KeyError
            If required fields are missing.
        ValueError
            If field types are wrong.
        """
        sig_hex: str = d.get("hmac_signature", "")
        return cls(
            robot_id=str(d["robot_id"]),
            monotonic_seq=int(d["monotonic_seq"]),
            timestamp=float(d["timestamp"]),
            trajectory_points=[
                (int(p[0]), int(p[1]), int(p[2]))
                for p in d["trajectory_points"]
            ],
            hmac_signature=bytes.fromhex(sig_hex) if sig_hex else b"",
        )


# ---------------------------------------------------------------------------
# Canonical payload serialisation (used for HMAC)
# ---------------------------------------------------------------------------

def _canonical_payload(packet: IntentPacket) -> bytes:
    """
    Produce a deterministic byte string over the four non-signature fields.

    JSON with sorted keys and no superfluous whitespace ensures that two
    implementations encoding the same logical packet produce identical bytes.

    Returns
    -------
    bytes
        UTF-8-encoded JSON string of the four payload fields.
    """
    payload_dict = packet.to_dict(include_signature=False)
    return json.dumps(payload_dict, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


# ---------------------------------------------------------------------------
# Signing & Verification
# ---------------------------------------------------------------------------

def sign_packet(packet: IntentPacket, shared_key: bytes) -> IntentPacket:
    """
    Compute the HMAC-SHA256 signature over *packet*'s payload and return a
    **new** ``IntentPacket`` with the ``hmac_signature`` field set.

    Parameters
    ----------
    packet : IntentPacket
        Unsigned packet (``hmac_signature`` is ignored during computation).
    shared_key : bytes
        Pre-shared fleet key.  Must be >= 16 bytes for adequate security.

    Returns
    -------
    IntentPacket
        A copy of *packet* with ``hmac_signature`` set to the 16-byte digest.
    """
    if len(shared_key) < 16:
        raise ValueError(
            f"sign_packet: shared_key must be >= 16 bytes, got {len(shared_key)}."
        )
    payload = _canonical_payload(packet)
    digest = _hmac.new(shared_key, payload, _HMAC_ALGORITHM).digest()
    return IntentPacket(
        robot_id=packet.robot_id,
        monotonic_seq=packet.monotonic_seq,
        timestamp=packet.timestamp,
        trajectory_points=packet.trajectory_points,
        hmac_signature=digest[:HMAC_TRUNCATE_BYTES],
    )


def verify_packet(packet: IntentPacket, shared_key: bytes) -> bool:
    """
    Verify the ``hmac_signature`` of *packet* using *shared_key*.

    Uses a constant-time comparison (``hmac.compare_digest``) to prevent
    timing side-channel attacks.

    Parameters
    ----------
    packet : IntentPacket
        The packet to verify.  Must have ``hmac_signature`` set.
    shared_key : bytes
        The same pre-shared key used when the packet was signed.

    Returns
    -------
    bool
        ``True`` if the signature is valid and non-empty, ``False`` otherwise.
    """
    if not packet.hmac_signature:
        return False
    payload = _canonical_payload(packet)
    expected = _hmac.new(shared_key, payload, _HMAC_ALGORITHM).digest()
    expected_truncated = expected[:HMAC_TRUNCATE_BYTES]
    return _hmac.compare_digest(packet.hmac_signature, expected_truncated)


# ---------------------------------------------------------------------------
# Wire serialisation / deserialisation
# ---------------------------------------------------------------------------

def serialise(packet: IntentPacket) -> bytes:
    """
    Serialise a signed ``IntentPacket`` to a compact UTF-8 JSON byte string
    suitable for a single UDP datagram.

    Parameters
    ----------
    packet : IntentPacket
        Must be signed (``hmac_signature != b""``).

    Returns
    -------
    bytes
        Serialised packet bytes.
    """
    return json.dumps(
        packet.to_dict(include_signature=True),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def deserialise(data: bytes) -> IntentPacket:
    """
    Deserialise a UDP datagram back into an ``IntentPacket``.

    Parameters
    ----------
    data : bytes
        Raw bytes received from the network.

    Returns
    -------
    IntentPacket
        Parsed packet (unverified — call ``verify_packet()`` before trusting).

    Raises
    ------
    ValueError
        If JSON is malformed or required fields are missing.
    """
    try:
        d = json.loads(data.decode("utf-8"))
        return IntentPacket.from_dict(d)
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"deserialise: malformed packet — {exc}") from exc


# ---------------------------------------------------------------------------
# AntiReplayGuard
# ---------------------------------------------------------------------------

class AntiReplayGuard:
    """
    Per-receiver tracker of the highest observed sequence number for each
    sending robot.

    Thread-safety
    -------------
    This class is NOT thread-safe.  In async or multi-threaded code the
    caller must protect access with an ``asyncio.Lock`` or ``threading.Lock``.

    Rules
    -----
    * First packet from a new sender (seq any value) → ACCEPTED and
      recorded.
    * Subsequent packets must satisfy ``seq > last_seen_seq`` → ACCEPTED.
    * ``seq <= last_seen_seq`` → REJECTED as potential replay.
    """

    def __init__(self) -> None:
        # robot_id -> highest accepted sequence number
        self._last_seq: Dict[str, int] = {}

    def is_fresh(self, robot_id: str, seq: int) -> bool:
        """
        Return ``True`` if *seq* is strictly greater than the last seen
        sequence for *robot_id* (or if *robot_id* is encountered for the
        first time).
        """
        last = self._last_seq.get(robot_id)
        return last is None or seq > last

    def record(self, robot_id: str, seq: int) -> None:
        """
        Mark *seq* as the latest accepted sequence number for *robot_id*.
        Should be called only after both HMAC verification and ``is_fresh``
        pass.
        """
        self._last_seq[robot_id] = seq

    def accept(self, robot_id: str, seq: int) -> bool:
        """
        Combined ``is_fresh`` + ``record`` in a single atomic call.

        Returns
        -------
        bool
            ``True`` if accepted and recorded; ``False`` if rejected as
            replay.
        """
        if self.is_fresh(robot_id, seq):
            self.record(robot_id, seq)
            return True
        return False

    def last_seen(self, robot_id: str) -> Optional[int]:
        """Return the last accepted sequence number for *robot_id*, or None."""
        return self._last_seq.get(robot_id)

    def __repr__(self) -> str:
        return f"AntiReplayGuard(tracked_robots={len(self._last_seq)})"
