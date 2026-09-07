"""
telemetry/godot_bridge.py
==========================
Lightweight UDP telemetry bridge that streams per-robot state to a
Godot visualizer (or any UDP listener) at ``127.0.0.1:4242``.

Wire format (JSON, one datagram per agent per flush)
-----------------------------------------------------
    {
        "robot_id":   "AMR_001",
        "x":          5.0,
        "y":          8.0,
        "heading":    90.0,
        "state":      "NAVIGATING",
        "battery":    88.4,
        "laser_trail": [[5,9],[5,10],[6,10]],
        "speed_frac": 1.0,
        "tick":       142,
        "ts":         1725736234.871
    }

Design notes
------------
* Non-blocking: uses a single ``SOCK_DGRAM`` socket shared across all agents.
  A dropped frame (network hiccup) never stalls the control loop.
* Each ``flush()`` call iterates the fleet dict and sends one datagram per
  agent.  At 10 Hz with 4 agents that is 4 × ~200 bytes = ~800 bytes/s —
  negligible on loopback.
* The bridge can be disabled (``enabled=False``) for pure benchmark runs
  where Godot is not running, avoiding socket errors.

Godot-side GDScript to receive
-------------------------------
    var socket = PacketPeerUDP.new()
    socket.bind(4242)
    func _process(_delta):
        while socket.get_available_packet_count() > 0:
            var data = socket.get_packet().get_string_from_utf8()
            var json = JSON.parse_string(data)
            update_robot(json["robot_id"], json["x"], json["y"], json["state"])

Zero external dependencies — ``json``, ``socket``, ``logging``,
``dataclasses``, ``typing`` from the standard library.
"""

from __future__ import annotations

import json
import logging
import socket
import time as _time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default endpoints
# ---------------------------------------------------------------------------
GODOT_HOST:    str = "127.0.0.1"
GODOT_PORT:    int = 4242
MAX_DGRAM:     int = 65_507


# ---------------------------------------------------------------------------
# Fleet-level telemetry snapshot
# ---------------------------------------------------------------------------
@dataclass
class FleetTelemetry:
    """
    Aggregated view of the whole fleet at one point in time.

    Used by the BenchmarkRunner to collect per-tick metrics without
    socket overhead.
    """
    tick:       int
    timestamp:  float = field(default_factory=_time.time)
    agents:     Dict[str, dict] = field(default_factory=dict)

    def add(self, snapshot: dict) -> None:
        """Add/update one agent's telemetry."""
        self.agents[snapshot["robot_id"]] = snapshot

    def __repr__(self) -> str:
        return f"FleetTelemetry(tick={self.tick}, agents={list(self.agents)})"


# ---------------------------------------------------------------------------
# GodotBridge
# ---------------------------------------------------------------------------
class GodotBridge:
    """
    UDP telemetry bridge streaming robot state to Godot (or any listener).

    Parameters
    ----------
    host : str
        Destination IP (default ``127.0.0.1``).
    port : int
        Destination UDP port (default ``4242``).
    enabled : bool
        Set False to disable all socket activity (pure benchmark mode).
    cell_size_m : float
        Metres per grid cell — multiplied into x/y for Godot world coords.
        Default 0.5 m/cell.  Set 1.0 for 1:1 grid → world mapping.
    """

    def __init__(
        self,
        host:        str   = GODOT_HOST,
        port:        int   = GODOT_PORT,
        enabled:     bool  = True,
        cell_size_m: float = 0.5,
    ) -> None:
        self._host       = host
        self._port       = port
        self._enabled    = enabled
        self._cell       = cell_size_m
        self._sock: Optional[socket.socket] = None
        self._tick:      int   = 0
        self._sent:      int   = 0
        self._errors:    int   = 0

        if enabled:
            try:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._sock.setblocking(False)
                logger.info(
                    "GodotBridge: UDP socket ready -> %s:%d.", host, port
                )
            except OSError as exc:
                logger.error("GodotBridge: socket creation failed: %s", exc)
                self._enabled = False

    # ------------------------------------------------------------------
    # Single-agent send
    # ------------------------------------------------------------------
    def send(self, snapshot: dict, tick: int) -> bool:
        """
        Serialise and send one agent's telemetry snapshot.

        Parameters
        ----------
        snapshot : dict
            Agent's ``telemetry`` dict (produced by ``AMRAgent._update_telemetry()``).
        tick : int
            Current simulation tick.

        Returns
        -------
        bool
            True if the datagram was sent without error.
        """
        if not self._enabled or self._sock is None:
            return False

        payload = dict(snapshot)
        payload["tick"] = tick
        payload["ts"]   = _time.time()
        # Scale grid coords to world metres for Godot
        payload["x"]    = snapshot.get("x", 0.0) * self._cell
        payload["y"]    = snapshot.get("y", 0.0) * self._cell

        try:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            if len(data) > MAX_DGRAM:
                data = data[:MAX_DGRAM]
            self._sock.sendto(data, (self._host, self._port))
            self._sent += 1
            return True
        except BlockingIOError:
            # Socket buffer full — drop frame silently (non-blocking)
            return False
        except OSError as exc:
            self._errors += 1
            if self._errors <= 3:   # suppress spam after repeated failures
                logger.warning("GodotBridge: send error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Fleet flush (call once per tick after all agents have ticked)
    # ------------------------------------------------------------------
    def flush(self, agents: list, tick: int) -> FleetTelemetry:
        """
        Send one telemetry datagram per agent and return a ``FleetTelemetry``
        snapshot for the benchmark runner.

        Parameters
        ----------
        agents : list of AMRAgent
            The active fleet.
        tick : int
            Current simulation tick.

        Returns
        -------
        FleetTelemetry
            In-memory snapshot of all agents (always populated, even when
            ``enabled=False``).
        """
        self._tick = tick
        ft = FleetTelemetry(tick=tick)
        for agent in agents:
            ft.add(dict(agent.telemetry))
            self.send(agent.telemetry, tick)
        return ft

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release the UDP socket."""
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        logger.info(
            "GodotBridge closed. sent=%d errors=%d", self._sent, self._errors
        )

    def __enter__(self) -> "GodotBridge":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"GodotBridge(target={self._host}:{self._port}, "
            f"enabled={self._enabled}, sent={self._sent})"
        )
