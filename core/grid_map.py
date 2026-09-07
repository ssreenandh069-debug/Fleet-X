"""
core/grid_map.py
================
Discrete 2D occupancy grid for AMR warehouse navigation.

Cell semantics
--------------
    WALKWAY  (0) : Freely traversable aisle cell.
    SHELF    (1) : Impassable obstacle (occupied by rack).
    PICKUP   (2) : Pickup station — passable, special semantics for tasks.
    DROP     (3) : Drop/delivery station — passable, special semantics.

Virtual Highways
----------------
Each cell may carry a *preferred direction* expressed as a unit (dx, dy)
vector.  Moving **with** the preferred direction costs 1.0x (no penalty).
Moving **against** the preferred direction (exactly opposite) costs 10x.
Movement perpendicular to the preferred direction costs 1.5x.
Cells with no preferred direction (None) are penalty-free in all directions.

Design goals
------------
* Zero external dependencies — pure standard library.
* Fully typed throughout.
* sub-15 ms query paths (pure dict / list lookups, no numpy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants: cell types
# ---------------------------------------------------------------------------
WALKWAY = 0
SHELF   = 1
PICKUP  = 2
DROP    = 3

# Passable cell types (robots may occupy these)
PASSABLE_TYPES: frozenset[int] = frozenset({WALKWAY, PICKUP, DROP})

# Movement directions (action -> (dx, dy))
# dx = column-delta (East positive), dy = row-delta (South positive)
DIRECTION_VECTORS: Dict[str, Tuple[int, int]] = {
    "NORTH": ( 0, -1),
    "SOUTH": ( 0,  1),
    "EAST" : ( 1,  0),
    "WEST" : (-1,  0),
}

# Virtual highway penalty multipliers
_AGAINST_PENALTY:      float = 10.0
_PERPENDICULAR_PENALTY: float = 1.5


# ---------------------------------------------------------------------------
# HighwayRule dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HighwayRule:
    """
    Associates a preferred travel direction with a rectangular zone of cells.

    Attributes
    ----------
    preferred_dx : int
        Column delta of the preferred direction (+1 East, -1 West, 0 neutral).
    preferred_dy : int
        Row delta of the preferred direction (+1 South, -1 North, 0 neutral).
    row_start : int
        Inclusive start row of the highway zone.
    row_end : int
        Inclusive end row of the highway zone.
    col_start : int
        Inclusive start column of the highway zone.
    col_end : int
        Inclusive end column of the highway zone.
    """
    preferred_dx: int
    preferred_dy: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int


# ---------------------------------------------------------------------------
# GridMap
# ---------------------------------------------------------------------------
@dataclass
class GridMap:
    """
    Discrete 2D occupancy grid representing a warehouse floor.

    Parameters
    ----------
    grid : List[List[int]]
        Row-major 2D list of cell types.  ``grid[row][col]`` gives the type
        at row *row*, column *col*.
    highway_rules : list of HighwayRule, optional
        Directional biases applied to rectangular zones of the grid.

    Notes
    -----
    Coordinate convention used throughout this module and by the planner:
        * ``col`` (x in the planner) increases **East**.
        * ``row`` (y in the planner) increases **South**.
    The planner stores positions as ``(col, row)`` i.e. ``(x, y)``.
    """

    grid: List[List[int]]
    highway_rules: List[HighwayRule] = field(default_factory=list)

    # Derived attributes populated in __post_init__
    rows: int = field(init=False)
    cols: int = field(init=False)

    # Cached per-cell preferred direction: (row, col) -> (dx, dy) | None
    _cell_direction: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = field(
        init=False, default_factory=dict
    )

    # Per-cell inflation cost overlay (added on top of base move_cost).
    # Populated by HazardManager around detected obstacles.
    # Key: (col, row); Value: additional cost float (>= 0).
    _inflation_overlay: Dict[Tuple[int, int], float] = field(
        init=False, default_factory=dict
    )

    # Set of cells that were originally passable but later blocked at runtime
    _runtime_blocked: Set[Tuple[int, int]] = field(
        init=False, default_factory=set
    )

    def __post_init__(self) -> None:
        if not self.grid or not self.grid[0]:
            raise ValueError("GridMap: grid must be non-empty.")

        self.rows = len(self.grid)
        self.cols = len(self.grid[0])

        # Validate rectangular shape
        for r, row in enumerate(self.grid):
            if len(row) != self.cols:
                raise ValueError(
                    f"GridMap: row {r} has {len(row)} cols, expected {self.cols}."
                )

        # Build per-cell direction cache from highway rules.
        # Later rules override earlier ones for overlapping zones.
        self._cell_direction = {}
        for rule in self.highway_rules:
            direction: Tuple[int, int] = (rule.preferred_dx, rule.preferred_dy)
            for r in range(rule.row_start, rule.row_end + 1):
                for c in range(rule.col_start, rule.col_end + 1):
                    self._cell_direction[(r, c)] = direction

    # ------------------------------------------------------------------
    # Basic queries
    # ------------------------------------------------------------------

    def in_bounds(self, col: int, row: int) -> bool:
        """Return True if (col, row) is within the grid boundaries."""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def cell_type(self, col: int, row: int) -> int:
        """Return the cell type at (col, row)."""
        return self.grid[row][col]

    def is_passable(self, col: int, row: int) -> bool:
        """Return True if a robot may occupy (col, row)."""
        return self.in_bounds(col, row) and self.grid[row][col] in PASSABLE_TYPES

    # ------------------------------------------------------------------
    # Runtime cell mutation (used by HazardManager)
    # ------------------------------------------------------------------

    def block_cell(self, col: int, row: int) -> bool:
        """
        Permanently block a cell at runtime (stamp it as SHELF).

        Called by ``HazardManager`` when an unmapped obstacle is detected.
        Records the original type in ``_runtime_blocked`` so the change can
        be reversed via ``unblock_cell()``.

        Returns True if the cell was passable before and is now blocked.
        Returns False if it was already impassable.
        """
        if not self.in_bounds(col, row):
            return False
        if self.grid[row][col] not in PASSABLE_TYPES:
            return False  # already impassable
        self._runtime_blocked.add((col, row))
        self.grid[row][col] = SHELF
        return True

    def unblock_cell(self, col: int, row: int) -> bool:
        """
        Restore a previously runtime-blocked cell to WALKWAY.

        Returns True if the cell was runtime-blocked and is now restored.
        """
        if (col, row) not in self._runtime_blocked:
            return False
        self._runtime_blocked.discard((col, row))
        self.grid[row][col] = WALKWAY
        return True

    def set_inflation_cost(self, col: int, row: int, extra_cost: float) -> None:
        """
        Set (or clear) an additional traversal cost on a passable cell.

        The STA* planner calls ``move_cost()`` which reads this overlay,
        discouraging robots from cutting corners near hazards without making
        the cell fully impassable.

        Parameters
        ----------
        extra_cost : float
            Additional cost to add on top of the base 1.0.  Pass 0.0 to
            clear a previous inflation entry.
        """
        if extra_cost <= 0.0:
            self._inflation_overlay.pop((col, row), None)
        else:
            self._inflation_overlay[(col, row)] = extra_cost

    def inflation_cost(self, col: int, row: int) -> float:
        """Return the inflation overlay cost for (col, row), or 0.0."""
        return self._inflation_overlay.get((col, row), 0.0)

    def preferred_direction(
        self, col: int, row: int
    ) -> Optional[Tuple[int, int]]:
        """
        Return the preferred travel (dx, dy) for a cell, or None if no rule
        applies.
        """
        return self._cell_direction.get((row, col), None)

    # ------------------------------------------------------------------
    # Movement cost
    # ------------------------------------------------------------------

    def move_cost(
        self,
        from_col: int,
        from_row: int,
        to_col: int,
        to_row: int,
    ) -> float:
        """
        Euclidean base cost for moving from (from_col, from_row) to
        (to_col, to_row), scaled by any virtual-highway penalty.

        The penalty is applied based on the preferred direction of the
        **source** cell (the cell the robot is *leaving*), which models
        aisle direction enforcement at the point of departure.

        Returns
        -------
        float
            Travel cost >= 1.0.
        """
        dcol = to_col - from_col
        drow = to_row - from_row

        # Base cost: 1.0 for cardinal moves
        base_cost: float = 1.0

        # Look up preferred direction of the departure cell
        pref: Optional[Tuple[int, int]] = self.preferred_direction(from_col, from_row)
        if pref is not None:
            pdx, pdy = pref
            # Dot product of movement vector and preferred direction
            dot: int = dcol * pdx + drow * pdy
            if dot < 0:
                # Moving exactly against preferred direction
                base_cost = _AGAINST_PENALTY
            elif dot == 0:
                # Perpendicular to preferred direction
                base_cost = _PERPENDICULAR_PENALTY
            # dot > 0 -> aligned with preferred direction: no penalty

        # Add hazard inflation cost on the *destination* cell so the planner
        # is discouraged from routing through cells adjacent to obstacles.
        base_cost += self.inflation_cost(to_col, to_row)

        return base_cost

    # ------------------------------------------------------------------
    # Neighbour enumeration (used by A*)
    # ------------------------------------------------------------------

    def passable_neighbours(
        self, col: int, row: int
    ) -> List[Tuple[int, int, float]]:
        """
        Return all passable cardinal neighbours of (col, row) with their
        movement costs.

        Returns
        -------
        list of (to_col, to_row, cost)
        """
        results: List[Tuple[int, int, float]] = []
        for dx, dy in DIRECTION_VECTORS.values():
            nc, nr = col + dx, row + dy
            if self.is_passable(nc, nr):
                cost = self.move_cost(col, row, nc, nr)
                results.append((nc, nr, cost))
        return results

    # ------------------------------------------------------------------
    # Heuristic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def octile_distance(
        c1: int, r1: int, c2: int, r2: int
    ) -> float:
        """
        Octile distance heuristic — admissible for 4-connected grids and
        tighter than Manhattan for diagonal-free grids as well.
        For pure cardinal movement, this reduces to Chebyshev which is still
        admissible and consistent.
        """
        dx = abs(c1 - c2)
        dy = abs(r1 - r2)
        return max(dx, dy) + (2 ** 0.5 - 1) * min(dx, dy)

    @staticmethod
    def manhattan_distance(c1: int, r1: int, c2: int, r2: int) -> int:
        """Standard Manhattan distance — admissible lower bound."""
        return abs(c1 - c2) + abs(r1 - r2)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"GridMap(rows={self.rows}, cols={self.cols}, "
            f"highway_rules={len(self.highway_rules)})"
        )

    def ascii_render(self) -> str:
        """
        Return a human-readable ASCII art of the grid.

        Symbol legend:
            '.' -> Walkway (0)
            '#' -> Shelf / obstacle (1)
            'P' -> Pickup station (2)
            'D' -> Drop station (3)
            '?' -> Unknown type
        """
        symbols = {WALKWAY: ".", SHELF: "#", PICKUP: "P", DROP: "D"}
        lines: List[str] = []
        for row in self.grid:
            lines.append("".join(symbols.get(c, "?") for c in row))
        return "\n".join(lines)
