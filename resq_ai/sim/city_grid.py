"""
resq_ai.sim.city_grid
=====================
Procedural 50x50 urban environment generator with road networks, river, bridges,
elevation gradient, and vehicle-specific traversability modeling.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional
import numpy as np

from resq_ai.core.models import CellType, VehicleType, GridCell, ZoneStatus, HazardLevel
from resq_ai.geo.inundation import InundationModel


class CityGrid:
    """Represents the 2D urban topology for routing and hazard tracking."""

    def __init__(self, width: int = 50, height: int = 50, seed: int = 42):
        self.width = width
        self.height = height
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        self.cells: Dict[Tuple[int, int], GridCell] = {}
        self.zones: Dict[str, ZoneStatus] = {}
        self.bridge_locations: List[Tuple[int, int]] = []
        self.river_cells: List[Tuple[int, int]] = []
        
        self._initialize_zones()
        self._generate_terrain()
        self._init_inundation_engine()

    def _init_inundation_engine(self) -> None:
        """Initializes continuous elevation and water depth arrays for physics-based inundation."""
        self.elevation_grid = np.zeros((self.height, self.width), dtype=float)
        self.river_mask = np.zeros((self.height, self.width), dtype=bool)
        for (x, y), cell in self.cells.items():
            self.elevation_grid[y, x] = cell.elevation
            if cell.cell_type == CellType.RIVER:
                self.river_mask[y, x] = True
        self.inundation_model = InundationModel(
            elevation=self.elevation_grid,
            river_mask=self.river_mask,
            baseline_river_stage=2.0,
            drainage_rate=0.04
        )

    def _initialize_zones(self) -> None:
        """Configures the four major civil defense zones."""
        self.zones = {
            "zone_north": ZoneStatus(
                zone_id="zone_north",
                name="North Uptown",
                bounds=(0, 0, self.width, 15),
                base_vulnerability=0.2,
                belief_distribution={HazardLevel.SAFE: 0.90, HazardLevel.ADVISORY: 0.08, HazardLevel.CRITICAL: 0.015, HazardLevel.EVACUATE: 0.005}
            ),
            "zone_central": ZoneStatus(
                zone_id="zone_central",
                name="Central Downtown",
                bounds=(0, 15, self.width, 30),
                base_vulnerability=0.5,
                belief_distribution={HazardLevel.SAFE: 0.80, HazardLevel.ADVISORY: 0.15, HazardLevel.CRITICAL: 0.04, HazardLevel.EVACUATE: 0.01}
            ),
            "zone_riverfront": ZoneStatus(
                zone_id="zone_riverfront",
                name="Riverfront Marina",
                bounds=(0, 30, self.width, 40),
                base_vulnerability=0.85,
                belief_distribution={HazardLevel.SAFE: 0.65, HazardLevel.ADVISORY: 0.25, HazardLevel.CRITICAL: 0.08, HazardLevel.EVACUATE: 0.02}
            ),
            "zone_south": ZoneStatus(
                zone_id="zone_south",
                name="South Valley",
                bounds=(0, 40, self.width, self.height),
                base_vulnerability=0.6,
                belief_distribution={HazardLevel.SAFE: 0.75, HazardLevel.ADVISORY: 0.18, HazardLevel.CRITICAL: 0.05, HazardLevel.EVACUATE: 0.02}
            ),
        }

    def get_zone_for_coord(self, x: int, y: int) -> str:
        for zid, zone in self.zones.items():
            min_x, min_y, max_x, max_y = zone.bounds
            if min_x <= x < max_x and min_y <= y < max_y:
                return zid
        return "zone_central"

    def _generate_terrain(self) -> None:
        """Procedurally shapes roads, river, bridges, elevation, and buildings."""
        # 1. Base initialization with elevation gradient (higher in north, valley in south)
        for y in range(self.height):
            for x in range(self.width):
                # North elevation ~ 25m, South elevation ~ 4m
                elevation = 25.0 - (y / self.height) * 21.0 + self.rng.normal(0, 0.5)
                zone_id = self.get_zone_for_coord(x, y)
                self.cells[(x, y)] = GridCell(
                    x=x,
                    y=y,
                    cell_type=CellType.ROAD,
                    base_cost=1.0,
                    elevation=round(elevation, 2),
                    water_level=0.0,
                    is_blocked=False,
                    zone_id=zone_id
                )

        # 2. Carve a meandering river across y ~ 32 to 36
        river_center_y = 34
        for x in range(self.width):
            # Sine wave meander
            meander = int(2.5 * math.sin(x * 0.18))
            center = river_center_y + meander
            for dy in (-1, 0, 1):
                ry = center + dy
                if 0 <= ry < self.height:
                    cell = self.cells[(x, ry)]
                    cell.cell_type = CellType.RIVER
                    cell.water_level = 2.5
                    cell.elevation = min(cell.elevation, 3.0)
                    self.river_cells.append((x, ry))

        # 3. Construct 4 Bridges across the river (evenly spaced)
        bridge_xs = [8, 20, 32, 44]
        for bx in bridge_xs:
            # Find river cells at column bx
            for by in range(river_center_y - 4, river_center_y + 5):
                if (bx, by) in self.cells and self.cells[(bx, by)].cell_type == CellType.RIVER:
                    cell = self.cells[(bx, by)]
                    cell.cell_type = CellType.BRIDGE
                    cell.water_level = 0.0  # Bridges stay dry initially
                    self.bridge_locations.append((bx, by))

        # 4. Lay Highway and Arterial Roads
        # Highway horizontal across y=12, y=42
        for x in range(self.width):
            for hy in (12, 42):
                if (x, hy) in self.cells and self.cells[(x, hy)].cell_type != CellType.RIVER:
                    self.cells[(x, hy)].cell_type = CellType.HIGHWAY
                    self.cells[(x, hy)].base_cost = 0.6

        # Highway vertical across x=25 (excluding non-bridge river)
        for y in range(self.height):
            if (25, y) in self.cells and self.cells[(25, y)].cell_type != CellType.RIVER:
                if self.cells[(25, y)].cell_type != CellType.BRIDGE:
                    self.cells[(25, y)].cell_type = CellType.HIGHWAY
                    self.cells[(25, y)].base_cost = 0.6

        # Arterial grid every 10 cells
        for y in range(self.height):
            for x in range(self.width):
                if (x % 10 == 0 or y % 10 == 0) and self.cells[(x, y)].cell_type == CellType.ROAD:
                    self.cells[(x, y)].cell_type = CellType.ARTERIAL
                    self.cells[(x, y)].base_cost = 0.8

        # 5. Fill residential/commercial buildings in pockets between roads
        for y in range(self.height):
            for x in range(self.width):
                cell = self.cells[(x, y)]
                if cell.cell_type == CellType.ROAD and (x % 5 in (2, 3)) and (y % 5 in (2, 3)):
                    # 75% chance building, 25% park
                    if self.rng.random() < 0.75:
                        cell.cell_type = CellType.BUILDING
                        cell.base_cost = 999.0
                    else:
                        cell.cell_type = CellType.PARK
                        cell.base_cost = 1.4

    def get_cell(self, x: int, y: int) -> Optional[GridCell]:
        return self.cells.get((x, y))

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def get_neighbors(self, coord: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Returns 8-way neighbors within grid bounds."""
        x, y = coord
        neighbors: List[Tuple[int, int]] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.in_bounds(nx, ny):
                    neighbors.append((nx, ny))
        return neighbors

    def get_step_cost(self, u: Tuple[int, int], v: Tuple[int, int], vtype: VehicleType) -> float:
        """Calculates directed travel cost from cell u to neighbor v for given vehicle type."""
        target_cell = self.get_cell(v[0], v[1])
        if target_cell is None or not target_cell.traversable_for(vtype):
            return float("inf")

        # Base cost of target
        base_cost = target_cell.travel_cost_for(vtype)
        if math.isinf(base_cost):
            return float("inf")

        # Diagonal movement factor with strict corner-cutting prevention
        dx = abs(u[0] - v[0])
        dy = abs(u[1] - v[1])
        if dx == 1 and dy == 1:
            # Diagonal transit: both adjacent orthogonal cells must be traversable
            adj1 = self.get_cell(u[0], v[1])
            adj2 = self.get_cell(v[0], u[1])
            if adj1 is None or not adj1.traversable_for(vtype):
                return float("inf")
            if adj2 is None or not adj2.traversable_for(vtype):
                return float("inf")
            distance_mult = 1.41421356
        else:
            distance_mult = 1.0

        return base_cost * distance_mult

    def set_water_level(self, x: int, y: int, level: float) -> None:
        cell = self.get_cell(x, y)
        if cell:
            cell.water_level = max(0.0, round(level, 3))

    def set_blocked(self, x: int, y: int, is_blocked: bool) -> None:
        cell = self.get_cell(x, y)
        if cell:
            cell.is_blocked = is_blocked

    def step_inundation(
        self,
        dt_minutes: float = 5.0,
        rain_rate_mm_hr: float = 0.0,
        river_stage: Optional[float] = None,
        breach_locations: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Tuple[int, int]]:
        """
        Executes a physics-inspired flood step and updates internal GridCells.
        Returns list of cell coordinates whose flood depth changed significantly.
        """
        if river_stage is not None:
            self.inundation_model.set_river_stage(river_stage)

        depth_field = self.inundation_model.step(
            dt_minutes=dt_minutes,
            rain_rate_mm_hr=rain_rate_mm_hr,
            breach_locations=breach_locations,
        )

        changed_coords: List[Tuple[int, int]] = []
        for y in range(self.height):
            for x in range(self.width):
                cell = self.cells.get((x, y))
                if cell and cell.cell_type != CellType.BRIDGE:
                    new_depth = float(depth_field[y, x])
                    if abs(cell.water_level - new_depth) > 0.05:
                        cell.water_level = round(new_depth, 3)
                        changed_coords.append((x, y))
        return changed_coords

    def count_flooded_cells(self) -> int:
        return sum(1 for c in self.cells.values() if c.is_flooded)

    def count_blocked_cells(self) -> int:
        return sum(1 for c in self.cells.values() if c.is_blocked)
