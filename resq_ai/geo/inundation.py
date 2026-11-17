"""
resq_ai.geo.inundation
======================
Physics-inspired hydrologic inundation engine.
Computes time-varying 2D flood depth fields using:
- Digital Elevation Model (DEM)
- River stage dynamics with overtopping and breach mechanics
- Topographically connected hydraulic diffusion (priority-flow bathtub)
- Vehicle-specific passability gating and speed degradation factors
"""

from __future__ import annotations
from typing import Tuple, Optional, Dict
import numpy as np
from scipy.ndimage import gaussian_filter


class InundationModel:
    """Physics-inspired hydrologic inundation model over 2D terrain."""

    AMBULANCE_MAX_DEPTH_M: float = 0.35
    BOAT_MIN_DRAFT_M: float = 0.20
    HEAVY_RESCUE_MAX_DEPTH_M: float = 0.60

    def __init__(
        self,
        elevation: np.ndarray,
        river_mask: np.ndarray,
        baseline_river_stage: float = 2.0,
        drainage_rate: float = 0.05,
    ):
        """
        Args:
            elevation: 2D float array of terrain elevation in meters Z(x, y).
            river_mask: 2D boolean array where True indicates the river channel.
            baseline_river_stage: Normal river water surface elevation in meters.
            drainage_rate: Water infiltration/drainage rate in meters per timestep.
        """
        self.elevation = np.asarray(elevation, dtype=float)
        self.height, self.width = self.elevation.shape
        self.river_mask = np.asarray(river_mask, dtype=bool)
        self.baseline_stage = float(baseline_river_stage)
        self.current_stage = float(baseline_river_stage)
        self.drainage_rate = float(drainage_rate)

        # Inundation depth in meters above ground level D(x, y)
        self.depth_field = np.zeros((self.height, self.width), dtype=float)
        # Initialize river depth
        river_depth = np.maximum(0.0, self.baseline_stage - self.elevation)
        self.depth_field[self.river_mask] = np.maximum(river_depth[self.river_mask], 1.5)

    def set_river_stage(self, stage: float) -> None:
        """Sets the river stage water elevation in meters."""
        self.current_stage = float(stage)

    def step(
        self,
        dt_minutes: float = 5.0,
        rain_rate_mm_hr: float = 0.0,
        breach_locations: Optional[list[Tuple[int, int]]] = None,
    ) -> np.ndarray:
        """
        Advances the hydrologic simulation by dt_minutes.
        Uses topographically constrained hydraulic diffusion from river and breach points.

        Returns:
            Updated 2D water depth array in meters.
        """
        # 1. Update river channel water elevation
        river_water_elev = self.current_stage
        
        # 2. Add local rainfall runoff: rain_mm / 1000 = meters of water
        rain_m = (rain_rate_mm_hr / 1000.0) * (dt_minutes / 60.0)
        if rain_m > 0:
            self.depth_field += rain_m

        # 3. Topographically connected flood propagation from river and breaches
        # Water surface elevation W = Z + D
        water_surface = self.elevation + self.depth_field
        
        # River cells are pinned to at least the river stage
        water_surface[self.river_mask] = np.maximum(
            water_surface[self.river_mask],
            river_water_elev
        )

        # Priority propagation using queue/stencil diffusion
        # We perform iterative local hydraulic leveling
        # Water flows from higher water surface to lower water surface if elevation allows
        diffused_surface = water_surface.copy()
        
        # 4-connected neighbors kernel
        for _ in range(3):  # 3 relaxation iterations per timestep
            n_up = np.roll(diffused_surface, -1, axis=0)
            n_down = np.roll(diffused_surface, 1, axis=0)
            n_left = np.roll(diffused_surface, -1, axis=1)
            n_right = np.roll(diffused_surface, 1, axis=1)

            # Prevent wrap-around boundaries
            n_up[-1, :] = diffused_surface[-1, :]
            n_down[0, :] = diffused_surface[0, :]
            n_left[:, -1] = diffused_surface[:, -1]
            n_right[:, 0] = diffused_surface[:, 0]

            # Water flows outward if adjacent water head exceeds terrain elevation
            can_receive_up = n_up > self.elevation
            can_receive_down = n_down > self.elevation
            can_receive_left = n_left > self.elevation
            can_receive_right = n_right > self.elevation

            valid_heads = np.stack([
                np.where(can_receive_up, n_up, self.elevation),
                np.where(can_receive_down, n_down, self.elevation),
                np.where(can_receive_left, n_left, self.elevation),
                np.where(can_receive_right, n_right, self.elevation),
                diffused_surface
            ], axis=0)

            # Max connected water head with slight hydraulic friction slope
            max_head = np.max(valid_heads, axis=0) - 0.02
            diffused_surface = np.maximum(diffused_surface, max_head)

        # Re-pin river
        diffused_surface[self.river_mask] = np.maximum(
            diffused_surface[self.river_mask],
            river_water_elev
        )

        # Handle specific levee breach points (inject localized water head)
        if breach_locations:
            for bx, by in breach_locations:
                if 0 <= by < self.height and 0 <= bx < self.width:
                    diffused_surface[by, bx] = max(
                        diffused_surface[by, bx],
                        self.elevation[by, bx] + 1.2
                    )

        # 4. Compute resulting depth field
        new_depth = np.maximum(0.0, diffused_surface - self.elevation)
        
        # Apply natural soil drainage/percolation to non-river cells
        drain = self.drainage_rate * (dt_minutes / 60.0)
        non_river = ~self.river_mask
        new_depth[non_river] = np.maximum(0.0, new_depth[non_river] - drain)

        # Smooth slightly to simulate physical viscous fluid behavior
        self.depth_field = gaussian_filter(new_depth, sigma=0.4)
        # Preserve river minimum
        self.depth_field[self.river_mask] = np.maximum(self.depth_field[self.river_mask], 1.2)
        
        return self.depth_field

    def get_water_depth(self, x: int, y: int) -> float:
        """Returns water depth at (x, y) in meters."""
        if 0 <= y < self.height and 0 <= x < self.width:
            return float(self.depth_field[y, x])
        return 0.0

    def is_passable(self, x: int, y: int, vehicle_type_str: str) -> bool:
        """
        Determines whether cell (x, y) is physically passable for vehicle_type.
        """
        depth = self.get_water_depth(x, y)
        is_river = self.river_mask[y, x] if (0 <= y < self.height and 0 <= x < self.width) else False

        vtype = vehicle_type_str.lower()
        if "boat" in vtype:
            # Boats can navigate in river or any flooded street with depth >= 0.20m
            return bool(is_river or (depth >= self.BOAT_MIN_DRAFT_M))
        
        if "ambulance" in vtype:
            # Ambulances cannot enter river unless on bridge (handled by road net)
            # and cannot traverse water deeper than 0.35m
            if is_river:
                return False
            return bool(depth <= self.AMBULANCE_MAX_DEPTH_M)

        if "heavy" in vtype or "fire" in vtype:
            if is_river:
                return False
            return bool(depth <= self.HEAVY_RESCUE_MAX_DEPTH_M)

        return bool(depth <= self.AMBULANCE_MAX_DEPTH_M)

    def get_speed_multiplier(self, x: int, y: int, vehicle_type_str: str) -> float:
        """
        Returns speed degradation multiplier [0.0 to 1.0] based on water depth.
        """
        depth = self.get_water_depth(x, y)
        vtype = vehicle_type_str.lower()

        if "boat" in vtype:
            if depth < self.BOAT_MIN_DRAFT_M:
                return 0.0
            return 1.0  # Full speed on water

        # Road vehicle
        if depth > self.AMBULANCE_MAX_DEPTH_M:
            return 0.0
        # Linear degradation: at depth 0 -> 1.0; at max depth -> 0.15
        factor = 1.0 - (depth / self.AMBULANCE_MAX_DEPTH_M) * 0.85
        return float(np.clip(factor, 0.15, 1.0))
