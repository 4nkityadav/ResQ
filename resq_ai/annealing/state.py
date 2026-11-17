"""
resq_ai.annealing.state
=======================
State representation and objective evaluation for multi-vehicle rescue dispatch.
Models:
- Pickup-and-Delivery with capacity-based hospital drop-offs
- Boat-to-Road marine dock handoff points
- Dynamic casualty risk weights from Bayesian Network
- Warm-start route stability penalty (prevents en-route thrashing)
- Shift time and fuel limits
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from resq_ai.core.models import Vehicle, DistressCall, VehicleType


# Default coordinates for civil defense hospital trauma centers and boat handoff docks
DEFAULT_HOSPITALS = [(5, 5), (25, 25), (30, 45)]
DEFAULT_BOAT_DOCKS = [(20, 34), (35, 33)]


def _min_dist_to_points(from_loc: Tuple[int, int], points: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], float]:
    """Finds nearest point and returns (point, travel_minutes)."""
    best_pt = points[0]
    min_d = float("inf")
    for pt in points:
        dx = abs(from_loc[0] - pt[0])
        dy = abs(from_loc[1] - pt[1])
        d = max(dx, dy) + 0.41421356 * min(dx, dy)
        if d < min_d:
            min_d = d
            best_pt = pt
    # 0.2 km per cell at 45 km/h -> travel time in minutes
    t_min = (min_d * 0.2 / 45.0) * 60.0
    return best_pt, t_min


class ScheduleState:
    """
    Candidate solution encoding vehicle routes and unassigned emergency calls.
    """

    def __init__(
        self,
        vehicles: Dict[str, Vehicle],
        calls: Dict[str, DistressCall],
        routes: Optional[Dict[str, List[str]]] = None,
        unassigned: Optional[List[str]] = None,
        cost_matrix: Optional[Dict[Tuple[str, str], float]] = None,
        previous_routes: Optional[Dict[str, List[str]]] = None,
    ):
        self.vehicles = vehicles
        self.calls = calls
        self.cost_matrix = cost_matrix or {}
        self.previous_routes = previous_routes or {}

        # Map vehicle_id -> list of call_ids in visiting order
        if routes is not None:
            self.routes = {vid: list(r) for vid, r in routes.items()}
        else:
            self.routes = {vid: [] for vid in self.vehicles}

        # Unassigned calls
        if unassigned is not None:
            self.unassigned = list(unassigned)
        else:
            assigned_set = set(c for r in self.routes.values() for c in r)
            self.unassigned = [cid for cid in self.calls if cid not in assigned_set]

    def copy(self) -> ScheduleState:
        return ScheduleState(
            vehicles=self.vehicles,
            calls=self.calls,
            routes={vid: list(r) for vid, r in self.routes.items()},
            unassigned=list(self.unassigned),
            cost_matrix=self.cost_matrix,
            previous_routes=self.previous_routes,
        )

    def is_vehicle_compatible(self, vehicle_id: str, call_id: str) -> bool:
        """
        Hard constraint check: Vehicle-type compatibility.
        """
        v = self.vehicles[vehicle_id]
        c = self.calls[call_id]

        if v.vehicle_type == VehicleType.AMBULANCE:
            if c.reported_flood_m >= 0.35:
                return False
        elif v.vehicle_type == VehicleType.RESCUE_BOAT:
            if c.reported_flood_m < 0.20:
                return False
        return True

    def get_travel_time(self, from_id: str, to_call_id: str, vehicle_type: VehicleType) -> float:
        """
        Retrieves travel time between two entities from cost matrix or Octile estimate.
        """
        if (from_id, to_call_id) in self.cost_matrix:
            val = self.cost_matrix[(from_id, to_call_id)]
            if not math.isinf(val):
                return val

        if from_id in self.vehicles:
            from_loc = self.vehicles[from_id].current_location
            speed = self.vehicles[from_id].speed_kmh
        elif from_id in self.calls:
            from_loc = self.calls[from_id].location
            speed = 40.0
        else:
            from_loc = (0, 0)
            speed = 40.0

        to_loc = self.calls[to_call_id].location
        dx = abs(from_loc[0] - to_loc[0])
        dy = abs(from_loc[1] - to_loc[1])
        grid_dist = max(dx, dy) + 0.41421356 * min(dx, dy)
        km = grid_dist * 0.2
        return (km / max(10.0, speed)) * 60.0

    def evaluate_cost(
        self,
        alpha_travel: float = 0.40,
        capacity_penalty_weight: float = 250.0,
        unserved_penalty_weight: float = 200.0,
        shift_limit_min: float = 90.0,
        shift_penalty_weight: float = 12.0,
        stability_penalty_weight: float = 40.0,
    ) -> float:
        """
        Calculates total objective cost under realistic Pickup-and-Delivery:
        - When onboard passengers reach capacity, unit routes to hospital/dock drop-off
        - Bayesian priority risk-weighted waiting time: sum_k (W_k * T_arrival(k))
        - Warm-start stability penalty if en-route dispatch is altered
        - Soft shift / fuel limit penalties
        """
        total_cost = 0.0

        # 1. Cost of vehicle routes
        for vid, call_sequence in self.routes.items():
            if not call_sequence:
                continue

            vehicle = self.vehicles[vid]
            vtype = vehicle.vehicle_type
            curr_loc_id = vid
            curr_coords = vehicle.current_location
            cumulative_time = 0.0
            current_onboard = 0

            for cid in call_sequence:
                call = self.calls[cid]

                # Hard capacity exceedance for a single call (e.g. 4 patients for 2-capacity unit)
                if call.occupant_count > vehicle.capacity:
                    excess_single = call.occupant_count - vehicle.capacity
                    total_cost += capacity_penalty_weight * (excess_single ** 2)
                
                # Check if picking up these occupants would exceed capacity
                # If so, the vehicle must execute a hospital drop-off run first
                if (current_onboard + call.occupant_count) > vehicle.capacity:
                    # Drop off at nearest hospital or marine dock
                    if vtype == VehicleType.AMBULANCE:
                        drop_pt, drop_time = _min_dist_to_points(curr_coords, DEFAULT_HOSPITALS)
                    else:
                        drop_pt, drop_time = _min_dist_to_points(curr_coords, DEFAULT_BOAT_DOCKS)

                    cumulative_time += drop_time + 5.0
                    curr_coords = drop_pt
                    current_onboard = 0  # Patients delivered, capacity reset

                # Travel leg to emergency scene
                leg_time = self.get_travel_time(curr_loc_id, cid, vtype)
                cumulative_time += leg_time
                curr_loc_id = cid
                curr_coords = call.location
                current_onboard += call.occupant_count

                # Casualty risk-weighted waiting time: W_i * T_arrival
                risk_weight = call.risk_score
                total_cost += risk_weight * cumulative_time

            # Final return drop-off if carrying patients at end of tour
            if current_onboard > 0:
                if vtype == VehicleType.AMBULANCE:
                    _, drop_time = _min_dist_to_points(curr_coords, DEFAULT_HOSPITALS)
                else:
                    _, drop_time = _min_dist_to_points(curr_coords, DEFAULT_BOAT_DOCKS)
                cumulative_time += drop_time

            # Travel expenditure penalty
            total_cost += alpha_travel * cumulative_time

            # Shift / Fuel Limit Penalty
            if cumulative_time > shift_limit_min:
                excess_time = cumulative_time - shift_limit_min
                total_cost += shift_penalty_weight * (excess_time ** 2)

            # Warm-start route stability penalty
            if self.previous_routes and vid in self.previous_routes:
                prev_route = self.previous_routes[vid]
                if prev_route and call_sequence:
                    if prev_route[0] != call_sequence[0]:
                        total_cost += stability_penalty_weight

        # 2. Penalty for unserved calls (weighted heavily by Bayesian risk score)
        for ucid in self.unassigned:
            ucall = self.calls[ucid]
            total_cost += unserved_penalty_weight * ucall.risk_score

        return round(total_cost, 2)

    def is_valid(self) -> bool:
        """Checks if all calls are accounted for exactly once and vehicle types are valid."""
        assigned = []
        for vid, route in self.routes.items():
            for cid in route:
                assigned.append(cid)
                if not self.is_vehicle_compatible(vid, cid):
                    return False

        all_calls = set(assigned) | set(self.unassigned)
        if len(all_calls) != len(self.calls):
            return False
        if len(assigned) != len(set(assigned)):
            return False
        return True
