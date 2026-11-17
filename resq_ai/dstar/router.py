"""
resq_ai.dstar.router
====================
Fleet routing and ETA computation engine.
Maintains active D* Lite instances for en-route vehicles, computes the pairwise
travel-cost matrix for Simulated Annealing, and calculates route ETA deltas.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional, Any

from resq_ai.core.models import Vehicle, DistressCall, VehicleType
from resq_ai.sim.city_grid import CityGrid
from resq_ai.dstar.dstar_lite import DStarLite
from resq_ai.dstar.astar import AStar


Coord = Tuple[int, int]


class FleetRouter:
    """Manages multi-vehicle navigation and dynamic replanning across the fleet."""

    def __init__(self, grid: CityGrid):
        self.grid = grid
        # Active D* Lite instances keyed by vehicle_id
        self.active_dstar: Dict[str, DStarLite] = {}
        # Backup A* searcher for fast one-off matrix distance estimates
        self.astar_ambulance = AStar(grid, VehicleType.AMBULANCE)
        self.astar_boat = AStar(grid, VehicleType.RESCUE_BOAT)

    def get_route(
        self,
        vehicle: Vehicle,
        destination: Coord,
    ) -> Tuple[List[Coord], float, str]:
        """
        Plans or replans path for a vehicle to a destination.
        Returns: (path, cost_minutes, route_summary)
        """
        vid = vehicle.vehicle_id
        vtype = vehicle.vehicle_type
        start = vehicle.current_location

        # Create or reuse D* Lite instance
        dstar = DStarLite(self.grid, start=start, goal=destination, vehicle_type=vtype)
        path, cost, _ = dstar.plan()
        self.active_dstar[vid] = dstar

        # Convert grid cost units to minutes (assuming ~1 unit = ~0.2 km at speed km/h)
        # travel_time_hours = (cost * 0.2 km) / speed_kmh
        travel_time_min = (cost * 0.2 / max(10.0, vehicle.speed_kmh)) * 60.0
        travel_time_min = round(travel_time_min, 1)

        summary = f"{vid} ({vtype.value}) route to {destination}: {len(path)} steps (~{travel_time_min} min)"
        return path, travel_time_min, summary

    def handle_obstacle_update(
        self,
        blocked_coord: Coord,
    ) -> List[Dict[str, Any]]:
        """
        Notifies all active D* Lite vehicle instances of a blocked or flooded cell.
        Replans locally if the vehicle's active path passes through or near the cell.
        Returns ETA deltas for affected vehicles.
        """
        replan_reports: List[Dict[str, Any]] = []

        for vid, dstar in list(self.active_dstar.items()):
            # Notify cell change
            dstar.notify_cell_changed(blocked_coord)
            
            # Check if previous path crossed this cell
            # Replan
            old_cost = dstar.get_g(dstar.s_start)
            new_path, new_cost, reexpanded = dstar.replan()
            
            if math.isinf(new_cost):
                replan_reports.append({
                    "vehicle_id": vid,
                    "rerouted": True,
                    "delta_min": float("inf"),
                    "summary": f"{vid} path completely blocked! No alternative traversable route found.",
                    "reexpanded_count": len(reexpanded),
                    "new_path": [],
                })
            else:
                cost_diff = new_cost - old_cost
                delta_min = round((cost_diff * 0.2 / 40.0) * 60.0, 1)
                delta_str = f"+{delta_min} min" if delta_min >= 0 else f"{delta_min} min"
                
                replan_reports.append({
                    "vehicle_id": vid,
                    "rerouted": True,
                    "delta_min": delta_min,
                    "summary": f"{vid} rerouted ({delta_str}, {len(reexpanded)} cells repaired locally)",
                    "reexpanded_count": len(reexpanded),
                    "new_path": new_path,
                })

        return replan_reports

    def compute_cost_matrix(
        self,
        vehicles: List[Vehicle],
        calls: List[DistressCall],
    ) -> Dict[Tuple[str, str], float]:
        """
        Computes pairwise travel-time cost matrix between vehicles and distress locations.
        Cost is in minutes. Incompatible vehicle-location pairs return infinity.
        """
        cost_matrix: Dict[Tuple[str, str], float] = {}

        for v in vehicles:
            searcher = self.astar_boat if v.vehicle_type == VehicleType.RESCUE_BOAT else self.astar_ambulance
            vloc = v.current_location

            for c in calls:
                cloc = c.location
                target_cell = self.grid.get_cell(cloc[0], cloc[1])

                # Check basic compatibility before running search
                if target_cell and not target_cell.traversable_for(v.vehicle_type):
                    cost_matrix[(v.vehicle_id, c.call_id)] = float("inf")
                    continue

                _, path_cost, _ = searcher.search(vloc, cloc)
                if math.isinf(path_cost):
                    cost_matrix[(v.vehicle_id, c.call_id)] = float("inf")
                else:
                    time_min = (path_cost * 0.2 / max(10.0, v.speed_kmh)) * 60.0
                    cost_matrix[(v.vehicle_id, c.call_id)] = round(time_min, 2)

        return cost_matrix
