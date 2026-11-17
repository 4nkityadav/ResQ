"""
resq_ai.core.world_state
========================
Unified, canonical WorldState and event-driven data flow core.
Single source of truth for simulation clock, spatial road topology,
environmental hazard fields, heterogeneous fleet status, hospitals,
boat-to-road handoff points, and active casualty registries.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
import copy
import time

from resq_ai.core.models import (
    GridCell,
    Vehicle,
    VehicleType,
    VehicleStatus,
    DistressCall,
    PriorityLevel,
    HazardLevel,
    SensorObservation,
    ZoneStatus,
    EventType,
    SimEvent,
)
from resq_ai.sim.city_grid import CityGrid


@dataclass
class Hospital:
    """Medical center / drop-off depot for patient care."""
    hospital_id: str
    name: str
    location: Tuple[int, int]
    capacity: int = 50
    current_patients: int = 0


@dataclass
class BoatHandoffPoint:
    """Designated dock/ramp where boats transfer rescued victims to ambulances."""
    handoff_id: str
    name: str
    location: Tuple[int, int]
    dock_capacity: int = 4
    current_waiting_victims: int = 0


@dataclass
class WorldState:
    """
    Canonical, centralized world model that all algorithmic modules read and write.
    No module is permitted to maintain an uncoordinated private copy of grid or fleet.
    """
    step: int = 0
    time_minutes: float = 0.0
    grid: CityGrid = field(default_factory=lambda: CityGrid(width=50, height=50, seed=42))
    vehicles: Dict[str, Vehicle] = field(default_factory=dict)
    distress_calls: Dict[str, DistressCall] = field(default_factory=dict)
    hospitals: Dict[str, Hospital] = field(default_factory=dict)
    handoff_points: Dict[str, BoatHandoffPoint] = field(default_factory=dict)
    zone_hazards: Dict[str, ZoneStatus] = field(default_factory=dict)
    routes: Dict[str, List[str]] = field(default_factory=dict)
    event_history: List[SimEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def initialize_default_infrastructure(self) -> None:
        """Sets up designated hospitals and boat-to-road handoff points."""
        # Primary Municipal Hospital in North Central
        self.hospitals["HOSP-01"] = Hospital(
            hospital_id="HOSP-01",
            name="General Civil Defense Hospital",
            location=(5, 5),
            capacity=100,
        )
        # Secondary Emergency Trauma Center in Uptown
        self.hospitals["HOSP-02"] = Hospital(
            hospital_id="HOSP-02",
            name="Uptown Trauma Clinic",
            location=(45, 5),
            capacity=50,
        )

        # Designated Boat-to-Ambulance Handoff Docks along the riverfront
        # These are cells accessible to both boats (water/ramp) and ambulances (road access)
        self.handoff_points["DOCK-01"] = BoatHandoffPoint(
            handoff_id="DOCK-01",
            name="West Marina Transfer Slip",
            location=(8, 33),
        )
        self.handoff_points["DOCK-02"] = BoatHandoffPoint(
            handoff_id="DOCK-02",
            name="Central Pier Handoff Station",
            location=(25, 33),
        )
        self.handoff_points["DOCK-03"] = BoatHandoffPoint(
            handoff_id="DOCK-03",
            name="East River Ramp Station",
            location=(44, 33),
        )

    def get_unserved_calls(self) -> List[DistressCall]:
        return [c for c in self.distress_calls.values() if not c.is_served]

    def get_served_calls(self) -> List[DistressCall]:
        return [c for c in self.distress_calls.values() if c.is_served]

    def get_p1_calls(self) -> List[DistressCall]:
        return [c for c in self.distress_calls.values() if c.priority == PriorityLevel.P1 and not c.is_served]

    def get_cell(self, x: int, y: int) -> Optional[GridCell]:
        return self.grid.get_cell(x, y)

    def get_vehicle(self, vid: str) -> Optional[Vehicle]:
        return self.vehicles.get(vid)

    def record_event(self, event: SimEvent) -> None:
        self.event_history.append(event)

    def snapshot(self) -> Dict[str, Any]:
        """Produces a JSON-serializable snapshot of the current world state."""
        return {
            "step": self.step,
            "time_minutes": self.time_minutes,
            "grid_dimensions": (self.grid.width, self.grid.height),
            "flooded_cells_count": self.grid.count_flooded_cells(),
            "blocked_cells_count": self.grid.count_blocked_cells(),
            "vehicles": {
                vid: {
                    "type": v.vehicle_type.value,
                    "loc": v.current_location,
                    "status": v.status.value,
                    "fuel_hours": round(v.fuel_hours_remaining, 2),
                    "total_served": v.total_calls_served,
                    "current_load": v.current_load,
                    "assigned_calls": list(v.assigned_call_ids),
                    "path_len": len(v.current_path),
                }
                for vid, v in self.vehicles.items()
            },
            "unserved_calls_count": len(self.get_unserved_calls()),
            "p1_calls_count": len(self.get_p1_calls()),
            "hospitals": {
                hid: {"name": h.name, "loc": h.location, "patients": h.current_patients}
                for hid, h in self.hospitals.items()
            },
            "handoff_points": {
                hid: {"name": d.name, "loc": d.location, "waiting": d.current_waiting_victims}
                for hid, d in self.handoff_points.items()
            },
            "zone_hazards": {
                zid: {
                    "map_state": z.current_hazard_state.value,
                    "alert": z.alert_triggered,
                    "p_crit_evac": round(float(z.belief_distribution.get(HazardLevel.CRITICAL, 0.0) + z.belief_distribution.get(HazardLevel.EVACUATE, 0.0)), 4)
                }
                for zid, z in self.zone_hazards.items()
            },
            "routes": {vid: list(r) for vid, r in self.routes.items()},
        }
