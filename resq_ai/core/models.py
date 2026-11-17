"""
resq_ai.core.models
===================
Data models, enums, and state representations for ResQ-AI.
Strictly typed dataclasses for clear state transitions and serialization.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional
import time


class CellType(str, Enum):
    ROAD = "road"
    ARTERIAL = "arterial"
    HIGHWAY = "highway"
    BRIDGE = "bridge"
    RIVER = "river"
    BUILDING = "building"
    PARK = "park"


class VehicleType(str, Enum):
    AMBULANCE = "ambulance"
    RESCUE_BOAT = "boat"


class VehicleStatus(str, Enum):
    IDLE = "idle"
    EN_ROUTE_TO_SCENE = "en_route_to_scene"
    ON_SCENE = "on_scene"
    EN_ROUTE_TO_BASE = "en_route_to_base"


class PriorityLevel(str, Enum):
    P1 = "P1"  # Immediate life threat (<15 min target response)
    P2 = "P2"  # Serious / urgent (<45 min target)
    P3 = "P3"  # Non-urgent / Delayed evacuation (<120 min target)


class HazardLevel(str, Enum):
    SAFE = "Safe"
    ADVISORY = "Advisory"
    CRITICAL = "Critical"
    EVACUATE = "Evacuate"


class OccupantType(str, Enum):
    ELDERLY = "Elderly"
    CHILDREN = "Children"
    ADULTS = "Adults"


class EventType(str, Enum):
    NEW_CALL = "NEW_CALL"
    SENSOR_UPDATE = "SENSOR_UPDATE"
    ROAD_BLOCKED = "ROAD_BLOCKED"
    ROAD_CLEARED = "ROAD_CLEARED"
    VEHICLE_FREE = "VEHICLE_FREE"
    DISPATCH_COMMAND = "DISPATCH_COMMAND"


@dataclass
class GridCell:
    """Represents a discrete coordinate cell on the city map."""
    x: int
    y: int
    cell_type: CellType = CellType.ROAD
    base_cost: float = 1.0
    elevation: float = 10.0  # Meters above sea level
    water_level: float = 0.0  # In meters
    is_blocked: bool = False
    zone_id: str = "zone_central"

    @property
    def coord(self) -> Tuple[int, int]:
        return (self.x, self.y)

    @property
    def is_flooded(self) -> bool:
        return self.water_level >= 0.20

    def traversable_for(self, vtype: VehicleType) -> bool:
        """Determines if this cell is passable by the given vehicle type."""
        if self.is_blocked:
            return False
        if vtype == VehicleType.AMBULANCE:
            # Cannot cross water deeper than 0.35m or river cells
            if self.cell_type == CellType.RIVER:
                return False
            if self.water_level >= 0.35:
                return False
            return self.cell_type in (
                CellType.ROAD,
                CellType.ARTERIAL,
                CellType.HIGHWAY,
                CellType.BRIDGE,
                CellType.PARK,
            )
        elif vtype == VehicleType.RESCUE_BOAT:
            # Boat can only traverse river or flooded cells with water >= 0.20m
            if self.cell_type == CellType.RIVER:
                return True
            return self.water_level >= 0.20
        return False

    def travel_cost_for(self, vtype: VehicleType) -> float:
        """Calculates traversal time cost for a vehicle traversing this cell."""
        if not self.traversable_for(vtype):
            return float("inf")
        # Speed modifier by road type
        speed_mod = {
            CellType.HIGHWAY: 0.5,
            CellType.ARTERIAL: 0.75,
            CellType.ROAD: 1.0,
            CellType.BRIDGE: 1.0,
            CellType.PARK: 1.5,
            CellType.RIVER: 1.2,
            CellType.BUILDING: 999.0,
        }.get(self.cell_type, 1.0)

        # High water resistance adds cost for ambulance
        water_penalty = (self.water_level * 2.0) if vtype == VehicleType.AMBULANCE else 0.0
        return max(0.2, self.base_cost * speed_mod + water_penalty)


@dataclass
class DistressCall:
    """Incoming emergency rescue or evacuation call."""
    call_id: str
    location: Tuple[int, int]
    timestamp_step: int
    caller_name: str
    occupants_type: OccupantType = OccupantType.ADULTS
    occupant_count: int = 1
    building_age: str = "Old"       # "Old", "Moderate", "New"
    construction_type: str = "Masonry"  # "Wood", "Masonry", "ReinforcedConcrete"
    reported_flood_m: float = 0.5
    fire_present: bool = False
    
    # Inferred by Bayesian Network
    priority: PriorityLevel = PriorityLevel.P2
    p_collapse: float = 0.50
    p_hypothermia: float = 0.50
    risk_score: float = 5.0
    explanation: str = "Pending evaluation"
    
    # Dispatch lifecycle
    is_served: bool = False
    assigned_vehicle_id: Optional[str] = None
    served_timestamp_step: Optional[int] = None
    waiting_time_steps: int = 0


@dataclass
class Vehicle:
    """Rescue vehicle (Ambulance or Boat)."""
    vehicle_id: str
    vehicle_type: VehicleType
    capacity: int
    speed_kmh: float
    base_location: Tuple[int, int]
    current_location: Tuple[int, int]
    current_load: int = 0
    assigned_call_ids: List[str] = field(default_factory=list)
    current_path: List[Tuple[int, int]] = field(default_factory=list)
    status: VehicleStatus = VehicleStatus.IDLE
    fuel_hours_remaining: float = 8.0
    total_calls_served: int = 0
    total_distance_traveled: float = 0.0


@dataclass
class SensorObservation:
    """Continuous sensor readings for a zone at a specific timestep."""
    step: int
    zone_id: str
    water_level_m: float
    rainfall_mmh: float
    wind_speed_kmh: float

    def to_vector(self) -> List[float]:
        return [self.water_level_m, self.rainfall_mmh, self.wind_speed_kmh]


@dataclass
class ZoneStatus:
    """Tracks current real-time state and HMM belief for a geographical zone."""
    zone_id: str
    name: str
    bounds: Tuple[int, int, int, int]  # (min_x, min_y, max_x, max_y)
    base_vulnerability: float
    current_hazard_state: HazardLevel = HazardLevel.SAFE
    belief_distribution: Dict[HazardLevel, float] = field(
        default_factory=lambda: {
            HazardLevel.SAFE: 0.85,
            HazardLevel.ADVISORY: 0.10,
            HazardLevel.CRITICAL: 0.04,
            HazardLevel.EVACUATE: 0.01,
        }
    )
    history_observations: List[SensorObservation] = field(default_factory=list)
    alert_triggered: bool = False


@dataclass
class SimEvent:
    """Discrete simulation event in the event-driven orchestrator."""
    step: int
    event_type: EventType
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
