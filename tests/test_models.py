"""
Unit tests for resq_ai.core.models
"""

import pytest
from resq_ai.core.models import (
    CellType,
    VehicleType,
    VehicleStatus,
    PriorityLevel,
    HazardLevel,
    GridCell,
    DistressCall,
    Vehicle,
    OccupantType,
)


def test_grid_cell_ambulance_traversability():
    # Dry road should be traversable by ambulance
    dry_road = GridCell(x=5, y=5, cell_type=CellType.ROAD, water_level=0.0)
    assert dry_road.traversable_for(VehicleType.AMBULANCE) is True
    assert dry_road.traversable_for(VehicleType.RESCUE_BOAT) is False

    # Blocked road should not be traversable by either
    blocked_road = GridCell(x=5, y=5, cell_type=CellType.ROAD, is_blocked=True)
    assert blocked_road.traversable_for(VehicleType.AMBULANCE) is False
    assert blocked_road.traversable_for(VehicleType.RESCUE_BOAT) is False

    # Deep water (0.5m) should block ambulance, allow boat
    flooded_road = GridCell(x=5, y=5, cell_type=CellType.ROAD, water_level=0.5)
    assert flooded_road.traversable_for(VehicleType.AMBULANCE) is False
    assert flooded_road.traversable_for(VehicleType.RESCUE_BOAT) is True


def test_grid_cell_boat_traversability():
    # River should allow boat, forbid ambulance
    river_cell = GridCell(x=10, y=34, cell_type=CellType.RIVER, water_level=2.5)
    assert river_cell.traversable_for(VehicleType.AMBULANCE) is False
    assert river_cell.traversable_for(VehicleType.RESCUE_BOAT) is True

    # Bridge over river should allow ambulance if not flooded
    bridge_cell = GridCell(x=8, y=34, cell_type=CellType.BRIDGE, water_level=0.0)
    assert bridge_cell.traversable_for(VehicleType.AMBULANCE) is True
    assert bridge_cell.traversable_for(VehicleType.RESCUE_BOAT) is False


def test_travel_cost_calculation():
    highway = GridCell(x=0, y=12, cell_type=CellType.HIGHWAY, base_cost=1.0)
    normal_road = GridCell(x=0, y=0, cell_type=CellType.ROAD, base_cost=1.0)
    
    # Highway cost should be lower than normal road for ambulance
    cost_hwy = highway.travel_cost_for(VehicleType.AMBULANCE)
    cost_road = normal_road.travel_cost_for(VehicleType.AMBULANCE)
    assert cost_hwy < cost_road


def test_distress_call_creation():
    call = DistressCall(
        call_id="CALL-001",
        location=(15, 25),
        timestamp_step=1,
        caller_name="Alice",
        occupants_type=OccupantType.ELDERLY,
        occupant_count=2,
        reported_flood_m=0.8,
        fire_present=False
    )
    assert call.call_id == "CALL-001"
    assert call.is_served is False
    assert call.priority == PriorityLevel.P2
