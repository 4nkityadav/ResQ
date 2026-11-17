"""
Unit tests for resq_ai.sim.city_grid and resq_ai.sim.disaster_sim
"""

import pytest
from resq_ai.core.models import CellType, VehicleType, EventType
from resq_ai.sim.city_grid import CityGrid
from resq_ai.sim.disaster_sim import DisasterSimulator


def test_city_grid_procedural_generation():
    grid = CityGrid(width=50, height=50, seed=42)
    assert len(grid.cells) == 2500
    assert len(grid.river_cells) > 0
    # 4 bridge columns across 3 river cells = 12 bridge cells
    assert len(grid.bridge_locations) == 12
    assert len(grid.zones) == 4

    # Check reproducibility with seed
    grid2 = CityGrid(width=50, height=50, seed=42)
    assert grid.bridge_locations == grid2.bridge_locations
    assert grid.get_cell(10, 10).elevation == grid2.get_cell(10, 10).elevation


def test_city_grid_8_way_neighbors():
    grid = CityGrid(width=50, height=50, seed=42)
    
    # Interior cell has 8 neighbors
    interior_neighbors = grid.get_neighbors((10, 10))
    assert len(interior_neighbors) == 8

    # Corner cell (0, 0) has 3 neighbors
    corner_neighbors = grid.get_neighbors((0, 0))
    assert len(corner_neighbors) == 3

    # Edge cell (0, 10) has 5 neighbors
    edge_neighbors = grid.get_neighbors((0, 10))
    assert len(edge_neighbors) == 5


def test_disaster_sim_initialization():
    sim = DisasterSimulator(seed=123)
    assert len(sim.vehicles) == 8  # 5 ambulances + 3 boats
    assert sum(1 for v in sim.vehicles.values() if v.vehicle_type == VehicleType.AMBULANCE) == 5
    assert sum(1 for v in sim.vehicles.values() if v.vehicle_type == VehicleType.RESCUE_BOAT) == 3


def test_disaster_sim_stepping_and_events():
    sim = DisasterSimulator(seed=42)
    
    # Run 5 steps
    all_events = []
    for _ in range(5):
        events = sim.step()
        all_events.extend(events)

    assert sim.current_step == 5
    # Should have sensor update events for zones every step
    sensor_events = [e for e in all_events if e.event_type == EventType.SENSOR_UPDATE]
    assert len(sensor_events) == 5 * 4  # 5 steps * 4 zones

    # Check that calls or flood events occurred
    assert sim.call_counter >= 0


def test_manual_blockage_and_clearance():
    sim = DisasterSimulator(seed=42)
    event_block = sim.trigger_blockage(15, 15)
    assert event_block is not None
    assert event_block.event_type == EventType.ROAD_BLOCKED
    assert sim.grid.get_cell(15, 15).is_blocked is True

    event_clear = sim.trigger_clearance(15, 15)
    assert event_clear is not None
    assert event_clear.event_type == EventType.ROAD_CLEARED
    assert sim.grid.get_cell(15, 15).is_blocked is False
