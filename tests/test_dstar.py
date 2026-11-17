"""
Unit tests for resq_ai.dstar (IndexedPriorityQueue, DStarLite, AStar, FleetRouter).
Verifies path optimality against A*, local vertex repair, and vehicle traversability.
"""

import pytest
import math
import numpy as np

from resq_ai.core.models import VehicleType, Vehicle, DistressCall, OccupantType
from resq_ai.sim.city_grid import CityGrid
from resq_ai.dstar.priority_queue import IndexedPriorityQueue
from resq_ai.dstar.dstar_lite import DStarLite
from resq_ai.dstar.astar import AStar
from resq_ai.dstar.router import FleetRouter


def test_indexed_priority_queue():
    pq = IndexedPriorityQueue()
    assert pq.is_empty()
    assert pq.top_key() == (float("inf"), float("inf"))

    pq.insert("node_A", (5.0, 1.0))
    pq.insert("node_B", (2.0, 4.0))
    pq.insert("node_C", (2.0, 2.0))

    assert len(pq) == 3
    assert pq.contains("node_B")
    assert pq.top_key() == (2.0, 2.0)  # node_C is smaller lexicographically

    # Update key of node_A to be lowest
    pq.update("node_A", (1.0, 0.5))
    assert pq.top_key() == (1.0, 0.5)

    item, key = pq.pop_min()
    assert item == "node_A"
    assert key == (1.0, 0.5)

    # Remove node_B
    pq.remove("node_B")
    assert not pq.contains("node_B")
    assert len(pq) == 1

    item2, _ = pq.pop_min()
    assert item2 == "node_C"
    assert pq.is_empty()


def test_dstar_optimality_matches_astar():
    grid = CityGrid(width=50, height=50, seed=42)
    start = (5, 5)
    goal = (20, 20)
    vtype = VehicleType.AMBULANCE

    astar = AStar(grid, vehicle_type=vtype)
    path_a, cost_a, exp_a = astar.search(start, goal)

    dstar = DStarLite(grid, start=start, goal=goal, vehicle_type=vtype)
    path_d, cost_d, exp_d = dstar.plan()

    assert len(path_d) > 0
    assert len(path_a) > 0
    assert path_d[0] == start
    assert path_d[-1] == goal
    # Path costs should match within numerical margin
    assert np.isclose(cost_a, cost_d, atol=0.2)


def test_dstar_dynamic_local_replanning():
    grid = CityGrid(width=50, height=50, seed=42)
    start = (10, 10)
    goal = (30, 10)
    vtype = VehicleType.AMBULANCE

    dstar = DStarLite(grid, start=start, goal=goal, vehicle_type=vtype)
    path1, cost1, _ = dstar.plan()
    assert len(path1) > 0

    # Choose a cell along the middle of the path and block it
    obstacle_cell = path1[len(path1) // 2]
    grid.set_blocked(obstacle_cell[0], obstacle_cell[1], True)

    # Notify D* Lite and replan
    dstar.notify_cell_changed(obstacle_cell)
    path2, cost2, reexpanded = dstar.replan()

    assert len(path2) > 0
    # Blocked cell must NOT be in the newly replanned path
    assert obstacle_cell not in path2
    # Rerouted path cost should be >= original cost
    assert cost2 >= cost1
    # Local repair should only re-expand a fraction of the 2,500 grid cells
    assert len(reexpanded) < 1500


def test_dstar_vs_astar_randomized_obstacle_invariance():
    """Property test: D* Lite post-repair cost must strictly match cold-start A*."""
    rng = np.random.default_rng(123)
    grid = CityGrid(width=50, height=50, seed=42)
    start = (5, 5)
    goal = (20, 20)
    vtype = VehicleType.AMBULANCE

    dstar = DStarLite(grid, start=start, goal=goal, vehicle_type=vtype)
    path_d, cost_d, _ = dstar.plan()
    assert len(path_d) > 0

    # Inject 5 sequential dynamic obstacles and verify D* Lite matches fresh A*
    for _ in range(5):
        if len(path_d) <= 2:
            break
        obs_idx = rng.integers(1, len(path_d) - 1)
        obs_coord = path_d[obs_idx]
        grid.set_blocked(obs_coord[0], obs_coord[1], True)

        dstar.notify_cell_changed(obs_coord)
        path_d, cost_d, _ = dstar.replan()

        # Compare against cold-start A* on modified grid
        astar = AStar(grid, vehicle_type=vtype)
        path_a, cost_a, _ = astar.search(start, goal)

        if math.isinf(cost_a):
            assert math.isinf(cost_d)
        else:
            assert np.isclose(cost_d, cost_a, atol=0.25)



def test_vehicle_specific_routing():
    grid = CityGrid(width=50, height=50, seed=42)
    
    # Destination inside the river
    river_target = grid.river_cells[0]
    start_point = (25, 10)

    # Ambulance should fail to find path into river
    dstar_amb = DStarLite(grid, start=start_point, goal=river_target, vehicle_type=VehicleType.AMBULANCE)
    path_amb, cost_amb, _ = dstar_amb.plan()
    assert len(path_amb) == 0
    assert math.isinf(cost_amb)

    # Boat located at dock/river should successfully navigate river cells
    boat_start = grid.river_cells[0]
    boat_dest = grid.river_cells[5]
    dstar_boat = DStarLite(grid, start=boat_start, goal=boat_dest, vehicle_type=VehicleType.RESCUE_BOAT)
    path_boat, cost_boat, _ = dstar_boat.plan()
    assert len(path_boat) > 0
    assert not math.isinf(cost_boat)


def test_fleet_router_cost_matrix():
    grid = CityGrid(width=50, height=50, seed=42)
    router = FleetRouter(grid)

    river_cell = grid.river_cells[0]
    vehicles = [
        Vehicle(vehicle_id="AMB-1", vehicle_type=VehicleType.AMBULANCE, capacity=2, speed_kmh=45.0,
                base_location=(5, 5), current_location=(5, 5)),
        Vehicle(vehicle_id="BOAT-1", vehicle_type=VehicleType.RESCUE_BOAT, capacity=4, speed_kmh=30.0,
                base_location=river_cell, current_location=river_cell),
    ]
    calls = [
        DistressCall(call_id="CALL-1", location=(12, 12), timestamp_step=1, caller_name="P1"),
        DistressCall(call_id="CALL-2", location=river_cell, timestamp_step=1, caller_name="P2"),  # River call
    ]

    cost_matrix = router.compute_cost_matrix(vehicles, calls)
    
    # Ambulance to land call: valid cost
    assert not math.isinf(cost_matrix[("AMB-1", "CALL-1")])
    assert cost_matrix[("AMB-1", "CALL-1")] > 0.0

    # Ambulance to river call: inf
    assert math.isinf(cost_matrix[("AMB-1", "CALL-2")])

    # Boat to river call: valid cost
    assert not math.isinf(cost_matrix[("BOAT-1", "CALL-2")])
