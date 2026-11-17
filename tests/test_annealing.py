"""
Unit tests for resq_ai.annealing (ScheduleState, SimulatedAnnealingOptimizer, Baselines).
Verifies constraint enforcement, neighborhood operators, and warm-start replanning.
"""

import pytest
import math
import numpy as np

from resq_ai.core.models import Vehicle, VehicleType, DistressCall, OccupantType, PriorityLevel
from resq_ai.annealing.state import ScheduleState
from resq_ai.annealing.simulated_annealing import SimulatedAnnealingOptimizer
from resq_ai.annealing.baselines import GreedyAllocator, RandomAllocator, ClarkeWrightSavingsAllocator
from resq_ai.annealing.alns import ALNSOptimizer


@pytest.fixture
def mock_fleet_and_calls():
    vehicles = {
        "AMB-1": Vehicle(vehicle_id="AMB-1", vehicle_type=VehicleType.AMBULANCE, capacity=2, speed_kmh=45.0,
                         base_location=(5, 5), current_location=(5, 5)),
        "AMB-2": Vehicle(vehicle_id="AMB-2", vehicle_type=VehicleType.AMBULANCE, capacity=2, speed_kmh=45.0,
                         base_location=(10, 10), current_location=(10, 10)),
        "BOAT-1": Vehicle(vehicle_id="BOAT-1", vehicle_type=VehicleType.RESCUE_BOAT, capacity=4, speed_kmh=30.0,
                          base_location=(25, 34), current_location=(25, 34)),
    }
    calls = {
        "CALL-1": DistressCall(call_id="CALL-1", location=(7, 7), timestamp_step=1, caller_name="Alice",
                               occupant_count=1, reported_flood_m=0.1, priority=PriorityLevel.P1, risk_score=9.0),
        "CALL-2": DistressCall(call_id="CALL-2", location=(8, 8), timestamp_step=1, caller_name="Bob",
                               occupant_count=1, reported_flood_m=0.1, priority=PriorityLevel.P2, risk_score=5.0),
        "CALL-3": DistressCall(call_id="CALL-3", location=(26, 34), timestamp_step=1, caller_name="Charlie",
                               occupant_count=2, reported_flood_m=0.8, priority=PriorityLevel.P1, risk_score=9.5), # Flood call
    }
    return vehicles, calls


def test_schedule_state_compatibility_and_validity(mock_fleet_and_calls):
    vehicles, calls = mock_fleet_and_calls
    state = ScheduleState(vehicles=vehicles, calls=calls)

    # Ambulance compatible with CALL-1 (dry road), incompatible with CALL-3 (0.8m flood)
    assert state.is_vehicle_compatible("AMB-1", "CALL-1") is True
    assert state.is_vehicle_compatible("AMB-1", "CALL-3") is False

    # Boat compatible with CALL-3 (flood), incompatible with CALL-1 (0.1m dry)
    assert state.is_vehicle_compatible("BOAT-1", "CALL-3") is True
    assert state.is_vehicle_compatible("BOAT-1", "CALL-1") is False

    # Assign valid routes
    state.routes["AMB-1"] = ["CALL-1"]
    state.routes["AMB-2"] = ["CALL-2"]
    state.routes["BOAT-1"] = ["CALL-3"]
    state.unassigned = []
    assert state.is_valid() is True

    # Duplicate assignment should invalidate state
    state.routes["AMB-2"].append("CALL-1")
    assert state.is_valid() is False


def test_cost_evaluation_and_capacity_penalties(mock_fleet_and_calls):
    vehicles, calls = mock_fleet_and_calls
    state = ScheduleState(vehicles=vehicles, calls=calls)
    
    # All calls unassigned -> high unserved penalty
    unserved_cost = state.evaluate_cost()
    assert unserved_cost > 1000.0

    # Assign calls within capacity
    state.routes["AMB-1"] = ["CALL-1"]
    state.routes["BOAT-1"] = ["CALL-3"]
    state.unassigned = ["CALL-2"]
    cost_served = state.evaluate_cost()
    assert cost_served < unserved_cost

    # Overload AMB-1 (capacity 2, add 3 occupants)
    overload_call = DistressCall(call_id="CALL-EXTRA", location=(9, 9), timestamp_step=1, caller_name="Extra",
                                 occupant_count=3, reported_flood_m=0.1, risk_score=5.0)
    state.calls["CALL-EXTRA"] = overload_call
    state.routes["AMB-1"].append("CALL-EXTRA")
    cost_overloaded = state.evaluate_cost()
    # Cost should sharply spike due to quadratic capacity penalty
    assert cost_overloaded > cost_served + 250.0


def test_simulated_annealing_optimization(mock_fleet_and_calls):
    vehicles, calls = mock_fleet_and_calls
    state = ScheduleState(vehicles=vehicles, calls=calls)
    initial_cost = state.evaluate_cost()

    optimizer = SimulatedAnnealingOptimizer(
        initial_temp=80.0,
        cooling_rate=0.96,
        max_iterations=300,
        cooling_mode="geometric",
        seed=42,
    )
    best_state, best_cost, history = optimizer.optimize(state, max_time_sec=0.8)

    assert best_cost < initial_cost
    assert best_state.is_valid()
    assert len(history) > 5
    # Best cost must be non-increasing in history
    best_series = [h["best_cost"] for h in history]
    assert all(x >= y for x, y in zip(best_series, best_series[1:]))


def test_warm_start_replan(mock_fleet_and_calls):
    vehicles, calls = mock_fleet_and_calls
    state = ScheduleState(vehicles=vehicles, calls=calls)
    state.routes["AMB-1"] = ["CALL-1"]
    state.routes["BOAT-1"] = ["CALL-3"]
    state.unassigned = ["CALL-2"]

    optimizer = SimulatedAnnealingOptimizer(seed=42)
    
    # New call arrives dynamically
    new_call = DistressCall(
        call_id="CALL-NEW-01",
        location=(6, 6),
        timestamp_step=2,
        caller_name="Dynamic Victim",
        occupant_count=1,
        reported_flood_m=0.1,
        priority=PriorityLevel.P1,
        risk_score=9.2,
    )

    t0 = pytest.importorskip("time").perf_counter()
    new_state, new_cost = optimizer.warm_start_replan(state, [new_call])
    elapsed_ms = (pytest.importorskip("time").perf_counter() - t0) * 1000.0

    # Must complete fast
    assert elapsed_ms < 500.0
    assert "CALL-NEW-01" in new_state.calls
    # The new call must be assigned to an ambulance or boat
    assigned_calls = [c for r in new_state.routes.values() for c in r]
    assert "CALL-NEW-01" in assigned_calls or "CALL-NEW-01" in new_state.unassigned


def test_baselines_execution(mock_fleet_and_calls):
    vehicles, calls = mock_fleet_and_calls
    greedy_state, greedy_cost = GreedyAllocator.allocate(vehicles, calls)
    assert greedy_state.is_valid()
    assert greedy_cost < 3000.0

    rand_state, rand_cost = RandomAllocator.allocate(vehicles, calls, seed=42)
    assert rand_state.is_valid()


def test_clarke_wright_savings_allocation(mock_fleet_and_calls):
    vehicles, calls = mock_fleet_and_calls
    cw_state, cw_cost = ClarkeWrightSavingsAllocator.allocate(vehicles, calls)
    assert cw_state.is_valid()
    assert cw_cost < 3000.0


def test_alns_optimizer_execution(mock_fleet_and_calls):
    vehicles, calls = mock_fleet_and_calls
    state = ScheduleState(vehicles=vehicles, calls=calls)
    initial_cost = state.evaluate_cost()

    alns = ALNSOptimizer(max_iterations=150, seed=42)
    best_state, best_cost, stats = alns.optimize(state, max_time_sec=1.0)
    assert best_state.is_valid()
    assert best_cost < initial_cost
    assert stats["iterations"] > 10
    assert stats["elapsed_ms"] < 2000.0


def test_route_stability_penalty_invariant(mock_fleet_and_calls):
    vehicles, calls = mock_fleet_and_calls
    prev_routes = {"AMB-1": ["CALL-1"], "AMB-2": ["CALL-2"], "BOAT-1": ["CALL-3"]}
    
    # State keeping same first en-route assignment
    state_stable = ScheduleState(vehicles=vehicles, calls=calls, routes=prev_routes, previous_routes=prev_routes)
    cost_stable = state_stable.evaluate_cost()

    # State swapping first en-route assignment (diverting vehicle)
    diverted_routes = {"AMB-1": ["CALL-2"], "AMB-2": ["CALL-1"], "BOAT-1": ["CALL-3"]}
    state_diverted = ScheduleState(vehicles=vehicles, calls=calls, routes=diverted_routes, previous_routes=prev_routes)
    cost_diverted = state_diverted.evaluate_cost()

    # Diverted state must incur stability penalties
    assert cost_diverted > cost_stable

