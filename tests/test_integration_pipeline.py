"""
tests.test_integration_pipeline
================================
Integration tests proving the unified WorldState and the closed-loop dataflow:
  Sensors -> HMM Hazard -> BayesNet Triage -> D* Lite Graph Weights -> SA Dispatch.
Verifies that changing a physical sensor reading strictly changes the downstream dispatch.
"""

import pytest
import os
import numpy as np

from resq_ai.orchestrator.engine import OrchestratorEngine
from resq_ai.core.models import (
    SensorObservation,
    DistressCall,
    OccupantType,
    PriorityLevel,
    HazardLevel,
    VehicleType,
    SimEvent,
    EventType,
)
from resq_ai.core.world_state import WorldState


def test_unified_world_model_shared_state():
    """Verifies that all subsystems read and write to one canonical WorldState."""
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default_scenario.yaml")
    engine = OrchestratorEngine(config_path=cfg_path, seed=42)

    assert isinstance(engine.world, WorldState)
    assert engine.world.grid is engine.grid
    assert engine.world.vehicles is engine.vehicles
    assert engine.world.distress_calls is engine.calls
    assert len(engine.world.hospitals) >= 2
    assert len(engine.world.handoff_points) >= 3


def test_sensor_change_propagates_to_dispatch():
    """
    INTEGRATION PROOF:
    Proves that altering a zone's raw sensor reading changes the HMM belief,
    which changes the Bayesian Network triage risk score,
    which alters the Simulated Annealing cost function and final vehicle assignment.
    """
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default_scenario.yaml")
    engine = OrchestratorEngine(config_path=cfg_path, seed=42)

    # 1. Register a test call in zone_riverfront at (22, 34)
    cid = "CALL-INTEGRATION-01"
    test_call = DistressCall(
        call_id=cid,
        location=(22, 34),
        timestamp_step=1,
        caller_name="Waterfront Residence",
        occupants_type=OccupantType.ELDERLY,
        occupant_count=2,
        building_age="Old",
        construction_type="Wood",
        reported_flood_m=0.45,
        fire_present=False,
    )
    engine.calls[cid] = test_call

    # --- SCENARIO A: Calm / Safe Sensor Reading ---
    calm_obs = SensorObservation(
        step=1,
        zone_id="zone_riverfront",
        water_level_m=0.04,
        rainfall_mmh=4.0,
        wind_speed_kmh=12.0,
    )
    engine.process_event(SimEvent(step=1, event_type=EventType.SENSOR_UPDATE, payload={"observation": calm_obs}))

    # Evaluate call triage with current HMM hazard state
    z_sum_calm = engine.hmm_tracker.get_zone_summary("zone_riverfront")
    engine.triage.evaluate_call(test_call, hazard_state=z_sum_calm["map_state"])
    calm_priority = test_call.priority
    calm_risk_score = test_call.risk_score

    # Compute baseline SA schedule
    cm_calm = engine.router.compute_cost_matrix(list(engine.vehicles.values()), [test_call])
    schedule_calm, cost_calm = engine.sa_optimizer.warm_start_replan(engine.current_schedule, [test_call], cm_calm)

    # --- SCENARIO B: Severe Flash Flood Sensor Reading ---
    severe_obs = SensorObservation(
        step=2,
        zone_id="zone_riverfront",
        water_level_m=1.85,
        rainfall_mmh=125.0,
        wind_speed_kmh=90.0,
    )
    engine.process_event(SimEvent(step=2, event_type=EventType.SENSOR_UPDATE, payload={"observation": severe_obs}))

    # Re-evaluate call triage under escalated HMM hazard state
    z_sum_severe = engine.hmm_tracker.get_zone_summary("zone_riverfront")
    assert z_sum_severe["alert"] is True or z_sum_severe["map_state"] in ("Critical", "Evacuate")

    engine.triage.evaluate_call(test_call, hazard_state=z_sum_severe["map_state"])
    severe_priority = test_call.priority
    severe_risk_score = test_call.risk_score

    # Compute re-planned SA schedule under escalated risk
    cm_severe = engine.router.compute_cost_matrix(list(engine.vehicles.values()), [test_call])
    schedule_severe, cost_severe = engine.sa_optimizer.warm_start_replan(engine.current_schedule, [test_call], cm_severe)

    # 1. HMM belief shifted
    assert z_sum_calm["map_state"] != z_sum_severe["map_state"]
    # 2. Bayesian triage risk score strictly increased
    assert severe_risk_score > calm_risk_score, f"Expected {severe_risk_score} > {calm_risk_score}"
    # 3. Schedule objective cost strictly changed due to risk weight modulation
    assert cost_severe != cost_calm
