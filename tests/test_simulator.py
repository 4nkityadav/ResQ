"""
tests.test_simulator
====================
Verification tests for discrete-event simulator (DES), scenario schema validation,
and seedable bit-identical determinism.
"""

import pytest
from resq_ai.sim.scenario_schema import ScenarioConfig, ScenarioInject, ScenarioFleetConfig
from resq_ai.sim.simulator import DiscreteEventSimulator


def test_scenario_schema_validation():
    valid_data = {
        "scenario_name": "Test Scenario",
        "disaster_type": "flash_flood",
        "seed": 99,
        "fleet": {"ambulances": 4, "boats": 2},
        "injects": [
            {"step": 5, "inject_type": "bridge_collapse", "parameters": {"location": [20, 34]}}
        ]
    }
    cfg = ScenarioConfig.from_dict(valid_data)
    assert cfg.scenario_name == "Test Scenario"
    assert len(cfg.injects) == 1
    assert cfg.fleet.ambulances == 4

    # Invalid disaster type
    with pytest.raises(ValueError, match="unrecognized"):
        ScenarioConfig.from_dict({"disaster_type": "alien_invasion"})

    # Invalid fleet (0 ambulances)
    with pytest.raises(ValueError, match="at least 1 ambulance"):
        ScenarioConfig.from_dict({"fleet": {"ambulances": 0}})

    # Invalid inject type
    with pytest.raises(ValueError, match="Invalid inject_type"):
        ScenarioConfig.from_dict({
            "injects": [{"step": 1, "inject_type": "tornado_f5"}]
        })


def test_compound_disaster_scenario_loading():
    sim = DiscreteEventSimulator(config_path="configs/scenarios/compound_disaster.yaml")
    assert sim.config.disaster_type == "compound"
    assert len(sim.config.injects) == 4
    assert sim.engine.vehicles["AMB-2"].status.value == "idle"


def test_discrete_event_simulator_inject_execution():
    sim = DiscreteEventSimulator(config_path="configs/scenarios/compound_disaster.yaml", seed=404)
    
    # Run 20 steps to cover all injects (steps 5, 8, 12, 18)
    events = sim.run(steps=20)
    assert len(events) > 20

    # Verify bridge collapse inject executed at step 8
    bridge_events = [e for e in events if "Bridge" in str(e.payload.get("description", "")) and "INJECT" in str(e.payload.get("description", ""))]
    assert len(bridge_events) >= 1
    assert sim.engine.grid.get_cell(20, 34).is_blocked is True

    # Verify unit breakdown inject executed at step 18
    breakdown_events = [e for e in events if e.payload.get("breakdown") is True]
    assert len(breakdown_events) >= 1


def test_end_to_end_scenario_determinism():
    """Invariant: Running same scenario with fixed seed produces bit-identical telemetry."""
    sim1 = DiscreteEventSimulator(config_path="configs/scenarios/compound_disaster.yaml", seed=777)
    events1 = sim1.run(steps=15)
    
    sim2 = DiscreteEventSimulator(config_path="configs/scenarios/compound_disaster.yaml", seed=777)
    events2 = sim2.run(steps=15)

    assert len(events1) == len(events2)
    for e1, e2 in zip(events1, events2):
        assert e1.step == e2.step
        assert e1.event_type == e2.event_type
        assert str(e1.payload.get("description")) == str(e2.payload.get("description"))

    # Vehicle positions must be bit-identical
    for vid in sim1.engine.vehicles:
        loc1 = sim1.engine.vehicles[vid].current_location
        loc2 = sim2.engine.vehicles[vid].current_location
        assert loc1 == loc2
