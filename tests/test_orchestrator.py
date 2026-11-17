"""
Unit tests for resq_ai.orchestrator.engine
Verifies end-to-end integration and sub-second acceptance demo requirements.
"""

import pytest
import time
import os

from resq_ai.orchestrator.engine import OrchestratorEngine
from resq_ai.core.models import PriorityLevel, HazardLevel


def test_orchestrator_initialization_and_stepping():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default_scenario.yaml")
    engine = OrchestratorEngine(config_path=cfg_path, seed=42)

    assert len(engine.vehicles) == 8
    assert len(engine.grid.zones) == 4

    # Run 3 simulation steps
    step_results = []
    for _ in range(3):
        res = engine.step()
        step_results.append(res)

    assert engine.sim.current_step == 3
    state = engine.get_dashboard_state()
    assert state["step"] == 3
    assert len(state["zones"]) == 4


def test_acceptance_demo_end_to_end_under_one_second():
    """
    ACCEPTANCE DEMO:
    Dispatcher: 'Dispatch rescue boat to Block C. Reports say Main Street is submerged.'
    -> Main Street edges marked impassable
    -> D* Lite replans incrementally and reports ETA delta
    -> Bayes Net computes Block C priority
    -> SA re-optimizes fleet assignment
    -> HMM hazard level tracked
    -> All in under 1 second!
    """
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default_scenario.yaml")
    engine = OrchestratorEngine(config_path=cfg_path, seed=42)

    # Prime with a step to generate initial sensor and call state
    engine.step()

    command = "Dispatch rescue boat to Block C. Reports say Main Street is submerged."

    t0 = time.perf_counter()
    response = engine.handle_natural_language_command(command)
    latency_sec = time.perf_counter() - t0

    # 1. Verification of latency requirement (< 1.0s)
    assert latency_sec < 1.0, f"Acceptance demo took {latency_sec:.3f}s, exceeding 1.0s limit!"

    # 2. Verification of multi-module transparent feedback
    assert "Main Street" in response
    assert "impassable" in response
    assert "D*" in response
    assert "Bayesian triage" in response
    assert "Block C" in response
    assert "Priority 1" in response or "P1" in response
    assert "SA reassigns" in response

    # 3. Verify state modifications
    # Main street (25, 20) must be blocked
    main_st_cell = engine.grid.get_cell(25, 20)
    assert main_st_cell.is_blocked is True
    assert main_st_cell.water_level >= 0.70
