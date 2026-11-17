"""
Unit tests for resq_ai.bayesnet.factor, exact_inference, and triage.
Verifies factor algebra, variable elimination, and the 10-node CPT triage model.
"""

import pytest
import numpy as np
import os

from resq_ai.bayesnet.factor import DiscreteFactor
from resq_ai.bayesnet.exact_inference import BayesianNetwork
from resq_ai.bayesnet.triage import TriageEngine
from resq_ai.core.models import DistressCall, OccupantType, PriorityLevel


def test_factor_product_and_marginalize():
    # phi1(A, B) where A has 2 states, B has 2 states
    # phi2(B, C) where B has 2 states, C has 2 states
    phi1 = DiscreteFactor(
        variables=["A", "B"],
        cardinalities=[2, 2],
        values=np.array([[0.1, 0.2], [0.3, 0.4]]),
        state_names={"A": ["a0", "a1"], "B": ["b0", "b1"]},
    )
    phi2 = DiscreteFactor(
        variables=["B", "C"],
        cardinalities=[2, 2],
        values=np.array([[0.5, 0.5], [0.1, 0.9]]),
        state_names={"B": ["b0", "b1"], "C": ["c0", "c1"]},
    )

    phi_prod = phi1.product(phi2)
    assert phi_prod.variables == ["A", "B", "C"]
    assert phi_prod.cardinalities == [2, 2, 2]

    # Verify specific point: A=a0, B=b0, C=c0: 0.1 * 0.5 = 0.05
    val_a0_b0_c0 = phi_prod.get_probability({"A": "a0", "B": "b0", "C": "c0"})
    assert np.isclose(val_a0_b0_c0, 0.05)

    # Marginalize B out
    phi_marg = phi_prod.marginalize(["B"])
    assert phi_marg.variables == ["A", "C"]
    assert np.isclose(np.sum(phi_marg.values), np.sum(phi_prod.values))


def test_factor_observe_and_normalize():
    phi = DiscreteFactor(
        variables=["A", "B"],
        cardinalities=[2, 2],
        values=np.array([[0.1, 0.2], [0.3, 0.4]]),
        state_names={"A": ["a0", "a1"], "B": ["b0", "b1"]},
    )
    # Condition on B=b1
    phi_red = phi.observe({"B": "b1"})
    assert phi_red.variables == ["A"]
    # Values should be slice of B=b1: [0.2, 0.4]
    assert np.allclose(phi_red.values, [0.2, 0.4])

    phi_red.normalize()
    assert np.isclose(np.sum(phi_red.values), 1.0)
    assert np.isclose(phi_red.get_probability({"A": "a0"}), 0.2 / 0.6)


def test_dag_validation_and_cycle_detection():
    bn = BayesianNetwork()
    bn.add_node("A", states=["s0", "s1"])
    bn.add_node("B", states=["s0", "s1"], parents=["A"])
    bn.add_node("C", states=["s0", "s1"], parents=["B"])
    # No cycle
    bn.validate_dag()

    # Create cycle: A depends on C
    bn.parents["A"] = ["C"]
    bn.children["C"].append("A")
    with pytest.raises(ValueError, match="Cycle detected"):
        bn.validate_dag()


def test_exact_inference_on_resq_10_node_network():
    cpt_path = os.path.join(os.path.dirname(__file__), "..", "configs", "bayes_cpt.yaml")
    bn = BayesianNetwork(config_path=cpt_path)
    assert len(bn.nodes) == 10

    # 1. Prior query without evidence: PriorityScore must sum to 1.0
    f_prior = bn.query("PriorityScore")
    assert np.isclose(np.sum(f_prior.values), 1.0)
    assert f_prior.variables == ["PriorityScore"]

    # 2. Severe disaster evidence: Old Wood, High Flood, Active Fire, Elderly
    evidence_severe = {
        "BuildingAge": "Old",
        "ConstructionType": "Wood",
        "FloodLevel": "High",
        "FirePresent": "True",
        "OccupantType": "Elderly",
    }
    f_p1 = bn.query("PriorityScore", evidence=evidence_severe)
    assert np.isclose(np.sum(f_p1.values), 1.0)
    p1_prob = f_p1.get_probability({"PriorityScore": "P1"})
    p3_prob = f_p1.get_probability({"PriorityScore": "P3"})
    assert p1_prob > 0.70
    assert p3_prob < 0.05

    # 3. Benign evidence: New ReinforcedConcrete, Low Flood, No Fire, Adults
    evidence_calm = {
        "BuildingAge": "New",
        "ConstructionType": "ReinforcedConcrete",
        "FloodLevel": "Low",
        "FirePresent": "False",
        "OccupantType": "Adults",
    }
    f_p3 = bn.query("PriorityScore", evidence=evidence_calm)
    p1_prob_calm = f_p3.get_probability({"PriorityScore": "P1"})
    p3_prob_calm = f_p3.get_probability({"PriorityScore": "P3"})
    assert p3_prob_calm > 0.65
    assert p1_prob_calm < 0.10


def test_triage_engine_evaluation():
    triage = TriageEngine()
    
    # Severe call
    severe_call = DistressCall(
        call_id="CALL-TEST-01",
        location=(10, 10),
        timestamp_step=5,
        caller_name="Elderly Care Home",
        occupants_type=OccupantType.ELDERLY,
        occupant_count=4,
        building_age="Old",
        construction_type="Wood",
        reported_flood_m=1.40,
        fire_present=True,
    )
    evaluated = triage.evaluate_call(severe_call, hazard_state="Critical")
    assert evaluated.priority == PriorityLevel.P1
    assert evaluated.p_collapse >= 0.70
    assert evaluated.risk_score >= 8.0
    assert "collapse risk" in evaluated.explanation.lower()

    # Mild call
    mild_call = DistressCall(
        call_id="CALL-TEST-02",
        location=(5, 5),
        timestamp_step=5,
        caller_name="Office Tower",
        occupants_type=OccupantType.ADULTS,
        occupant_count=1,
        building_age="New",
        construction_type="ReinforcedConcrete",
        reported_flood_m=0.10,
        fire_present=False,
    )
    evaluated_mild = triage.evaluate_call(mild_call, hazard_state="Safe")
    assert evaluated_mild.priority in (PriorityLevel.P2, PriorityLevel.P3)
    assert evaluated_mild.p_collapse < 0.10
    assert evaluated_mild.risk_score < 5.0


def test_value_of_information_and_question_recommendation():
    triage = TriageEngine()
    unclear_call = DistressCall(
        call_id="CALL-UNCLEAR",
        location=(15, 15),
        timestamp_step=1,
        caller_name="Resident",
        occupants_type=OccupantType.ADULTS,
        building_age="Moderate",
        construction_type="Masonry",
        reported_flood_m=0.40,
        fire_present=False,
    )
    # Target variable is PriorityScore
    top_var, prompt, gain = triage.recommend_next_question(unclear_call, hazard_state="Advisory")
    assert top_var is not None
    assert prompt is not None
    assert gain >= 0.0
    assert "?" in prompt


def test_sensitivity_analysis():
    cpt_path = os.path.join(os.path.dirname(__file__), "..", "configs", "bayes_cpt.yaml")
    bn = BayesianNetwork(config_path=cpt_path)
    swings = bn.sensitivity_analysis(
        target_var="PriorityScore",
        target_state="P1",
        candidate_vars=["FirePresent", "ConstructionType", "BuildingAge"],
        current_evidence={"FloodLevel": "Moderate"}
    )
    assert "FirePresent" in swings or "ConstructionType" in swings
    for var, swing in swings.items():
        assert 0.0 <= swing <= 1.0


def test_temporal_repeated_call_compounding():
    triage = TriageEngine()
    call = DistressCall(
        call_id="CALL-REPEAT-1",
        location=(20, 20),
        timestamp_step=5,
        caller_name="Flooded Basement",
        occupants_type=OccupantType.CHILDREN,
        building_age="Old",
        construction_type="Wood",
        reported_flood_m=0.50,
        fire_present=False,
    )
    eval1 = triage.evaluate_call(call, hazard_state="Critical")
    prev_collapse = eval1.p_collapse
    prev_hypo = eval1.p_hypothermia

    # Repeated call after 20 minutes with rising water
    call.reported_flood_m = 0.90
    eval2 = triage.evaluate_repeated_call(call, prev_collapse, prev_hypo, elapsed_minutes=20.0, hazard_state="Critical")
    # Compounded collapse and hypothermia must strictly be greater
    assert eval2.p_collapse >= prev_collapse
    assert eval2.p_hypothermia >= prev_hypo
    assert eval2.priority == PriorityLevel.P1

