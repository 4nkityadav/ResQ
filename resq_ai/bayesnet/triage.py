"""
resq_ai.bayesnet.triage
=======================
High-level emergency triage evaluator powered by Bayesian Network exact inference.
Translates call parameters and HMM hazard states into probabilistic risk scores,
decision-theoretic Expected Utility rankings (START/SALT-aligned),
and Value-of-Information (VoI) recommendations for 911 dispatchers.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import os

from resq_ai.core.models import DistressCall, PriorityLevel, OccupantType
from resq_ai.bayesnet.exact_inference import BayesianNetwork


DISPATCHER_QUESTION_PROMPTS: Dict[str, str] = {
    "FirePresent": "Is there an active fire, smoke, or gas leak on the premises?",
    "OccupantType": "Are there vulnerable occupants (infants, elderly, or mobility-impaired) inside?",
    "BuildingAge": "What is the approximate age of the building (Pre-1970 unreinforced vs modern)?",
    "ConstructionType": "What is the structural construction type (wood frame, brick/masonry, concrete)?",
    "FloodLevel": "What is the current water level relative to electrical outlets or floor level?",
}

# Decision-Theoretic Utility Matrix U(action, true_state)
# Aligned with emergency triage START/SALT protocol (high penalty for under-triaging P1)
UTILITY_MATRIX: Dict[PriorityLevel, Dict[str, float]] = {
    PriorityLevel.P1: {"P1": 100.0, "P2": 15.0, "P3": -50.0},
    PriorityLevel.P2: {"P1": -40.0, "P2": 70.0, "P3": 10.0},
    PriorityLevel.P3: {"P1": -250.0, "P2": -20.0, "P3": 60.0},
}


class TriageEngine:
    """Evaluates incoming distress calls using exact Bayesian inference and decision utility."""

    def __init__(self, cpt_config_path: Optional[str] = None):
        if cpt_config_path is None:
            default_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "bayes_cpt.yaml")
            cpt_config_path = os.path.abspath(default_path)
            
        self.bn = BayesianNetwork(config_path=cpt_config_path)

    def _build_evidence(self, call: DistressCall, hazard_state: Optional[str] = None) -> Dict[str, str]:
        """Constructs evidence dictionary from call data."""
        if call.reported_flood_m < 0.30:
            flood_category = "Low"
        elif call.reported_flood_m < 0.85:
            flood_category = "Moderate"
        else:
            flood_category = "High"

        occ_str = call.occupants_type.value if hasattr(call.occupants_type, "value") else str(call.occupants_type)

        evidence: Dict[str, str] = {
            "BuildingAge": str(call.building_age),
            "ConstructionType": str(call.construction_type),
            "OccupantType": occ_str,
        }
        if call.fire_present:
            evidence["FirePresent"] = "True"

        if hazard_state:
            evidence["HazardState"] = hazard_state
            if call.reported_flood_m >= 1.20 and hazard_state in ("Critical", "Evacuate"):
                evidence["FloodLevel"] = "High"
            elif call.reported_flood_m < 0.20 and hazard_state == "Safe":
                evidence["FloodLevel"] = "Low"
        else:
            evidence["FloodLevel"] = flood_category

        return evidence

    def evaluate_call(
        self,
        call: DistressCall,
        hazard_state: Optional[str] = None,
    ) -> DistressCall:
        """
        Runs Variable Elimination and Expected Utility ranking to triage the call.
        """
        evidence = self._build_evidence(call, hazard_state)

        # 1. Exact Inference via Variable Elimination
        f_collapse = self.bn.query("StructuralCollapse", evidence=evidence)
        f_hypothermia = self.bn.query("HypothermiaRisk", evidence=evidence)
        f_priority = self.bn.query("PriorityScore", evidence=evidence)

        p_collapse_true = f_collapse.get_probability({"StructuralCollapse": "True"})
        p_hypo_high = f_hypothermia.get_probability({"HypothermiaRisk": "High"})
        
        prob_p1 = f_priority.get_probability({"PriorityScore": "P1"})
        prob_p2 = f_priority.get_probability({"PriorityScore": "P2"})
        prob_p3 = f_priority.get_probability({"PriorityScore": "P3"})

        # 2. Decision-Theoretic Expected Utility Ranking:
        # EU(a) = sum_{s} P(s | e) * U(a, s)
        eu_scores: Dict[PriorityLevel, float] = {}
        state_probs = {"P1": prob_p1, "P2": prob_p2, "P3": prob_p3}
        for action, utility_row in UTILITY_MATRIX.items():
            eu = sum(state_probs[s] * utility_row[s] for s in ["P1", "P2", "P3"])
            eu_scores[action] = eu

        # Optimal action maximizes expected utility
        assigned_priority = max(eu_scores.keys(), key=lambda a: eu_scores[a])

        # Continuous risk score in [1.0, 10.0] for Simulated Annealing optimization
        raw_score = (10.0 * prob_p1) + (5.0 * prob_p2) + (1.0 * prob_p3) + (2.0 * p_collapse_true)
        risk_score = round(min(10.0, max(1.0, raw_score)), 2)

        # 3. Generate transparent evidence explanation in plain English
        drivers = []
        if p_collapse_true >= 0.50:
            drivers.append(f"high building collapse risk ({p_collapse_true:.0%}) due to {call.building_age.lower()} {call.construction_type.lower()} structure")
        if p_hypo_high >= 0.40:
            occ_str = call.occupants_type.value if hasattr(call.occupants_type, "value") else str(call.occupants_type)
            drivers.append(f"cold-water hypothermia danger ({p_hypo_high:.0%}) for {occ_str.lower()}")
        if call.fire_present:
            drivers.append("active building fire reported")
        if call.reported_flood_m >= 0.85:
            drivers.append(f"deep floodwaters ({call.reported_flood_m:.1f}m)")

        if not drivers:
            drivers.append("routine flood alert for this district")

        p_val = assigned_priority.value
        p_desc = "Immediate Life Risk" if p_val == "P1" else ("Urgent Attention" if p_val == "P2" else "Non-Critical")
        explanation = f"{p_val} ({p_desc}): " + "; ".join(drivers)

        call.priority = assigned_priority
        call.p_collapse = round(p_collapse_true, 3)
        call.p_hypothermia = round(p_hypo_high, 3)
        call.risk_score = risk_score
        call.explanation = explanation

        return call

    def recommend_next_question(
        self,
        call: DistressCall,
        hazard_state: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str], float]:
        """
        Uses Value of Information (VoI) to determine the single best question
        a 911 dispatcher should prompt next to minimize priority entropy.
        
        Returns:
            (variable_name, prompt_question_string, information_gain_bits)
        """
        evidence = self._build_evidence(call, hazard_state)
        candidate_vars = ["FirePresent", "BuildingAge", "ConstructionType", "OccupantType", "FloodLevel"]
        
        voi_dict = self.bn.value_of_information("PriorityScore", candidate_vars, evidence)
        if not voi_dict:
            return None, None, 0.0

        top_var, max_gain = next(iter(voi_dict.items()))
        prompt = DISPATCHER_QUESTION_PROMPTS.get(top_var, f"Can you confirm details regarding {top_var}?")
        return top_var, prompt, max_gain

    def evaluate_repeated_call(
        self,
        call: DistressCall,
        previous_p_collapse: float,
        previous_p_hypothermia: float,
        elapsed_minutes: float = 10.0,
        hazard_state: Optional[str] = None,
    ) -> DistressCall:
        """
        Temporal Dynamic Bayesian Network slice for repeated calls from the same address.
        Compounds structural fatigue and hypothermia exposure over elapsed time.
        """
        updated_call = self.evaluate_call(call, hazard_state=hazard_state)
        
        # Temporal compounding transition
        # Fatigue compounding factor
        time_factor = min(1.0, elapsed_minutes / 60.0)
        
        # Structural degradation accumulates
        compounded_collapse = updated_call.p_collapse + (1.0 - updated_call.p_collapse) * (0.25 * time_factor * previous_p_collapse)
        # Hypothermia risk compounds with cold flood exposure duration
        compounded_hypo = updated_call.p_hypothermia + (1.0 - updated_call.p_hypothermia) * (0.35 * time_factor * previous_p_hypothermia)

        updated_call.p_collapse = round(min(1.0, compounded_collapse), 3)
        updated_call.p_hypothermia = round(min(1.0, compounded_hypo), 3)

        if updated_call.p_collapse >= 0.65 or updated_call.p_hypothermia >= 0.60:
            updated_call.priority = PriorityLevel.P1
            updated_call.explanation += f" [Repeated Call: Compounded exposure +{elapsed_minutes:.0f}m]"

        return updated_call
