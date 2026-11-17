"""
resq_ai.hmm.tracker
===================
Real-time hazard tracking manager across civil defense zones.
Provides:
- Online belief updates with spatial coupling across adjacent zones
- Marginalized missing sensor telemetry handling
- Predictive multi-horizon escalation forecasts (15m, 30m, 60m)
- Viterbi trajectory decoding
- Escalation alert triggering for the orchestrator and Bayesian Network
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from resq_ai.core.models import HazardLevel, SensorObservation, ZoneStatus
from resq_ai.hmm.continuous_hmm import GaussianHMM


# Canonical civil defense zone topological adjacency
DEFAULT_ZONE_ADJACENCY: Dict[str, List[str]] = {
    "zone_north": ["zone_central"],
    "zone_central": ["zone_north", "zone_riverfront"],
    "zone_riverfront": ["zone_central", "zone_south"],
    "zone_south": ["zone_riverfront"],
}


class ZoneHazardTracker:
    """Manages continuous hazard monitoring, spatial coupling, and forecasts."""

    def __init__(
        self,
        hmm_model: Optional[GaussianHMM] = None,
        escalation_threshold: float = 0.60,
        spatial_coupling_lambda: float = 0.15,
        zone_adjacency: Optional[Dict[str, List[str]]] = None,
    ):
        self.hmm = hmm_model or GaussianHMM()
        self.escalation_threshold = escalation_threshold
        self.spatial_coupling = spatial_coupling_lambda
        self.adjacency = zone_adjacency or DEFAULT_ZONE_ADJACENCY
        
        # State mapping
        self.state_enum_map = {
            0: HazardLevel.SAFE,
            1: HazardLevel.ADVISORY,
            2: HazardLevel.CRITICAL,
            3: HazardLevel.EVACUATE,
        }
        self.enum_state_map = {v: k for k, v in self.state_enum_map.items()}

        # Online belief distribution per zone: shape (4,)
        self.zone_beliefs: Dict[str, np.ndarray] = {}
        self.zone_observations: Dict[str, List[SensorObservation]] = {}
        self.active_alerts: Dict[str, bool] = {}

    def register_zone(self, zone_id: str, initial_belief: Optional[Dict[HazardLevel, float]] = None) -> None:
        """Initializes tracking for a civil defense zone."""
        if initial_belief:
            vec = np.zeros(4, dtype=np.float64)
            for lvl, prob in initial_belief.items():
                vec[self.enum_state_map[lvl]] = prob
            vec = vec / np.sum(vec)
        else:
            vec = self.hmm.pi.copy()

        self.zone_beliefs[zone_id] = vec
        self.zone_observations[zone_id] = []
        self.active_alerts[zone_id] = False

    def update_zone_observation(
        self,
        obs: SensorObservation,
    ) -> Tuple[Dict[HazardLevel, float], bool, str]:
        """
        Processes a new sensor observation for a zone.
        Applies HMM filter and spatial coupling from neighboring zones.
        """
        zid = obs.zone_id
        if zid not in self.zone_beliefs:
            self.register_zone(zid)

        # Record observation
        self.zone_observations[zid].append(obs)

        # 1. Single-step filter update
        obs_vec = np.array(obs.to_vector(), dtype=np.float64)
        current_belief = self.zone_beliefs[zid]
        filtered_belief = self.hmm.filter_single_step(current_belief, obs_vec)

        # 2. Spatial coupling with adjacent zones
        neighbors = self.adjacency.get(zid, [])
        valid_neighbor_beliefs = [
            self.zone_beliefs[n] for n in neighbors if n in self.zone_beliefs
        ]

        if valid_neighbor_beliefs and self.spatial_coupling > 0:
            avg_neighbor = np.mean(valid_neighbor_beliefs, axis=0)
            coupled_belief = (1.0 - self.spatial_coupling) * filtered_belief + self.spatial_coupling * avg_neighbor
            coupled_belief = coupled_belief / np.sum(coupled_belief)
        else:
            coupled_belief = filtered_belief

        self.zone_beliefs[zid] = coupled_belief

        # 3. Escalation risk: P(Critical) + P(Evacuate)
        p_critical_or_evacuate = float(coupled_belief[2] + coupled_belief[3])
        alert_triggered = p_critical_or_evacuate >= self.escalation_threshold
        self.active_alerts[zid] = alert_triggered

        # Determine MAP current state
        map_idx = int(np.argmax(coupled_belief))
        map_state = self.state_enum_map[map_idx]

        # Build belief dictionary
        belief_dict = {
            self.state_enum_map[i]: float(coupled_belief[i])
            for i in range(4)
        }

        # Multi-horizon forecast
        forecast = self.hmm.forecast_critical_risk(coupled_belief)

        summary = (
            f"Zone '{zid}': state={map_state.value} "
            f"(P(Crit/Evac)={p_critical_or_evacuate:.2f}, 30m-Forecast={forecast.get('30m', 0.0):.2f}, "
            f"Alert={'ACTIVE' if alert_triggered else 'NORMAL'})"
        )

        return belief_dict, alert_triggered, summary

    def get_forecast(self, zone_id: str) -> Dict[str, float]:
        """Returns multi-horizon risk forecast {15m, 30m, 60m} for zone."""
        if zone_id not in self.zone_beliefs:
            self.register_zone(zone_id)
        return self.hmm.forecast_critical_risk(self.zone_beliefs[zone_id])

    def get_viterbi_trajectory(self, zone_id: str) -> List[HazardLevel]:
        """Runs offline Viterbi decoding over full historical observation sequence for a zone."""
        history = self.zone_observations.get(zone_id, [])
        if not history:
            return []

        obs_matrix = np.array([o.to_vector() for o in history], dtype=np.float64)
        best_path_indices, _ = self.hmm.viterbi(obs_matrix)
        return [self.state_enum_map[idx] for idx in best_path_indices]

    def get_belief_vector(self, zone_id: str) -> np.ndarray:
        """Returns raw numpy vector of probabilities [Safe, Advisory, Critical, Evacuate]."""
        if zone_id not in self.zone_beliefs:
            self.register_zone(zone_id)
        return self.zone_beliefs[zone_id]

    def get_zone_summary(self, zone_id: str) -> Dict[str, Any]:
        """Provides full telemetry snapshot with multi-horizon predictive forecasts."""
        if zone_id not in self.zone_beliefs:
            self.register_zone(zone_id)

        vec = self.zone_beliefs[zone_id]
        p_crit_evac = float(vec[2] + vec[3])
        map_idx = int(np.argmax(vec))
        forecast = self.hmm.forecast_critical_risk(vec)
        
        return {
            "zone_id": zone_id,
            "map_state": self.state_enum_map[map_idx].value,
            "belief": {self.state_enum_map[i].value: round(float(vec[i]), 4) for i in range(4)},
            "p_critical_or_evacuate": round(p_crit_evac, 4),
            "forecast_15m": forecast.get("15m", 0.0),
            "forecast_30m": forecast.get("30m", 0.0),
            "forecast_60m": forecast.get("60m", 0.0),
            "alert": self.active_alerts.get(zone_id, False),
            "observation_count": len(self.zone_observations.get(zone_id, [])),
        }
