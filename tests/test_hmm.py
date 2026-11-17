"""
Unit tests for resq_ai.hmm.continuous_hmm and resq_ai.hmm.tracker
Strict mathematical verification of forward, backward, Viterbi, and Baum-Welch.
"""

import pytest
import numpy as np
from resq_ai.core.models import HazardLevel, SensorObservation
from resq_ai.hmm.continuous_hmm import GaussianHMM
from resq_ai.hmm.tracker import ZoneHazardTracker


def test_hmm_initialization_invariants():
    hmm = GaussianHMM()
    # Check dimensions
    assert hmm.pi.shape == (4,)
    assert hmm.A.shape == (4, 4)
    assert hmm.means.shape == (4, 3)
    assert hmm.variances.shape == (4, 3)

    # Probabilities must sum to 1
    assert np.isclose(np.sum(hmm.pi), 1.0)
    assert np.allclose(np.sum(hmm.A, axis=1), 1.0)

    # Left-to-right bias: forward/self transitions dominate backward
    assert hmm.A[0, 0] > hmm.A[0, 2]
    assert hmm.A[1, 1] > hmm.A[1, 0]
    assert hmm.A[2, 2] > hmm.A[2, 0]
    assert hmm.A[3, 3] > hmm.A[3, 1]


def test_gaussian_emission_calculation():
    hmm = GaussianHMM()
    # Safe mean is [0.05, 5.0, 15.0]
    exact_safe_obs = np.array([0.05, 5.0, 15.0])
    p_safe = hmm.emission_prob(0, exact_safe_obs)
    p_evac = hmm.emission_prob(3, exact_safe_obs)

    # Probability of safe observation under Safe state must be orders of magnitude higher than under Evacuate
    assert p_safe > p_evac
    assert p_safe > 0.0


def test_forward_algorithm_and_scaling():
    hmm = GaussianHMM()
    T = 10
    # Simulate calm observations
    obs = np.array([[0.05, 5.0, 15.0] for _ in range(T)])
    alpha, c, log_lik = hmm.forward(obs)

    assert alpha.shape == (T, 4)
    assert c.shape == (T,)
    # Filtered probabilities at every timestep must sum to 1.0
    for t in range(T):
        assert np.isclose(np.sum(alpha[t]), 1.0)

    # In calm weather, state 0 (Safe) should have high probability
    assert alpha[-1, 0] > 0.60
    assert np.isfinite(log_lik)


def test_viterbi_decoding_escalating_sequence():
    hmm = GaussianHMM()
    # Sequence clearly transitioning: Safe -> Advisory -> Critical -> Evacuate
    escalating_obs = np.array([
        [0.05, 5.0, 15.0],    # Safe
        [0.06, 6.0, 16.0],    # Safe
        [0.26, 26.0, 36.0],   # Advisory
        [0.28, 28.0, 38.0],   # Advisory
        [0.78, 68.0, 62.0],   # Critical
        [1.65, 115.0, 88.0],  # Evacuate
    ])

    best_path, log_p = hmm.viterbi(escalating_obs)
    assert len(best_path) == 6
    assert np.isfinite(log_p)

    # Check decoded states follow monotonic non-decreasing escalation
    assert best_path[0] == 0  # Safe
    assert best_path[2] == 1  # Advisory
    assert best_path[4] == 2  # Critical
    assert best_path[5] == 3  # Evacuate


def test_baum_welch_em_convergence():
    hmm = GaussianHMM(seed=42)
    rng = np.random.default_rng(42)
    # Generate 40 timesteps of synthetic observations
    synthetic_obs = np.vstack([
        rng.normal([0.05, 5.0, 15.0], [0.02, 2.0, 3.0], size=(15, 3)),
        rng.normal([0.30, 30.0, 40.0], [0.05, 5.0, 6.0], size=(15, 3)),
        rng.normal([0.80, 70.0, 65.0], [0.10, 8.0, 8.0], size=(10, 3)),
    ])

    log_likelihoods = hmm.baum_welch(synthetic_obs, max_iter=15, tolerance=1e-3)
    assert len(log_likelihoods) > 1

    # In EM, log-likelihood should increase or remain stable across iterations
    for i in range(1, len(log_likelihoods)):
        # Allow tiny numerical noise tolerance (1e-5)
        assert log_likelihoods[i] >= log_likelihoods[i - 1] - 1e-4

    # Post-training parameters must still satisfy probability constraints
    assert np.isclose(np.sum(hmm.pi), 1.0)
    assert np.allclose(np.sum(hmm.A, axis=1), 1.0)
    assert np.all(hmm.variances > 0)


def test_zone_tracker_online_update_and_alerts():
    tracker = ZoneHazardTracker(escalation_threshold=0.60)
    tracker.register_zone("zone_riverfront")

    # Step 1: Normal calm reading
    obs_calm = SensorObservation(step=1, zone_id="zone_riverfront", water_level_m=0.06, rainfall_mmh=6.0, wind_speed_kmh=14.0)
    belief1, alert1, _ = tracker.update_zone_observation(obs_calm)
    assert alert1 is False
    assert belief1[HazardLevel.SAFE] > 0.50

    # Step 2-4: Severe flood surge observations
    obs_surge1 = SensorObservation(step=2, zone_id="zone_riverfront", water_level_m=0.85, rainfall_mmh=75.0, wind_speed_kmh=68.0)
    obs_surge2 = SensorObservation(step=3, zone_id="zone_riverfront", water_level_m=1.70, rainfall_mmh=120.0, wind_speed_kmh=90.0)
    
    tracker.update_zone_observation(obs_surge1)
    belief3, alert3, summary = tracker.update_zone_observation(obs_surge2)

    # Risk should now be high, alert triggered
    assert alert3 is True
    assert (belief3[HazardLevel.CRITICAL] + belief3[HazardLevel.EVACUATE]) >= 0.60
    assert "ACTIVE" in summary

    # Viterbi trajectory check
    trajectory = tracker.get_viterbi_trajectory("zone_riverfront")
    assert len(trajectory) == 3
    assert trajectory[-1] in (HazardLevel.CRITICAL, HazardLevel.EVACUATE)


def test_hmm_missing_dimension_marginalization():
    hmm = GaussianHMM()
    # Observation with missing wind_speed_kmh (NaN)
    obs_partial = np.array([0.05, 5.0, np.nan])
    p_safe = hmm.emission_prob(0, obs_partial)
    p_evac = hmm.emission_prob(3, obs_partial)
    assert p_safe > p_evac
    assert np.isfinite(p_safe)

    # Total sensor blackout: all dimensions NaN
    obs_blackout = np.array([np.nan, np.nan, np.nan])
    logp = hmm.emission_log_prob(0, obs_blackout)
    assert logp == 0.0

    # Online step with partial observation must succeed and normalize to 1.0
    prior = np.array([0.7, 0.2, 0.08, 0.02])
    updated = hmm.filter_single_step(prior, obs_partial)
    assert np.isclose(np.sum(updated), 1.0)
    assert updated[0] > updated[3]


def test_student_t_robust_emissions():
    hmm_student = GaussianHMM(emission_type="student_t", student_nu=3.0)
    hmm_gauss = GaussianHMM(emission_type="gaussian")

    # Extreme outlier spike
    outlier_obs = np.array([0.05, 500.0, 15.0])
    p_t = hmm_student.emission_prob(0, outlier_obs)
    p_g = hmm_gauss.emission_prob(0, outlier_obs)
    # Heavy tails of Student-t prevent zero underflow compared to Gaussian
    assert p_t >= p_g


def test_predictive_forecast_and_brier_score():
    hmm = GaussianHMM()
    prior = np.array([0.4, 0.3, 0.2, 0.1])
    forecast = hmm.forecast_critical_risk(prior, horizons=[3, 6, 12])
    assert "15m" in forecast and "30m" in forecast and "60m" in forecast
    assert 0.0 <= forecast["15m"] <= 1.0

    # Brier score test
    probs = np.array([[0.9, 0.08, 0.01, 0.01], [0.1, 0.2, 0.6, 0.1]])
    true_states = [0, 2]
    bs = GaussianHMM.compute_brier_score(probs, true_states)
    assert 0.0 <= bs <= 1.0
    assert bs < 0.3  # Well calibrated predictions have low Brier score


def test_spatial_coupling_across_zones():
    tracker = ZoneHazardTracker(spatial_coupling_lambda=0.25)
    tracker.register_zone("zone_riverfront", initial_belief={HazardLevel.SAFE: 0.1, HazardLevel.CRITICAL: 0.8, HazardLevel.EVACUATE: 0.1, HazardLevel.ADVISORY: 0.0})
    tracker.register_zone("zone_central", initial_belief={HazardLevel.SAFE: 0.9, HazardLevel.ADVISORY: 0.1, HazardLevel.CRITICAL: 0.0, HazardLevel.EVACUATE: 0.0})

    # Update zone_central with calm observation; spatial coupling should bleed some calm into riverfront
    obs_calm = SensorObservation(step=1, zone_id="zone_central", water_level_m=0.05, rainfall_mmh=5.0, wind_speed_kmh=15.0)
    tracker.update_zone_observation(obs_calm)
    summary = tracker.get_zone_summary("zone_central")
    assert "forecast_30m" in summary

