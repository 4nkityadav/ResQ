"""
experiments.benchmark_hmm_tracking
==================================
Evaluates HMM online state tracking accuracy, filtered belief state progression,
and alert lead-time against ground-truth disaster escalation trajectories.
Generates publication plot: experiments/hmm_tracking_accuracy.png
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from resq_ai.core.models import HazardLevel, SensorObservation
from resq_ai.sim.disaster_sim import DisasterSimulator
from resq_ai.hmm.tracker import ZoneHazardTracker


def evaluate_hmm_tracking(steps: int = 50, seed: int = 42):
    print(f"\n--- Running HMM State Tracking Evaluation ({steps} Timesteps) ---")
    sim = DisasterSimulator(seed=seed)
    tracker = ZoneHazardTracker(escalation_threshold=0.60)

    for zid in sim.grid.zones:
        tracker.register_zone(zid)

    zone_target = "zone_riverfront"
    ground_truth = []
    filtered_p_crit_evac = []
    map_states = []
    observations = []

    state_to_num = {
        HazardLevel.SAFE: 0,
        HazardLevel.ADVISORY: 1,
        HazardLevel.CRITICAL: 2,
        HazardLevel.EVACUATE: 3,
    }

    correct_predictions = 0

    for step_idx in range(1, steps + 1):
        events = sim.step()
        gt_state = sim.ground_truth_hazard[zone_target]
        ground_truth.append(state_to_num[gt_state])

        # Find sensor event for target zone
        for ev in events:
            if ev.event_type.value == "SENSOR_UPDATE" and ev.payload["zone_id"] == zone_target:
                obs: SensorObservation = ev.payload["observation"]
                belief, alert, summary = tracker.update_zone_observation(obs)
                p_crit = belief[HazardLevel.CRITICAL] + belief[HazardLevel.EVACUATE]
                filtered_p_crit_evac.append(p_crit)
                observations.append(obs.water_level_m)

                # MAP state
                map_idx = max(range(4), key=lambda i: list(belief.values())[i])
                map_states.append(map_idx)
                if map_idx == state_to_num[gt_state]:
                    correct_predictions += 1
                break

    accuracy = (correct_predictions / steps) * 100.0

    print("=================== HMM TRACKING METRICS ===================")
    print(f"Target Zone:                 {zone_target}")
    print(f"Evaluation Steps:            {steps}")
    print(f"MAP State Accuracy:          {accuracy:.1f}%")
    print(f"Final Filtered P(Crit/Evac): {filtered_p_crit_evac[-1]:.4f}")
    print("============================================================\n")

    # Plot
    plot_path = os.path.join(os.path.dirname(__file__), "hmm_tracking_accuracy.png")
    _plot_hmm_results(steps, ground_truth, map_states, filtered_p_crit_evac, observations, plot_path)
    print(f"HMM evaluation plot saved to: {plot_path}")


def _plot_hmm_results(steps, gt, map_s, p_crit, water_obs, out_path):
    time_steps = list(range(1, steps + 1))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Subplot 1: Ground Truth vs Decoded MAP state
    ax1.step(time_steps, gt, label="Ground Truth Severity", color="#ef4444", linewidth=2.2, where="post")
    ax1.plot(time_steps, map_s, "o--", label="HMM Filtered MAP State", color="#06b6d4", alpha=0.85, markersize=5)
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(["Safe", "Advisory", "Critical", "Evacuate"], fontsize=9, fontweight="bold")
    ax1.set_ylabel("Hazard State", fontsize=10, fontweight="bold")
    ax1.set_title("HMM Real-Time Online State Tracking vs Ground Truth (Riverfront Zone)", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Probability of Escalation
    ax2.plot(time_steps, p_crit, color="#f59e0b", linewidth=2.0, label="P(Critical or Evacuate | y_1:t)")
    ax2.axhline(0.60, color="#ef4444", linestyle=":", label="Alert Threshold (0.60)")
    ax2.set_xlabel("Simulation Timestep (minutes)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Belief Probability", fontsize=10, fontweight="bold")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    evaluate_hmm_tracking(steps=50)
