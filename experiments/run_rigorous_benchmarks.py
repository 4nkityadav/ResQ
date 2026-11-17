"""
experiments.run_rigorous_benchmarks
===================================
Runs comprehensive, reproducible benchmarks across all core algorithms:
1. Multi-Vehicle Dispatch Optimization (30 seeds):
   - Compares: Random, Greedy/Nearest-Neighbor, Clarke-Wright Savings, Simulated Annealing, ALNS
   - Computes: Mean, Std Dev, 95% Confidence Intervals, Paired t-tests (p-values)
2. D* Lite vs Cold-Start A* Benchmark (100 obstacle injections):
   - Measures: Replan latency (ms), node expansions, path cost optimality gap (%)
3. HMM Tracking & Calibration Benchmark:
   - Measures: State classification accuracy, Brier calibration score, forecast latency
4. End-to-End Incident Command Replan Latency Benchmark:
   - Verifies end-to-end mission loop latency remains strictly < 100 ms on 50x50 urban grid
"""

from __future__ import annotations
import math
import time
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Ensure project root in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from resq_ai.core.models import Vehicle, VehicleType, DistressCall, OccupantType, PriorityLevel
from resq_ai.sim.city_grid import CityGrid
from resq_ai.dstar.router import FleetRouter
from resq_ai.dstar.dstar_lite import DStarLite
from resq_ai.dstar.astar import AStar
from resq_ai.annealing.state import ScheduleState
from resq_ai.annealing.simulated_annealing import SimulatedAnnealingOptimizer
from resq_ai.annealing.alns import ALNSOptimizer
from resq_ai.annealing.baselines import GreedyAllocator, RandomAllocator, ClarkeWrightSavingsAllocator
from resq_ai.hmm.continuous_hmm import GaussianHMM
from resq_ai.orchestrator.engine import OrchestratorEngine


def run_dispatch_benchmark(num_seeds: int = 30) -> Dict[str, Any]:
    print(f"\n[1/4] Running Fleet Dispatch Benchmark over {num_seeds} randomized seeds...")
    methods = ["Random", "Greedy_NN", "Clarke_Wright", "Simulated_Annealing", "ALNS"]
    costs: Dict[str, List[float]] = {m: [] for m in methods}
    times_ms: Dict[str, List[float]] = {m: [] for m in methods}

    for seed_idx in range(num_seeds):
        seed = 1000 + seed_idx
        grid = CityGrid(width=50, height=50, seed=seed)
        rng = np.random.default_rng(seed)

        # 5 Ambulances, 3 Boats
        vehicles: Dict[str, Vehicle] = {}
        amb_locs = [(5, 5), (45, 5), (10, 20), (40, 20), (25, 10)]
        for i in range(5):
            vid = f"AMB-{i+1}"
            vehicles[vid] = Vehicle(
                vehicle_id=vid, vehicle_type=VehicleType.AMBULANCE, capacity=2, speed_kmh=45.0,
                base_location=amb_locs[i], current_location=amb_locs[i]
            )
        boat_locs = [(8, 35), (20, 33), (32, 34)]
        for i in range(3):
            vid = f"BOAT-{i+1}"
            vehicles[vid] = Vehicle(
                vehicle_id=vid, vehicle_type=VehicleType.RESCUE_BOAT, capacity=4, speed_kmh=30.0,
                base_location=boat_locs[i], current_location=boat_locs[i]
            )

        # 20 Distress Calls (15 land, 5 flood)
        calls: Dict[str, DistressCall] = {}
        for c_idx in range(15):
            cid = f"CALL-LAND-{c_idx+1}"
            calls[cid] = DistressCall(
                call_id=cid,
                location=(int(rng.integers(2, 48)), int(rng.integers(2, 30))),
                timestamp_step=1,
                caller_name=f"Resident {c_idx+1}",
                occupants_type=OccupantType.ELDERLY if rng.random() < 0.4 else OccupantType.ADULTS,
                occupant_count=int(rng.integers(1, 3)),
                building_age="Old" if rng.random() < 0.5 else "Moderate",
                construction_type="Wood" if rng.random() < 0.5 else "Masonry",
                reported_flood_m=float(rng.uniform(0.0, 0.25)),
                priority=PriorityLevel.P1 if rng.random() < 0.3 else PriorityLevel.P2,
                risk_score=float(round(rng.uniform(5.0, 9.5), 1)),
            )

        for c_idx in range(5):
            cid = f"CALL-WATER-{c_idx+1}"
            r_cell = grid.river_cells[int(rng.integers(0, len(grid.river_cells)))]
            calls[cid] = DistressCall(
                call_id=cid,
                location=r_cell,
                timestamp_step=1,
                caller_name=f"Marina Victim {c_idx+1}",
                occupants_type=OccupantType.CHILDREN if rng.random() < 0.5 else OccupantType.ADULTS,
                occupant_count=int(rng.integers(1, 4)),
                building_age="Old",
                construction_type="Wood",
                reported_flood_m=float(round(rng.uniform(0.8, 2.2), 2)),
                priority=PriorityLevel.P1,
                risk_score=float(round(rng.uniform(8.5, 10.0), 1)),
            )

        router = FleetRouter(grid)
        cost_matrix = router.compute_cost_matrix(list(vehicles.values()), list(calls.values()))

        # 1. Random
        t0 = time.perf_counter()
        _, c_rand = RandomAllocator.allocate(vehicles, calls, cost_matrix=cost_matrix, seed=seed)
        t_rand = (time.perf_counter() - t0) * 1000.0
        costs["Random"].append(c_rand)
        times_ms["Random"].append(t_rand)

        # 2. Greedy NN
        t0 = time.perf_counter()
        _, c_greedy = GreedyAllocator.allocate(vehicles, calls, cost_matrix=cost_matrix)
        t_greedy = (time.perf_counter() - t0) * 1000.0
        costs["Greedy_NN"].append(c_greedy)
        times_ms["Greedy_NN"].append(t_greedy)

        # 3. Clarke-Wright Savings
        t0 = time.perf_counter()
        _, c_cw = ClarkeWrightSavingsAllocator.allocate(vehicles, calls, cost_matrix=cost_matrix)
        t_cw = (time.perf_counter() - t0) * 1000.0
        costs["Clarke_Wright"].append(c_cw)
        times_ms["Clarke_Wright"].append(t_cw)

        # 4. Simulated Annealing
        t0 = time.perf_counter()
        init_state = ScheduleState(vehicles=vehicles, calls=calls, cost_matrix=cost_matrix)
        sa = SimulatedAnnealingOptimizer(max_iterations=600, seed=seed)
        _, c_sa, _ = sa.optimize(init_state, max_time_sec=0.25)
        t_sa = (time.perf_counter() - t0) * 1000.0
        costs["Simulated_Annealing"].append(c_sa)
        times_ms["Simulated_Annealing"].append(t_sa)

        # 5. ALNS
        t0 = time.perf_counter()
        alns = ALNSOptimizer(max_iterations=300, seed=seed)
        _, c_alns, _ = alns.optimize(init_state, max_time_sec=0.25)
        t_alns = (time.perf_counter() - t0) * 1000.0
        costs["ALNS"].append(c_alns)
        times_ms["ALNS"].append(t_alns)

    # Statistical summary
    summary = {}
    for m in methods:
        arr = np.array(costs[m])
        t_arr = np.array(times_ms[m])
        mean_c = float(np.mean(arr))
        std_c = float(np.std(arr, ddof=1))
        ci95 = float(1.96 * (std_c / math.sqrt(num_seeds)))
        summary[m] = {
            "mean_cost": round(mean_c, 2),
            "std_cost": round(std_c, 2),
            "ci95_cost": round(ci95, 2),
            "mean_time_ms": round(float(np.mean(t_arr)), 2),
        }

    # Hypothesis tests: Paired t-tests comparing ALNS vs Greedy and SA vs Greedy
    t_stat_alns, p_val_alns = stats.ttest_rel(costs["ALNS"], costs["Greedy_NN"])
    t_stat_sa, p_val_sa = stats.ttest_rel(costs["Simulated_Annealing"], costs["Greedy_NN"])

    summary["tests"] = {
        "alns_vs_greedy_p_value": float(p_val_alns),
        "alns_vs_greedy_t_stat": float(t_stat_alns),
        "sa_vs_greedy_p_value": float(p_val_sa),
        "sa_vs_greedy_t_stat": float(t_stat_sa),
    }

    # Generate publication plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)

    # Boxplot of costs
    box_data = [costs[m] for m in methods]
    labels = ["Random", "Greedy NN", "Clarke-Wright", "Sim. Annealing", "ALNS"]
    colors = ["#94A3B8", "#F59E0B", "#3B82F6", "#10B981", "#06B6D4"]

    bplot = ax1.boxplot(box_data, patch_artist=True, tick_labels=labels)
    for patch, col in zip(bplot["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)

    ax1.set_title("Dispatch Objective Cost by Method (30 Seeds)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Total Objective Cost (Lower is Better)", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Execution Time vs Quality Scatter / Bar
    means = [summary[m]["mean_cost"] for m in methods]
    errs = [summary[m]["ci95_cost"] for m in methods]
    bars = ax2.bar(labels, means, yerr=errs, capsize=5, color=colors, alpha=0.85, edgecolor="#334155")
    ax2.set_title("Mean Cost with 95% Confidence Intervals", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Mean Objective Cost", fontsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, m in zip(bars, methods):
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 150, f"{yval:.0f}\n±{summary[m]['ci95_cost']:.0f}",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig("experiments/dispatch_benchmarks.png")
    plt.close()

    print(f"Dispatch Benchmark Complete! ALNS Mean: {summary['ALNS']['mean_cost']} ± {summary['ALNS']['ci95_cost']} (p < {p_val_alns:.1e})")
    return summary


def run_dstar_vs_astar_benchmark(num_trials: int = 100) -> Dict[str, Any]:
    print(f"\n[2/4] Running D* Lite vs A* Replan Benchmark over {num_trials} dynamic obstacle injections...")
    grid = CityGrid(width=50, height=50, seed=42)
    start = (5, 5)
    goal = (42, 42)
    vtype = VehicleType.AMBULANCE

    dstar = DStarLite(grid, start=start, goal=goal, vehicle_type=vtype)
    init_path, init_cost, _ = dstar.plan()

    rng = np.random.default_rng(42)
    dstar_times_ms = []
    astar_times_ms = []
    optimality_gaps = []
    dstar_expansions = []
    astar_expansions = []

    curr_path = list(init_path)

    for t in range(num_trials):
        # Pick random road cell to block
        rx = int(rng.integers(2, 48))
        ry = int(rng.integers(2, 30))
        grid.set_blocked(rx, ry, True)

        # D* Lite incremental replan
        t0 = time.perf_counter()
        dstar.notify_cell_changed((rx, ry))
        d_path, d_cost, d_exp = dstar.replan()
        t_dstar = (time.perf_counter() - t0) * 1000.0

        # Cold-start A*
        t0 = time.perf_counter()
        astar = AStar(grid, vehicle_type=vtype)
        a_path, a_cost, a_exp = astar.search(start, goal)
        t_astar = (time.perf_counter() - t0) * 1000.0

        dstar_times_ms.append(t_dstar)
        astar_times_ms.append(t_astar)
        dstar_expansions.append(len(d_exp))
        astar_expansions.append(len(a_exp))

        if not math.isinf(a_cost) and not math.isinf(d_cost) and a_cost > 0:
            gap = abs(d_cost - a_cost) / a_cost
            optimality_gaps.append(gap)

    mean_dstar_t = float(np.mean(dstar_times_ms))
    mean_astar_t = float(np.mean(astar_times_ms))
    speedup = mean_astar_t / max(mean_dstar_t, 1e-4)
    mean_gap = float(np.mean(optimality_gaps)) if optimality_gaps else 0.0

    summary = {
        "trials": num_trials,
        "dstar_mean_time_ms": round(mean_dstar_t, 3),
        "astar_mean_time_ms": round(mean_astar_t, 3),
        "dstar_speedup_factor": round(speedup, 2),
        "mean_optimality_gap_pct": round(mean_gap * 100.0, 4),
        "dstar_mean_expansions": round(float(np.mean(dstar_expansions)), 1),
        "astar_mean_expansions": round(float(np.mean(astar_expansions)), 1),
    }

    print(f"D* Lite: {mean_dstar_t:.2f} ms vs A*: {mean_astar_t:.2f} ms ({speedup:.1f}x faster, gap: {mean_gap*100:.3f}%)")
    return summary


def run_hmm_calibration_benchmark() -> Dict[str, Any]:
    print("\n[3/4] Running HMM Calibration & Forecast Benchmark...")
    hmm = GaussianHMM(seed=42)
    rng = np.random.default_rng(42)

    # Synthesize realistic time series across 4 stages (Safe, Advisory, Critical, Evacuate)
    T = 200
    true_states = []
    observations = []

    for t in range(T):
        if t < 50:
            st = 0
            obs = rng.normal([0.05, 5.0, 15.0], [0.02, 2.0, 3.0])
        elif t < 100:
            st = 1
            obs = rng.normal([0.25, 25.0, 35.0], [0.05, 4.0, 5.0])
        elif t < 150:
            st = 2
            obs = rng.normal([0.75, 65.0, 60.0], [0.08, 6.0, 6.0])
        else:
            st = 3
            obs = rng.normal([1.60, 110.0, 85.0], [0.15, 10.0, 10.0])

        true_states.append(st)
        observations.append(obs)

    obs_matrix = np.array(observations)

    # Filtered probabilities
    alpha, _, log_lik = hmm.forward(obs_matrix)
    pred_states = np.argmax(alpha, axis=1)

    acc = float(np.mean(pred_states == true_states))
    brier = GaussianHMM.compute_brier_score(alpha, true_states)

    # 15m/30m forecast latency
    t0 = time.perf_counter()
    for _ in range(100):
        _ = hmm.forecast_critical_risk(alpha[-1])
    forecast_latency_us = ((time.perf_counter() - t0) / 100.0) * 1e6

    summary = {
        "sequence_length": T,
        "classification_accuracy": round(acc * 100.0, 2),
        "brier_score": round(brier, 4),
        "forecast_latency_microseconds": round(forecast_latency_us, 2),
    }

    print(f"HMM Accuracy: {acc*100:.1f}%, Brier Score: {brier:.4f}, Forecast: {forecast_latency_us:.1f} µs")
    return summary


def run_orchestrator_latency_benchmark(num_steps: int = 15) -> Dict[str, Any]:
    print(f"\n[4/4] Running End-to-End Orchestrator Loop Latency Benchmark ({num_steps} steps)...")
    engine = OrchestratorEngine(seed=42)

    step_times_ms = []
    for _ in range(num_steps):
        t0 = time.perf_counter()
        engine.step()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        step_times_ms.append(elapsed_ms)

    # NLP Dispatch latency
    t0 = time.perf_counter()
    resp = engine.handle_natural_language_command("Dispatch boat to Block C. Reports say Main Street is submerged.")
    nlp_dispatch_ms = (time.perf_counter() - t0) * 1000.0

    mean_step_ms = float(np.mean(step_times_ms))
    p95_step_ms = float(np.percentile(step_times_ms, 95))

    summary = {
        "mean_step_latency_ms": round(mean_step_ms, 2),
        "p95_step_latency_ms": round(p95_step_ms, 2),
        "nlp_dispatch_latency_ms": round(nlp_dispatch_ms, 2),
        "strictly_under_100ms": bool(p95_step_ms < 100.0 and nlp_dispatch_ms < 100.0),
    }

    print(f"Orchestrator Step Latency: Mean = {mean_step_ms:.2f} ms, P95 = {p95_step_ms:.2f} ms, NLP Replan = {nlp_dispatch_ms:.2f} ms")
    return summary


def main():
    print("=================================================================")
    print("      RESQ-AI: COMPREHENSIVE EMPIRICAL ACCEPTANCE BENCHMARKS     ")
    print("=================================================================")
    
    dispatch_results = run_dispatch_benchmark(num_seeds=30)
    dstar_results = run_dstar_vs_astar_benchmark(num_trials=100)
    hmm_results = run_hmm_calibration_benchmark()
    orch_results = run_orchestrator_latency_benchmark(num_steps=15)

    all_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dispatch": dispatch_results,
        "dstar": dstar_results,
        "hmm": hmm_results,
        "orchestrator_latency": orch_results,
    }

    out_file = Path("experiments/benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\nAll benchmark results successfully written to experiments/benchmark_results.json")


if __name__ == "__main__":
    main()
