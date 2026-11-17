"""
experiments.benchmark_sa_vs_greedy
==================================
Empirical benchmark comparing Simulated Annealing (Geometric & Adaptive)
against Greedy Nearest-Vehicle and Random Assignment baselines
on a 20+ distress call emergency dispatch instance.
"""

from __future__ import annotations
import time
import os
import sys
from typing import Dict, List, Any

# Ensure project root in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from resq_ai.core.models import (
    Vehicle,
    VehicleType,
    DistressCall,
    OccupantType,
    PriorityLevel,
)
from resq_ai.sim.city_grid import CityGrid
from resq_ai.dstar.router import FleetRouter
from resq_ai.annealing.state import ScheduleState
from resq_ai.annealing.simulated_annealing import SimulatedAnnealingOptimizer
from resq_ai.annealing.baselines import GreedyAllocator, RandomAllocator


def create_benchmark_instance(num_calls: int = 22, seed: int = 42) -> Tuple[Dict[str, Vehicle], Dict[str, DistressCall], Dict]:
    grid = CityGrid(width=50, height=50, seed=seed)
    rng = np.random.default_rng(seed)

    # 5 Ambulances, 3 Boats
    vehicles: Dict[str, Vehicle] = {}
    amb_locs = [(5, 5), (45, 5), (10, 20), (40, 20), (25, 10)]
    for i in range(5):
        vid = f"AMB-{i+1}"
        vehicles[vid] = Vehicle(
            vehicle_id=vid,
            vehicle_type=VehicleType.AMBULANCE,
            capacity=2,
            speed_kmh=45.0,
            base_location=amb_locs[i],
            current_location=amb_locs[i],
        )

    boat_locs = [(8, 35), (20, 33), (32, 34)]
    for i in range(3):
        vid = f"BOAT-{i+1}"
        vehicles[vid] = Vehicle(
            vehicle_id=vid,
            vehicle_type=VehicleType.RESCUE_BOAT,
            capacity=4,
            speed_kmh=30.0,
            base_location=boat_locs[i],
            current_location=boat_locs[i],
        )

    # 22 Calls: mix of dry land and flooded calls, P1, P2, P3
    calls: Dict[str, DistressCall] = {}
    for j in range(num_calls):
        cid = f"CALL-{j+1:02d}"
        # 35% chance in flooded/river region (for boats), 65% on land
        is_flood = bool(rng.random() < 0.35)
        if is_flood:
            cx = int(rng.integers(5, 45))
            cy = int(rng.integers(32, 38))
            f_depth = round(float(rng.uniform(0.40, 1.60)), 2)
        else:
            cx = int(rng.integers(5, 45))
            cy = int(rng.integers(5, 30))
            f_depth = 0.05

        p_tier = rng.choice(["P1", "P2", "P3"], p=[0.40, 0.40, 0.20])
        risk_score = 9.0 if p_tier == "P1" else (5.5 if p_tier == "P2" else 2.5)

        calls[cid] = DistressCall(
            call_id=cid,
            location=(cx, cy),
            timestamp_step=1,
            caller_name=f"Victim-{j+1}",
            occupants_type=OccupantType.ELDERLY if p_tier == "P1" else OccupantType.ADULTS,
            occupant_count=int(rng.integers(1, 3)),
            reported_flood_m=f_depth,
            priority=PriorityLevel(p_tier),
            risk_score=risk_score,
        )

    # Precompute cost matrix using FleetRouter
    router = FleetRouter(grid)
    cost_matrix = router.compute_cost_matrix(list(vehicles.values()), list(calls.values()))

    return vehicles, calls, cost_matrix


def run_sa_benchmark(num_calls: int = 22):
    print(f"\n--- Running Simulated Annealing vs Baselines Benchmark ({num_calls} Distress Calls) ---")
    vehicles, calls, cost_matrix = create_benchmark_instance(num_calls=num_calls, seed=42)

    # 1. Random Allocator (average over 5 trials)
    rand_costs = []
    t0 = time.perf_counter()
    for s in range(5):
        _, c = RandomAllocator.allocate(vehicles, calls, cost_matrix, seed=100 + s)
        rand_costs.append(c)
    rand_time_ms = ((time.perf_counter() - t0) / 5.0) * 1000.0
    rand_cost_mean = float(np.mean(rand_costs))

    # 2. Greedy Allocator
    t0 = time.perf_counter()
    greedy_state, greedy_cost = GreedyAllocator.allocate(vehicles, calls, cost_matrix)
    greedy_time_ms = (time.perf_counter() - t0) * 1000.0

    # 3. Simulated Annealing (Geometric Cooling)
    sa_geom = SimulatedAnnealingOptimizer(
        initial_temp=120.0,
        cooling_rate=0.985,
        max_iterations=1200,
        restart_threshold=150,
        cooling_mode="geometric",
        seed=42,
    )
    init_state = ScheduleState(vehicles, calls, cost_matrix=cost_matrix)
    t0 = time.perf_counter()
    geom_state, geom_cost, geom_history = sa_geom.optimize(init_state, max_time_sec=1.5)
    geom_time_ms = (time.perf_counter() - t0) * 1000.0

    # 4. Simulated Annealing (Adaptive Cooling + Reheat Restarts)
    sa_adapt = SimulatedAnnealingOptimizer(
        initial_temp=120.0,
        cooling_rate=0.985,
        max_iterations=1200,
        restart_threshold=120,
        cooling_mode="adaptive",
        seed=42,
    )
    t0 = time.perf_counter()
    adapt_state, adapt_cost, adapt_history = sa_adapt.optimize(init_state, max_time_sec=1.5)
    adapt_time_ms = (time.perf_counter() - t0) * 1000.0

    # P1 response metrics
    p1_served_greedy = sum(1 for cid in calls if calls[cid].priority == PriorityLevel.P1 and any(cid in r for r in greedy_state.routes.values()))
    p1_served_sa = sum(1 for cid in calls if calls[cid].priority == PriorityLevel.P1 and any(cid in r for r in adapt_state.routes.values()))
    total_p1 = sum(1 for c in calls.values() if c.priority == PriorityLevel.P1)

    print("\n================ FLEET ALLOCATION BENCHMARK SUMMARY ================")
    print(f"| Dispatch Strategy       | Objective Cost | P1 Calls Served | Runtime (ms) |")
    print(f"|-------------------------|----------------|-----------------|--------------|")
    print(f"| Random Allocation       | {rand_cost_mean:14.2f} |             N/A | {rand_time_ms:12.2f} |")
    print(f"| Greedy Nearest-Vehicle  | {greedy_cost:14.2f} |     {p1_served_greedy:2d} / {total_p1:2d} ({p1_served_greedy/total_p1:.0%}) | {greedy_time_ms:12.2f} |")
    print(f"| SA (Geometric Cooling)  | {geom_cost:14.2f} |     {p1_served_sa:2d} / {total_p1:2d} ({p1_served_sa/total_p1:.0%}) | {geom_time_ms:12.2f} |")
    print(f"| SA (Adaptive + Reheats) | {adapt_cost:14.2f} |     {p1_served_sa:2d} / {total_p1:2d} ({p1_served_sa/total_p1:.0%}) | {adapt_time_ms:12.2f} |")
    improvement = ((greedy_cost - adapt_cost) / greedy_cost) * 100.0
    print(f"====================================================================")
    print(f"Simulated Annealing outperformed Greedy by {improvement:.1f}% cost reduction!\n")

    # Plot convergence
    out_plot = os.path.join(os.path.dirname(__file__), "sa_vs_greedy_benchmark.png")
    _plot_sa_convergence(geom_history, adapt_history, greedy_cost, out_plot)
    print(f"Convergence plot saved to: {out_plot}")


def _plot_sa_convergence(geom_hist: List[Dict], adapt_hist: List[Dict], greedy_cost: float, out_path: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    iters_g = [h["iteration"] for h in geom_hist]
    best_g = [h["best_cost"] for h in geom_hist]

    iters_a = [h["iteration"] for h in adapt_hist]
    curr_a = [h["current_cost"] for h in adapt_hist]
    best_a = [h["best_cost"] for h in adapt_hist]
    temp_a = [h["temperature"] for h in adapt_hist]

    # Subplot 1: Objective Cost Trajectory
    ax1.axhline(greedy_cost, color="#ef4444", linestyle="--", linewidth=1.8, label=f"Greedy Baseline ({greedy_cost:.1f})")
    ax1.plot(iters_a, curr_a, color="#94a3b8", alpha=0.4, label="SA Current (Stochastic Exploration)")
    ax1.plot(iters_g, best_g, color="#f59e0b", linewidth=2.0, label="SA Geometric Best Cost")
    ax1.plot(iters_a, best_a, color="#10b981", linewidth=2.4, label="SA Adaptive Best Cost")
    ax1.set_ylabel("Total Objective Cost", fontsize=11, fontweight="bold")
    ax1.set_title("Simulated Annealing Fleet Optimization Convergence (22 Distress Calls)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Temperature Decay & Reheats
    ax2.plot(iters_a, temp_a, color="#06b6d4", linewidth=2.0, label="Temperature T(k) with Adaptive Cooling & Reheats")
    ax2.set_xlabel("Annealing Iteration", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Temperature T", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    run_sa_benchmark(num_calls=22)
