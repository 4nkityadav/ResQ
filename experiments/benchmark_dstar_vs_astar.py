"""
experiments.benchmark_dstar_vs_astar
====================================
Empirical benchmark comparing D* Lite (incremental repair) vs Static A* (re-search from scratch)
over 100+ random obstacle/flooding change events on a 50x50 city grid.
Generates metrics table and benchmark plot.
"""

from __future__ import annotations
import time
import math
import os
import sys
from typing import List, Dict, Any

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from resq_ai.core.models import VehicleType
from resq_ai.sim.city_grid import CityGrid
from resq_ai.dstar.dstar_lite import DStarLite
from resq_ai.dstar.astar import AStar


def run_benchmark(num_events: int = 100, seed: int = 42) -> Dict[str, Any]:
    print(f"--- Running D* Lite vs Static A* Benchmark ({num_events} Edge Change Events) ---")
    grid = CityGrid(width=50, height=50, seed=seed)
    rng = np.random.default_rng(seed)

    start = (5, 5)
    goal = (45, 45)
    vtype = VehicleType.AMBULANCE

    # Initial plans
    astar = AStar(grid, vehicle_type=vtype)
    dstar = DStarLite(grid, start=start, goal=goal, vehicle_type=vtype)

    t0 = time.perf_counter()
    init_path_dstar, init_cost_dstar, init_exp_dstar = dstar.plan()
    init_dstar_time = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    init_path_astar, init_cost_astar, init_exp_astar = astar.search(start, goal)
    init_astar_time = (time.perf_counter() - t0) * 1000.0

    print(f"Initial Plan: A* Cost={init_cost_astar:.2f} ({len(init_exp_astar)} nodes, {init_astar_time:.2f}ms) | "
          f"D* Cost={init_cost_dstar:.2f} ({len(init_exp_dstar)} nodes, {init_dstar_time:.2f}ms)")

    events_data = []
    current_path = init_path_dstar.copy()

    for i in range(num_events):
        # Pick cell to change: 60% chance pick a cell on or adjacent to current active path, 40% random
        if current_path and len(current_path) > 4 and rng.random() < 0.60:
            target_idx = int(rng.integers(2, len(current_path) - 2))
            cx, cy = current_path[target_idx]
        else:
            cx = int(rng.integers(5, 45))
            cy = int(rng.integers(5, 45))

        cell = grid.get_cell(cx, cy)
        if not cell or cell.coord == start or cell.coord == goal:
            continue

        # Toggle blockage or increase water level
        was_blocked = cell.is_blocked
        cell.is_blocked = not was_blocked

        # 1. Measure D* Lite incremental replan
        t_dstar_0 = time.perf_counter()
        dstar.notify_cell_changed((cx, cy))
        dstar_path, dstar_cost, dstar_expanded = dstar.replan()
        dstar_time_ms = (time.perf_counter() - t_dstar_0) * 1000.0

        # 2. Measure A* search from scratch
        t_astar_0 = time.perf_counter()
        astar_path, astar_cost, astar_expanded = astar.search(start, goal)
        astar_time_ms = (time.perf_counter() - t_astar_0) * 1000.0

        if dstar_path:
            current_path = dstar_path.copy()

        events_data.append({
            "event": i + 1,
            "dstar_nodes": len(dstar_expanded),
            "astar_nodes": len(astar_expanded),
            "dstar_time_ms": dstar_time_ms,
            "astar_time_ms": astar_time_ms,
            "dstar_cost": dstar_cost,
            "astar_cost": astar_cost,
        })

    # Summary statistics
    avg_dstar_nodes = float(np.mean([e["dstar_nodes"] for e in events_data]))
    avg_astar_nodes = float(np.mean([e["astar_nodes"] for e in events_data]))
    avg_dstar_time = float(np.mean([e["dstar_time_ms"] for e in events_data]))
    avg_astar_time = float(np.mean([e["astar_time_ms"] for e in events_data]))
    node_reduction = ((avg_astar_nodes - avg_dstar_nodes) / max(1.0, avg_astar_nodes)) * 100.0
    speedup = avg_astar_time / max(0.001, avg_dstar_time)

    print("\n================ BENCHMARK SUMMARY (100 Events) ================")
    print(f"| Metric                   | D* Lite (Incremental) | Static A* (Scratch)  | Improvement    |")
    print(f"|--------------------------|-----------------------|----------------------|----------------|")
    print(f"| Mean Nodes Expanded      | {avg_dstar_nodes:21.1f} | {avg_astar_nodes:20.1f} | {node_reduction:6.1f}% fewer |")
    print(f"| Mean Execution Time (ms) | {avg_dstar_time:21.2f} | {avg_astar_time:20.2f} | {speedup:6.1f}x speedup |")
    print(f"=================================================================\n")

    # Generate Publication Plot
    plot_path = os.path.join(os.path.dirname(__file__), "dstar_vs_astar_benchmark.png")
    _plot_benchmark_results(events_data, plot_path)
    print(f"Benchmark plot successfully saved to: {plot_path}")

    return {
        "avg_dstar_nodes": avg_dstar_nodes,
        "avg_astar_nodes": avg_astar_nodes,
        "avg_dstar_time_ms": avg_dstar_time,
        "avg_astar_time_ms": avg_astar_time,
        "node_reduction_pct": node_reduction,
        "speedup_factor": speedup,
        "plot_path": plot_path,
    }


def _plot_benchmark_results(events_data: List[Dict[str, Any]], out_path: str) -> None:
    events = [e["event"] for e in events_data]
    dstar_nodes = [e["dstar_nodes"] for e in events_data]
    astar_nodes = [e["astar_nodes"] for e in events_data]
    dstar_time = [e["dstar_time_ms"] for e in events_data]
    astar_time = [e["astar_time_ms"] for e in events_data]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")

    # Plot 1: Nodes Expanded
    ax1.plot(events, astar_nodes, label="Static A* (Scratch)", color="#ef4444", alpha=0.75, linewidth=1.5)
    ax1.plot(events, dstar_nodes, label="D* Lite (Local Repair)", color="#06b6d4", linewidth=1.8)
    ax1.set_ylabel("Nodes Expanded", fontsize=11, fontweight="bold")
    ax1.set_title("D* Lite vs Static A*: Node Expansions over 100 Dynamic Edge Changes (50x50 Grid)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Wall-Clock Latency
    ax2.plot(events, astar_time, label="Static A* Latency (ms)", color="#f97316", alpha=0.75, linewidth=1.5)
    ax2.plot(events, dstar_time, label="D* Lite Latency (ms)", color="#10b981", linewidth=1.8)
    ax2.set_xlabel("Dynamic Obstacle / Flooding Event Sequence", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Execution Latency (ms)", fontsize=11, fontweight="bold")
    ax2.set_yscale("log")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    run_benchmark(num_events=100)
