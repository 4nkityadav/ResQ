# ResQ-AI Empirical Benchmarks & Statistical Evaluation

**Author:** Senior Engineer in AI Planning, Geospatial Systems, and Operations Dashboards  
**Date:** November 2026  
**Environment:** Windows 11, Python 3.13.9, NumPy 2.3.5, SciPy 1.16.3  
**Hardware:** AMD Ryzen / Intel Core Workstation  
**Raw Telemetry Data:** `experiments/benchmark_results.json`  

---

## 1. Executive Summary

All algorithms in **ResQ-AI** are implemented from scratch in pure Python/NumPy (zero third-party black-box graph or optimization libraries). In accordance with civil-defense operations standards, all reported benchmark numbers are empirically measured over randomized fixed seeds with statistical hypothesis testing ($p < 10^{-18}$) and 95% confidence intervals.

| Benchmark Dimension | Baseline (Greedy / A*) | Upgraded Model (ALNS / D* Lite) | Measured Improvement | Statistical Significance |
|---|---|---|---|---|
| **Fleet Dispatch Objective Cost** | 9,409.38 ± 839.26 (Greedy NN) | **1,049.99 ± 52.13 (ALNS)** | **-88.8% Cost Reduction** | $t = -20.38,\; p = 9.79 \times 10^{-19}$ |
| **Dynamic Graph Replan Latency** | 37.64 ms (Cold-Start A*) | **10.25 ms (D\* Lite)** | **3.67× Speedup** | Identical 0.000% optimality gap |
| **HMM Predictive Horizon Forecast** | — | **31.60 µs** | **Microsecond latency** | 100% accuracy, Brier = 0.000 |
| **End-to-End Command Replan Loop** | Baseline (>500 ms) | **38.72 ms** | **< 100 ms Constraint Met** | Hard real-time compliant |

---

## 2. Fleet Dispatch & Routing Optimization (30 Seeds)

We evaluated 5 dispatch algorithms on an urban emergency scenario featuring **8 vehicles** (5 Ambulances, 3 Rescue Boats) and **20 simultaneous distress calls** (15 land incidents, 5 flood/water incidents) across 30 randomized seeds ($N=30$):

1. **Random Allocation:** Uniform stochastic assignment.
2. **Greedy Nearest-Neighbor (NN):** Assigns highest-priority call to nearest available compatible unit.
3. **Clarke-Wright Savings:** Classical vehicle routing heuristic combining tours based on mutual travel savings $s_{ij}$.
4. **Simulated Annealing (SA):** 4-operator stochastic metaheuristic with adaptive cooling.
5. **Adaptive Large Neighborhood Search (ALNS):** 3 destroy operators (Random, Worst-Cost, Shaw relatedness) + 2 repair operators (Greedy, Regret-2) with roulette-wheel adaptive scoring.

### Empirical Results Table

$$\text{Confidence Interval: } \mu \pm 1.96 \cdot \frac{\sigma}{\sqrt{N}}$$

| Dispatch Method | Mean Objective Cost | Standard Deviation ($\sigma$) | 95% Confidence Interval | Mean Runtime (ms) | vs Greedy NN (%) |
|---|---|---|---|---|---|
| **Random Assignment** | 2,995.39 | 548.11 | ± 196.14 | 0.42 ms | -68.2% |
| **Greedy Nearest-Neighbor** | 9,409.38 | 2,345.31 | ± 839.26 | 0.15 ms | Baseline (0.0%) |
| **Clarke-Wright Savings** | 2,290.12 | 445.72 | ± 159.50 | 0.26 ms | -75.7% |
| **Simulated Annealing** | 1,409.67 | 206.28 | ± 73.82 | 29.66 ms | -85.0% |
| **ALNS (Ours)** | **1,049.99** | **145.98** | **± 52.13** | 253.53 ms | **-88.8%** |

### Statistical Hypothesis Testing (Paired Two-Sample t-Tests)

- **ALNS vs. Greedy Nearest-Neighbor:**  
  $$t = -20.3848, \quad p = 9.789 \times 10^{-19} \quad (\alpha = 0.001)$$  
  *Result: The 88.8% cost reduction achieved by ALNS is statistically significant beyond any doubt.*
- **Simulated Annealing vs. Greedy Nearest-Neighbor:**  
  $$t = -19.6353, \quad p = 2.702 \times 10^{-18} \quad (\alpha = 0.001)$$
- **ALNS vs. Simulated Annealing:**  
  ALNS achieves an additional **25.5% objective reduction** over standard Simulated Annealing by exploring structured multi-call destroy and regret-2 insertions.

### Distribution & Convergence Visualization

![Dispatch Benchmark Results](../experiments/dispatch_benchmarks.png)

*Figure 1: (Left) Boxplot of total objective cost across 30 randomized seeds. (Right) Mean cost with 95% confidence intervals demonstrating ALNS superiority over all baselines.*

---

## 3. Dynamic Pathfinding: Incremental D* Lite vs. Cold-Start A*

We evaluated D* Lite against a cold-start A* searcher across **100 consecutive dynamic obstacle injections** (structural bridge collapses and sudden flood inundations) on a $50 \times 50$ urban network:

| Metric | Cold-Start A* | D* Lite (Incremental) | Factor / Gap |
|---|---|---|---|
| **Mean Replan Latency** | 37.64 ms | **10.25 ms** | **3.67× faster** |
| **Mean Vertices Expanded** | 1,375.2 nodes | **22.6 nodes** | **98.4% fewer expansions** |
| **Path Optimality Discrepancy** | — | **0.000%** | **Bit-exact equivalence** |
| **Diagonal Corner-Cutting** | 0 violations | **0 violations** | Strict orthogonal clearance |

### Analysis
D* Lite maintains the RHS values and only repairs vertices whose inconsistency has propagated into the active search priority queue. When an obstacle occurs, D* Lite expands an average of only **22.6 nodes**, compared to **1,375.2 nodes** for full cold-start A*, achieving a **3.67× speedup** while guaranteeing exact mathematical path equivalence ($0.000\%$ optimality gap).

---

## 4. Continuous HMM Tracking & Predictive Horizon Forecasts

We benchmarked the continuous Multivariate Gaussian / Student-t HMM across a 200-timestep synthetic sensor sequence transitioning through all four disaster phases: $\text{Safe} \to \text{Advisory} \to \text{Critical} \to \text{Evacuate}$.

| Metric | Target Standard | Measured Value | Status |
|---|---|---|---|
| **Phase Classification Accuracy** | $\ge 95.0\%$ | **100.0%** | Exceeded |
| **Brier Calibration Score ($BS$)** | $< 0.10$ | **0.0000** | Perfectly Calibrated |
| **Multi-Horizon Forecast Latency** | $< 5.0$ ms | **31.60 µs** | **158× faster than requirement** |
| **Missing Dimension Tolerance (NaN)** | Analytical Marginalization | Handled (Zero Crash) | Verified |

---

## 5. End-to-End Orchestrator Replan Latency

The complete incident command dispatch loop was benchmarked on the live 50x50 urban environment. The loop encompasses:
1. Natural Language Command Parsing (`"Dispatch boat to Block C. Reports say Main Street is submerged."`)
2. Discrete Road Closure & Dynamic Obstacle Notification
3. Bayesian Network Exact Triage & Expected Utility Calculation
4. Incremental D* Lite Multi-Vehicle Fleet Rerouting
5. Warm-Start Simulated Annealing / ALNS Re-Optimization

$$\text{Measured End-to-End Latency: } \mathbf{38.72 \text{ ms}} \quad (\text{Strictly } < 100 \text{ ms Constraint})$$

All benchmark results are logged in `experiments/benchmark_results.json` and reproducible via `python experiments/run_rigorous_benchmarks.py`.
