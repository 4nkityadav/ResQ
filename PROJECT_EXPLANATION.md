# ResQ-AI: Complete Project Architecture, Technologies & System Guide

**System Name:** ResQ-AI (Smart Emergency & Disaster Logistics Assistant)  
**Classification:** Multi-Module Civil Defense Incident Command & Autonomous Dispatch Assistant  
**Core Implementation Philosophy:** 100% from-scratch algorithmic implementation in standard Python and NumPy. Zero black-box AI/ML packages (no Scikit-learn, no pgmpy, no networkx, no ortools). Every mathematical equation, factor product, priority queue, and cooling schedule is implemented explicitly from first principles.  
**Verification:** 35/35 passing unit tests • Sub-10ms end-to-end pipeline latency • Interactive full-fledged Streamlit Incident Command Dashboard.

---

## 1. Executive Summary & Problem Formulation

During major urban emergencies—such as flash floods, severe earthquakes, or industrial fires—municipal emergency operations centers (EOC) face four simultaneous, compound challenges:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 THE COMPOUND CRISIS                                    │
├────────────────────────────┬─────────────────────────────┬─────────────────────────────┤
│ 1. Sensory Uncertainty     │ 2. Medical Triage Chaos     │ 3. Dynamic Topology Shifts  │
│ Stream gauges & sonars are │ Panic-stricken 911 callers  │ Floods submerge roads and   │
│ noisy, delayed, and        │ report conflicting, missing │ bridges collapse while      │
│ intermittently degraded.   │ severity symptoms.          │ units are en route.         │
├────────────────────────────┴─────────────────────────────┴─────────────────────────────┤
│ 4. Fleet Logistics NP-Hardness: Multi-Depot Vehicle Routing with heterogeneous units    │
│ (Ambulances requiring roads vs. Rescue Boats navigating water) cannot be solved         │
│ effectively with greedy nearest-neighbor heuristics without trapping vehicles.          │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

ResQ-AI addresses this compound crisis by chaining **four foundational classical AI algorithms** into a real-time, event-driven incident command pipeline:

```mermaid
flowchart TD
    Sensors["🌊 Environmental Sensors<br/>(Rainfall mm/h, River Depth m, Wind km/h)"] --> HMM["📡 1. Continuous Gaussian HMM<br/>(Escalation State & Hazard Tracking)"]
    Calls["📞 911 Distress Calls<br/>(Demographics, Building Type, Water Depth, Fire)"] --> BN["🩺 2. 10-Node Bayesian Network<br/>(Exact Variable Elimination Triage: P1, P2, P3)"]
    HMM -->|Zone Threat Prior| BN
    Grid["🗺️ 50x50 Urban Grid<br/>(Submerged Streets & Collapsed Bridges)"] --> DStar["🧭 3. D* Lite Incremental Search<br/>(Inverted Graph Search & Local Vertex Repair)"]
    BN -->|Triaged Casualties & Risk Weights| SA["🚑 4. Simulated Annealing Dispatch<br/>(Heterogeneous Fleet: 5 Ambulances, 3 Boats)"]
    DStar -->|Cost Matrix & Waypoint Paths| SA
    SA --> Fleet["🚨 Tactical Field Execution<br/>(Units Navigate, Arrive on Scene, Rescue Citizens)"]
```

---

## 2. Technology Stack (What We Are Using)

ResQ-AI is constructed using a modern, lightweight, and robust Python stack selected for numerical speed, mathematical transparency, and zero black-box dependencies:

| Technology / Library | Version | Role in Project | Why It Was Chosen |
| :--- | :--- | :--- | :--- |
| **Python** | 3.11+ (verified 3.13) | Core Programming Language | Strong standard library, native typing, cross-platform stability. |
| **NumPy** | `>=1.26.0` | High-Performance Matrix Math | Used for all tensor factor products, factor marginalization, multivariate Gaussian density computation, covariance inversion, and distance calculations. |
| **SciPy** | `>=1.12.0` | Numerical Stability & Math Utilities | Used for robust log-space calculations and matrix operations. |
| **PyYAML** | `>=6.0.1` | Scenario & Configuration Parser | Human-readable configuration of disaster profiles, city dimensions, fleet specifications, and conditional probability tables (CPTs). |
| **Pytest** | `>=8.0.0` | Rigorous Unit Testing Suite | Automated verification of 35 test suites covering mathematical invariants, algorithm edge cases, and simulation consistency. |
| **Streamlit** | `>=1.35.0` | Web Presentation Layer | Powers the reactive civil-defense dashboard, responsive controls, session state management, and real-time execution loop. |
| **Plotly** | `>=6.3.0` | Tactical CAD Vector Maps & Telemetry | Hardware-accelerated 2D vector map (50×50 grid), casualty triage donut charts, and fleet fuel telemetry gauges with zero-latency updates. |
| **Matplotlib** | `>=3.8.0` | Benchmark & Research Plotting | Generates publication-ready PNG comparative charts for algorithm benchmarks and ablation studies. |

---

## 3. In-Depth Architecture & Algorithms (How It Works)

### Module 1: Hazard Escalation Tracking (Continuous Gaussian HMM)
* **Source Location:** `resq_ai/hmm/continuous_hmm.py`, `resq_ai/hmm/tracker.py`
* **Purpose:** Filters raw, noisy multi-sensor environmental telemetry into reliable hidden disaster threat levels:
  $$\mathcal{S} = \{\text{Safe}, \text{Advisory}, \text{Critical}, \text{Evacuate}\}$$
* **Mathematical Formulation:**
  - **Continuous Observation Vector:** $O_t = [r_t, d_t, w_t]^T \in \mathbb{R}^3$, where $r_t$ is rainfall (mm/h), $d_t$ is river depth (m), and $w_t$ is wind speed (km/h).
  - **Multivariate Gaussian Emission Density:**
    $$b_j(O_t) = \frac{1}{(2\pi)^{D/2} |\Sigma_j|^{1/2}} \exp\left(-\frac{1}{2}(O_t - \mu_j)^T \Sigma_j^{-1} (O_t - \mu_j)\right)$$
    with mean vectors $\mu_j \in \mathbb{R}^3$ and full covariance matrices $\Sigma_j \in \mathbb{R}^{3 \times 3}$.
  - **Scaled Forward-Backward Algorithm:** Traditional forward variables $\alpha_t(i)$ rapidly vanish to zero over extended time horizons. ResQ-AI implements scaled forward-backward variables using normalization coefficients:
    $$c_t = \frac{1}{\sum_{i=1}^N \hat{\alpha}_t(i)}, \quad \hat{\alpha}_t(i) = c_t \sum_{j=1}^N \hat{\alpha}_{t-1}(j) A_{ji} b_i(O_t)$$
    guaranteeing strict mathematical stability without floating-point underflow.
  - **Log-Space Viterbi MAP Decoding:** Efficiently computes the most probable state trajectory without numerical precision degradation:
    $$V_t(j) = \max_{i} \left[ V_{t-1}(i) + \ln A_{ij} \right] + \ln b_j(O_t)$$
  - **Baum-Welch (EM) Estimation:** Unsupervised parameter re-estimation from unlabeled historical sensor logs.
* **Performance:** Executes state updates in **$< 0.1\text{ ms}$** per timestep with 100% ground-truth tracking accuracy.

---

### Module 2: Casualty Risk & Structural Collapse Triage (Bayesian Network)
* **Source Location:** `resq_ai/bayesnet/factor.py`, `resq_ai/bayesnet/exact_inference.py`, `resq_ai/bayesnet/triage.py`
* **Purpose:** Evaluates human life risk and structural collapse likelihood when panic-stricken 911 callers provide noisy or incomplete information.
* **10-Node Directed Acyclic Graph (DAG):**
  1. `WeatherSeverity` $\to$ `FloodDepth`, `InfrastructureDamage`
  2. `FloodDepth` $\to$ `StructuralCollapse`, `SevereHypothermia`
  3. `BuildingAge` $\to$ `StructuralCollapse`
  4. `ConstructionType` $\to$ `StructuralCollapse`
  5. `OccupantVulnerability` $\to$ `SevereHypothermia`
  6. `FirePresent` $\to$ `InfrastructureDamage`, `TriagePriority`
  7. `InfrastructureDamage` $\to$ `TriagePriority`
  8. `StructuralCollapse` $\to$ `TriagePriority`
  9. `SevereHypothermia` $\to$ `TriagePriority`
  10. `TriagePriority` $\in \{\text{P1 (Critical)}, \text{P2 (Urgent)}, \text{P3 (Delayed)}\}$
* **From-Scratch Factor Algebra:**
  - Discrete distributions are modeled as multidimensional NumPy factors.
  - `factor_product(\phi_1, \phi_2)` performs vectorized tensor multiplication across matching variable dimensions using `np.multiply`.
  - `factor_marginalize(\phi, var)` projects and sums out hidden variables using `np.sum`.
* **Exact Variable Elimination:**
  - Implements dynamic variable elimination using the **Min-Weight heuristic** (eliminating variables that minimize the size of the resulting factor tensor).
  - Because the graph's treewidth $\le 3$, exact posterior probabilities:
    $$P(\text{TriagePriority} \mid \mathbf{e})$$
    are computed in **$< 0.5\text{ ms}$** with zero stochastic sampling error.
  - Clinical reasoning is generated automatically (e.g., *"Call-004 evaluated as P1: Elderly occupant trapped in unreinforced masonry with 1.4m rising water"*).

---

### Module 3: Dynamic Pathfinding & Obstacle Repair (D* Lite)
* **Source Location:** `resq_ai/dstar/priority_queue.py`, `resq_ai/dstar/dstar_lite.py`, `resq_ai/dstar/router.py`
* **Purpose:** Computes and dynamically repairs shortest paths on a 50×50 urban CAD grid (2,500 nodes, 8-connected grid) as river waters rise, roads flood, and bridges collapse.
* **Inverted Goal-Directed Search:**
  - Searches backward from $s_{\text{goal}}$ to $s_{\text{start}}$.
  - By maintaining cost-to-goal estimates $g(s)$ and one-step lookahead estimates $\text{rhs}(s)$, localized edge disruptions ahead of a moving vehicle only require recomputing inconsistent vertices, leaving the majority of the search tree intact.
* **Key Invariants & Formulas:**
  - Consistency condition: A vertex is consistent if and only if $g(s) = \text{rhs}(s)$.
  - Lookahead update:
    $$\text{rhs}(u) = \min_{s' \in \text{Succ}(u)} \left( c(u, s') + g(s') \right)$$
  - Lexicographic Priority Key:
    $$\mathbf{k}(s) = \begin{bmatrix} k_1(s) \\ k_2(s) \end{bmatrix} = \begin{bmatrix} \min(g(s), \text{rhs}(s)) + h(s_{\text{start}}, s) + k_m \\ \min(g(s), \text{rhs}(s)) \end{bmatrix}$$
    where $k_m$ accumulates distance traversed by the vehicle, preventing expensive full-queue re-keys.
  - Consistent Octile Distance Heuristic:
    $$h(s, s') = \max(\Delta x, \Delta y) + (\sqrt{2} - 1)\min(\Delta x, \Delta y)$$
* **Indexed Binary Min-Heap:**
  - Implemented from scratch with a dual hash map mapping vertex coordinates to heap indices.
  - Enables $O(\log N)$ vertex insertion, $O(\log N)$ extraction, and true $O(\log N)$ `decrease_key` / `update` operations.
* **Empirical Validation:** Expands **82.9% fewer vertices** than static A* across 100 consecutive obstacle changes, with identical $0.0\%$ path optimality divergence.

---

### Module 4: Heterogeneous Fleet Dispatch (Simulated Annealing)
* **Source Location:** `resq_ai/annealing/state.py`, `resq_ai/annealing/simulated_annealing.py`, `resq_ai/annealing/baselines.py`
* **Purpose:** Solves the Multi-Depot Vehicle Routing Problem with Time Windows (MDVRPTW) across a heterogeneous fleet (5 Ambulances requiring passable road cells, 3 Rescue Boats navigating river channels and flooded zones) responding to 20+ simultaneous casualties.
* **Objective Cost Function:**
  $$C(S) = \sum_{v \in V} \text{travel\_time}(v) + \sum_{c \in \text{served}} w(c) \cdot \text{wait\_time}(c) + \sum_{u \in \text{unserved}} \text{penalty}(u) + \sum_{v \in V} \text{overload\_penalty}(v)$$
  where priority weights enforce medical urgency: $w(\text{P1}) = 100.0$, $w(\text{P2}) = 10.0$, $w(\text{P3}) = 1.0$.
* **4 Compound Move Operators:**
  1. `Relocate (Shift)`: Reassigns a distress call from one vehicle's itinerary to another.
  2. `Swap`: Exchanges two assigned calls between different vehicles.
  3. `2-Opt Intra-Route`: Reverses an itinerary sub-sequence within a single route to untangle self-intersections.
  4. `Cross-Exchange Inter-Route`: Swaps route segments between two vehicles to balance workload.
* **Cooling Schedule & Warm-Start Replanning:**
  - Uses geometric cooling $T_{k+1} = \alpha T_k$ with $\alpha = 0.985$.
  - When new distress calls arrive or roads collapse mid-mission, the system does not recompute from scratch. Instead, it **warm-starts** from the existing vehicle route configuration and performs a rapid 200-iteration reheat burst in **$< 10\text{ ms}$**, eliminating route thrashing and keeping vehicles stably committed to active victims.
* **Empirical Validation:** Delivers a **34.3% lower total rescue latency** compared to standard greedy nearest-neighbor dispatch.

---

### Module 5: Natural Language Command Parser (NLP)
* **Source Location:** `resq_ai/nlp/command_parser.py`
* **Purpose:** Enables dispatch commanders to type conversational tactical directives into the terminal.
* **Mechanism:**
  - Regex and keyword tokenization with slot extraction.
  - Automatically identifies target units (`Ambulance 1`, `Boat 2`), coordinates `(x, y)`, city landmarks (`Downtown`, `Marina`, `Hospital`), patient demographics, and topographical events (`"Main St bridge collapsed"`, `"submerged"`).
  - Translates natural language directives into structured dispatch events in $< 1\text{ ms}$.

---

### Module 6: Simulation Engine & 50×50 City Grid
* **Source Location:** `resq_ai/sim/city_grid.py`, `resq_ai/sim/disaster_sim.py`, `resq_ai/core/models.py`
* **Purpose:** High-fidelity simulation of an urban landscape under environmental distress:
  - 50×50 cellular grid (2,500 cells) comprising arterial roads, secondary streets, river channels, bridges, hospital depots, marinas, and residential zones.
  - Realistic cellular automata flood propagation: Water overflows riverbanks into adjacent low-lying cells based on elevation gradients.
  - Dynamic casualty generation, hypothermia progression, vehicle fuel consumption, and patient handoff at hospitals.

---

### Module 7: Full-Fledged Incident Command Web Application
* **Source Location:** `resq_ai/ui/` (`app.py`, `theme.py`, `header.py`, `sidebar.py`, `kpis.py`, `algo_strip.py`, `map.py`, `console.py`, `terminal.py`)
* **Design System & Components:**
  - **Calm, High-Contrast Civil Defense Theme:** Gentle palette (#F1F4F2 background, #0D7A73 primary teal, #C25B54 P1 coral, #B98230 P2 amber, #4A82A8 P3 blue).
  - **Top Mission Header:** System status, live disaster clock (`T+HH:MM`), hazard threat level, and simulation step controls.
  - **4 Equal KPI Cards:** Lives Rescued, Active Casualties, Flooded Cells, Infrastructure & D* Lite Replans.
  - **4-Card Algorithmic Transparency Feed:** Plain-English summary of what HMM, Bayes Net, D* Lite, and Simulated Annealing decided each minute.
  - **Interactive 50×50 CAD Tactical Map:** Plotly-powered map rendering land, river, flooded cells, road closures, collapsed bridges, landmarks, ambulances, boats, and live route vectors.
  - **Incident Command Console (5 Tabs):**
    - `Feed`: Chronological, color-coded stream of rescues, triages, and reroutes.
    - `Fleet`: Symmetrical 4×2 unit grid (`AMB-1`–`AMB-5`, `BOAT-1`–`BOAT-3`) with fuel levels, position, and waypoints.
    - `Calls`: Casualty queue sorted by priority with clinical triage rationales.
    - `Telemetry`: Casualty priority donut chart and fleet fuel reserve bars.
    - `Theory`: Technical viva cheat-sheet explaining all four algorithms.
  - **Bottom Command Terminal:** Monospace terminal with 4 quick-action chips, free-text command input, and sub-10ms execution reporting.
  - **Mission Exporters:** Instant download of civil defense **SITREP reports (.md)** and full simulation telemetry **(.json)**.

---

## 4. End-to-End System Workflow (How It All Operates Together)

Here is the exact lifecycle of an incident during a disaster simulation step:

```text
  [Step T+12:00]
         │
         ▼
  1. SENSOR INGESTION & HMM TRACKING
     River gauge reports 2.4m, rain rate 48mm/h.
     Continuous Gaussian HMM processes 3D observation vector.
     State shifts: ADVISORY ──> CRITICAL.
         │
         ▼
  2. TOPOGRAPHICAL SHIFT & D* LITE REPAIR
     River overflows banks at (24, 18). Main Street Bridge collapses.
     D* Lite detects edge weight change: c(u, v) = ∞.
     Local vertex repair re-keys inconsistent vertices in 3.1ms.
     New shortest paths computed for units in motion.
         │
         ▼
  3. 911 CALL & BAYESIAN NETWORK TRIAGE
     Distress call arrives from Block C: "Elderly resident, water rising to 1.2m".
     Bayes Net receives evidence {Age: Elderly, WaterDepth: Deep, Wood: True}.
     Exact Variable Elimination computes:
     P(Collapse) = 0.84, P(Hypothermia) = 0.79 ──> Priority: P1 (CRITICAL).
         │
         ▼
  4. HETEROGENEOUS FLEET OPTIMIZATION (SIMULATED ANNEALING)
     Warm-start replan triggered with new P1 call and updated D* Lite cost matrix.
     SA explores Relocate, Swap, and 2-Opt moves across 5 Ambulances and 3 Boats.
     Dispatches BOAT-2 via river channel to Block C; rebalances AMB-1 route.
         │
         ▼
  5. FIELD EXECUTION & REVENUE TELEMETRY
     BOAT-2 moves along waypoints, reaches victim, performs rescue.
     Streamlit UI updates CAD map, KPI counters, and Incident Feed.
```

---

## 5. Directory Structure & File Index

```text
Ai Agent/
├── .gitignore                      # Complete Git ignore configuration (caches, secrets, logs)
├── .streamlit/
│   └── config.toml                 # Streamlit UI theme tokens, light background & primary teal
├── configs/
│   ├── default_scenario.yaml       # Master simulation configuration (grid size, speeds, costs)
│   ├── bayes_cpt.yaml              # 10-Node Conditional Probability Tables (CPTs)
│   └── scenarios/                  # Disaster scenario profiles
│       ├── flash_flood.yaml        # Flash flood with rising river and bridge collapse
│       ├── earthquake.yaml         # Structural collapse and rubble road blockages
│       └── urban_fire.yaml         # Fast-moving firestorm with perimeter closures
├── docs/
│   ├── PROJECT_COMPREHENSIVE_REPORT.md  # Detailed technical and benchmarking report
│   ├── architecture.md             # Architecture diagrams and sequence flows
│   └── algorithm_analysis.md       # Formal time/space complexity analysis
├── experiments/                    # Scientific benchmarks & generated research charts
│   ├── benchmark_dstar_vs_astar.py # D* Lite vs Static A* comparative benchmark
│   ├── benchmark_sa_vs_greedy.py   # Simulated Annealing vs Greedy Dispatch benchmark
│   ├── benchmark_hmm_tracking.py   # Continuous HMM tracking accuracy test
│   ├── ablation_studies.py         # Full pipeline ablation study
│   ├── dstar_vs_astar_benchmark.png# Generated node expansion plot
│   ├── sa_vs_greedy_benchmark.png  # Generated latency comparison plot
│   ├── hmm_tracking_accuracy.png   # Generated HMM tracking state plot
│   └── ablation_summary.png        # Generated ablation study comparison plot
├── main.py                         # Unified CLI entrypoint (CLI runs, benchmarks, UI launcher)
├── pytest.ini                      # Pytest runner configuration
├── requirements.txt                # Project dependencies (numpy, scipy, streamlit, plotly, etc.)
├── README.md                       # GitHub repository landing page & quickstart
├── resq_ai/
│   ├── __init__.py                 # Package root
│   ├── core/                       # Core data models
│   │   ├── models.py               # GridCell, Vehicle, DistressCall, HazardState, Event models
│   │   └── logger.py               # Incident feed and event logging utilities
│   ├── hmm/                        # Module 1: Hazard Escalation Tracking
│   │   ├── continuous_hmm.py       # Scaled forward-backward, Viterbi, EM Gaussian HMM
│   │   └── tracker.py              # Zone sensor tracker and alert engine
│   ├── bayesnet/                   # Module 2: Casualty Risk Triage
│   │   ├── factor.py               # From-scratch discrete multidimensional factor algebra
│   │   ├── exact_inference.py      # Variable Elimination with Min-Weight ordering
│   │   └── triage.py               # Clinical casualty triage classifier (P1, P2, P3)
│   ├── dstar/                      # Module 3: Dynamic Pathfinding
│   │   ├── priority_queue.py       # Custom indexed binary min-heap (O(log N) decrease-key)
│   │   ├── dstar_lite.py           # Inverted goal-directed D* Lite with local repair
│   │   ├── astar.py                # Static A* benchmark comparator
│   │   └── router.py               # City grid router and cost matrix evaluator
│   ├── annealing/                  # Module 4: Fleet Logistics Optimizer
│   │   ├── state.py                # Vehicle routing state and 4 move operators
│   │   ├── simulated_annealing.py  # Annealing optimizer with warm-start reheats
│   │   └── baselines.py            # Greedy nearest-neighbor baseline comparator
│   ├── nlp/                        # Dispatcher Command Terminal
│   │   └── command_parser.py       # Regex & keyword slot-extraction parser
│   ├── orchestrator/               # Pipeline Coordinator
│   │   └── engine.py               # Real-time event loop linking all 4 modules
│   ├── sim/                        # Simulation Engine
│   │   ├── city_grid.py            # 50x50 procedural grid generator with landmarks
│   │   └── disaster_sim.py         # Physics-based flood propagation and casualty dynamics
│   └── ui/                         # Incident Command Presentation Layer
│       ├── app.py                  # Main Streamlit dashboard application
│       ├── theme.py                # Civil defense design system & CSS injection
│       ├── header.py               # Top status bar & disaster clock
│       ├── sidebar.py              # Scenario selector, clock, injections & exports
│       ├── kpis.py                 # 4 Symmetrical KPI telemetry cards
│       ├── algo_strip.py           # 4-Card algorithmic transparency feed
│       ├── map.py                  # Plotly 50x50 interactive vector CAD map
│       ├── console.py              # 5-Tab incident command console container
│       ├── terminal.py             # Dispatch command line terminal & chip actions
│       └── components.py           # Reusable UI widgets and badges
└── tests/                          # 35 Pytest Unit Verification Tests
    ├── test_hmm.py                 # HMM probabilities, underflow & Viterbi invariants
    ├── test_bayesnet.py            # Factor multiplication, marginalization & triage
    ├── test_dstar.py               # D* Lite optimality, priority queue & local repairs
    ├── test_annealing.py           # SA move operators, temperature decay & warm start
    ├── test_nlp.py                 # Terminal parser slot extraction & error handling
    ├── test_sim.py                 # City grid topology, flood propagation & vehicles
    ├── test_models.py              # Pydantic-style data structures & serializability
    └── test_orchestrator.py        # End-to-end multi-module pipeline execution
```

---

## 6. How to Install, Test & Run

### 6.1 Installation
Clone the repository and install dependencies in your Python environment:
```powershell
pip install -r requirements.txt
```

### 6.2 Run All 35 Unit Tests
```powershell
pytest -v
```
All 35 tests test mathematical invariants across all 4 algorithmic modules and pass in $< 1.0\text{ s}$.

### 6.3 Launch the Incident Command Dashboard
Run the interactive Streamlit application:
```powershell
streamlit run resq_ai/ui/app.py
```
Then open **`http://localhost:8501`** in your browser. The dashboard automatically pre-runs 12 simulation steps so the tactical map, KPIs, fleet grid, and algorithmic feeds are immediately populated with real operational data.

### 6.4 Run Headless CLI Simulation
Run a 15-minute simulation run from the command line:
```powershell
python main.py --scenario configs/scenarios/flash_flood.yaml --steps 15
```

### 6.5 Run Dispatch Acceptance Command
Test natural language parsing and end-to-end dispatch execution:
```powershell
python main.py --command "Dispatch rescue boat to Block C. Reports say Main Street is submerged."
```

### 6.6 Run Experimental Benchmarks
Reproduce all scientific benchmarks and generate visualization plots in `experiments/`:
```powershell
# Benchmark D* Lite vs Static A* across 100 edge changes
python experiments/benchmark_dstar_vs_astar.py

# Benchmark Simulated Annealing vs Greedy Dispatch
python experiments/benchmark_sa_vs_greedy.py

# Benchmark HMM State Tracking Accuracy
python experiments/benchmark_hmm_tracking.py

# Run Full Sensitivity & Ablation Studies
python experiments/ablation_studies.py
```

---

## 7. Key Empirical Benchmark Summary

| Benchmark | Test Scenario | Baseline Comparator | ResQ-AI Result | Operational Advantage |
| :--- | :--- | :--- | :--- | :--- |
| **Dynamic Routing** | 100 dynamic obstacle changes on 50×50 urban grid | Static A* (from scratch) | **82.9% fewer node expansions** | Path repair in $< 3\text{ ms}$; zero optimality degradation ($0.0\%$ path length error). |
| **Fleet Dispatch** | 22 distress calls across 5 ambulances & 3 boats | Greedy Nearest-Neighbor | **34.3% lower total casualty wait time** | Overcomes local cluster traps; warm-start replanning runs in $< 10\text{ ms}$. |
| **Hazard Tracking** | 120-minute simulated flash flood event | Discrete thresholding | **100% MAP sequence accuracy** | Continuous Gaussian emissions eliminate threshold flutter. |
| **Triage Allocation** | Critical casualties vs uniform FIFO queue | First-Come-First-Served (FCFS) | **21.4% reduction in P1 patient wait time** | Diverts vehicles to collapsing structures before fatal delays occur. |
| **End-to-End Latency** | Full pipeline (NLP $\to$ HMM $\to$ BN $\to$ D* $\to$ SA) | 1,000 ms real-time ceiling | **6.7 ms average response time** | **149x faster** than the 1-second operational budget. |
