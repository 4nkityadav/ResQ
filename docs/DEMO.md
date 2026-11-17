# 🎬 ResQ-AI: Two-Minute Live Demonstration Guide

This guide details the scripted 2-minute end-to-end demonstration (`demo_2min.py`) and live UI workflow for **ResQ-AI**, showcasing the integrated civil-defense decision pipeline built 100% from scratch in pure Python, NumPy, and SciPy.

---

## ⚡ Quick Start

### 1. Run Automated Terminal Demonstration
Run the paced demonstration script (~75 seconds with readable pauses):
```powershell
python demo_2min.py
```

Or run in non-blocking fast mode (~5 seconds):
```powershell
python demo_2min.py --fast
```

### 2. Launch Interactive Incident Command Console
```powershell
streamlit run resq_ai/ui/app.py
```
*Live dashboard operates at: `http://localhost:8501`*

---

## 🧭 Demonstration Walkthrough Stages

The automated demonstration exercises all 5 mathematical decision modules in sequence:

### Stage 1: Unified World Model & Geospatial Initialization
* **Architecture:** Initializes the canonical `WorldState` holding the 50x50 urban DEM grid (2,500 cells), road network graph, 8 fleet units (5 ambulances, 3 rescue boats), 2 municipal hospitals, and 3 boat handoff docks.
* **Integrity Invariant:** Proves that all algorithmic modules read and write to one centralized world model with zero uncoordinated private grid copies.

### Stage 2: Robust Continuous HMM Hazard Tracking
* **Input:** Simulates a sudden storm surge and cloudburst in Zone `Riverfront` ($1.85$m water level, $2.40$m/s flow, $62.0$mm/h rainfall).
* **Mathematics:** Gaussian/Student-t emissions ($\nu = 4.0$) marginalize over missing sensor telemetry dimensions. Sticky transitions ($\kappa = 0.08$) prevent threshold chatter.
* **Outcome:** Posterior belief rapidly elevates to **Critical/Evacuate**. The multi-horizon forecast projects critical risk at $+15$m ($0.42$), $+30$m ($0.59$), and $+60$m ($0.75$), generating an early escalation alert.

### Stage 3: Bayesian Network Exact Triage & Value-of-Information (VoI)
* **Input:** Ingests 911 distress call `CALL-DEMO-01` with partial evidence: `HazardState=Critical`, `FloodLevel=High`, `OccupantType=Elderly`.
* **Mathematics:** Exact Variable Elimination with greedy Min-Weight ordering calculates exact posterior marginals:
  - $P(\text{Building Collapse} = \text{True}) = 55.2\%$
  - $P(\text{Hypothermia} = \text{High}) = 80.0\%$
* **Triage Decision:** Maps to **P1 (Immediate Red)** under the START/SALT expected-utility framework ($EU = 84.2 / 100.0$).
* **Value-of-Information (VoI):** Shannon mutual information ranks unasked questions by information gain:
  1. `BuildingAge`: $0.0203$ bits
  2. `ConstructionType`: $0.0148$ bits
  3. `FirePresent`: $0.0145$ bits
* **Dispatcher Prompt:** *"Ask caller immediately about Building Age to collapse triage uncertainty fastest."*

### Stage 4: Dynamic Infrastructure Failure & D* Lite Incremental Repair
* **Event:** Mid-mission inject: Central River Bridge at coordinate `(25, 20)` collapses due to hydrodynamic force.
* **Passability Rules:** Enforces vehicle class limits (Ambulance max water depth $= 0.35$m, Boat navigable $\ge 0.20$m) and forbids diagonal corner-cutting.
* **Mathematics:** D* Lite updates edge costs dynamically via reversed $rhs$-values and indexed priority queue keys ($k_1, k_2$).
* **Verification:** Routes for all affected units are updated in $< 1$ ms with **0.000% optimality gap** against fresh cold-start A*, expanding $98.4\%$ fewer nodes.

### Stage 5: Capacity-Constrained Fleet Routing (ALNS / SA Pickup & Delivery)
* **Constraints:** Vehicle patient capacity (Ambulance $= 2$, Boat $= 4$), mandatory hospital delivery drop-offs, and boat-to-road handoff docks.
* **Optimization:** Runs Adaptive Large Neighborhood Search (ALNS) with 3 destroy operators (Random, Worst-Cost, Shaw relatedness) and 2 repair operators (Greedy, Regret-2).
* **Performance:** Executes in $13.5$ ms (tactical budget: $< 100$ ms) achieving an $88.8\%$ cost reduction over greedy nearest-neighbor dispatch without unserved casualty penalties.

### Stage 6: Dispatcher Natural Language Terminal & Undo Invariant
* **Input:** `"Dispatch rescue boat to Block C. Reports say Main Street is submerged."`
* **Parsing & Execution:** High-confidence intent parsing identifies `DISPATCH` intent, maps `Block C` to `(22, 34)`, closes `Main Street`, reroutes fleet, and assigns `BOAT-2` in $12.8$ ms.
* **Deterministic Undo:** Demonstrates transactional safety by triggering `undo_last_command()`, fully rolling back the WorldState to its pre-command baseline.

### Stage 7: Mission SITREP & Telemetry Generation
* **Outputs:** Generates an incident commander Situation Report (`SITREP.md`) with executive casualty summaries, fleet disposition, and tabular hazard status.
* **GIS Interoperability:** Serializes live spatial features to GeoJSON `FeatureCollection` for MapLibre GL, QGIS, or deck.gl integration.

---

## 🖥️ Live UI Demonstration Script (Streamlit Command Console)

When presenting the live UI (`http://localhost:8501`), follow this sequence:

1. **Top Bar Overview:**
   - Note the real-time simulation clock ($T+00\text{m}$), operational threat status, and scenario selector.
2. **Geospatial Map Canvas (Center 60%):**
   - Show the continuous DEM topography, color-shaded flood depth contours, road network hierarchies (highways, arterials, local streets), bridges, hospitals, and vehicle markers with directional heading needles.
3. **Left Sidebar Controls:**
   - Toggle map layers: *City Landmarks*, *Vehicle Routes*, *Flooded Streets*, *River Channel*.
   - Click **"Inject Bridge Collapse"** and observe the live yellow/red obstacle appear on the river crossing.
4. **Right Operations Panel (5 Tabs):**
   - **Incidents Queue:** View incoming 911 calls sorted by triage priority (P1 Immediate, P2 Delayed, P3 Minimal) with occupancy and flood depth.
   - **Fleet Overview:** Click unit `AMB-1` or `BOAT-2` to inspect remaining fuel hours, current patient load, and active route stops.
   - **Hazard Forecast:** Review multi-horizon $+15$m, $+30$m, and $+60$m predictive failure risks across urban districts.
   - **Explanations ("Why"):** Review plain-language causal reasoning explaining why specific patients were prioritized and which routes were rejected due to high flood depth.
   - **Terminal & Undo:** Type `Dispatch rescue boat to Block C` and click Execute. Observe the instant fleet reallocation, then click **"↩ Undo Last Command"** to verify immediate rollback.
5. **Mission Scrubber (Bottom Bar):**
   - Scrub through past mission milestones to inspect historical flood propagation.
6. **Telemetry Export:**
   - Download the instant **SITREP (.md)** or **GIS Telemetry (.geojson)** files directly from the sidebar.
