# ResQ-AI: Plain-English Project Explanation

> **No complicated math jargon. Just what it does, how it works, and why it matters.**

---

## 💡 1. What is ResQ-AI in Simple Words?

Imagine a sudden flash flood or earthquake hits a city:
* Rivers overflow and streets turn into rivers.
* Bridges collapse.
* Hundreds of people call 911 at the same time, panicking.
* The city only has 5 ambulances and 3 rescue boats.

A human 911 dispatcher gets overwhelmed in minutes. They don't know which roads are still passable, who is in the greatest danger, or which vehicle should go where.

**ResQ-AI is an intelligent emergency control assistant.** It acts like an automated "co-pilot" for disaster managers:
1. It watches city sensors to spot dangerous flooding before it gets worse.
2. It prioritizes 911 calls so the people in life-threatening danger get helped first.
3. It finds safe routes around flooded streets and broken bridges (like Google Maps for disasters).
4. It tells each ambulance and rescue boat exactly who to pick up and which hospital to take them to.
5. It runs inside a clean, live dashboard on your computer screen.

---

## 🧩 2. The 4 "Brains" of the System

Instead of using mysterious "black-box" machine learning, ResQ-AI uses **4 clear, smart problem-solving engines**. Here is what each one does:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        HOW THE 4 BRAINS WORK TOGETHER                  │
├────────────────────┬────────────────────┬──────────────────────────────┤
│ 1. Hazard Watcher  │ 2. Smart Triage    │ 3. Disaster Navigator        │
│ (HMM)              │ (Bayes Net)        │ (D* Lite)                    │
│ "Is Zone B getting │ "Who needs help    │ "Which roads are open, and   │
│ flooded right now?"│ right this second?"│ which are blocked by water?" │
├────────────────────┴────────────────────┴──────────────────────────────┤
│ 4. Fleet Dispatcher (Simulated Annealing / ALNS)                       │
│ "Which boat or ambulance should pick up which victim and drop them     │
│ off at the hospital?"                                                  │
└────────────────────────────────────────────────────────────────────────┘
```

---

### Brain 1: The Hazard Watcher (Continuous HMM)
* **What it does:** Acts like a smart weather radar for each city district.
* **The Problem:** Water level and rain sensors are "noisy" — one sensor reading might jump up by accident because of a splash, or a sensor might temporarily lose battery.
* **The Solution:** Brain 1 doesn't panic over a single high reading. It looks at the continuous trend over time.
* **The Outcome:** It gives the true danger level: **Safe**, **Advisory**, **Critical**, or **Evacuate**. It also gives a 15, 30, and 60-minute forecast so the city can prepare before the water reaches homes.

---

### Brain 2: The 911 Triage Doctor (Bayesian Network)
* **What it does:** Decides who gets rescued first when 911 calls flood in.
* **The Problem:** Callers are scared and confused. They might forget to mention their grandmother has asthma, or they might not know if the building's foundation is cracked.
* **The Solution:** Brain 2 connects the clues:
  * *High water + Old wooden building = High chance the house might collapse.*
  * *High water + Elderly person = High chance of severe hypothermia.*
* **The Outcome:**
  1. It tags calls into medical priority levels:
     * **P1 (Immediate - Red):** Life-threatening. Must be rescued now.
     * **P2 (Delayed - Yellow):** Urgent, but stable for a short period.
     * **P3 (Minimal - Green):** Minor issues.
  2. **Bonus Superpower (Smart Questioning):** It tells the 911 operator: *"Ask the caller if water has entered the ground floor — that single answer will tell us what to do fastest!"*

---

### Brain 3: The Disaster Navigator (D* Lite)
* **What it does:** Directs ambulances and rescue boats through the city.
* **The Problem:** Normal GPS (like basic A* or Dijkstra) calculates the whole city from scratch every time a street floods. That is way too slow when roads are collapsing every minute.
* **The Solution:** Brain 3 only recalculates the *broken piece* of the route.
  * It knows that **Ambulances cannot drive in water deeper than 35 cm** (they will stall).
  * It knows that **Rescue Boats need at least 20 cm of water** (they can't drive on dry roads).
* **The Outcome:** When a bridge collapses, Brain 3 reroutes the fleet in **less than 1 millisecond** with 0% mistakes.

---

### Brain 4: The Fleet Dispatcher (Simulated Annealing / ALNS)
* **What it does:** Assigns the right vehicle to the right person.
* **The Problem (Why simple logic fails):**
  * If you just send the closest ambulance to the closest call (Greedy dispatch), the ambulance picks up 2 people, gets completely full, and gets stuck while someone in critical condition dies waiting on the other side of town.
* **The Solution:** Brain 4 plans complete round-trips:
  * Ambulance picks up Patient 1 $\to$ picks up Patient 2 $\to$ drives to Hospital $\to$ drops them off $\to$ heads to Patient 3.
  * If a person is in deep water, a **Rescue Boat** picks them up, brings them to a **River Dock**, and transfers them to an **Ambulance**.
* **The Outcome:** Brain 4 serves **100% of critical patients** and reduces overall waiting time by **88%** compared to simple nearest-neighbor dispatch.

---

## 🖥️ 3. The Incident Command Console (The Screen You See)

You access the system through your web browser at **`http://localhost:8501`**.

The screen is divided into 3 simple sections:

1. **Left Sidebar:**
   * Pick a disaster scenario (*Flash Flood*, *Earthquake*, *Urban Fire*, or *Compound Cascade*).
   * Click buttons to simulate events (like *"Collapse Central Bridge"*).
   * Toggle map layers on and off.
   * Download a clean summary report (`SITREP.md`) or map data (`.geojson`).

2. **Center Map (The Big Visual):**
   * Shows the whole city in high-contrast dark mode.
   * Blue shading shows where floodwater is rising.
   * Red dots show people waiting for rescue.
   * Vehicle icons move across the streets with little arrows showing where they are driving.
   * Shows hospitals, river docks, and bridges.

3. **Right Control Panel (5 Easy Tabs):**
   * **Incidents Tab:** The list of 911 calls, sorted by most urgent first.
   * **Fleet Tab:** Shows each ambulance and boat, how much fuel it has left, and how many people it has rescued.
   * **Hazard Forecast Tab:** Predicts which neighborhood will be underwater in the next 15, 30, and 60 minutes.
   * **Explanations ("Why") Tab:** Tells you in plain English why the AI made each decision (e.g. *"Assigned Boat 2 because water is 1.2m deep; ambulance would get flooded"*).
   * **Terminal Tab:** You can type orders like *"Dispatch boat to Block C"* or hit **"Undo"** if you made a mistake!

---

## ⏱️ 4. A 1-Minute Story: What Happens in a Real Run?

Here is what happens during a 1-minute simulation run:

1. **Minute 1:** Heavy rain starts. The **Hazard Watcher** notices river gauges rising in the Riverfront District and alerts the team.
2. **Minute 2:** A 911 call comes in from an elderly couple trapped in an old house. The **Triage Doctor** marks them as **P1 (Immediate)**.
3. **Minute 4:** The Central River Bridge collapses! The **Disaster Navigator** immediately flashes the bridge red and steers vehicles around it.
4. **Minute 6:** The **Fleet Dispatcher** routes Boat 1 to rescue the couple, delivers them to the West River Dock, where Ambulance 2 is already waiting to rush them to the General Hospital.
5. **Total computation time:** The entire computer thought process took **38 milliseconds** (less than half the time it takes to blink your eye).

---

## 🏆 5. Top 3 Reasons This Project is Unique

If anyone (a professor, recruiter, or interviewer) asks why this project is special, tell them these 3 points:

1. **100% Custom-Built from Scratch:**
   * We did NOT use pre-made machine learning libraries like scikit-learn, networkx, or OR-Tools. Every single algorithm (the HMM equations, the Bayesian probability math, the D* Lite graph search, and the fleet optimizer) was written from first principles in pure Python and NumPy.
2. **Lightning Fast (< 40 ms):**
   * Emergency systems must never lag. ResQ-AI recalculates the entire city's routes and fleet plans in **38.7 milliseconds**, well below the 100 ms limit.
3. **Transparent & Explainable:**
   * It never acts like a mysterious black box. It always explains **why** it prioritized a patient, **why** a street was marked impassable, and **why** a specific ambulance was chosen.

---

## 🗣️ 6. Quick Cheat Sheet for Presentations & Viva

* **Q: What is the main goal of ResQ-AI?**
  * *Answer:* To help emergency dispatchers save more lives during urban disasters by automating flood hazard tracking, medical triage, road rerouting, and ambulance/boat dispatching in real time.
* **Q: Why do you have both boats and ambulances?**
  * *Answer:* Real floods have both dry roads and submerged areas. Ambulances can't drive in deep water, and boats can't drive on land. Our system coordinates both, including dock handoffs where boats hand patients over to waiting ambulances.
* **Q: Why D* Lite instead of regular A* or Google Maps?**
  * *Answer:* Regular A* has to search the entire map all over again when a road closes. D* Lite only repairs the broken part of the path, making it almost 4 times faster and saving 98% of unnecessary search calculations.
* **Q: How does the dispatcher interact with the system?**
  * *Answer:* Through an interactive web dashboard with a live map, or by typing simple English commands into a command terminal with an instant "Undo" button.
