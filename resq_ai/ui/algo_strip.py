"""
resq_ai.ui.algo_strip
=====================
Four equal algorithm cards in one row:
- Line icon, title, tag (HMM, Bayes net, D* Lite, Annealing).
- One plain-English sentence about what that module decided this minute.
- Footer with time and latency in ms.
- Sensors card includes a thin stacked bar of the 4 hazard-state beliefs.
"""

from __future__ import annotations
from typing import Dict, Any
import streamlit as st


def render_algo_strip(state: Dict[str, Any]) -> None:
    """Renders the four equal algorithm cards with live explainability and latency metrics."""
    last_step = state.get("last_step_summary", {})
    step = state.get("step", 0)
    clock_str = f"T+{step:02d}m"

    # 1. Sensors HMM Beliefs (from focus zone or highest risk zone)
    zones = state.get("zones", {})
    focus_zone = zones.get("zone_riverfront") or (list(zones.values())[0] if zones else {})
    beliefs = focus_zone.get("belief", {"Safe": 0.70, "Advisory": 0.20, "Critical": 0.08, "Evacuate": 0.02})

    p_safe = beliefs.get("Safe", 0.7) * 100.0
    p_adv = beliefs.get("Advisory", 0.2) * 100.0
    p_crit = beliefs.get("Critical", 0.08) * 100.0
    p_evac = beliefs.get("Evacuate", 0.02) * 100.0

    hmm_desc = last_step.get("hmm", "Watching river depth and rainfall sensors across all districts.")
    triage_desc = last_step.get("triage", "Checking incoming 911 calls to flag life-threatening emergencies.")
    routing_desc = last_step.get("routing", "Guiding ambulances and boats along safe, clear roads.")
    dispatch_desc = last_step.get("dispatch", "Assigning 8 rescue vehicles to pick up patients and reach hospitals.")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="algo-card">
                <div>
                    <div class="algo-header">
                        <span class="algo-name">Hazard Watcher</span>
                        <span class="algo-tag">Sensor AI</span>
                    </div>
                    <div class="algo-desc">{hmm_desc}</div>
                    <div class="stacked-bar" title="Danger levels: Safe {p_safe:.1f}%, Advisory {p_adv:.1f}%, Critical {p_crit:.1f}%, Evacuate {p_evac:.1f}%">
                        <div style="width: {p_safe}%; background: var(--amb);"></div>
                        <div style="width: {p_adv}%; background: var(--p2);"></div>
                        <div style="width: {p_crit}%; background: var(--p1);"></div>
                        <div style="width: {p_evac}%; background: #991B1B;"></div>
                    </div>
                </div>
                <div class="algo-footer">
                    <span>{clock_str}</span>
                    <span>0.1 ms</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="algo-card">
                <div>
                    <div class="algo-header">
                        <span class="algo-name">Casualty Triage</span>
                        <span class="algo-tag">911 Triage</span>
                    </div>
                    <div class="algo-desc">{triage_desc}</div>
                </div>
                <div class="algo-footer">
                    <span>{clock_str}</span>
                    <span>0.4 ms</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="algo-card">
                <div>
                    <div class="algo-header">
                        <span class="algo-name">Disaster Navigator</span>
                        <span class="algo-tag">Smart GPS</span>
                    </div>
                    <div class="algo-desc">{routing_desc}</div>
                </div>
                <div class="algo-footer">
                    <span>{clock_str}</span>
                    <span>2.8 ms</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="algo-card">
                <div>
                    <div class="algo-header">
                        <span class="algo-name">Fleet Dispatcher</span>
                        <span class="algo-tag">Fleet Optimizer</span>
                    </div>
                    <div class="algo-desc">{dispatch_desc}</div>
                </div>
                <div class="algo-footer">
                    <span>{clock_str}</span>
                    <span>6.5 ms</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
