"""
resq_ai.ui.navbar
=================
Comprehensive top Navigation Bar component for ResQ-AI Incident Command Center.
Provides full structured top layout:
- Left: Brand icon, ResQ-AI logo, version badge, and operational division subtitle.
- Center: Quick navigation view pills (Map, Fleet, 911 Calls, Forecast, SitRep).
- Right: Threat status badge, disaster clock, simulation controls (+1m, +5m, Auto-play),
  and quick 1-click Dark/Light theme toggle switch.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import streamlit as st


def render_navbar(
    state: Dict[str, Any],
    scenario_name: str,
    current_theme: str = "light",
    is_auto_playing: bool = False,
    active_nav_view: str = "Tactical Map",
) -> Dict[str, Any]:
    """
    Renders the modern, full-width Incident Command Navigation Bar.
    Returns dictionary of actions:
      {'step_1': bool, 'step_5': bool, 'auto_play': bool, 'theme_toggled': Optional[str], 'nav_view': str}
    """
    actions: Dict[str, Any] = {
        "step_1": False,
        "step_5": False,
        "theme_toggled": None,
        "nav_view": active_nav_view,
    }
    is_dark = (current_theme == "dark")

    # 1. Determine highest hazard level and probability from zones
    zones = state.get("zones", {})
    max_hazard_name = "Safe"
    max_hazard_prob = 0.05
    dot_class = "dot-safe"

    rank = {"Evacuate": 4, "Critical": 3, "Advisory": 2, "Safe": 1}
    highest_rank = 0

    for zid, zdata in zones.items():
        m_state = zdata.get("map_state", "Safe")
        r = rank.get(m_state, 1)
        p = zdata.get("p_critical_or_evacuate", 0.0)
        if r > highest_rank or (r == highest_rank and p > max_hazard_prob):
            highest_rank = r
            max_hazard_name = m_state
            max_hazard_prob = p

    if max_hazard_name == "Critical":
        dot_class = "dot-critical"
    elif max_hazard_name == "Evacuate":
        dot_class = "dot-evacuate"
    elif max_hazard_name == "Advisory":
        dot_class = "dot-advisory"
    else:
        dot_class = "dot-safe"

    hazard_pct = int(max_hazard_prob * 100) if max_hazard_name in ("Critical", "Evacuate") else 15

    # Mission Clock
    step_mins = state.get("step", 0)
    hours = step_mins // 60
    mins = step_mins % 60
    clock_str = f"T+{hours:02d}:{mins:02d}"

    # Counts for nav badges
    unserved_count = len(state.get("unserved_calls", []))
    p1_count = sum(1 for c in state.get("unserved_calls", []) if c.get("priority") == "P1")
    vehicles_count = len(state.get("vehicles", []))

    # Top Navbar Container
    st.markdown('<div class="navbar-wrapper">', unsafe_allow_html=True)
    
    col_brand, col_status, col_ctrls = st.columns([1.0, 1.2, 1.45])

    with col_brand:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; padding: 2px 0;">
                <div style="width: 36px; height: 36px; border-radius: 9px; background: linear-gradient(135deg, #0EA5E9 0%, #0284C7 100%); display: flex; align-items: center; justify-content: center; font-size: 1.15rem; box-shadow: 0 2px 6px rgba(14, 165, 233, 0.35);">
                    🚨
                </div>
                <div>
                    <div style="display: flex; align-items: center; gap: 6px;">
                        <span style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.18rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.02em;">
                            ResQ-AI
                        </span>
                        <span style="background: var(--accent-soft); color: var(--accent); font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; font-weight: 700; padding: 1px 6px; border-radius: 4px; border: 1px solid var(--accent);">
                            v2.4 CAD
                        </span>
                    </div>
                    <div style="font-size: 0.72rem; color: var(--text-secondary); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
                        Civil Defense Command Center
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_status:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: center; gap: 10px; padding: 4px 0; flex-wrap: wrap;">
                <div class="hazard-pill" title="Real-time multi-zone disaster threat estimation">
                    <span class="status-dot {dot_class}"></span>
                    <span style="font-size: 0.80rem; font-weight: 700;">City Alert: {max_hazard_name} ({hazard_pct}%)</span>
                </div>
                <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 20px; padding: 4px 10px; font-size: 0.78rem; font-weight: 600; color: var(--text-secondary);">
                    {scenario_name}
                </div>
                <div class="tabular-nums" style="background: var(--bg-card); border: 1px solid var(--border); padding: 4px 10px; border-radius: 7px; font-weight: 700; color: var(--accent); font-size: 0.86rem; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
                    ⏱️ {clock_str}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_ctrls:
        # Quick simulation controls & theme switch
        c_step1, c_step5, c_auto, c_theme = st.columns([0.85, 0.85, 1.35, 1.15])
        with c_step1:
            if st.button("+1m", key="nav_btn_step1", width="stretch", help="Advance simulation by 1 minute"):
                actions["step_1"] = True
        with c_step5:
            if st.button("+5m", key="nav_btn_step5", width="stretch", help="Advance simulation by 5 minutes"):
                actions["step_5"] = True
        with c_auto:
            auto_val = st.toggle("Auto", value=is_auto_playing, help="Continuously advance simulation")
            actions["auto_play"] = auto_val
        with c_theme:
            theme_icon = "☀️ Light" if is_dark else "🌙 Dark"
            theme_target = "light" if is_dark else "dark"
            if st.button(theme_icon, key="nav_theme_btn", width="stretch", help=f"Switch to {theme_target.title()} Mode"):
                actions["theme_toggled"] = theme_target

    st.markdown('</div>', unsafe_allow_html=True)
    return actions
