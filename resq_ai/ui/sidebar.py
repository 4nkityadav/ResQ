"""
resq_ai.ui.sidebar
==================
Left sidebar component (~270px):
- Scenario: three selectable cards with one-line descriptions.
- Simulation clock: speed slider (0.5-3.0s) and restart button.
- Inject an event: 911 call, collapse bridge, breach levee.
- Map layers: landmarks, vehicle routes, flooded streets, river channel.
- Mission reports: SITREP (.md) and telemetry (.json) download buttons.
"""

from __future__ import annotations
import json
from typing import Dict, Any, Tuple
import streamlit as st

from resq_ai.core.models import DistressCall, OccupantType
from resq_ai.ui.components import generate_sitrep_markdown


SCENARIOS = {
    "flash_flood.yaml": {
        "title": "Flash Flood",
        "desc": "River levee breach, submerged roads, rising casualties",
        "icon": "🌊",
    },
    "earthquake.yaml": {
        "title": "Earthquake",
        "desc": "Structural rubble blockages and urgent building collapse triage",
        "icon": "⚡",
    },
    "urban_fire.yaml": {
        "title": "Urban Fire",
        "desc": "Active thermal hazards and smoke-related medical evacuations",
        "icon": "🔥",
    },
    "compound_disaster.yaml": {
        "title": "Compound Cascade",
        "desc": "Flood, levee breach, bridge failure & breakdown",
        "icon": "⚠️",
    },
}


def render_sidebar(
    engine: Any,
    state: Dict[str, Any],
    selected_scenario: str,
    theme_choice: str,
) -> Dict[str, Any]:
    """Renders the incident command left sidebar and returns user control selections."""
    results = {
        "scenario_changed": None,
        "restart_sim": False,
        "inject_call": False,
        "inject_bridge": False,
        "inject_levee": False,
        "speed_delay": 1.2,
        "show_landmarks": True,
        "show_routes": True,
        "show_flood": True,
        "show_river": True,
        "theme": theme_choice,
    }

    with st.sidebar:
        # Theme toggle (Clean, full width, zero widget key conflicts)
        st.markdown("<div class='sidebar-section-title'>THEME</div>", unsafe_allow_html=True)
        is_dark = (st.session_state.get("theme", "light") == "dark")
        new_is_dark = st.toggle(
            "🌙 Dark Mode" if is_dark else "☀️ Light Mode",
            value=is_dark,
            help="Switch between Dark and Light mode",
        )
        if new_is_dark != is_dark:
            new_theme = "dark" if new_is_dark else "light"
            st.session_state.theme = new_theme
            if hasattr(st, "query_params"):
                st.query_params["theme"] = new_theme
            st.rerun()
        results["theme"] = st.session_state.get("theme", "light")

        # SCENARIO SECTION
        st.markdown("<div class='sidebar-section-title'>SCENARIO</div>", unsafe_allow_html=True)
        scenario_options = list(SCENARIOS.keys())
        scen_labels = [f"{SCENARIOS[k]['title']} — {SCENARIOS[k]['desc']}" for k in scenario_options]
        
        curr_idx = scenario_options.index(selected_scenario) if selected_scenario in scenario_options else 0
        new_scen = st.radio(
            "Select Scenario",
            options=scenario_options,
            format_func=lambda k: f"{SCENARIOS[k]['title']}\n{SCENARIOS[k]['desc']}",
            index=curr_idx,
            label_visibility="collapsed",
            key="sidebar_scen_radio",
        )
        if new_scen != selected_scenario:
            results["scenario_changed"] = new_scen

        # SIMULATION CLOCK SECTION
        st.markdown("<div class='sidebar-section-title'>SIMULATION CLOCK</div>", unsafe_allow_html=True)
        speed = st.slider(
            "Step Delay (s)",
            min_value=0.5,
            max_value=3.0,
            value=1.2,
            step=0.1,
            label_visibility="collapsed",
            key="sidebar_speed_slider",
        )
        results["speed_delay"] = speed
        st.caption(f"Playback Speed: {speed:.1f}s / step")

        if st.button("Reset Baseline Simulation", key="sidebar_restart_btn", width="stretch", help="Re-initialize scenario to T+12 baseline"):
            results["restart_sim"] = True

        # INJECT AN EVENT SECTION (Symmetrical full-width action buttons)
        st.markdown("<div class='sidebar-section-title'>INJECT AN EVENT</div>", unsafe_allow_html=True)
        if st.button("+ 911 Distress Call", key="sidebar_add_call_btn", width="stretch", help="Generate emergency caller"):
            results["inject_call"] = True

        if st.button("Collapse River Bridge", key="sidebar_break_bridge_btn", width="stretch", help="Damage bridge infrastructure"):
            results["inject_bridge"] = True

        if st.button("Breach River Levee", key="sidebar_breach_levee_btn", width="stretch", help="Trigger levee barrier failure"):
            results["inject_levee"] = True

        # MAP LAYERS SECTION
        st.markdown("<div class='sidebar-section-title'>MAP LAYERS</div>", unsafe_allow_html=True)
        results["show_landmarks"] = st.checkbox("City landmarks", value=True, key="layer_landmarks")
        results["show_routes"] = st.checkbox("Vehicle routes", value=True, key="layer_routes")
        results["show_flood"] = st.checkbox("Flooded streets", value=True, key="layer_flood")
        results["show_river"] = st.checkbox("River channel", value=True, key="layer_river")

        # MISSION REPORTS SECTION
        st.markdown("<div class='sidebar-section-title'>MISSION REPORTS</div>", unsafe_allow_html=True)
        st.caption("📄 SITREP, 📊 JSON, & 🗺️ GeoJSON downloads ready in bottom footer.")

    return results
