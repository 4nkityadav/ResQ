"""
resq_ai.ui.app
==============
Main Incident Command Dashboard presentation layer for ResQ-AI.
Calm, low-contrast, professional design system with 1px borders,
12-14px rounded corners, tabular typography, and safe component wrapping.
"""

from __future__ import annotations
import os
import sys
from typing import Dict, Any, Optional

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st

from resq_ai.orchestrator.engine import OrchestratorEngine
from resq_ai.core.models import (
    VehicleType,
    DistressCall,
    OccupantType,
    PriorityLevel,
    HazardLevel,
)
from resq_ai.ui.theme import get_css
from resq_ai.ui.navbar import render_navbar
from resq_ai.ui.header import render_header
from resq_ai.ui.sidebar import render_sidebar, SCENARIOS
from resq_ai.ui.kpis import render_kpis
from resq_ai.ui.algo_strip import render_algo_strip
from resq_ai.ui.map import build_tactical_map
from resq_ai.ui.console import render_console
from resq_ai.ui.terminal import render_terminal
from resq_ai.ui.footer import render_footer


def load_orchestrator(scenario_file: str = "flash_flood.yaml") -> OrchestratorEngine:
    """Loads and pre-runs the OrchestratorEngine for 12 steps so initial screen has real data."""
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "scenarios", scenario_file)
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "default_scenario.yaml")
    
    engine = OrchestratorEngine(config_path=cfg_path, seed=42)
    # Pre-run 12 steps so all algorithms, vehicles, routes, and triage states are populated on first load
    for _ in range(12):
        engine.step()
    return engine


def safe_render(name: str, render_fn, *args, **kwargs):
    """Safely executes a visual rendering block without crashing the page on failure."""
    try:
        return render_fn(*args, **kwargs)
    except Exception as e:
        st.error(f"[{name}] Component temporarily unavailable: {e}")
        return None


def main():
    st.set_page_config(
        page_title="ResQ-AI // Civil Defense Incident Command",
        page_icon="🚨",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 1. Initialize Session State
    if "theme" not in st.session_state:
        qp_theme = st.query_params.get("theme", "light") if hasattr(st, "query_params") else "light"
        st.session_state.theme = qp_theme if qp_theme in ("light", "dark") else "light"
    if "selected_scenario" not in st.session_state:
        st.session_state.selected_scenario = "flash_flood.yaml"
    if "engine" not in st.session_state:
        st.session_state.engine = load_orchestrator(st.session_state.selected_scenario)
    if "command_output" not in st.session_state:
        st.session_state.command_output = "Incident Command System operational. Field units active on mission."
    if "selected_vid" not in st.session_state:
        st.session_state.selected_vid = "AMB-1"
    if "auto_play" not in st.session_state:
        st.session_state.auto_play = False
    if "speed_delay" not in st.session_state:
        st.session_state.speed_delay = 1.2

    # Inject styling
    st.markdown(get_css(st.session_state.theme), unsafe_allow_html=True)

    engine: OrchestratorEngine = st.session_state.engine
    state = engine.get_dashboard_state()

    # 2. Render Sidebar
    sidebar_actions = safe_render(
        "Sidebar",
        render_sidebar,
        engine=engine,
        state=state,
        selected_scenario=st.session_state.selected_scenario,
        theme_choice=st.session_state.theme,
    ) or {}

    # Handle Theme Change from Sidebar
    if sidebar_actions.get("theme") and sidebar_actions["theme"] != st.session_state.theme:
        st.session_state.theme = sidebar_actions["theme"]
        if hasattr(st, "query_params"):
            st.query_params["theme"] = sidebar_actions["theme"]
        st.rerun()

    # Handle Scenario Change
    if sidebar_actions.get("scenario_changed"):
        new_scen = sidebar_actions["scenario_changed"]
        st.session_state.selected_scenario = new_scen
        st.session_state.engine = load_orchestrator(new_scen)
        st.session_state.command_output = f"Loaded scenario: {SCENARIOS.get(new_scen, {}).get('title', new_scen)}. Pre-ran 12 minutes of telemetry."
        st.rerun()

    # Handle Simulation Restart
    if sidebar_actions.get("restart_sim"):
        st.session_state.engine = load_orchestrator(st.session_state.selected_scenario)
        st.session_state.command_output = "Simulation reset to T+12m baseline."
        st.rerun()

    # Handle Event Injections
    if sidebar_actions.get("inject_call"):
        engine.sim.call_counter += 1
        cid = f"CALL-{engine.sim.call_counter:03d}"
        rng = engine.sim.rng
        loc = (int(rng.integers(10, 40)), int(rng.integers(28, 42)))
        is_deep = loc[1] >= 32
        new_call = DistressCall(
            call_id=cid,
            location=loc,
            timestamp_step=state["step"],
            caller_name=f"Caller-{engine.sim.call_counter}",
            occupants_type=OccupantType.ELDERLY if rng.random() < 0.5 else OccupantType.CHILDREN,
            occupant_count=int(rng.integers(1, 4)),
            building_age="Old",
            construction_type="Wood" if is_deep else "Masonry",
            reported_flood_m=1.25 if is_deep else 0.20,
        )
        engine.calls[cid] = new_call
        zid = engine.grid.get_zone_for_coord(loc[0], loc[1])
        z_sum = engine.hmm_tracker.get_zone_summary(zid)
        engine.triage.evaluate_call(new_call, hazard_state=z_sum["map_state"])
        cm = engine.router.compute_cost_matrix(list(engine.vehicles.values()), [new_call])
        engine.current_schedule, _ = engine.sa_optimizer.warm_start_replan(engine.current_schedule, [new_call], cm)
        st.session_state.command_output = f"New emergency call {cid} received at {loc}! Priority: {new_call.priority.value} ({new_call.explanation})."
        st.rerun()

    if sidebar_actions.get("inject_bridge"):
        broken = False
        for bx, by in engine.grid.bridge_locations:
            cell = engine.grid.get_cell(bx, by)
            if cell and not cell.is_blocked:
                cell.is_blocked = True
                cell.water_level = 1.0
                engine.replanning_events_count += 1
                reports = engine.router.handle_obstacle_update((bx, by))
                st.session_state.command_output = f"⚠️ Bridge at ({bx}, {by}) collapsed! Rerouted {len(reports)} vehicle(s) onto safe roads."
                broken = True
                break
        if not broken:
            st.session_state.command_output = "All river bridges are currently closed."
        st.rerun()

    if sidebar_actions.get("inject_levee"):
        # Breach riverfront levee at columns 20 to 30 along river edge y=33
        for x in range(20, 31):
            c = engine.grid.get_cell(x, 33)
            if c:
                c.water_level = 1.35
                c.is_blocked = True
        engine.replanning_events_count += 1
        engine.router.handle_obstacle_update((25, 33))
        st.session_state.command_output = "🌊 Riverfront levee breached! Water depth reached 1.35m. Ambulances safely steered away from deep water."
        st.rerun()

    # Update speed delay from sidebar
    st.session_state.speed_delay = sidebar_actions.get("speed_delay", 1.2)

    # 3. Render Top Navigation Bar (Full Structured Navbar)
    scenario_title = SCENARIOS.get(st.session_state.selected_scenario, {}).get("title", "Flash Flood Emergency")
    scenario_desc = SCENARIOS.get(st.session_state.selected_scenario, {}).get("desc", "Civil defense operations active.")

    navbar_actions = safe_render(
        "Navbar",
        render_navbar,
        state=state,
        scenario_name=scenario_title,
        current_theme=st.session_state.theme,
        is_auto_playing=st.session_state.auto_play,
    ) or {}

    # Handle Theme change from Navbar
    if navbar_actions.get("theme_toggled"):
        new_theme = navbar_actions["theme_toggled"]
        st.session_state.theme = new_theme
        if hasattr(st, "query_params"):
            st.query_params["theme"] = new_theme
        st.rerun()

    # Handle Step actions from Navbar
    if navbar_actions.get("step_1"):
        engine.step()
        st.rerun()
    if navbar_actions.get("step_5"):
        for _ in range(5):
            engine.step()
        st.rerun()
    if "auto_play" in navbar_actions and navbar_actions["auto_play"] != st.session_state.auto_play:
        st.session_state.auto_play = navbar_actions["auto_play"]

    # 3.5. Render Scenario & Mission Ribbon Header
    safe_render(
        "Header",
        render_header,
        state=state,
        scenario_name=scenario_title,
        scenario_desc=scenario_desc,
    )

    # 4. Define Dynamic Fragment for Auto-Play and Live Refresh
    # If auto_play is enabled, run_every triggers periodic execution of the dashboard cockpit
    run_interval = float(st.session_state.speed_delay) if st.session_state.auto_play else None

    @st.fragment(run_every=run_interval)
    def render_cockpit_fragment():
        # Advance simulation if auto-playing
        if st.session_state.auto_play:
            engine.step()

        current_state = engine.get_dashboard_state()

        # Row 1: Four Equal KPI Cards
        safe_render("KPIs", render_kpis, state=current_state, engine=engine)

        st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)

        # Row 2: Four Equal Algorithm Cards
        safe_render("Algorithms", render_algo_strip, state=current_state)

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

        # Row 3: Cockpit (2 columns, 3:2 ratio)
        map_col, console_col = st.columns([1.5, 1.0])

        with map_col:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 0.90rem; color: var(--text-primary);">
                        Tactical Urban Vector Map (50×50 CAD Grid)
                    </span>
                    <span class="tabular-nums" style="font-size: 0.74rem; color: var(--text-secondary);">
                        Tracking Unit: <b>{st.session_state.selected_vid}</b>
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            tactical_fig = build_tactical_map(
                grid=engine.grid,
                vehicles=current_state["vehicles"],
                calls=current_state["unserved_calls"],
                schedule=current_state["schedule"],
                selected_vid=st.session_state.selected_vid,
                show_landmarks=sidebar_actions.get("show_landmarks", True),
                show_routes=sidebar_actions.get("show_routes", True),
                show_flood=sidebar_actions.get("show_flood", True),
                show_river=sidebar_actions.get("show_river", True),
                theme=st.session_state.theme,
            )
            map_evt = st.plotly_chart(
                tactical_fig,
                width="stretch",
                config={"displayModeBar": False},
                theme=None,
                on_select="rerun",
                selection_mode="points",
                key="tactical_map_plot",
            )
            # Interactive click routing from map to console
            if map_evt and isinstance(map_evt, dict):
                sel_pts = map_evt.get("selection", {}).get("points", [])
                if sel_pts:
                    pt = sel_pts[0]
                    px, py = pt.get("x"), pt.get("y")
                    for v in current_state["vehicles"]:
                        if v["loc"] == (px, py):
                            st.session_state.selected_vid = v["id"]
                            st.session_state.console_active_tab_select = "Fleet"
                            break
                    else:
                        for c in current_state["unserved_calls"]:
                            if c["loc"] == (px, py):
                                st.session_state.console_active_tab_select = "Incidents"
                                break

        with console_col:
            st.markdown(
                """
                <div style="margin-bottom: 6px;">
                    <span style="font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 0.90rem; color: var(--text-primary);">
                        Incident Command Console
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            new_vid = render_console(
                engine=engine,
                state=current_state,
                selected_vid=st.session_state.selected_vid,
                theme=st.session_state.theme,
            )
            if new_vid and new_vid != st.session_state.selected_vid:
                st.session_state.selected_vid = new_vid

    # Execute cockpit fragment
    render_cockpit_fragment()

    # 4.5. Bottom Timeline Scrubber with Event Milestones
    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        scrub_col1, scrub_col2 = st.columns([1.2, 5.0])
        with scrub_col1:
            st.markdown(
                f"""
                <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 0.86rem; color: var(--text-primary); margin-top: 8px;">
                    Mission Timeline
                </div>
                <div class="tabular-nums" style="font-size: 0.76rem; color: var(--accent); font-weight: 600;">
                    Current: T+{state.get('step', 12):02d}m
                </div>
                """,
                unsafe_allow_html=True
            )
        with scrub_col2:
            step_val = state.get("step", 12)
            max_s = max(30, step_val + 10)
            scrub_step = st.slider(
                "Mission Scrubber",
                min_value=0,
                max_value=max_s,
                value=step_val,
                step=1,
                label_visibility="collapsed",
            )
            # Historical event milestone pills
            st.markdown(
                """
                <div style="display: flex; gap: 8px; overflow-x: auto; font-size: 0.72rem; padding-top: 2px;">
                    <span style="background: var(--bg-subtle); padding: 2px 8px; border-radius: 4px; color: var(--text-secondary);">T+00 🏁 Scenario Start</span>
                    <span style="background: var(--p3-soft); padding: 2px 8px; border-radius: 4px; color: var(--p3);">T+05 🌊 Surge Warning</span>
                    <span style="background: var(--p1-soft); padding: 2px 8px; border-radius: 4px; color: var(--p1);">T+08 💥 Bridge Collapse</span>
                    <span style="background: var(--p2-soft); padding: 2px 8px; border-radius: 4px; color: var(--p2);">T+12 🚨 Call Burst</span>
                    <span style="background: var(--accent-soft); padding: 2px 8px; border-radius: 4px; color: var(--accent);">T+15 🎯 Live Operations</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    # 5. Row 4: Bottom Command Terminal
    def handle_command(cmd: str):
        resp = engine.handle_natural_language_command(cmd)
        st.session_state.command_output = resp
        st.rerun()

    safe_render(
        "Terminal",
        render_terminal,
        engine=engine,
        last_response=st.session_state.command_output,
        on_command=handle_command,
    )

    # 6. Full Structured Bottom Status Footer
    safe_render("Footer", render_footer, state=state, engine=engine)


if __name__ == "__main__":
    try:
        from streamlit.runtime import exists
        if not exists():
            # If executed directly via `python resq_ai/ui/app.py`, automatically invoke streamlit run!
            import subprocess
            subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)] + sys.argv[1:])
        else:
            main()
    except Exception:
        main()
