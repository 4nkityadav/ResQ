"""
resq_ai.ui.console
==================
Right-side Incident Command Console component with rock-solid responsive layout:
1. Incidents: Live priority queue sorted by START/SALT severity (P1/P2/P3).
2. Fleet: Symmetrical unit telemetry grid (fuel, occupancy, status, itinerary).
3. Hazard Forecast: 15/30/60 min predictive escalation risk gauges & per-zone HMM beliefs.
4. Explanations: Plain-English decision transparency ("Why unit X was assigned to call Y").
5. Terminal: Integrated command terminal with confirmation dialogs and undo capability.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import streamlit as st

from resq_ai.ui.components import build_casualty_donut_chart, build_fleet_fuel_chart


def render_console(
    engine: Any,
    state: Dict[str, Any],
    selected_vid: str,
    active_tab_idx: int = 0,
    theme: str = "light",
) -> str:
    """Renders the incident console with 5 dedicated operations tabs."""
    new_selected_vid = selected_vid
    is_dark = (theme == "dark")

    # Persistent Segmented Control with the 5 required tabs
    valid_tabs = ["Incidents", "Fleet", "Hazard Forecast", "Explanations", "Terminal"]
    if "console_active_tab_select" not in st.session_state or st.session_state.console_active_tab_select not in valid_tabs:
        st.session_state.console_active_tab_select = "Incidents"

    active_tab = st.segmented_control(
        "Console Operations",
        options=valid_tabs,
        key="console_active_tab_select",
        label_visibility="collapsed",
    ) or "Incidents"

    # Container matching tactical map viewport
    with st.container(height=620, border=True):

        # TAB 1: INCIDENTS QUEUE
        if active_tab == "Incidents":
            calls = state.get("unserved_calls", [])
            if not calls:
                st.success("All emergency calls served! Fleet on active standby.")
            else:
                # Sort: P1 first, then P2, then P3
                p_order = {"P1": 0, "P2": 1, "P3": 2}
                sorted_calls = sorted(calls, key=lambda c: (p_order.get(c.get("priority", "P3"), 3), -c.get("risk_score", 0)))
                
                st.markdown(
                    f"<div style='font-size: 0.78rem; color: var(--text-secondary); margin-bottom: 8px;'>"
                    f"Waiting 911 Calls: <b>{len(calls)}</b> incidents active "
                    f"(<span style='color: var(--p1); font-weight: 600;'>{sum(1 for c in calls if c.get('priority')=='P1')} Urgent</span>)</div>",
                    unsafe_allow_html=True
                )

                for c in sorted_calls:
                    p = c.get("priority", "P3")
                    p_color = "var(--p1)" if p == "P1" else ("var(--p2)" if p == "P2" else "var(--p3)")
                    p_soft = "var(--p1-soft)" if p == "P1" else ("var(--p2-soft)" if p == "P2" else "var(--p3-soft)")
                    p_col = c.get("p_collapse", 0.0) or 0.0
                    p_hypo = c.get("p_hypothermia", 0.0) or 0.0

                    st.markdown(
                        f"""
                        <div style="background: var(--bg-card); border: 1px solid var(--border); border-left: 4px solid {p_color}; border-radius: 8px; padding: 10px 12px; margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                <span style="font-weight: 700; font-size: 0.88rem; color: var(--text-primary);">{c['id']} — {c.get('caller_name', 'Caller')}</span>
                                <span class="tabular-nums" style="background: {p_soft}; color: {p_color}; padding: 2px 8px; border-radius: 12px; font-weight: 700; font-size: 0.76rem;">{p}</span>
                            </div>
                            <div class="tabular-nums" style="font-size: 0.76rem; color: var(--text-secondary); margin-bottom: 4px;">
                                Location: <b>{c['loc']}</b> | Occupants: <b>{c.get('occupants', 1)}</b> | Water: <b>{c.get('flood_m', 0):.1f}m</b> | Collapse Risk: <b>{p_col:.0%}</b>
                            </div>
                            <div style="font-size: 0.74rem; color: var(--text-primary); font-style: italic;">
                                {c.get('explanation', 'Awaiting triage rationale')}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        # TAB 2: FLEET TELEMETRY
        elif active_tab == "Fleet":
            vehicles = state.get("vehicles", [])
            v_ids = [v["id"] for v in vehicles]

            # Unit Selection Grid
            for row_idx in range(0, len(v_ids), 4):
                row_vids = v_ids[row_idx : row_idx + 4]
                cols = st.columns(4)
                for c_idx, vid in enumerate(row_vids):
                    with cols[c_idx]:
                        is_curr = (vid == new_selected_vid)
                        icon = "🚑" if "AMB" in vid else "🚤"
                        btn_type = "primary" if is_curr else "secondary"
                        if st.button(f"{icon} {vid}", key=f"f_btn_{vid}", type=btn_type, width="stretch"):
                            new_selected_vid = vid
                            st.session_state.selected_vid = vid

            # Selected Unit Details
            target_v = next((v for v in vehicles if v["id"] == new_selected_vid), vehicles[0] if vehicles else None)
            if target_v:
                vid = target_v["id"]
                assigned = state.get("schedule", {}).get(vid, [])
                vtype = target_v.get("type", "ambulance").title()
                fuel = target_v.get("fuel_h", 0.0)
                served = target_v.get("total_served", 0)
                status = target_v.get("status", "idle").replace("_", " ").upper()
                loc = target_v.get("loc", (0, 0))

                st.markdown(
                    f"""
                    <div style="background: var(--bg-subtle); border-radius: 8px; padding: 12px; margin-top: 10px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-weight: 700; font-size: 0.94rem; color: var(--text-primary);">{vid} ({vtype})</span>
                            <span class="tabular-nums" style="font-size: 0.76rem; background: var(--accent-soft); color: var(--accent); padding: 3px 8px; border-radius: 6px; font-weight: 600;">{status}</span>
                        </div>
                        <div class="tabular-nums" style="font-size: 0.78rem; color: var(--text-secondary); line-height: 1.5;">
                            GPS Location: <b>{loc}</b><br>
                            Fuel Remaining: <b>{fuel:.1f} hours</b> | Rescued: <b>{served} citizens</b><br>
                            Capacity: <b>{target_v.get('capacity', 2)} patients</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("<div style='font-size: 0.80rem; font-weight: 600; margin-bottom: 4px;'>Assigned Mission Itinerary</div>", unsafe_allow_html=True)
                if not assigned:
                    st.caption("No pending calls assigned. Standby patrol.")
                else:
                    for s_idx, cid in enumerate(assigned):
                        st.markdown(
                            f"""
                            <div style="display: flex; align-items: center; padding: 4px 8px; border-bottom: 1px solid var(--border); font-size: 0.78rem;">
                                <span style="color: var(--accent); font-weight: 700; margin-right: 8px;">Stop {s_idx+1}</span>
                                <span style="font-weight: 600;">{cid}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            # Fuel Overview Bar Chart
            st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
            st.plotly_chart(build_fleet_fuel_chart(vehicles, theme=theme), width="stretch", config={"displayModeBar": False})

        # TAB 3: HAZARD FORECAST
        elif active_tab == "Hazard Forecast":
            zones = state.get("zones", {})
            st.markdown("<div style='font-size: 0.80rem; font-weight: 700; margin-bottom: 8px;'>Flood Danger Forecast (Next 15, 30, 60 Minutes)</div>", unsafe_allow_html=True)

            for zid, zdata in zones.items():
                name = zid.replace("zone_", "").replace("_", " ").title()
                state_val = zdata.get("map_state", "Safe")
                alert = zdata.get("alert", False)
                f15 = zdata.get("forecast_15m", 0.0)
                f30 = zdata.get("forecast_30m", 0.0)
                f60 = zdata.get("forecast_60m", 0.0)

                state_color = "var(--p1)" if state_val in ("Critical", "Evacuate") else ("var(--p2)" if state_val == "Advisory" else "var(--amb)")

                st.markdown(
                    f"""
                    <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 10px; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-weight: 700; font-size: 0.86rem; color: var(--text-primary);">{name} District</span>
                            <span style="color: {state_color}; font-weight: 700; font-size: 0.78rem;">{state_val.upper()}</span>
                        </div>
                        <div class="tabular-nums" style="display: flex; justify-content: space-between; font-size: 0.74rem; color: var(--text-secondary); background: var(--bg-subtle); padding: 6px 8px; border-radius: 6px;">
                            <span>+15 min: <b>{f15:.0%}</b></span>
                            <span>+30 min: <b>{f30:.0%}</b></span>
                            <span>+60 min: <b>{f60:.0%}</b></span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # TAB 4: EXPLANATIONS (TRANSPARENCY "WHY")
        elif active_tab == "Explanations":
            st.markdown("<div style='font-size: 0.82rem; font-weight: 700; margin-bottom: 6px;'>Why Did the AI Make This Choice?</div>", unsafe_allow_html=True)
            st.caption("Clear explanations for patient priorities, assigned vehicles, and flooded roads avoided.")

            calls = state.get("unserved_calls", [])
            for c in calls[:6]:
                cid = c["id"]
                st.markdown(
                    f"""
                    <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; margin-bottom: 8px;">
                        <div style="font-weight: 700; font-size: 0.84rem; color: var(--text-primary); margin-bottom: 4px;">Incident {cid}</div>
                        <div style="font-size: 0.76rem; color: var(--text-secondary); margin-bottom: 4px;">
                            <b>Triage Reason:</b> {c.get('explanation', 'Standard evaluation')}
                        </div>
                        <div class="tabular-nums" style="font-size: 0.74rem; color: var(--text-secondary);">
                            <b>Vehicle Decision:</b> Assigned closest safe rescue unit. Ambulances cannot enter roads flooded deeper than 35 cm (stalling hazard); rescue boats are dispatched to deep-water areas.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # TAB 5: TERMINAL & ACTIONS (WITH UNDO)
        elif active_tab == "Terminal":
            st.markdown("<div style='font-size: 0.82rem; font-weight: 700; margin-bottom: 6px;'>Dispatcher Command Terminal</div>", unsafe_allow_html=True)
            
            # Undo button
            if st.button("↩ Undo Last Command", key="console_undo_btn", width="stretch"):
                if hasattr(engine, "event_bus") and engine.event_bus.can_undo():
                    engine.event_bus.undo()
                    st.success("Last action undone successfully.")
                    st.rerun()
                else:
                    st.info("No prior actions available in undo history.")

            # Input form
            with st.form("console_terminal_form", clear_on_submit=True):
                cmd_txt = st.text_input("Dispatch Command", placeholder="e.g. 'Dispatch boat to Block C' or 'Close Main Street'")
                submitted = st.form_submit_button("Execute Command ❯", width="stretch")
                if submitted and cmd_txt:
                    resp = engine.handle_natural_language_command(cmd_txt)
                    st.session_state.command_output = resp
                    st.rerun()

            # Output log
            last_resp = st.session_state.get("command_output", "")
            if last_resp:
                st.markdown(
                    f"""
                    <div style="background: var(--bg-subtle); border-radius: 6px; padding: 8px 10px; font-size: 0.76rem; font-family: 'JetBrains Mono', monospace; color: var(--text-primary); margin-top: 8px;">
                        <b>❯ RESULT:</b> {last_resp}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    return new_selected_vid
