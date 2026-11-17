"""
resq_ai.ui.kpis
===============
Four equal KPI cards in one row:
- Lives rescued (number, percent, thin progress bar)
- Active casualties (number plus P1/P2/P3 chips)
- Flooded urban cells (number, percent, bar)
- Infrastructure (road cells closed, bridges open, D* Lite replan count)
"""

from __future__ import annotations
from typing import Dict, Any
import streamlit as st


def render_kpis(state: Dict[str, Any], engine: Any) -> None:
    """Renders the four equal KPI cards aligned to the grid."""
    vehicles = state.get("vehicles", [])
    unserved = state.get("unserved_calls", [])
    
    # 1. Lives Rescued
    rescued = sum(v.get("total_served", 0) for v in vehicles)
    total_calls = rescued + len(unserved)
    rescue_pct = (rescued / total_calls * 100.0) if total_calls > 0 else 100.0
    
    # 2. Active Casualties
    p1_count = sum(1 for c in unserved if c.get("priority") == "P1")
    p2_count = sum(1 for c in unserved if c.get("priority") == "P2")
    p3_count = sum(1 for c in unserved if c.get("priority") == "P3")
    
    # 3. Flooded Cells
    flooded = state.get("flooded_count", 0)
    total_cells = state.get("grid_width", 50) * state.get("grid_height", 50)
    flood_pct = (flooded / max(1, total_cells)) * 100.0
    
    # 4. Infrastructure
    blocked = state.get("blocked_count", 0)
    open_bridges = 0
    total_bridges = len(engine.grid.bridge_locations) if hasattr(engine, "grid") else 4
    if hasattr(engine, "grid"):
        for bx, by in engine.grid.bridge_locations:
            cell = engine.grid.get_cell(bx, by)
            if cell and not cell.is_blocked:
                open_bridges += 1
    replans = getattr(engine, "replanning_events_count", 0)

    # 4 Columns
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="kpi-box">
                <div>
                    <div class="kpi-title">Lives Rescued</div>
                    <div class="kpi-val">{rescued} <span style="font-size:0.9rem; font-weight:500; color:var(--text-secondary);">citizens</span></div>
                </div>
                <div>
                    <div class="kpi-footer">
                        <span>{rescue_pct:.1f}% of total calls</span>
                    </div>
                    <div class="progress-bar-bg">
                        <div class="progress-bar-fill" style="width: {min(100.0, rescue_pct)}%; background: var(--amb);"></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="kpi-box">
                <div>
                    <div class="kpi-title">Active Casualties</div>
                    <div class="kpi-val">{len(unserved)} <span style="font-size:0.9rem; font-weight:500; color:var(--text-secondary);">waiting</span></div>
                </div>
                <div>
                    <div class="kpi-footer" style="gap: 5px; margin-top: 4px;">
                        <span class="chip-p1">P1: {p1_count}</span>
                        <span class="chip-p2">P2: {p2_count}</span>
                        <span class="chip-p3">P3: {p3_count}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="kpi-box">
                <div>
                    <div class="kpi-title">Flooded Urban Cells</div>
                    <div class="kpi-val">{flooded} <span style="font-size:0.85rem; font-weight:500; color:var(--text-secondary);">/ {total_cells}</span></div>
                </div>
                <div>
                    <div class="kpi-footer">
                        <span>{flood_pct:.1f}% city coverage</span>
                    </div>
                    <div class="progress-bar-bg">
                        <div class="progress-bar-fill" style="width: {min(100.0, flood_pct)}%; background: var(--boat);"></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="kpi-box">
                <div>
                    <div class="kpi-title">Infrastructure Status</div>
                    <div class="kpi-val">{blocked} <span style="font-size:0.85rem; font-weight:500; color:var(--text-secondary);">closed</span></div>
                </div>
                <div>
                    <div class="kpi-footer">
                        <span>Bridges open: <b>{open_bridges}/{total_bridges}</b> • Replans: <b>{replans}</b></span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
