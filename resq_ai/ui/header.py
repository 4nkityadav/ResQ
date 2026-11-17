"""
resq_ai.ui.header
=================
Mission Scenario & Operational Ribbon component for ResQ-AI Incident Command Center.
Displays the active scenario title, real-time mission phase, and operational brief.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import streamlit as st


def render_header(
    state: Dict[str, Any],
    scenario_name: str,
    scenario_desc: str = "River levee breach, submerged roads, and rising casualties.",
) -> None:
    """Renders the clean mission scenario ribbon."""
    st.markdown(
        f"""
        <div class="app-header">
            <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
                <span style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.05rem; font-weight: 800; color: var(--text-primary);">
                    {scenario_name}
                </span>
                <span style="font-size: 0.80rem; color: var(--text-secondary); border-left: 2px solid var(--border); padding-left: 10px;">
                    {scenario_desc}
                </span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; background: var(--bg-subtle); color: var(--text-secondary); padding: 3px 8px; border-radius: 4px; border: 1px solid var(--border);">
                    Phase: Dynamic Response
                </span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; background: var(--accent-soft); color: var(--accent); padding: 3px 8px; border-radius: 4px; border: 1px solid var(--accent); font-weight: 700;">
                    8 Units Deployed
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
