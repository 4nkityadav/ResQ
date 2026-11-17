"""
resq_ai.ui.components
=====================
High-performance, interactive visual components for ResQ-AI using Plotly.
Provides tactical vector maps with landmark pins, active route vector lines,
casualty distribution charts, fleet fuel telemetry, and situation report generators.
"""

from __future__ import annotations
import json
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import plotly.graph_objects as go

from resq_ai.core.models import CellType, VehicleType, PriorityLevel
from resq_ai.sim.city_grid import CityGrid


# Predefined major urban landmarks
CITY_LANDMARKS = [
    {"name": "🏥 Hospital", "pos": (5, 5), "color": "#10b981"},
    {"name": "🏢 North Uptown", "pos": (15, 8), "color": "#38bdf8"},
    {"name": "🏙️ Downtown Center", "pos": (25, 25), "color": "#818cf8"},
    {"name": "⚓ River Marina", "pos": (25, 34), "color": "#06b6d4"},
    {"name": "🏘️ South Valley", "pos": (30, 45), "color": "#fbbf24"},
    {"name": "📦 Block A", "pos": (10, 10), "color": "#64748b"},
    {"name": "📦 Block B", "pos": (35, 12), "color": "#64748b"},
    {"name": "📦 Block C", "pos": (22, 34), "color": "#64748b"},
    {"name": "📦 Block D", "pos": (40, 42), "color": "#64748b"},
]


def build_interactive_map(
    grid: CityGrid,
    vehicles: List[Dict[str, Any]],
    calls: List[Dict[str, Any]],
    schedule: Dict[str, List[str]],
    selected_vid: Optional[str] = None,
    show_landmarks: bool = True,
    show_routes: bool = True,
    show_flood: bool = True,
    show_river: bool = True,
) -> go.Figure:
    """
    Constructs an interactive, high-contrast Plotly vector map of the 50x50 urban disaster grid.
    Supports layer toggles, vehicle selection highlighting, and landmark labels.
    """
    fig = go.Figure()
    W, H = grid.width, grid.height

    # 1. Base Grid Layer: Ambient Streets, River, and Flooded Water
    river_x, river_y, river_text = [], [], []
    flood_x, flood_y, flood_text = [], [], []
    road_x, road_y = [], []

    for y in range(0, H, 2):  # Downsample ambient street dots for crisp 60fps rendering
        for x in range(0, W, 2):
            cell = grid.get_cell(x, y)
            if not cell:
                continue
            if cell.cell_type == CellType.RIVER:
                river_x.append(x)
                river_y.append(y)
                river_text.append(f"🌊 <b>River Channel</b> ({x}, {y})<br>Depth: {cell.water_level:.1f}m<br><i>Traversable by rescue boats only</i>")
            elif cell.is_flooded:
                flood_x.append(x)
                flood_y.append(y)
                flood_text.append(f"💧 <b>Flooded Street</b> ({x}, {y})<br>Water: {cell.water_level:.2f}m")
            else:
                road_x.append(x)
                road_y.append(y)

    # Ambient city road network
    fig.add_trace(go.Scatter(
        x=road_x, y=road_y,
        mode="markers",
        marker=dict(size=3, color="#172033", opacity=0.7),
        hoverinfo="skip",
        name="Streets",
        showlegend=False,
    ))

    # River Channel
    if show_river and river_x:
        fig.add_trace(go.Scatter(
            x=river_x, y=river_y,
            mode="markers",
            marker=dict(size=8, color="#0284c7", symbol="square", opacity=0.85),
            hovertext=river_text,
            hoverinfo="text",
            name="River Channel",
        ))

    # Flooded Streets
    if show_flood and flood_x:
        fig.add_trace(go.Scatter(
            x=flood_x, y=flood_y,
            mode="markers",
            marker=dict(size=7, color="#06b6d4", symbol="square", opacity=0.65),
            hovertext=flood_text,
            hoverinfo="text",
            name="Flood Water",
        ))

    # Bridges
    bridge_x, bridge_y, bridge_text = [], [], []
    for bx, by in grid.bridge_locations:
        bridge_x.append(bx)
        bridge_y.append(by)
        cell = grid.get_cell(bx, by)
        b_status = "COLLAPSED (Blocked)" if (cell and cell.is_blocked) else "OPERATIONAL"
        bridge_text.append(f"🌉 <b>Bridge</b> ({bx}, {by})<br>Status: {b_status}")

    fig.add_trace(go.Scatter(
        x=bridge_x, y=bridge_y,
        mode="markers",
        marker=dict(size=9, color="#94a3b8", symbol="diamond", line=dict(width=1, color="#f8fafc")),
        hovertext=bridge_text,
        hoverinfo="text",
        name="Bridges",
    ))

    # 2. Blocked / Collapsed Roads (Red X markers)
    block_x, block_y, block_text = [], [], []
    for y in range(H):
        for x in range(W):
            cell = grid.get_cell(x, y)
            if cell and cell.is_blocked:
                block_x.append(x)
                block_y.append(y)
                block_text.append(f"🚫 <b>ROAD HAZARD</b> at ({x}, {y})<br>Impassable: D* Lite rerouted traffic")

    if block_x:
        fig.add_trace(go.Scatter(
            x=block_x, y=block_y,
            mode="markers",
            marker=dict(size=12, color="#ef4444", symbol="x", line=dict(width=2.5, color="#ffffff")),
            hovertext=block_text,
            hoverinfo="text",
            name="Closed Hazard",
        ))

    # 3. Landmark Pins
    if show_landmarks:
        lm_x = [lm["pos"][0] for lm in CITY_LANDMARKS]
        lm_y = [lm["pos"][1] for lm in CITY_LANDMARKS]
        lm_text = [f"📍 <b>{lm['name']}</b> ({lm['pos'][0]}, {lm['pos'][1]})" for lm in CITY_LANDMARKS]
        lm_labels = [lm["name"] for lm in CITY_LANDMARKS]
        fig.add_trace(go.Scatter(
            x=lm_x, y=lm_y,
            mode="markers+text",
            marker=dict(size=10, color="#6366f1", symbol="circle-open", line=dict(width=2, color="#a5b4fc")),
            text=lm_labels,
            textposition="top right",
            textfont=dict(size=9, color="#93c5fd", family="monospace"),
            hovertext=lm_text,
            hoverinfo="text",
            name="Landmarks",
            showlegend=False,
        ))

    # 4. Active Vehicle Routes (Dashed Lines)
    if show_routes:
        for v in vehicles:
            path = v.get("path", [])
            vid = v["id"]
            if path and len(path) > 1:
                px = [p[0] for p in path]
                py = [p[1] for p in path]
                is_selected = (selected_vid == vid)
                vcolor = "#34d399" if v["type"] == "ambulance" else "#38bdf8"
                line_width = 3.5 if is_selected else 2.0
                dash_style = "solid" if is_selected else "dot"
                fig.add_trace(go.Scatter(
                    x=px, y=py,
                    mode="lines",
                    line=dict(color="#f59e0b" if is_selected else vcolor, width=line_width, dash=dash_style),
                    hoverinfo="skip",
                    showlegend=False,
                ))

    # 5. Distress Calls (Pulsing Circles)
    p1_x, p1_y, p1_t = [], [], []
    p2_x, p2_y, p2_t = [], [], []
    p3_x, p3_y, p3_t = [], [], []

    for c in calls:
        cx, cy = c["loc"]
        info = (
            f"<b>🆘 {c['id']}</b> [{c['priority']}]<br>"
            f"Location: ({cx}, {cy})<br>"
            f"Occupants: {c.get('occupants', 1)} people<br>"
            f"Flood Water: {c.get('flood_m', 0.0):.2f}m<br>"
            f"Collapse Risk: {c['p_collapse']:.0%}<br>"
            f"<i>{c['explanation']}</i>"
        )
        if c["priority"] == "P1":
            p1_x.append(cx); p1_y.append(cy); p1_t.append(info)
        elif c["priority"] == "P2":
            p2_x.append(cx); p2_y.append(cy); p2_t.append(info)
        else:
            p3_x.append(cx); p3_y.append(cy); p3_t.append(info)

    if p1_x:
        fig.add_trace(go.Scatter(
            x=p1_x, y=p1_y,
            mode="markers+text",
            marker=dict(size=14, color="#ef4444", line=dict(width=2, color="#ffffff")),
            text=["P1" for _ in p1_x],
            textposition="top center",
            textfont=dict(size=9, color="#f87171", family="monospace"),
            hovertext=p1_t,
            hoverinfo="text",
            name="Critical (P1)",
        ))
    if p2_x:
        fig.add_trace(go.Scatter(
            x=p2_x, y=p2_y,
            mode="markers+text",
            marker=dict(size=11, color="#f59e0b", line=dict(width=1.5, color="#ffffff")),
            text=["P2" for _ in p2_x],
            textposition="top center",
            textfont=dict(size=8, color="#fbbf24", family="monospace"),
            hovertext=p2_t,
            hoverinfo="text",
            name="Urgent (P2)",
        ))
    if p3_x:
        fig.add_trace(go.Scatter(
            x=p3_x, y=p3_y,
            mode="markers",
            marker=dict(size=9, color="#10b981", line=dict(width=1, color="#34d399")),
            hovertext=p3_t,
            hoverinfo="text",
            name="Delayed (P3)",
        ))

    # 6. Rescue Fleet (Ambulances and Boats)
    amb_x, amb_y, amb_t, amb_lbl = [], [], [], []
    boat_x, boat_y, boat_t, boat_lbl = [], [], [], []

    for v in vehicles:
        vx, vy = v["loc"]
        assigned = schedule.get(v["id"], [])
        route_str = " ➔ ".join(assigned) if assigned else "Standby (Patrolling)"
        v_info = (
            f"<b>{v['id']} ({v['type'].upper()})</b><br>"
            f"Position: ({vx}, {vy})<br>"
            f"Status: {v['status']}<br>"
            f"Mission: {route_str}<br>"
            f"Rescued: {v.get('total_served', 0)} citizens<br>"
            f"Fuel Remaining: {v.get('fuel_h', 0):.1f}h"
        )
        if v["type"] == "ambulance":
            amb_x.append(vx); amb_y.append(vy); amb_t.append(v_info); amb_lbl.append(v["id"])
        else:
            boat_x.append(vx); boat_y.append(vy); boat_t.append(v_info); boat_lbl.append(v["id"])

    if amb_x:
        fig.add_trace(go.Scatter(
            x=amb_x, y=amb_y,
            mode="markers+text",
            marker=dict(size=18, color="#10b981", symbol="square", line=dict(width=2, color="#ffffff")),
            text=amb_lbl,
            textposition="bottom center",
            textfont=dict(size=10, color="#34d399", family="monospace"),
            hovertext=amb_t,
            hoverinfo="text",
            name="Ambulances (5)",
        ))

    if boat_x:
        fig.add_trace(go.Scatter(
            x=boat_x, y=boat_y,
            mode="markers+text",
            marker=dict(size=18, color="#06b6d4", symbol="triangle-up", line=dict(width=2, color="#ffffff")),
            text=boat_lbl,
            textposition="bottom center",
            textfont=dict(size=10, color="#38bdf8", family="monospace"),
            hovertext=boat_t,
            hoverinfo="text",
            name="Rescue Boats (3)",
        ))

    # 7. Highlight Selected Vehicle if requested
    if selected_vid:
        for v in vehicles:
            if v["id"] == selected_vid:
                svx, svy = v["loc"]
                fig.add_trace(go.Scatter(
                    x=[svx], y=[svy],
                    mode="markers",
                    marker=dict(size=28, color="rgba(245, 158, 11, 0.35)", symbol="circle", line=dict(width=2.5, color="#f59e0b")),
                    hoverinfo="skip",
                    showlegend=False,
                ))

    # Layout: High-contrast Dark Tactical CAD View
    fig.update_layout(
        plot_bgcolor="#070c18",
        paper_bgcolor="#070c18",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            range=[-1, W + 1],
            showgrid=True,
            gridcolor="#111827",
            zeroline=False,
            tickfont=dict(color="#475569", size=9),
            title=dict(text="Urban Grid Coordinate X (East)", font=dict(color="#64748b", size=10)),
        ),
        yaxis=dict(
            range=[H + 1, -1],  # Invert Y so (0,0) is top-left
            showgrid=True,
            gridcolor="#111827",
            zeroline=False,
            tickfont=dict(color="#475569", size=9),
            title=dict(text="Urban Grid Coordinate Y (South)", font=dict(color="#64748b", size=10)),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.14,
            xanchor="center",
            x=0.5,
            font=dict(color="#94a3b8", size=9),
        ),
        height=600,
    )

    return fig


def build_casualty_donut_chart(calls: List[Dict[str, Any]], theme: str = "light") -> go.Figure:
    """Renders a clean donut chart displaying the breakdown of waiting casualty priorities."""
    p1 = sum(1 for c in calls if c["priority"] == "P1")
    p2 = sum(1 for c in calls if c["priority"] == "P2")
    p3 = sum(1 for c in calls if c["priority"] == "P3")
    total = len(calls)
    is_dark = (theme == "dark")
    txt_color = "#E4EDEB" if is_dark else "#1E2B29"
    sub_color = "#8A9E9A" if is_dark else "#556663"

    fig = go.Figure()
    if total == 0:
        fig.add_annotation(
            text="All Clear<br>0 Waiting Calls",
            showarrow=False,
            font=dict(size=13, color="#4A8866", family="sans-serif"),
            xref="paper", yref="paper", x=0.5, y=0.5
        )
    else:
        fig.add_trace(go.Pie(
            labels=["Critical (P1)", "Urgent (P2)", "Delayed (P3)"],
            values=[p1, p2, p3],
            hole=0.64,
            marker=dict(colors=["#C25B54", "#B98230", "#4A82A8"], line=dict(color="#FFFFFF" if not is_dark else "#172221", width=1.5)),
            textinfo="value+percent",
            textfont=dict(size=10, color="#FFFFFF", family="JetBrains Mono, monospace"),
            hoverinfo="label+value+percent",
        ))
        fig.add_annotation(
            text=f"<b>{total}</b><br><span style='font-size:10px; color:{sub_color};'>CALLS</span>",
            showarrow=False,
            font=dict(size=15, color=txt_color, family="JetBrains Mono, monospace"),
            xref="paper", yref="paper", x=0.5, y=0.5
        )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=180,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0.5, xanchor="center", font=dict(color=sub_color, size=8)),
        transition=dict(duration=0),
    )
    return fig


def build_fleet_fuel_chart(vehicles: List[Dict[str, Any]], theme: str = "light") -> go.Figure:
    """Renders horizontal fuel level bars for all rescue units with seamless transparent background."""
    vids = [v["id"] for v in vehicles]
    fuels = [v.get("fuel_h", 0.0) for v in vehicles]
    is_dark = (theme == "dark")
    sub_color = "#8A9E9A" if is_dark else "#556663"
    txt_color = "#E4EDEB" if is_dark else "#1E2B29"
    grid_color = "#283C39" if is_dark else "#DEE6E3"

    colors = ["#4A8866" if f > 6.0 else ("#B98230" if f > 3.0 else "#C25B54") for f in fuels]

    fig = go.Figure(go.Bar(
        x=fuels,
        y=vids,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{f:.1f}h" for f in fuels],
        textposition="inside",
        textfont=dict(color="#FFFFFF", size=9, family="JetBrains Mono, monospace"),
    ))

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=190,
        xaxis=dict(
            range=[0, 14],
            showgrid=True,
            gridcolor=grid_color,
            title=dict(text="Fuel Remaining (Hours)", font=dict(color=sub_color, size=9)),
            tickfont=dict(color=sub_color, size=8, family="JetBrains Mono, monospace"),
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(color=txt_color, size=8, family="JetBrains Mono, monospace"),
            showgrid=False,
        ),
        transition=dict(duration=0),
    )
    return fig


def generate_sitrep_markdown(state: Dict[str, Any], engine: Any) -> str:
    """Generates a complete, downloadable civil defense situation report (SITREP)."""
    step = state["step"]
    rescued = sum(v.get("total_served", 0) for v in state["vehicles"])
    pending = len(state["unserved_calls"])
    p1 = sum(1 for c in state["unserved_calls"] if c["priority"] == "P1")
    flooded = state["flooded_count"]
    blocked = state["blocked_count"]

    report = f"""# 🚨 CIVIL DEFENSE INCIDENT SITREP // RESQ-AI
**Timestamp:** T+{step:02d} minutes  
**Incident Type:** Urban Flash Flood & Infrastructure Disruption  
**Fleet Disposition:** 5 Ambulances, 3 Rescue Boats  

---

## 1. Executive Summary
- **Total Rescued Lives:** {rescued} citizens
- **Active Waiting 911 Calls:** {pending} ({p1} Urgent P1 calls)
- **Flooded Road Network:** {flooded} blocks ({flooded/25:.1f}% of city)
- **Road Closures:** {blocked} bridges and major streets closed
- **Automatic Obstacle Reroutes:** {engine.replanning_events_count} route repairs around floods

---

## 2. District Threat Status
"""
    for zid, zdata in state["zones"].items():
        report += f"- **{zid.replace('zone_', '').title()} District**: Danger Level `{zdata['map_state'].upper()}` | Status: `{'ALERT' if zdata['alert'] else 'NORMAL'}`\n"

    report += """
---

## 3. Fleet Operational Status
| Unit ID | Type | Location | Current Mission | Citizens Rescued | Fuel Remaining |
|---|---|---|---|---|---|
"""
    for v in state["vehicles"]:
        mission = state["schedule"].get(v["id"], [])
        m_str = mission[0] if mission else "Patrol"
        report += f"| {v['id']} | {v['type'].title()} | {v['loc']} | {m_str} | {v.get('total_served', 0)} | {v['fuel_h']:.1f}h |\n"

    report += """
---

## 4. How the AI Automated Decision-Making
- **Flood Threat Tracking:** Continuously watches river level and rainfall sensors to forecast flooding 1 hour ahead.
- **911 Call Triage:** Prioritizes callers based on medical urgency, building collapse risk, and flood depth.
- **Real-Time Navigation:** Instantly steers ambulances and boats around flooded roads and collapsed bridges.
- **Fleet Dispatch:** Coordinates ambulances and boats to pick up patients, make hospital drops, and manage boat dock transfers.

*Report generated automatically by ResQ-AI Incident Command System.*
"""
    return report
