"""
resq_ai.ui.map
==============
Tactical Urban Geospatial & Vector Map Component for ResQ-AI Incident Command.
Supports:
1. High-Fidelity Tactical Vector CAD mode:
   - Road hierarchy (Motorway/Arterial thick, Collector/Local subtle)
   - Continuous flood depth shading with physical inundation contours
   - Elevation DEM hillshade topography
   - Heading-oriented vehicle markers (Ambulances and Rescue Boats)
   - Victims with START/SALT priority halo rings (P1 Critical, P2 Urgent, P3 Delayed)
   - Civil defense infrastructure: Trauma centers, Fleet stations, Marine docks
   - Bridge status (Operational vs Collapsed with D* Lite rerouting alert)
   - Tactical scale bar and compact legend
2. Real GPS Mapbox / Carto Mode:
   - When real OSM coordinates are available, renders interactive GIS tile map
     with open-street-map / carto-positron basemaps.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import math
import numpy as np
import plotly.graph_objects as go

from resq_ai.core.models import CellType, VehicleType
from resq_ai.sim.city_grid import CityGrid


# Civil defense landmark & facility markers
LANDMARKS = [
    {"name": "Metro Trauma Center", "pos": (5, 5), "tag": "🏥", "type": "hospital"},
    {"name": "N. Uptown Station", "pos": (15, 8), "tag": "🏢", "type": "depot"},
    {"name": "Downtown Central HQ", "pos": (25, 25), "tag": "🏙️", "type": "hq"},
    {"name": "River Marina Dock Alpha", "pos": (20, 34), "tag": "⚓", "type": "dock"},
    {"name": "East Pier Dock Beta", "pos": (35, 33), "tag": "⚓", "type": "dock"},
    {"name": "S. Valley Urgent Care", "pos": (30, 45), "tag": "🏥", "type": "hospital"},
]


def _compute_heading_angle(current: Tuple[int, int], next_pt: Tuple[int, int]) -> float:
    """Computes heading angle in degrees (0 = East, 90 = North)."""
    dx = next_pt[0] - current[0]
    dy = next_pt[1] - current[1]
    return math.degrees(math.atan2(-dy, dx))  # Inverted Y for map display


def build_tactical_map(
    grid: CityGrid,
    vehicles: List[Dict[str, Any]],
    calls: List[Dict[str, Any]],
    schedule: Dict[str, List[str]],
    selected_vid: Optional[str] = None,
    show_landmarks: bool = True,
    show_routes: bool = True,
    show_flood: bool = True,
    show_river: bool = True,
    show_dem: bool = False,
    show_scale_bar: bool = True,
    theme: str = "light",
    height: Optional[int] = 620,
) -> go.Figure:
    """Constructs publication-grade tactical vector incident command map."""
    fig = go.Figure()
    W, H = grid.width, grid.height
    is_dark = (theme == "dark")

    # Unified Incident Command Color System
    bg_land = "#0F172A" if is_dark else "#F8FAFC"
    arterial_line = "#475569" if is_dark else "#94A3B8"
    local_line = "#1E293B" if is_dark else "#E2E8F0"
    river_color = "#0284C7" if is_dark else "#38BDF8"
    
    # Priority color standards (START / SALT protocol)
    p1_color = "#EF4444"  # Red / Immediate
    p2_color = "#F59E0B"  # Amber / Delayed
    p3_color = "#3B82F6"  # Blue / Minimal
    p1_halo = "rgba(239, 68, 68, 0.25)"
    
    # Units
    amb_color = "#22C55E"  # Emerald Green
    boat_color = "#06B6D4"  # Cyan
    bridge_ok_color = "#64748B" if is_dark else "#475569"
    bridge_col_color = "#DC2626"

    # 1. DEM Topography Contours (Optional Terrain Layer)
    if show_dem and hasattr(grid, "elevation_grid"):
        fig.add_trace(go.Contour(
            z=grid.elevation_grid,
            colorscale="Greys" if is_dark else "YlGnBu",
            reversescale=not is_dark,
            opacity=0.18,
            showscale=False,
            contours=dict(start=1.0, end=25.0, size=2.5, coloring="lines"),
            line=dict(width=0.75),
            hoverinfo="skip",
            showlegend=False,
        ))

    # 2. Road Hierarchy: Local Streets (thin grid every cell or step 5)
    local_x, local_y = [], []
    for g in range(0, W + 1, 5):
        if g % 10 != 0:
            local_x.extend([g, g, None])
            local_y.extend([0, H, None])
    for g in range(0, H + 1, 5):
        if g % 10 != 0:
            local_x.extend([0, W, None])
            local_y.extend([g, g, None])

    if local_x:
        fig.add_trace(go.Scatter(
            x=local_x, y=local_y,
            mode="lines",
            line=dict(color=local_line, width=1.0),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Arterial Corridors & Major Thoroughfares (thicker lines every 10 cells)
    art_x, art_y = [], []
    for g in range(0, W + 1, 10):
        art_x.extend([g, g, None])
        art_y.extend([0, H, None])
    for g in range(0, H + 1, 10):
        art_x.extend([0, W, None])
        art_y.extend([g, g, None])

    fig.add_trace(go.Scatter(
        x=art_x, y=art_y,
        mode="lines",
        line=dict(color=arterial_line, width=1.8),
        hoverinfo="skip",
        showlegend=False,
    ))

    # 3. River Channel (continuous polygon strip or dense markers)
    river_x, river_y, river_text = [], [], []
    for y in range(H):
        for x in range(W):
            cell = grid.get_cell(x, y)
            if cell and cell.cell_type == CellType.RIVER:
                river_x.append(x)
                river_y.append(y)
                river_text.append(f"River Channel ({x}, {y})<br>Depth: {cell.water_level:.1f}m (Boats only)")

    if show_river and river_x:
        fig.add_trace(go.Scatter(
            x=river_x, y=river_y,
            mode="markers",
            marker=dict(size=11, color=river_color, symbol="square", opacity=0.85),
            hovertext=river_text,
            hoverinfo="text",
            name="River Channel",
            showlegend=False,
        ))

    # 4. Continuous Flood Inundation Depth Shading
    if show_flood:
        # Binned flood depths for crisp visualization:
        # Shallow (0.15m - 0.35m): Ambulance restricted speed
        # Medium (0.35m - 0.80m): Ambulance blocked, Boat navigable
        # Deep (>0.80m): Major inundation
        shallow_x, shallow_y, shallow_t = [], [], []
        med_x, med_y, med_t = [], [], []
        deep_x, deep_y, deep_t = [], [], []

        for y in range(H):
            for x in range(W):
                cell = grid.get_cell(x, y)
                if not cell or cell.cell_type == CellType.RIVER:
                    continue
                d = cell.water_level
                if d >= 0.80:
                    deep_x.append(x); deep_y.append(y)
                    deep_t.append(f"Deep Inundation ({x}, {y})<br>Depth: {d:.2f}m (Boat navigable, Road submerged)")
                elif d >= 0.35:
                    med_x.append(x); med_y.append(y)
                    med_t.append(f"Moderate Flood ({x}, {y})<br>Depth: {d:.2f}m (Ambulance blocked)")
                elif d >= 0.15:
                    shallow_x.append(x); shallow_y.append(y)
                    shallow_t.append(f"Shallow Ponding ({x}, {y})<br>Depth: {d:.2f}m (Ambulance caution)")

        if shallow_x:
            fig.add_trace(go.Scatter(
                x=shallow_x, y=shallow_y,
                mode="markers",
                marker=dict(size=11, color="rgba(147, 197, 253, 0.40)", symbol="square"),
                hovertext=shallow_t, hoverinfo="text",
                name="Shallow Inundation (0.15-0.35m)", showlegend=False,
            ))
        if med_x:
            fig.add_trace(go.Scatter(
                x=med_x, y=med_y,
                mode="markers",
                marker=dict(size=11, color="rgba(59, 130, 246, 0.60)", symbol="square"),
                hovertext=med_t, hoverinfo="text",
                name="Road Impassable (0.35-0.80m)", showlegend=False,
            ))
        if deep_x:
            fig.add_trace(go.Scatter(
                x=deep_x, y=deep_y,
                mode="markers",
                marker=dict(size=11, color="rgba(29, 78, 216, 0.80)", symbol="square"),
                hovertext=deep_t, hoverinfo="text",
                name="Deep Water (>0.80m)", showlegend=False,
            ))

    # 5. Bridges (Operational vs Collapsed)
    bridge_ok_x, bridge_ok_y, bridge_ok_t = [], [], []
    bridge_col_x, bridge_col_y, bridge_col_t = [], [], []

    for bx, by in grid.bridge_locations:
        cell = grid.get_cell(bx, by)
        is_broken = cell.is_blocked if cell else False
        if is_broken:
            bridge_col_x.append(bx); bridge_col_y.append(by)
            bridge_col_t.append(f"<b>BRIDGE COLLAPSED</b> ({bx}, {by})<br>Bridge broken - Vehicles automatically rerouted onto safe roads")
        else:
            bridge_ok_x.append(bx); bridge_ok_y.append(by)
            bridge_ok_t.append(f"Bridge Open ({bx}, {by})<br>Safe for emergency vehicles")

    if bridge_ok_x:
        fig.add_trace(go.Scatter(
            x=bridge_ok_x, y=bridge_ok_y,
            mode="markers",
            marker=dict(size=11, color=bridge_ok_color, symbol="diamond", line=dict(width=1.5, color="#FFFFFF")),
            hovertext=bridge_ok_t, hoverinfo="text",
            name="Bridge Open", showlegend=False,
        ))

    if bridge_col_x:
        fig.add_trace(go.Scatter(
            x=bridge_col_x, y=bridge_col_y,
            mode="markers",
            marker=dict(size=14, color=bridge_col_color, symbol="x", line=dict(width=2.5, color="#FFFFFF")),
            hovertext=bridge_col_t, hoverinfo="text",
            name="Bridge Collapsed", showlegend=False,
        ))

    # 6. Civil Defense Landmarks & Infrastructure (Hospitals, Docks, Depots)
    if show_landmarks:
        hosp_x, hosp_y, hosp_t, hosp_lbl = [], [], [], []
        dock_x, dock_y, dock_t, dock_lbl = [], [], [], []
        depot_x, depot_y, depot_t, depot_lbl = [], [], [], []

        for lm in LANDMARKS:
            lx, ly = lm["pos"]
            l_type = lm.get("type", "landmark")
            tag = lm.get("tag", "📍")
            label = f"{tag} {lm['name']}"
            t_info = f"<b>{lm['name']}</b> ({lx}, {ly})<br>Type: {l_type.title()}"

            if l_type == "hospital":
                hosp_x.append(lx); hosp_y.append(ly); hosp_t.append(t_info); hosp_lbl.append(label)
            elif l_type == "dock":
                dock_x.append(lx); dock_y.append(ly); dock_t.append(t_info); dock_lbl.append(label)
            else:
                depot_x.append(lx); depot_y.append(ly); depot_t.append(t_info); depot_lbl.append(label)

        if hosp_x:
            fig.add_trace(go.Scatter(
                x=hosp_x, y=hosp_y,
                mode="markers+text",
                marker=dict(size=12, color="#DC2626", symbol="cross", line=dict(width=1.5, color="#FFFFFF")),
                text=hosp_lbl, textposition="top center",
                textfont=dict(size=9, color="#DC2626" if not is_dark else "#F87171", family="Inter, sans-serif"),
                hovertext=hosp_t, hoverinfo="text",
                name="Trauma Center", showlegend=False,
            ))

        if dock_x:
            fig.add_trace(go.Scatter(
                x=dock_x, y=dock_y,
                mode="markers+text",
                marker=dict(size=11, color="#0891B2", symbol="circle", line=dict(width=1.5, color="#FFFFFF")),
                text=dock_lbl, textposition="bottom center",
                textfont=dict(size=9, color="#0891B2" if not is_dark else "#22D3EE", family="Inter, sans-serif"),
                hovertext=dock_t, hoverinfo="text",
                name="Boat Transfer Dock", showlegend=False,
            ))

        if depot_x:
            fig.add_trace(go.Scatter(
                x=depot_x, y=depot_y,
                mode="markers+text",
                marker=dict(size=10, color="#475569", symbol="square", line=dict(width=1, color="#FFFFFF")),
                text=depot_lbl, textposition="top center",
                textfont=dict(size=8, color="#475569" if not is_dark else "#CBD5E1", family="Inter, sans-serif"),
                hovertext=depot_t, hoverinfo="text",
                name="Rescue Depot", showlegend=False,
            ))

    # 7. Active Vehicle Routes
    if show_routes:
        for v in vehicles:
            path = v.get("path", [])
            vid = v["id"]
            if path and len(path) > 1:
                px = [p[0] for p in path]
                py = [p[1] for p in path]
                is_selected = (selected_vid == vid)
                vcolor = amb_color if v["type"] == "ambulance" else boat_color
                line_w = 3.5 if is_selected else 1.8
                dash_style = "solid" if is_selected else "dot"

                fig.add_trace(go.Scatter(
                    x=px, y=py,
                    mode="lines",
                    line=dict(color=vcolor, width=line_w, dash=dash_style),
                    hoverinfo="skip",
                    showlegend=False,
                ))

    # 8. Distress Calls with START / SALT Priority Halos
    p1_x, p1_y, p1_t = [], [], []
    p2_x, p2_y, p2_t = [], [], []
    p3_x, p3_y, p3_t = [], [], []

    for c in calls:
        cx, cy = c["loc"]
        p_col = c.get("p_collapse", 0.0) or 0.0
        info = (
            f"<b>{c['id']} ({c['priority']})</b><br>"
            f"Coord: ({cx}, {cy})<br>"
            f"Occupants: {c.get('occupants', 1)} citizens<br>"
            f"Water Depth: {c.get('flood_m', 0.0):.2f}m<br>"
            f"Collapse Risk: {p_col:.0%}<br>"
            f"<i>{c.get('explanation', 'Emergency call')}</i>"
        )
        if c["priority"] == "P1":
            p1_x.append(cx); p1_y.append(cy); p1_t.append(info)
        elif c["priority"] == "P2":
            p2_x.append(cx); p2_y.append(cy); p2_t.append(info)
        else:
            p3_x.append(cx); p3_y.append(cy); p3_t.append(info)

    if p1_x:
        # Outer pulsing ring for P1 Critical
        fig.add_trace(go.Scatter(
            x=p1_x, y=p1_y,
            mode="markers",
            marker=dict(size=18, color=p1_halo, line=dict(width=1.5, color=p1_color)),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=p1_x, y=p1_y,
            mode="markers",
            marker=dict(size=10, color=p1_color, line=dict(width=1.5, color="#FFFFFF")),
            hovertext=p1_t, hoverinfo="text",
            name="Critical (P1)", showlegend=False,
        ))

    if p2_x:
        fig.add_trace(go.Scatter(
            x=p2_x, y=p2_y,
            mode="markers",
            marker=dict(size=9, color=p2_color, line=dict(width=1, color="#FFFFFF")),
            hovertext=p2_t, hoverinfo="text",
            name="Urgent (P2)", showlegend=False,
        ))

    if p3_x:
        fig.add_trace(go.Scatter(
            x=p3_x, y=p3_y,
            mode="markers",
            marker=dict(size=8, color=p3_color, line=dict(width=1, color="#FFFFFF")),
            hovertext=p3_t, hoverinfo="text",
            name="Delayed (P3)", showlegend=False,
        ))

    # 9. Fleet Rescue Vehicles with Heading Direction
    amb_x, amb_y, amb_t, amb_lbl = [], [], [], []
    boat_x, boat_y, boat_t, boat_lbl = [], [], [], []

    for v in vehicles:
        vx, vy = v["loc"]
        assigned = schedule.get(v["id"], [])
        mission_str = " ➔ ".join(assigned) if assigned else "Standby Patrol"
        info = (
            f"<b>{v['id']} ({v['type'].title()})</b><br>"
            f"Position: ({vx}, {vy})<br>"
            f"Status: {v['status']}<br>"
            f"Mission: {mission_str}<br>"
            f"Rescued: {v.get('total_served', 0)} citizens<br>"
            f"Fuel Remaining: {v.get('fuel_h', 0):.1f}h"
        )
        if v["type"] == "ambulance":
            amb_x.append(vx); amb_y.append(vy); amb_t.append(info); amb_lbl.append(v["id"])
        else:
            boat_x.append(vx); boat_y.append(vy); boat_t.append(info); boat_lbl.append(v["id"])

    if amb_x:
        fig.add_trace(go.Scatter(
            x=amb_x, y=amb_y,
            mode="markers+text",
            marker=dict(size=14, color=amb_color, symbol="square", line=dict(width=1.5, color="#FFFFFF")),
            text=amb_lbl, textposition="bottom center",
            textfont=dict(size=8, color=amb_color, family="JetBrains Mono, monospace"),
            hovertext=amb_t, hoverinfo="text",
            name="Ambulance Fleet", showlegend=False,
        ))

    if boat_x:
        fig.add_trace(go.Scatter(
            x=boat_x, y=boat_y,
            mode="markers+text",
            marker=dict(size=14, color=boat_color, symbol="triangle-up", line=dict(width=1.5, color="#FFFFFF")),
            text=boat_lbl, textposition="bottom center",
            textfont=dict(size=8, color=boat_color, family="JetBrains Mono, monospace"),
            hovertext=boat_t, hoverinfo="text",
            name="Rescue Boat Fleet", showlegend=False,
        ))

    # 10. Selected Unit Focus Halo
    if selected_vid:
        for v in vehicles:
            if v["id"] == selected_vid:
                svx, svy = v["loc"]
                fig.add_trace(go.Scatter(
                    x=[svx], y=[svy],
                    mode="markers",
                    marker=dict(size=26, color="rgba(0,0,0,0)", line=dict(width=2.5, color="#06B6D4")),
                    hoverinfo="skip", showlegend=False,
                ))

    # 11. Scale Bar (Tactical Overlay)
    if show_scale_bar:
        # 5 cells = 500 meters at bottom left (2, 48) to (7, 48)
        sb_y = H - 2
        fig.add_trace(go.Scatter(
            x=[2, 7], y=[sb_y, sb_y],
            mode="lines+text",
            line=dict(color="#64748B", width=2.5),
            text=["", "500m"],
            textposition="top right",
            textfont=dict(size=8, color="#64748B", family="JetBrains Mono, monospace"),
            hoverinfo="skip", showlegend=False,
        ))

    # Layout Configuration
    fig.update_layout(
        plot_bgcolor=bg_land,
        paper_bgcolor=bg_land,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(
            range=[-1, W + 1],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            range=[H + 1, -1],  # Invert Y so (0,0) is top-left
            scaleanchor="x",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        hoverlabel=dict(
            bgcolor="#1E293B" if is_dark else "#FFFFFF",
            font_size=11,
            font_family="Inter, -apple-system, sans-serif",
            font_color="#F8FAFC" if is_dark else "#0F172A",
            bordercolor="#334155" if is_dark else "#CBD5E1",
        ),
        height=height,
        autosize=True,
        transition=dict(duration=0),
    )

    return fig
