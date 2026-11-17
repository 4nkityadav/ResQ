"""
resq_ai.geo.generate_osm_network
================================
Utility to generate an OpenStreetMap-compatible road network and infrastructure
dataset cached as GeoJSON / structured JSON for offline emergency planning.
"""

from __future__ import annotations
import json
import math
from pathlib import Path


def generate_cached_network(output_path: str = "configs/geo/osm_network.json") -> dict:
    bounds = {
        "min_lat": 42.3500,
        "max_lat": 42.3700,
        "min_lon": -71.0800,
        "max_lon": -71.0500,
    }
    center = {
        "lat": 42.3600,
        "lon": -71.0650,
    }

    cols = 8
    rows = 8

    nodes = {}
    node_list = []

    for r in range(rows):
        lat = bounds["max_lat"] - r * (bounds["max_lat"] - bounds["min_lat"]) / (rows - 1)
        for c in range(cols):
            lon = bounds["min_lon"] + c * (bounds["max_lon"] - bounds["min_lon"]) / (cols - 1)
            nid = f"N_{r}_{c}"
            # River flows around row 4 (lat approx 42.361)
            dist_to_river = abs(r - 4.2)
            elev = round(1.8 + dist_to_river * 4.0 + 0.5 * math.sin(c), 1)
            node_info = {
                "id": nid,
                "row": r,
                "col": c,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "elevation_m": elev,
            }
            nodes[nid] = node_info
            node_list.append(node_info)

    edges = []
    edge_id = 1

    def add_edge(u: str, v: str, name: str, road_class: str, speed: int, is_bridge: bool = False):
        nonlocal edge_id
        n1 = nodes[u]
        n2 = nodes[v]
        dlat = (n2["lat"] - n1["lat"]) * 111320
        dlon = (n2["lon"] - n1["lon"]) * 82400
        length = round(math.sqrt(dlat**2 + dlon**2), 1)
        edge = {
            "id": f"E_{edge_id:03d}",
            "source": u,
            "target": v,
            "name": name,
            "road_class": road_class,
            "speed_limit_kmh": speed,
            "length_m": length,
            "is_bridge": is_bridge,
            "geometry": [
                {"lat": n1["lat"], "lon": n1["lon"]},
                {"lat": n2["lat"], "lon": n2["lon"]},
            ],
        }
        edges.append(edge)
        edge_id += 1

    street_names = [
        "Uptown Heights Blvd",
        "Broad Canal St",
        "North Station Way",
        "Commercial Parkway",
        "Riverfront Esplanade",
        "Industrial South Rd",
        "Harbor View Way",
        "Valley Express Hwy",
    ]

    for r in range(rows):
        name = street_names[r]
        r_class = "motorway" if r in [0, 7] else ("primary" if r in [3, 4] else "secondary")
        speed = 70 if r_class == "motorway" else (50 if r_class == "primary" else 35)
        for c in range(cols - 1):
            u = f"N_{r}_{c}"
            v = f"N_{r}_{c+1}"
            add_edge(u, v, f"{name} Blk {c+1}", r_class, speed)
            add_edge(v, u, f"{name} Blk {c+1} (Rev)", r_class, speed)

    ave_names = [
        "West End Ave",
        "Medical Way",
        "State Parkway",
        "Central Bridge Blvd",
        "Civic Center St",
        "East Pier Rd",
        "Marina Approach",
        "Eastway Bypass",
    ]

    for c in range(cols):
        name = ave_names[c]
        for r in range(rows - 1):
            u = f"N_{r}_{c}"
            v = f"N_{r+1}_{c}"
            is_bridge = (r == 4) and (c in [1, 3, 5, 7])
            r_class = "primary" if c in [2, 3, 5] else "secondary"
            speed = 50 if r_class == "primary" else 35
            if is_bridge:
                r_class = "bridge"
                bridge_name = f"{name} Memorial Bridge"
                add_edge(u, v, bridge_name, r_class, speed, is_bridge=True)
                add_edge(v, u, f"{bridge_name} (Rev)", r_class, speed, is_bridge=True)
            else:
                add_edge(u, v, f"{name} Seg {r+1}", r_class, speed, is_bridge=False)
                add_edge(v, u, f"{name} Seg {r+1} (Rev)", r_class, speed, is_bridge=False)

    hospitals = [
        {
            "id": "HOSP_1",
            "name": "St. Jude Metro Trauma Center",
            "lat": nodes["N_1_1"]["lat"],
            "lon": nodes["N_1_1"]["lon"],
            "node_id": "N_1_1",
            "capacity": 50,
            "helipad": True,
        },
        {
            "id": "HOSP_2",
            "name": "Central District Hospital",
            "lat": nodes["N_2_4"]["lat"],
            "lon": nodes["N_2_4"]["lon"],
            "node_id": "N_2_4",
            "capacity": 35,
            "helipad": False,
        },
        {
            "id": "HOSP_3",
            "name": "Valley Memorial Care",
            "lat": nodes["N_6_2"]["lat"],
            "lon": nodes["N_6_2"]["lon"],
            "node_id": "N_6_2",
            "capacity": 25,
            "helipad": True,
        },
    ]

    depots = [
        {
            "id": "DEPOT_1",
            "name": "Central Fire & Rescue HQ",
            "lat": nodes["N_2_2"]["lat"],
            "lon": nodes["N_2_2"]["lon"],
            "node_id": "N_2_2",
            "ambulances": 4,
            "boats": 0,
        },
        {
            "id": "DEPOT_2",
            "name": "North Division Firehouse",
            "lat": nodes["N_0_4"]["lat"],
            "lon": nodes["N_0_4"]["lon"],
            "node_id": "N_0_4",
            "ambulances": 3,
            "boats": 0,
        },
        {
            "id": "DEPOT_3",
            "name": "River Marine Station",
            "lat": nodes["N_4_1"]["lat"],
            "lon": nodes["N_4_1"]["lon"],
            "node_id": "N_4_1",
            "ambulances": 1,
            "boats": 3,
        },
    ]

    boat_handoffs = [
        {
            "id": "DOCK_1",
            "name": "Marina Handoff Dock Alpha",
            "lat": nodes["N_4_1"]["lat"],
            "lon": nodes["N_4_1"]["lon"],
            "node_id": "N_4_1",
            "capacity": 4,
            "has_ambulance_ramp": True,
        },
        {
            "id": "DOCK_2",
            "name": "East Pier Rescue Ramp Beta",
            "lat": nodes["N_4_6"]["lat"],
            "lon": nodes["N_4_6"]["lon"],
            "node_id": "N_4_6",
            "capacity": 4,
            "has_ambulance_ramp": True,
        },
    ]

    river_channel = {
        "name": "Charles-Metro River",
        "baseline_stage_m": 1.2,
        "flood_stage_m": 3.5,
        "critical_stage_m": 5.0,
        "centerline": [
            {"lat": round(nodes[f"N_4_{c}"]["lat"] + 0.0005, 6), "lon": nodes[f"N_4_{c}"]["lon"]}
            for c in range(cols)
        ],
    }

    network_data = {
        "version": "2.0.0",
        "crs": "EPSG:4326",
        "region": "Central River Incident Command District",
        "bounds": bounds,
        "center": center,
        "nodes": node_list,
        "edges": edges,
        "facilities": {
            "hospitals": hospitals,
            "depots": depots,
            "boat_handoffs": boat_handoffs,
        },
        "hydrology": river_channel,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(network_data, f, indent=2)

    return network_data


if __name__ == "__main__":
    data = generate_cached_network()
    print(f"Generated network with {len(data['nodes'])} nodes and {len(data['edges'])} edges.")
