"""
resq_ai.geo.city_network
========================
Dual-mode urban transportation and civil defense network engine.
Supports:
1. Real Mode: Ingests OpenStreetMap GeoJSON networks with true GPS lat/lon,
   road hierarchy (motorway, primary, secondary, residential), speed limits,
   bridge spans, and civil defense assets (trauma centers, marine docks, firebases).
2. Synthetic Mode: Procedural believable city generator with multi-tier road hierarchy,
   meandering river channel, realistic elevation DEM gradient, and designated facilities.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
import math
import numpy as np

from resq_ai.core.models import VehicleType
from resq_ai.geo.inundation import InundationModel


@dataclass
class RoadNode:
    id: str
    lat: float
    lon: float
    x: float
    y: float
    elevation_m: float = 5.0
    district: str = "Central"


@dataclass
class RoadEdge:
    id: str
    source: str
    target: str
    name: str
    road_class: str  # motorway, primary, secondary, residential, bridge
    speed_limit_kmh: float
    length_m: float
    is_bridge: bool = False
    is_blocked: bool = False
    geometry: List[Tuple[float, float]] = field(default_factory=list)  # [(lat, lon), ...]


@dataclass
class FacilityAsset:
    id: str
    name: str
    kind: str  # hospital, depot, boat_dock
    lat: float
    lon: float
    node_id: str
    capacity: int = 20
    metadata: Dict[str, Any] = field(default_factory=dict)


class CityNetwork:
    """Base urban civil defense network with routing and spatial indexing."""

    def __init__(self):
        self.nodes: Dict[str, RoadNode] = {}
        self.edges: Dict[str, RoadEdge] = {}
        self.adjacency: Dict[str, List[RoadEdge]] = {}
        self.facilities: Dict[str, FacilityAsset] = {}
        self.bounds: Dict[str, float] = {}
        self.inundation_model: Optional[InundationModel] = None

    def add_node(self, node: RoadNode) -> None:
        self.nodes[node.id] = node
        if node.id not in self.adjacency:
            self.adjacency[node.id] = []

    def add_edge(self, edge: RoadEdge) -> None:
        self.edges[edge.id] = edge
        if edge.source not in self.adjacency:
            self.adjacency[edge.source] = []
        self.adjacency[edge.source].append(edge)

    def add_facility(self, facility: FacilityAsset) -> None:
        self.facilities[facility.id] = facility

    def get_hospitals(self) -> List[FacilityAsset]:
        return [f for f in self.facilities.values() if f.kind == "hospital"]

    def get_depots(self) -> List[FacilityAsset]:
        return [f for f in self.facilities.values() if f.kind == "depot"]

    def get_boat_docks(self) -> List[FacilityAsset]:
        return [f for f in self.facilities.values() if f.kind == "boat_dock"]

    def get_bridges(self) -> List[RoadEdge]:
        return [e for e in self.edges.values() if e.is_bridge]

    def set_bridge_status(self, edge_id: str, is_open: bool) -> None:
        """Toggles bridge passability (e.g. after structural collapse)."""
        if edge_id in self.edges:
            self.edges[edge_id].is_blocked = not is_open

    def is_edge_passable(
        self,
        edge: RoadEdge,
        vehicle_type: VehicleType,
        water_depth: float = 0.0,
    ) -> bool:
        """Determines whether a road edge can be traversed by a given vehicle."""
        if edge.is_blocked:
            return False

        if vehicle_type == VehicleType.RESCUE_BOAT:
            # Boat can only use water bodies or flooded roads with depth >= 0.20m
            # Cannot use standard bridges
            if edge.is_bridge:
                return False
            return water_depth >= InundationModel.BOAT_MIN_DRAFT_M

        if vehicle_type == VehicleType.AMBULANCE:
            if edge.is_bridge:
                # Bridges remain elevated above water unless structurally collapsed
                return not edge.is_blocked
            return water_depth <= InundationModel.AMBULANCE_MAX_DEPTH_M

        return not edge.is_blocked

    def edge_travel_time_seconds(
        self,
        edge: RoadEdge,
        vehicle_type: VehicleType,
        water_depth: float = 0.0,
    ) -> float:
        """Calculates expected transit time in seconds accounting for flood slowing."""
        if not self.is_edge_passable(edge, vehicle_type, water_depth):
            return float("inf")

        base_speed_mps = (edge.speed_limit_kmh * 1000.0) / 3600.0
        
        # Apply water slowing
        if vehicle_type == VehicleType.AMBULANCE and not edge.is_bridge and water_depth > 0:
            slow_factor = max(0.15, 1.0 - (water_depth / InundationModel.AMBULANCE_MAX_DEPTH_M) * 0.85)
            effective_speed = base_speed_mps * slow_factor
        elif vehicle_type == VehicleType.RESCUE_BOAT:
            # Boat travels at ~30 km/h (8.3 m/s) on water
            effective_speed = 8.3
        else:
            effective_speed = base_speed_mps

        return edge.length_m / max(effective_speed, 1.0)


class RealCityNetwork(CityNetwork):
    """
    Real-world GIS road network loaded from OpenStreetMap GeoJSON/GraphML.
    """

    def __init__(self, geojson_path: str = "configs/geo/osm_network.json"):
        super().__init__()
        self.source_path = geojson_path
        self._load_from_json(geojson_path)

    def _load_from_json(self, path: str) -> None:
        file_p = Path(path)
        if not file_p.exists():
            raise FileNotFoundError(f"OSM network cache not found at {path}")

        with open(file_p, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.bounds = data.get("bounds", {
            "min_lat": 42.350, "max_lat": 42.370,
            "min_lon": -71.080, "max_lon": -71.050
        })

        min_lat = self.bounds["min_lat"]
        max_lat = self.bounds["max_lat"]
        min_lon = self.bounds["min_lon"]
        max_lon = self.bounds["max_lon"]

        # 1. Ingest Nodes
        for nd in data.get("nodes", []):
            lat = nd["lat"]
            lon = nd["lon"]
            # Map lat/lon to 0..50 coordinate space for unified display
            norm_x = (lon - min_lon) / max(max_lon - min_lon, 1e-6)
            norm_y = (lat - min_lat) / max(max_lat - min_lat, 1e-6)
            grid_x = float(np.clip(norm_x * 50.0, 0.0, 50.0))
            grid_y = float(np.clip((1.0 - norm_y) * 50.0, 0.0, 50.0))

            node = RoadNode(
                id=nd["id"],
                lat=lat,
                lon=lon,
                x=grid_x,
                y=grid_y,
                elevation_m=float(nd.get("elevation_m", 5.0)),
                district="Downtown" if nd.get("row", 0) in [3, 4] else "Suburban",
            )
            self.add_node(node)

        # 2. Ingest Edges
        for ed in data.get("edges", []):
            geom = [(pt["lat"], pt["lon"]) for pt in ed.get("geometry", [])]
            edge = RoadEdge(
                id=ed["id"],
                source=ed["source"],
                target=ed["target"],
                name=ed.get("name", "Road"),
                road_class=ed.get("road_class", "secondary"),
                speed_limit_kmh=float(ed.get("speed_limit_kmh", 45.0)),
                length_m=float(ed.get("length_m", 250.0)),
                is_bridge=bool(ed.get("is_bridge", False)),
                geometry=geom,
            )
            self.add_edge(edge)

        # 3. Ingest Facilities
        facilities = data.get("facilities", {})
        for h in facilities.get("hospitals", []):
            self.add_facility(FacilityAsset(
                id=h["id"],
                name=h["name"],
                kind="hospital",
                lat=h["lat"],
                lon=h["lon"],
                node_id=h["node_id"],
                capacity=h.get("capacity", 40),
                metadata={"helipad": h.get("helipad", False)},
            ))

        for d in facilities.get("depots", []):
            self.add_facility(FacilityAsset(
                id=d["id"],
                name=d["name"],
                kind="depot",
                lat=d["lat"],
                lon=d["lon"],
                node_id=d["node_id"],
                metadata={"ambulances": d.get("ambulances", 3), "boats": d.get("boats", 1)},
            ))

        for b in facilities.get("boat_handoffs", []):
            self.add_facility(FacilityAsset(
                id=b["id"],
                name=b["name"],
                kind="boat_dock",
                lat=b["lat"],
                lon=b["lon"],
                node_id=b["node_id"],
                capacity=b.get("capacity", 5),
                metadata={"has_ambulance_ramp": b.get("has_ambulance_ramp", True)},
            ))


class SyntheticCityNetwork(CityNetwork):
    """
    Believable procedurally generated urban landscape for reproducible testing:
    - Multi-tier road hierarchy (Arterial, Collector, Residential)
    - Meandering river with designated bridge crossings
    - Continuous DEM elevation array
    - Designated civil defense assets
    """

    def __init__(self, width: int = 50, height: int = 50, seed: int = 42):
        super().__init__()
        self.width = width
        self.height = height
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.elevation_grid = np.zeros((height, width), dtype=float)
        self.river_mask = np.zeros((height, width), dtype=bool)
        self.river_y_center = 33

        self.bounds = {
            "min_lat": 42.350, "max_lat": 42.370,
            "min_lon": -71.080, "max_lon": -71.050
        }

        self._build_terrain_dem()
        self._build_procedural_network()
        self._build_facilities()

    def _build_terrain_dem(self) -> None:
        """Creates continuous elevation field sloping down towards river basin."""
        y_indices, x_indices = np.indices((self.height, self.width))
        
        # Base elevation: High north bluffs (~18m), sloping down to river at y=33 (~1.5m), then slight rise south (~8m)
        dist_to_river = np.abs(y_indices - self.river_y_center)
        self.elevation_grid = 1.5 + dist_to_river * 0.45 + 0.5 * np.sin(x_indices / 5.0)

        # Meandering river channel (width ~3 cells)
        for x in range(self.width):
            meander = int(self.river_y_center + 2.0 * math.sin(x / 4.0))
            for dy in range(-1, 2):
                ry = meander + dy
                if 0 <= ry < self.height:
                    self.river_mask[ry, x] = True
                    self.elevation_grid[ry, x] = 1.0  # river bed

        # Initialize physical inundation model
        self.inundation_model = InundationModel(
            elevation=self.elevation_grid,
            river_mask=self.river_mask,
            baseline_river_stage=1.8,
            drainage_rate=0.04
        )

    def _build_procedural_network(self) -> None:
        """Generates road hierarchy with bridges at designated river crossings."""
        # Nodes placed at road intersections: step of 5 units
        step = 5
        for y in range(0, self.height + 1, step):
            for x in range(0, self.width + 1, step):
                nid = f"SYN_{x}_{y}"
                elev = float(self.elevation_grid[min(y, self.height - 1), min(x, self.width - 1)])
                # lat/lon mapping
                lat = self.bounds["max_lat"] - (y / self.height) * (self.bounds["max_lat"] - self.bounds["min_lat"])
                lon = self.bounds["min_lon"] + (x / self.width) * (self.bounds["max_lon"] - self.bounds["min_lon"])
                self.add_node(RoadNode(
                    id=nid,
                    lat=lat,
                    lon=lon,
                    x=float(x),
                    y=float(y),
                    elevation_m=elev,
                    district="Riverfront" if 30 <= y <= 38 else ("Downtown" if 15 <= y < 30 else "Uptown"),
                ))

        # Horizontal edges
        eid = 1
        for y in range(0, self.height + 1, step):
            r_class = "motorway" if y in [0, 50] else ("primary" if y % 10 == 0 else "secondary")
            speed = 70.0 if r_class == "motorway" else (50.0 if r_class == "primary" else 35.0)
            for x in range(0, self.width, step):
                u = f"SYN_{x}_{y}"
                v = f"SYN_{x+step}_{y}"
                self.add_edge(RoadEdge(
                    id=f"E_H_{eid}",
                    source=u,
                    target=v,
                    name=f"Corridor {y} E",
                    road_class=r_class,
                    speed_limit_kmh=speed,
                    length_m=step * 100.0,
                    is_bridge=False,
                ))
                eid += 1
                self.add_edge(RoadEdge(
                    id=f"E_H_{eid}",
                    source=v,
                    target=u,
                    name=f"Corridor {y} W",
                    road_class=r_class,
                    speed_limit_kmh=speed,
                    length_m=step * 100.0,
                    is_bridge=False,
                ))
                eid += 1

        # Vertical edges & bridges across river (around y=30 to y=35)
        # Bridges placed at x = 10, 25, 40
        bridge_xs = [10, 25, 40]
        for x in range(0, self.width + 1, step):
            r_class = "primary" if x % 10 == 0 else "secondary"
            speed = 50.0 if r_class == "primary" else 35.0
            for y in range(0, self.height, step):
                u = f"SYN_{x}_{y}"
                v = f"SYN_{x}_{y+step}"
                is_river_span = (y == 30 and y + step == 35)
                is_bridge = is_river_span and (x in bridge_xs)
                
                # If crosses river and not a designated bridge, edge is impassable/nonexistent
                if is_river_span and not is_bridge:
                    continue

                b_class = "bridge" if is_bridge else r_class
                name = f"Bridge at Col {x}" if is_bridge else f"Ave {x}"
                self.add_edge(RoadEdge(
                    id=f"E_V_{eid}",
                    source=u,
                    target=v,
                    name=name,
                    road_class=b_class,
                    speed_limit_kmh=speed,
                    length_m=step * 100.0,
                    is_bridge=is_bridge,
                ))
                eid += 1
                self.add_edge(RoadEdge(
                    id=f"E_V_{eid}",
                    source=v,
                    target=u,
                    name=f"{name} (Rev)",
                    road_class=b_class,
                    speed_limit_kmh=speed,
                    length_m=step * 100.0,
                    is_bridge=is_bridge,
                ))
                eid += 1

    def _build_facilities(self) -> None:
        """Places standard hospitals, depots, and docks."""
        self.add_facility(FacilityAsset(
            id="HOSP_NORTH",
            name="North Trauma Center",
            kind="hospital",
            lat=42.368,
            lon=-71.070,
            node_id="SYN_5_5",
            capacity=40,
        ))
        self.add_facility(FacilityAsset(
            id="HOSP_DOWNTOWN",
            name="Downtown General Hospital",
            kind="hospital",
            lat=42.360,
            lon=-71.060,
            node_id="SYN_25_25",
            capacity=50,
        ))
        self.add_facility(FacilityAsset(
            id="DEPOT_CENTRAL",
            name="Central Rescue Fleet Base",
            kind="depot",
            lat=42.362,
            lon=-71.065,
            node_id="SYN_20_20",
        ))
        self.add_facility(FacilityAsset(
            id="DOCK_MARINA",
            name="Marina Boat Handoff Pier",
            kind="boat_dock",
            lat=42.355,
            lon=-71.065,
            node_id="SYN_25_35",
            capacity=5,
            metadata={"has_ambulance_ramp": True},
        ))
