"""
tests.test_geo
==============
Verification tests for RealCityNetwork, SyntheticCityNetwork, and InundationModel.
"""

import pytest
import numpy as np
from resq_ai.geo.city_network import RealCityNetwork, SyntheticCityNetwork
from resq_ai.geo.inundation import InundationModel
from resq_ai.core.models import VehicleType


def test_real_city_network_loading():
    net = RealCityNetwork("configs/geo/osm_network.json")
    assert len(net.nodes) == 64
    assert len(net.edges) > 100
    hospitals = net.get_hospitals()
    assert len(hospitals) == 3
    assert any("Trauma" in h.name for h in hospitals)
    docks = net.get_boat_docks()
    assert len(docks) == 2
    bridges = net.get_bridges()
    assert len(bridges) > 0


def test_synthetic_city_network_and_dem():
    net = SyntheticCityNetwork(width=50, height=50, seed=42)
    assert len(net.nodes) > 100
    assert len(net.edges) > 300
    assert net.elevation_grid.shape == (50, 50)
    assert net.elevation_grid.min() < 2.0  # River channel is low
    assert net.elevation_grid.max() > 10.0  # North bluffs are high
    assert len(net.get_hospitals()) >= 2
    assert len(net.get_boat_docks()) >= 1


def test_inundation_physics_and_passability():
    # 10x10 synthetic elevation: slope from 0.0 to 9.0
    elev = np.tile(np.linspace(0.0, 9.0, 10), (10, 1))
    river_mask = np.zeros((10, 10), dtype=bool)
    river_mask[:, 0:2] = True  # columns 0 and 1 are river

    model = InundationModel(elev, river_mask, baseline_river_stage=1.5, drainage_rate=0.01)
    # Stage at 3.0m overflows columns 0-2 (elevation < 3.0)
    model.set_river_stage(3.0)
    depth = model.step(dt_minutes=10, rain_rate_mm_hr=0.0)

    # In river/deep flood: Boat must be passable, Ambulance must be blocked
    assert model.is_passable(0, 0, "boat") is True
    assert model.is_passable(0, 0, "ambulance") is False

    # Deep water speed for ambulance is 0.0
    assert model.get_speed_multiplier(0, 0, "ambulance") == 0.0
    # Boat speed on water is 1.0
    assert model.get_speed_multiplier(0, 0, "boat") == 1.0

    # High dry ground (column 9, elevation ~9.0): Ambulance passable, Boat impassable
    assert model.is_passable(9, 5, "ambulance") is True
    assert model.is_passable(9, 5, "boat") is False


def test_bridge_passability_and_failure():
    net = SyntheticCityNetwork(seed=42)
    bridges = net.get_bridges()
    assert len(bridges) > 0
    bridge_edge = bridges[0]

    # Under normal condition, ambulance can cross bridge
    assert net.is_edge_passable(bridge_edge, VehicleType.AMBULANCE, water_depth=0.0) is True
    # Boat cannot cross bridge structure
    assert net.is_edge_passable(bridge_edge, VehicleType.RESCUE_BOAT, water_depth=0.0) is False

    # Simulate bridge collapse
    net.set_bridge_status(bridge_edge.id, is_open=False)
    assert net.is_edge_passable(bridge_edge, VehicleType.AMBULANCE, water_depth=0.0) is False
