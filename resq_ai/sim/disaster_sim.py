"""
resq_ai.sim.disaster_sim
========================
Disaster event generator and environmental simulator:
- Simulates flood water expansion & structural rubble blockages
- Simulates Poisson call arrivals with varied victim demographics
- Generates continuous noisy sensor feeds aligned with true ground-truth hazard states
- Emits discrete simulation events for the orchestrator
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional
import numpy as np

from resq_ai.core.models import (
    CellType,
    VehicleType,
    VehicleStatus,
    Vehicle,
    DistressCall,
    OccupantType,
    SensorObservation,
    EventType,
    SimEvent,
    HazardLevel,
)
from resq_ai.sim.city_grid import CityGrid


class DisasterSimulator:
    """Simulates real-time disaster escalation, call arrivals, and sensor feeds."""

    def __init__(self, config: Optional[Dict] = None, seed: int = 42):
        self.config = config or {}
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Initialize City
        grid_cfg = self.config.get("grid", {})
        self.grid = CityGrid(
            width=grid_cfg.get("width", 50),
            height=grid_cfg.get("height", 50),
            seed=seed
        )
        
        self.current_step = 0
        self.vehicles: Dict[str, Vehicle] = {}
        self.distress_calls: Dict[str, DistressCall] = {}
        self.call_counter = 0
        
        # Ground truth zone states for later HMM validation
        self.ground_truth_hazard: Dict[str, HazardLevel] = {
            "zone_north": HazardLevel.SAFE,
            "zone_central": HazardLevel.SAFE,
            "zone_riverfront": HazardLevel.ADVISORY,
            "zone_south": HazardLevel.SAFE,
        }
        
        self._initialize_fleet()

    def _initialize_fleet(self) -> None:
        """Sets up 5 ambulances and 3 rescue boats with realistic depot bases."""
        ambulance_depots = [(5, 5), (45, 5), (10, 20), (40, 20), (25, 10)]
        for i in range(5):
            vid = f"AMB-{i+1}"
            depot = ambulance_depots[i % len(ambulance_depots)]
            self.vehicles[vid] = Vehicle(
                vehicle_id=vid,
                vehicle_type=VehicleType.AMBULANCE,
                capacity=2,
                speed_kmh=45.0,
                base_location=depot,
                current_location=depot,
                status=VehicleStatus.IDLE,
                fuel_hours_remaining=8.0
            )

        boat_docks = [(8, 34), (25, 34), (44, 34)]
        for i in range(3):
            vid = f"BOAT-{i+1}"
            dock = boat_docks[i % len(boat_docks)]
            self.vehicles[vid] = Vehicle(
                vehicle_id=vid,
                vehicle_type=VehicleType.RESCUE_BOAT,
                capacity=4,
                speed_kmh=30.0,
                base_location=dock,
                current_location=dock,
                status=VehicleStatus.IDLE,
                fuel_hours_remaining=6.0
            )

    def trigger_blockage(self, x: int, y: int) -> Optional[SimEvent]:
        """Manually or dynamically blocks a cell and returns a ROAD_BLOCKED event."""
        cell = self.grid.get_cell(x, y)
        if cell and not cell.is_blocked:
            cell.is_blocked = True
            return SimEvent(
                step=self.current_step,
                event_type=EventType.ROAD_BLOCKED,
                payload={"coord": (x, y), "zone_id": cell.zone_id}
            )
        return None

    def trigger_clearance(self, x: int, y: int) -> Optional[SimEvent]:
        """Clears a blocked cell and returns a ROAD_CLEARED event."""
        cell = self.grid.get_cell(x, y)
        if cell and cell.is_blocked:
            cell.is_blocked = False
            return SimEvent(
                step=self.current_step,
                event_type=EventType.ROAD_CLEARED,
                payload={"coord": (x, y), "zone_id": cell.zone_id}
            )
        return None

    def step(self) -> List[SimEvent]:
        """Advances simulation by 1 timestep and yields generated events."""
        self.current_step += 1
        events: List[SimEvent] = []

        # 1. Update Ground Truth Hazard Progression
        self._update_hazard_progression()

        # 2. Diffuse Flooding / Environmental Water Expansion
        flood_events = self._diffuse_flood()
        events.extend(flood_events)

        # 3. Probabilistic Road Blockages & Bridge Incidents
        block_events = self._simulate_infrastructure_failures()
        events.extend(block_events)

        # 4. Generate Noisy Sensor Observations per Zone
        sensor_events = self._generate_sensor_readings()
        events.extend(sensor_events)

        # 5. Poisson Process Call Arrivals
        call_events = self._generate_poisson_calls()
        events.extend(call_events)

        return events

    def _update_hazard_progression(self) -> None:
        """Escalates ground-truth hazard levels over time based on scenario profile."""
        # Zone riverfront escalates first and fastest
        if self.current_step >= 5:
            self.ground_truth_hazard["zone_riverfront"] = HazardLevel.CRITICAL
        if self.current_step >= 20:
            self.ground_truth_hazard["zone_riverfront"] = HazardLevel.EVACUATE
            self.ground_truth_hazard["zone_central"] = HazardLevel.CRITICAL
            self.ground_truth_hazard["zone_south"] = HazardLevel.ADVISORY
        if self.current_step >= 45:
            self.ground_truth_hazard["zone_central"] = HazardLevel.EVACUATE
            self.ground_truth_hazard["zone_south"] = HazardLevel.CRITICAL
            self.ground_truth_hazard["zone_north"] = HazardLevel.ADVISORY

    def _diffuse_flood(self) -> List[SimEvent]:
        """Expands water outward from river cells to adjacent low-elevation cells."""
        events: List[SimEvent] = []
        if self.current_step < 3:
            return events

        # Water expansion rate
        surge_level = min(1.8, 0.04 * (self.current_step - 2))
        
        # Sample candidates around river to flood
        for rx, ry in self.grid.river_cells:
            if self.rng.random() < 0.25:
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nx, ny = rx + dx, ry + dy
                    cell = self.grid.get_cell(nx, ny)
                    if cell and cell.cell_type != CellType.RIVER:
                        # Water accumulates faster in low elevations
                        elev_factor = max(0.2, (20.0 - cell.elevation) / 20.0)
                        added_water = round(surge_level * elev_factor, 2)
                        prev_flooded = cell.is_flooded
                        cell.water_level = max(cell.water_level, added_water)
                        
                        # If cell just became impassable for ambulance, treat as blocked road
                        if not prev_flooded and cell.is_flooded and not cell.is_blocked:
                            # 10% chance of sudden physical obstacle debris too
                            if self.rng.random() < 0.10:
                                cell.is_blocked = True
                                events.append(SimEvent(
                                    step=self.current_step,
                                    event_type=EventType.ROAD_BLOCKED,
                                    payload={"coord": (nx, ny), "reason": "flooding_and_debris"}
                                ))
        return events

    def _simulate_infrastructure_failures(self) -> List[SimEvent]:
        """Simulates bridge overflow and random tree/rubble blockages."""
        events: List[SimEvent] = []
        
        # Check bridge failure probability
        p_bridge = self.config.get("disaster", {}).get("bridge_failure_probability_per_step", 0.02)
        for bx, by in self.grid.bridge_locations:
            bcell = self.grid.get_cell(bx, by)
            if bcell and not bcell.is_blocked and self.rng.random() < p_bridge:
                bcell.is_blocked = True
                bcell.water_level = 0.55
                events.append(SimEvent(
                    step=self.current_step,
                    event_type=EventType.ROAD_BLOCKED,
                    payload={"coord": (bx, by), "reason": "bridge_structural_failure"}
                ))

        # Check random arterial road blockages
        p_road = self.config.get("disaster", {}).get("road_blockage_probability_per_step", 0.015)
        if self.rng.random() < p_road:
            rand_x = int(self.rng.integers(0, self.grid.width))
            rand_y = int(self.rng.integers(0, self.grid.height))
            cell = self.grid.get_cell(rand_x, rand_y)
            if cell and cell.cell_type in (CellType.ROAD, CellType.ARTERIAL, CellType.HIGHWAY) and not cell.is_blocked:
                cell.is_blocked = True
                events.append(SimEvent(
                    step=self.current_step,
                    event_type=EventType.ROAD_BLOCKED,
                    payload={"coord": (rand_x, rand_y), "reason": "debris_obstruction"}
                ))
        return events

    def _generate_sensor_readings(self) -> List[SimEvent]:
        """Generates continuous noisy sensor readings for each zone based on ground truth."""
        events: List[SimEvent] = []
        
        # Sensor emission distributions per hazard state: [water_mean, rain_mean, wind_mean]
        # and standard deviations
        profiles = {
            HazardLevel.SAFE: ([0.05, 5.0, 15.0], [0.03, 3.0, 5.0]),
            HazardLevel.ADVISORY: ([0.25, 25.0, 35.0], [0.08, 6.0, 8.0]),
            HazardLevel.CRITICAL: ([0.75, 65.0, 60.0], [0.15, 12.0, 10.0]),
            HazardLevel.EVACUATE: ([1.60, 110.0, 85.0], [0.30, 20.0, 15.0]),
        }

        for zid, zone in self.grid.zones.items():
            state = self.ground_truth_hazard.get(zid, HazardLevel.SAFE)
            means, stds = profiles[state]
            
            # Sample continuous values with Gaussian noise
            water = max(0.0, float(self.rng.normal(means[0], stds[0])))
            rain = max(0.0, float(self.rng.normal(means[1], stds[1])))
            wind = max(0.0, float(self.rng.normal(means[2], stds[2])))
            
            obs = SensorObservation(
                step=self.current_step,
                zone_id=zid,
                water_level_m=round(water, 3),
                rainfall_mmh=round(rain, 2),
                wind_speed_kmh=round(wind, 2)
            )
            zone.history_observations.append(obs)
            events.append(SimEvent(
                step=self.current_step,
                event_type=EventType.SENSOR_UPDATE,
                payload={"zone_id": zid, "observation": obs}
            ))

        return events

    def _generate_poisson_calls(self) -> List[SimEvent]:
        """Samples incoming emergency distress calls using a Poisson arrival process."""
        events: List[SimEvent] = []
        lambda_rate = self.config.get("disaster", {}).get("call_arrival_poisson_lambda", 0.45)
        num_new_calls = int(self.rng.poisson(lambda_rate))

        for _ in range(num_new_calls):
            self.call_counter += 1
            call_id = f"CALL-{self.call_counter:03d}"
            
            # Select random coordinates biased towards more vulnerable/active zones
            zone_weights = [0.1, 0.35, 0.4, 0.15]
            chosen_zid = self.rng.choice(list(self.grid.zones.keys()), p=zone_weights)
            zone = self.grid.zones[chosen_zid]
            min_x, min_y, max_x, max_y = zone.bounds
            
            cx = int(self.rng.integers(min_x, max_x))
            cy = int(self.rng.integers(min_y, max_y))
            cell = self.grid.get_cell(cx, cy)
            
            # Caller demographics & situational attributes
            occ_str = str(self.rng.choice(["Elderly", "Children", "Adults"], p=[0.35, 0.30, 0.35]))
            occ_type = OccupantType(occ_str)
            occ_count = int(self.rng.integers(1, 5))
            b_age = self.rng.choice(["Old", "Moderate", "New"], p=[0.45, 0.35, 0.20])
            c_type = self.rng.choice(["Wood", "Masonry", "ReinforcedConcrete"], p=[0.30, 0.45, 0.25])
            
            local_water = cell.water_level if cell else 0.0
            reported_flood = round(float(local_water + abs(self.rng.normal(0.2, 0.15))), 2)
            fire_present = bool(self.rng.random() < 0.12)
            
            call = DistressCall(
                call_id=call_id,
                location=(cx, cy),
                timestamp_step=self.current_step,
                caller_name=f"Resident-{self.call_counter}",
                occupants_type=occ_type,
                occupant_count=occ_count,
                building_age=b_age,
                construction_type=c_type,
                reported_flood_m=reported_flood,
                fire_present=fire_present
            )
            self.distress_calls[call_id] = call
            events.append(SimEvent(
                step=self.current_step,
                event_type=EventType.NEW_CALL,
                payload={"call": call}
            ))

        return events
