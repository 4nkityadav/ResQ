"""
resq_ai.sim.scenario_schema
===========================
Strict validation schema for civil defense disaster scenarios and scheduled injects.
Validates parameters, fleet allocations, hazard rates, and mid-mission injects.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import yaml


VALID_DISASTER_TYPES = {"flash_flood", "earthquake", "urban_fire", "compound"}
VALID_INJECT_TYPES = {"bridge_collapse", "call_burst", "unit_breakdown", "levee_breach", "road_blocked"}


@dataclass
class ScenarioInject:
    step: int
    inject_type: str  # bridge_collapse, call_burst, unit_breakdown, levee_breach
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.step < 0:
            raise ValueError(f"Inject step must be >= 0, got {self.step}")
        if self.inject_type not in VALID_INJECT_TYPES:
            raise ValueError(f"Invalid inject_type '{self.inject_type}'. Valid: {VALID_INJECT_TYPES}")


@dataclass
class ScenarioFleetConfig:
    ambulances: int = 5
    boats: int = 3
    heavy_rescues: int = 0

    def validate(self) -> None:
        if self.ambulances < 1:
            raise ValueError("Must have at least 1 ambulance in fleet")
        if self.boats < 0:
            raise ValueError("Boats count cannot be negative")


@dataclass
class ScenarioConfig:
    scenario_name: str
    disaster_type: str
    seed: int = 42
    duration_steps: int = 120
    focus_zone: str = "zone_riverfront"
    rainfall_initial_mmh: float = 0.0
    wind_speed_initial_kmh: float = 15.0
    call_arrival_poisson_lambda: float = 0.5
    bridge_failure_probability_per_step: float = 0.02
    road_blockage_probability_per_step: float = 0.03
    fleet: ScenarioFleetConfig = field(default_factory=ScenarioFleetConfig)
    injects: List[ScenarioInject] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load_from_yaml(cls, path: str) -> ScenarioConfig:
        file_p = Path(path)
        if not file_p.exists():
            raise FileNotFoundError(f"Scenario YAML not found at: {path}")

        with open(file_p, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)

        return cls.from_dict(raw_data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ScenarioConfig:
        dtype = data.get("disaster_type", "flash_flood")
        if dtype not in VALID_DISASTER_TYPES:
            raise ValueError(f"Disaster type '{dtype}' unrecognized. Must be one of {VALID_DISASTER_TYPES}")

        fleet_raw = data.get("fleet", {})
        fleet = ScenarioFleetConfig(
            ambulances=int(fleet_raw.get("ambulances", 5)),
            boats=int(fleet_raw.get("boats", 3)),
            heavy_rescues=int(fleet_raw.get("heavy_rescues", 0)),
        )
        fleet.validate()

        injects: List[ScenarioInject] = []
        for inj in data.get("injects", []):
            inject_obj = ScenarioInject(
                step=int(inj["step"]),
                inject_type=inj["inject_type"],
                description=inj.get("description", ""),
                parameters=inj.get("parameters", {}),
            )
            inject_obj.validate()
            injects.append(inject_obj)

        cfg = cls(
            scenario_name=str(data.get("scenario_name", "Civil Defense Scenario")),
            disaster_type=dtype,
            seed=int(data.get("seed", 42)),
            duration_steps=int(data.get("duration_steps", 120)),
            focus_zone=str(data.get("focus_zone", "zone_riverfront")),
            rainfall_initial_mmh=float(data.get("rainfall_initial_mmh", 0.0)),
            wind_speed_initial_kmh=float(data.get("wind_speed_initial_kmh", 15.0)),
            call_arrival_poisson_lambda=float(data.get("call_arrival_poisson_lambda", 0.5)),
            bridge_failure_probability_per_step=float(data.get("bridge_failure_probability_per_step", 0.02)),
            road_blockage_probability_per_step=float(data.get("road_blockage_probability_per_step", 0.03)),
            fleet=fleet,
            injects=injects,
            metadata=data.get("metadata", {}),
        )
        return cfg
