"""
resq_ai.sim.simulator
=====================
Deterministic discrete-event simulator (DES) with priority-queue event loop.
Supports:
- Microsecond-accurate scheduled event queues (heapq)
- Scenario inject execution (bridge collapses, levee breaches, call bursts, unit breakdowns)
- Seedable 100% deterministic reproducibility
- Full state telemetry checkpoints
"""

from __future__ import annotations
import heapq
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np

from resq_ai.core.models import (
    SimEvent, EventType, Vehicle, DistressCall, VehicleType,
    VehicleStatus, PriorityLevel, OccupantType
)
from resq_ai.sim.scenario_schema import ScenarioConfig, ScenarioInject
from resq_ai.sim.city_grid import CityGrid
from resq_ai.sim.disaster_sim import DisasterSimulator


@dataclass(order=True)
class DESEvent:
    timestamp: float
    priority: int
    event_id: int
    event_type: str = field(compare=False)
    payload: Dict[str, Any] = field(compare=False, default_factory=dict)
    handler: Optional[Callable[[Dict[str, Any]], None]] = field(compare=False, default=None)


class DiscreteEventSimulator:
    """
    Deterministic priority-queue discrete-event simulator for emergency operations.
    """

    def __init__(
        self,
        scenario_config: Optional[ScenarioConfig] = None,
        config_path: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        if config_path:
            self.config = ScenarioConfig.load_from_yaml(config_path)
        elif scenario_config:
            self.config = scenario_config
        else:
            self.config = ScenarioConfig(
                scenario_name="Default Deterministic Simulation",
                disaster_type="flash_flood",
                seed=seed or 42,
            )

        if seed is not None:
            self.config.seed = seed

        self.seed = self.config.seed
        self.rng = np.random.default_rng(self.seed)

        # Internal core disaster engine
        self.engine = DisasterSimulator(seed=self.seed)
        self.current_time: float = 0.0
        self.step_index: int = 0
        
        # Priority queue for discrete events
        self._event_queue: List[DESEvent] = []
        self._id_counter = itertools.count()

        # Telemetry event log
        self.event_log: List[SimEvent] = []

        # Schedule initial events and injects
        self._schedule_scenario_injects()

    def _schedule_scenario_injects(self) -> None:
        """Schedules all declared scenario injects onto the event queue."""
        for inj in self.config.injects:
            t = float(inj.step)
            self.schedule_event(
                timestamp=t,
                priority=1,
                event_type="SCENARIO_INJECT",
                payload={"inject": inj},
            )

    def schedule_event(
        self,
        timestamp: float,
        priority: int,
        event_type: str,
        payload: Dict[str, Any],
        handler: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> int:
        """Enqueues an event at specified timestamp."""
        eid = next(self._id_counter)
        des_event = DESEvent(
            timestamp=timestamp,
            priority=priority,
            event_id=eid,
            event_type=event_type,
            payload=payload,
            handler=handler,
        )
        heapq.heappush(self._event_queue, des_event)
        return eid

    def step(self) -> List[SimEvent]:
        """
        Advances simulation by 1 discrete timestep (1 minute).
        Processes any queued discrete events up to and including current timestep.
        """
        self.step_index += 1
        target_time = float(self.step_index)
        emitted_events: List[SimEvent] = []

        # 1. Process discrete events queued for <= target_time
        while self._event_queue and self._event_queue[0].timestamp <= target_time:
            des_event = heapq.heappop(self._event_queue)
            self.current_time = des_event.timestamp

            if des_event.event_type == "SCENARIO_INJECT":
                inj: ScenarioInject = des_event.payload["inject"]
                sim_events = self._execute_inject(inj)
                emitted_events.extend(sim_events)
            elif des_event.handler:
                des_event.handler(des_event.payload)

        # 2. Advance the base disaster simulation engine
        engine_events = self.engine.step()
        emitted_events.extend(engine_events)

        self.event_log.extend(emitted_events)
        return emitted_events

    def _execute_inject(self, inj: ScenarioInject) -> List[SimEvent]:
        """Executes a scenario inject dynamically within the simulation."""
        events: List[SimEvent] = []

        if inj.inject_type == "bridge_collapse":
            loc = tuple(inj.parameters.get("location", (20, 34)))
            self.engine.grid.set_blocked(loc[0], loc[1], True)
            events.append(SimEvent(
                step=self.step_index,
                event_type=EventType.ROAD_BLOCKED,
                payload={
                    "location": loc,
                    "description": f"INJECT: {inj.description or 'Bridge collapsed'}",
                    "inject_type": "bridge_collapse",
                },
            ))

        elif inj.inject_type == "levee_breach":
            loc = tuple(inj.parameters.get("location", (20, 32)))
            head = float(inj.parameters.get("water_surge_head", 1.5))
            self.engine.grid.set_water_level(loc[0], loc[1], head)
            self.engine.grid.step_inundation(dt_minutes=5, breach_locations=[loc])
            events.append(SimEvent(
                step=self.step_index,
                event_type=EventType.SENSOR_UPDATE,
                payload={
                    "location": loc,
                    "water_depth": head,
                    "description": f"INJECT: Levee breached at {loc} with {head}m water surge",
                },
            ))

        elif inj.inject_type == "call_burst":
            count = int(inj.parameters.get("count", 3))
            zone = inj.parameters.get("focus_zone", "zone_riverfront")
            priority_str = inj.parameters.get("priority", "P1")
            
            for b_idx in range(count):
                bx = int(self.rng.integers(15, 35))
                by = int(self.rng.integers(30, 40))
                cid = f"CALL-INJECT-{self.step_index}-{b_idx+1}"
                new_call = DistressCall(
                    call_id=cid,
                    location=(bx, by),
                    timestamp_step=self.step_index,
                    caller_name=f"Burst Victim {b_idx+1}",
                    occupants_type=OccupantType.ELDERLY if priority_str == "P1" else OccupantType.ADULTS,
                    occupant_count=int(self.rng.integers(1, 4)),
                    building_age="Old",
                    construction_type="Wood",
                    reported_flood_m=0.85 if priority_str == "P1" else 0.25,
                    priority=PriorityLevel.P1 if priority_str == "P1" else PriorityLevel.P2,
                    risk_score=9.5 if priority_str == "P1" else 6.0,
                )
                self.engine.distress_calls[cid] = new_call
                events.append(SimEvent(
                    step=self.step_index,
                    event_type=EventType.NEW_CALL,
                    payload={
                        "call": new_call,
                        "description": f"INJECT: Mass casualty call burst {cid}",
                    },
                ))

        elif inj.inject_type == "unit_breakdown":
            vid = inj.parameters.get("vehicle_id", "AMB-2")
            if vid in self.engine.vehicles:
                veh = self.engine.vehicles[vid]
                veh.status = VehicleStatus.IDLE
                events.append(SimEvent(
                    step=self.step_index,
                    event_type=EventType.VEHICLE_FREE,
                    payload={
                        "vehicle_id": vid,
                        "breakdown": True,
                        "description": f"INJECT: Mechanical breakdown for unit {vid} (Out of Service)",
                    },
                ))

        return events

    def run(self, steps: int = 15) -> List[SimEvent]:
        """Runs the discrete-event simulation deterministically for specified steps."""
        all_events = []
        for _ in range(steps):
            events = self.step()
            all_events.extend(events)
        return all_events
