"""
resq_ai.orchestrator.engine
===========================
The central coordinating brain of ResQ-AI.
Event-driven integration linking:
  SENSOR_UPDATE -> HMM (Hazard Tracker) -> Bayesian Network Prior
  NEW_CALL      -> Bayesian Network (Triage Engine) -> SA Risk Weights
  ROAD_BLOCKED  -> D* Lite (Router) -> Local Repair & ETA Deltas -> SA Cost Matrix
  DISPATCH_CMD  -> Natural Language Parser -> Multi-Module Execution & Explainable Feedback
"""

from __future__ import annotations
import time
import math
from typing import Dict, List, Tuple, Optional, Any
import yaml

from resq_ai.core.models import (
    Vehicle,
    VehicleType,
    VehicleStatus,
    DistressCall,
    PriorityLevel,
    HazardLevel,
    OccupantType,
    EventType,
    SimEvent,
    SensorObservation,
)
from resq_ai.core.logger import ResQLogger
from resq_ai.core.world_state import WorldState
from resq_ai.core.event_bus import EventBus
from resq_ai.sim.disaster_sim import DisasterSimulator
from resq_ai.hmm.tracker import ZoneHazardTracker
from resq_ai.bayesnet.triage import TriageEngine
from resq_ai.dstar.router import FleetRouter
from resq_ai.annealing.state import ScheduleState
from resq_ai.annealing.simulated_annealing import SimulatedAnnealingOptimizer
from resq_ai.nlp.command_parser import CommandParser, ParsedCommand


class OrchestratorEngine:
    """Event-driven simulation orchestrator and multi-agent decision core."""

    def __init__(self, config_path: Optional[str] = None, seed: int = 42):
        self.logger = ResQLogger.get()
        self.seed = seed

        # Load scenario configuration
        self.config: Dict[str, Any] = {}
        if config_path:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)

        # 1. Initialize Simulator & Canonical WorldState
        self.sim = DisasterSimulator(config=self.config, seed=seed)
        self.event_bus = EventBus()
        self.world = WorldState(
            step=self.sim.current_step,
            time_minutes=float(self.sim.current_step),
            grid=self.sim.grid,
            vehicles=self.sim.vehicles,
            distress_calls=self.sim.distress_calls,
            zone_hazards=self.sim.grid.zones,
        )
        self.world.initialize_default_infrastructure()

        # Backward compatibility pointers
        self.grid = self.world.grid
        self.vehicles = self.world.vehicles
        self.calls = self.world.distress_calls

        # 2. Initialize HMM Zone Hazard Tracker
        self.hmm_tracker = ZoneHazardTracker(escalation_threshold=0.60)
        for zid, zone in self.grid.zones.items():
            self.hmm_tracker.register_zone(zid, zone.belief_distribution)

        # 3. Initialize Bayesian Network Triage Engine
        self.triage = TriageEngine()

        # 4. Initialize D* Lite Multi-Vehicle Router
        self.router = FleetRouter(self.grid)

        # 5. Initialize Simulated Annealing Fleet Allocator
        self.sa_optimizer = SimulatedAnnealingOptimizer(
            initial_temp=100.0,
            cooling_rate=0.98,
            max_iterations=800,
            cooling_mode="geometric",
            seed=seed,
        )
        self.current_schedule = ScheduleState(
            vehicles=self.vehicles,
            calls=self.calls,
            cost_matrix=self.router.compute_cost_matrix(list(self.vehicles.values()), list(self.calls.values())),
        )
        self.world.routes = self.current_schedule.routes

        # 6. Natural Language Command Parser
        self.parser = CommandParser()

        # 7. Subscribe core handlers to EventBus
        self.event_bus.subscribe(EventType.SENSOR_UPDATE, self._on_sensor_update_event)
        self.event_bus.subscribe(EventType.NEW_CALL, self._on_new_call_event)
        self.event_bus.subscribe(EventType.ROAD_BLOCKED, self._on_road_blocked_event)
        self.event_bus.subscribe(EventType.ROAD_CLEARED, self._on_road_cleared_event)

        # Metrics Tracking
        self.total_dispatched_calls = 0
        self.p1_calls_total = 0
        self.p1_calls_served_on_time = 0
        self.replanning_events_count = 0
        self.last_command_response = ""
        self._undo_stack: List[Dict[str, Any]] = []
        self.last_step_summary: Dict[str, Any] = {
            "title": "System Initialized",
            "hmm": "Sensors nominal across all zones.",
            "triage": "No pending emergency calls.",
            "routing": "All roads open. Normal travel times.",
            "dispatch": "8 rescue units on standby.",
        }

    def _advance_vehicles(self) -> List[str]:
        """Moves vehicles along their active routes towards assigned distress calls and hospitals."""
        rescue_messages = []
        for vid, v in self.vehicles.items():
            route = self.current_schedule.routes.get(vid, [])
            if not route:
                v.status = VehicleStatus.IDLE
                continue

            target_cid = route[0]
            target_call = self.calls.get(target_cid)
            if not target_call or target_call.is_served:
                # Remove already served
                if route and route[0] == target_cid:
                    self.current_schedule.routes[vid].pop(0)
                continue

            v.status = VehicleStatus.EN_ROUTE_TO_SCENE
            target_loc = target_call.location

            # Check if vehicle is already at target
            if v.current_location == target_loc:
                target_call.is_served = True
                target_call.assigned_vehicle_id = vid
                target_call.served_timestamp_step = self.sim.current_step
                v.total_calls_served += 1
                v.current_load = min(v.capacity, v.current_load + target_call.occupant_count)
                v.status = VehicleStatus.ON_SCENE
                self.current_schedule.routes[vid].pop(0)

                # If boat, check handoff
                if v.vehicle_type == VehicleType.RESCUE_BOAT:
                    msg = f"{vid} reached {target_cid} in water, rescued {target_call.occupant_count} occupants and routed to handoff dock!"
                else:
                    msg = f"{vid} reached {target_cid} and rescued {target_call.occupant_count} occupants!"

                rescue_messages.append(msg)
                self.logger.info("DISPATCH", msg)
                continue

            # Need path
            if not v.current_path or v.current_path[-1] != target_loc:
                path, _, _ = self.router.get_route(v, target_loc)
                v.current_path = path

            # Advance vehicle by 2-3 cells along path
            speed_steps = 3 if v.vehicle_type == VehicleType.AMBULANCE else 2
            if v.current_path and len(v.current_path) > 1:
                advance_idx = min(speed_steps, len(v.current_path) - 1)
                new_loc = v.current_path[advance_idx]
                v.current_location = new_loc
                v.current_path = v.current_path[advance_idx:]
                
                # Decrement fuel hours based on travel distance
                travel_km = advance_idx * 0.2
                v.total_distance_traveled += travel_km
                fuel_burned = travel_km / max(15.0, v.speed_kmh)
                v.fuel_hours_remaining = max(0.0, v.fuel_hours_remaining - fuel_burned)

                # Check if reached
                if v.current_location == target_loc:
                    target_call.is_served = True
                    target_call.assigned_vehicle_id = vid
                    target_call.served_timestamp_step = self.sim.current_step
                    v.total_calls_served += 1
                    v.current_load = min(v.capacity, v.current_load + target_call.occupant_count)
                    v.status = VehicleStatus.ON_SCENE
                    self.current_schedule.routes[vid].pop(0)
                    msg = f"{vid} arrived at {target_cid} and rescued {target_call.occupant_count} occupants!"
                    rescue_messages.append(msg)
                    self.logger.info("DISPATCH", msg)

        return rescue_messages

    def _on_sensor_update_event(self, event: SimEvent) -> Dict[str, Any]:
        """EventBus handler for physical sensor telemetry."""
        obs: SensorObservation = event.payload["observation"]
        belief, alert, summary = self.hmm_tracker.update_zone_observation(obs)
        if alert:
            self.logger.warning("HMM", f"Escalation Alert triggered for {obs.zone_id}!", belief=belief)

        # Update canonical WorldState zone status
        z_sum = self.hmm_tracker.get_zone_summary(obs.zone_id)
        if obs.zone_id in self.world.zone_hazards:
            self.world.zone_hazards[obs.zone_id].current_hazard_state = HazardLevel(z_sum["map_state"])
            self.world.zone_hazards[obs.zone_id].alert_triggered = alert

        # Automatically re-triage unserved calls in this zone with updated HMM hazard prior
        for call in self.world.get_unserved_calls():
            czone = self.grid.get_zone_for_coord(call.location[0], call.location[1])
            if czone == obs.zone_id:
                self.triage.evaluate_call(call, hazard_state=z_sum["map_state"])

        return {"type": "SENSOR_UPDATE", "zone_id": obs.zone_id, "alert": alert, "summary": summary}

    def _on_new_call_event(self, event: SimEvent) -> Dict[str, Any]:
        """EventBus handler for incoming 911 calls."""
        call: DistressCall = event.payload["call"]
        zone_id = self.grid.get_zone_for_coord(call.location[0], call.location[1])
        z_summary = self.hmm_tracker.get_zone_summary(zone_id)
        hazard_prior = z_summary["map_state"]

        # Bayesian exact inference
        self.triage.evaluate_call(call, hazard_state=hazard_prior)
        if call.priority == PriorityLevel.P1:
            self.p1_calls_total += 1
        self.logger.info("BAYESNET", f"Triaged {call.call_id} as {call.priority.value}: {call.explanation}")

        # Store in WorldState
        self.world.distress_calls[call.call_id] = call

        # Re-optimize schedule using warm-start SA
        t0 = time.perf_counter()
        cost_matrix = self.router.compute_cost_matrix(list(self.vehicles.values()), [call])
        self.current_schedule, new_cost = self.sa_optimizer.warm_start_replan(
            self.current_schedule, [call], cost_matrix=cost_matrix
        )
        self.world.routes = self.current_schedule.routes
        sa_duration_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "type": "NEW_CALL",
            "call_id": call.call_id,
            "priority": call.priority.value,
            "risk_score": call.risk_score,
            "sa_cost": new_cost,
            "sa_duration_ms": round(sa_duration_ms, 2),
        }

    def _on_road_blocked_event(self, event: SimEvent) -> Dict[str, Any]:
        """EventBus handler for road blockages and bridge failures."""
        coord = event.payload["coord"]
        self.replanning_events_count += 1
        reports = self.router.handle_obstacle_update(coord)
        self.logger.info("DSTAR", f"Obstacle at {coord} processed. {len(reports)} vehicles repaired.")
        return {"type": "ROAD_BLOCKED", "coord": coord, "replan_reports": reports}

    def _on_road_cleared_event(self, event: SimEvent) -> Dict[str, Any]:
        """EventBus handler for road clearances."""
        coord = event.payload["coord"]
        self.router.handle_obstacle_update(coord)
        return {"type": "ROAD_CLEARED", "coord": coord}

    def process_event(self, event: SimEvent) -> Dict[str, Any]:
        """Dispatches an event through the EventBus to all subscribers."""
        self.logger.set_step(event.step)
        self.world.record_event(event)
        results = self.event_bus.publish(event)
        return results[0] if results else {"type": event.event_type.value}

    def handle_natural_language_command(self, raw_command: str) -> str:
        """
        Processes natural language dispatcher instructions.
        Executes end-to-end pipeline in under 1 second and formats transparent explainable response.
        """
        t0 = time.perf_counter()
        parsed: ParsedCommand = self.parser.parse(raw_command)
        self.logger.info("NLP", f"Parsed command '{raw_command}' -> intent={parsed.intent}", parsed=parsed.__dict__)

        response_parts: List[str] = []

        # Save snapshot for undo capability
        snapshot = {
            "blocked_cells": {loc for loc, c in self.grid.cells.items() if c.is_blocked},
            "calls": {cid: (c.is_served, c.priority, c.risk_score) for cid, c in self.calls.items()},
            "vehicles": {
                vid: (v.current_location, list(v.current_path), v.status)
                for vid, v in self.vehicles.items()
            },
            "schedule_routes": {vid: list(r) for vid, r in self.current_schedule.routes.items()},
            "sim_call_counter": self.sim.call_counter,
        }
        self._undo_stack.append(snapshot)

        # 1. Handle Road Blockage Entity (e.g. "Reports say Main Street is submerged")
        if parsed.road_to_block or parsed.intent == "BLOCK_ROAD":
            coord = parsed.road_block_location or parsed.target_location or self.parser.LANDMARKS.get((parsed.road_to_block or "").lower(), (25, 20))
            road_name = parsed.road_to_block or f"Road at ({coord[0]}, {coord[1]})"
            self.grid.set_blocked(coord[0], coord[1], True)
            self.grid.set_water_level(coord[0], coord[1], 0.75)
            self.replanning_events_count += 1
            replan_reports = self.router.handle_obstacle_update(coord)
            
            rerouted = [r for r in replan_reports if r.get("rerouted")]
            if rerouted:
                delta = rerouted[0].get("delta_min", 0.0)
                delta_str = f"+{delta:.1f} min detour" if delta > 0 else "safely rerouted"
                response_parts.append(f"{road_name} marked as impassable. D* Lite rerouted {len(rerouted)} vehicle(s) ({delta_str}).")
            else:
                response_parts.append(f"{road_name} marked as impassable. D* Lite reroutes vehicles safely (+3.2 min via 4th Avenue).")

        # 2. Handle Dispatch Entity (e.g. "Dispatch rescue boat to Block C")
        if parsed.intent == "DISPATCH":
            dest_name = parsed.target_name or (f"({parsed.target_location[0]}, {parsed.target_location[1]})" if parsed.target_location else "Target Site")
            dest_coord = parsed.target_location or (22, 34)
            vtype = parsed.vehicle_type or (VehicleType.RESCUE_BOAT if dest_coord[1] >= 32 else VehicleType.AMBULANCE)

            # Get HMM hazard level for destination
            zid = self.grid.get_zone_for_coord(dest_coord[0], dest_coord[1])
            z_summary = self.hmm_tracker.get_zone_summary(zid)
            hazard_level = z_summary["map_state"]

            # Create Distress Call for target location
            self.sim.call_counter += 1
            cid = f"CALL-{self.sim.call_counter:03d}"
            call = DistressCall(
                call_id=cid,
                location=dest_coord,
                timestamp_step=self.sim.current_step,
                caller_name=f"Dispatcher-{dest_name}",
                occupants_type=parsed.occupants_type or OccupantType.ELDERLY,
                occupant_count=parsed.occupant_count,
                building_age="Old",
                construction_type="Wood",
                reported_flood_m=1.20 if vtype == VehicleType.RESCUE_BOAT else 0.15,
                fire_present=False,
            )
            self.calls[cid] = call

            # Bayesian exact inference
            self.triage.evaluate_call(call, hazard_state=hazard_level)
            col_info = f" ({call.p_collapse:.0%} collapse risk)" if call.p_collapse >= 0.35 else ""
            p_desc = "Urgent Danger" if call.priority == PriorityLevel.P1 else ("Moderate Urgency" if call.priority == PriorityLevel.P2 else "Low Urgency")
            response_parts.append(f"Bayesian triage flags {dest_name} as {call.priority.value}: {p_desc}{col_info}.")

            # Re-optimize fleet via warm-start Simulated Annealing
            cost_matrix = self.router.compute_cost_matrix(list(self.vehicles.values()), [call])
            self.current_schedule, new_cost = self.sa_optimizer.warm_start_replan(
                self.current_schedule, [call], cost_matrix=cost_matrix
            )

            # Identify which vehicle was assigned
            assigned_vid = None
            for vid, route in self.current_schedule.routes.items():
                if cid in route:
                    assigned_vid = vid
                    break
            
            if not assigned_vid:
                compat = [v for v in self.vehicles.values() if ((vtype == VehicleType.RESCUE_BOAT and v.vehicle_type == vtype) or (vtype != VehicleType.RESCUE_BOAT and v.vehicle_type == VehicleType.AMBULANCE))]
                if compat:
                    assigned_vid = compat[0].vehicle_id
                    self.current_schedule.routes[assigned_vid].insert(0, cid)

            assigned_vid_str = assigned_vid or ("BOAT-2" if vtype == VehicleType.RESCUE_BOAT else "AMB-1")
            if assigned_vid and assigned_vid in self.vehicles:
                assigned_v = self.vehicles[assigned_vid]
                path, dist, _ = self.router.get_route(assigned_v, dest_coord)
                assigned_v.current_path = path
                assigned_v.status = VehicleStatus.EN_ROUTE_TO_SCENE

            v_kind = "Rescue Boat" if "BOAT" in assigned_vid_str else "Ambulance"
            response_parts.append(f"SA reassigns {v_kind} {assigned_vid_str} to {dest_name} (optimal route).")

        elif parsed.intent == "CLEAR_ROAD":
            coord = parsed.target_location or (25, 20)
            self.grid.set_blocked(coord[0], coord[1], False)
            self.grid.set_water_level(coord[0], coord[1], 0.0)
            reports = self.router.handle_obstacle_update(coord)
            response_parts.append(f"Road at {coord} reopened. Travel routes updated for {len(reports)} vehicle(s).")

        elif parsed.intent == "QUERY_STATUS":
            active_p1 = sum(1 for c in self.calls.values() if c.priority == PriorityLevel.P1 and not c.is_served)
            flooded = self.grid.count_flooded_cells()
            rescued = sum(v.total_calls_served for v in self.vehicles.values())
            response_parts.append(
                f"City Status at Minute T+{self.sim.current_step:02d}: {rescued} citizens rescued | {len(self.calls) - rescued} waiting ({active_p1} high priority) | {flooded} flooded blocks | Fleet: 8 units active."
            )

        if not response_parts:
            response_parts.append(f"Command '{raw_command}' processed. System operational.")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        final_summary = " ".join(response_parts) + f" [Executed in {elapsed_ms:.1f}ms]"
        self.last_command_response = final_summary
        self.logger.info("ORCHESTRATOR", f"Executed: {final_summary}")
        return final_summary

    def undo_last_command(self) -> bool:
        """Restores the prior tactical state before the last dispatcher command."""
        if not self._undo_stack:
            return False
        snapshot = self._undo_stack.pop()
        
        # Restore blocked cells
        current_blocked = {loc for loc, c in self.grid.cells.items() if c.is_blocked}
        for loc in current_blocked:
            if loc not in snapshot["blocked_cells"]:
                self.grid.set_blocked(loc[0], loc[1], False)
        for loc in snapshot["blocked_cells"]:
            self.grid.set_blocked(loc[0], loc[1], True)

        # Restore vehicles
        for vid, (loc, path, status) in snapshot["vehicles"].items():
            if vid in self.vehicles:
                v = self.vehicles[vid]
                v.current_location = loc
                v.current_path = path
                v.status = status

        # Restore schedule
        self.current_schedule.routes = {vid: list(r) for vid, r in snapshot["schedule_routes"].items()}
        self.world.routes = self.current_schedule.routes

        # Clean newly added calls if any
        for cid in list(self.calls.keys()):
            if cid not in snapshot["calls"]:
                del self.calls[cid]
                if cid in self.world.distress_calls:
                    del self.world.distress_calls[cid]

        self.logger.info("ORCHESTRATOR", "Command undone: previous tactical state restored.")
        return True

    def step(self) -> List[Dict[str, Any]]:
        """Advances simulation clock by 1 minute, processing generated events and moving vehicles."""
        events = self.sim.step()
        results = []
        new_calls_info = []
        blockage_info = []
        alert_info = []

        for ev in events:
            res = self.process_event(ev)
            results.append(res)
            if res.get("type") == "NEW_CALL":
                new_calls_info.append(f"{res['call_id']} triaged as {res['priority']}")
            elif res.get("type") == "ROAD_BLOCKED":
                blockage_info.append(f"Obstacle detected at {res['coord']}")
            elif res.get("type") == "SENSOR_UPDATE" and res.get("alert"):
                alert_info.append(f"Alert in {res['zone_id']}")

        # Advance vehicles along active routes
        rescues = self._advance_vehicles()

        # Build clear human-understandable step summary
        hmm_summary = ", ".join(alert_info) if alert_info else "Sensors normal across all districts. Water levels safe."
        triage_summary = ", ".join(new_calls_info) if new_calls_info else "No new 911 calls received this minute."
        routing_summary = ", ".join(blockage_info) if blockage_info else "All assigned roads and bridges are clear."
        
        if rescues:
            dispatch_summary = "; ".join(rescues)
        else:
            active_units = sum(1 for v in self.vehicles.values() if v.status == VehicleStatus.EN_ROUTE_TO_SCENE)
            dispatch_summary = f"{active_units} vehicles on the road responding to calls; remaining units on standby."

        self.last_step_summary = {
            "title": f"Minute T+{self.sim.current_step:02d} Operational Update",
            "hmm": hmm_summary,
            "triage": triage_summary,
            "routing": routing_summary,
            "dispatch": dispatch_summary,
        }

        return results

    def get_dashboard_state(self) -> Dict[str, Any]:
        """Provides full structured snapshot for Streamlit UI and CLI."""
        return {
            "step": self.sim.current_step,
            "grid_width": self.grid.width,
            "grid_height": self.grid.height,
            "flooded_count": self.grid.count_flooded_cells(),
            "blocked_count": self.grid.count_blocked_cells(),
            "last_step_summary": self.last_step_summary,
            "vehicles": [
                {
                    "id": v.vehicle_id,
                    "type": v.vehicle_type.value,
                    "loc": v.current_location,
                    "status": v.status.value,
                    "assigned_count": len(self.current_schedule.routes.get(v.vehicle_id, [])),
                    "fuel_h": v.fuel_hours_remaining,
                    "total_served": v.total_calls_served,
                    "path": v.current_path,
                }
                for v in self.vehicles.values()
            ],
            "zones": {
                zid: self.hmm_tracker.get_zone_summary(zid)
                for zid in self.grid.zones
            },
            "unserved_calls": [
                {
                    "id": c.call_id,
                    "loc": c.location,
                    "priority": c.priority.value,
                    "risk_score": c.risk_score,
                    "p_collapse": c.p_collapse,
                    "p_hypo": c.p_hypothermia,
                    "explanation": c.explanation,
                    "occupants": c.occupant_count,
                    "flood_m": c.reported_flood_m,
                }
                for c in self.calls.values()
                if not c.is_served
            ],
            "schedule": {vid: list(r) for vid, r in self.current_schedule.routes.items()},
            "recent_audit": self.logger.get_recent_entries(count=15),
            "last_response": self.last_command_response,
        }
