"""
resq_ai.annealing.baselines
===========================
Baseline dispatch algorithms for empirical comparison:
1. Greedy Nearest-Vehicle Dispatch (prioritizes high-risk P1 calls to closest compatible unit)
2. Uniform Random Assignment Dispatch
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple
import numpy as np

from resq_ai.core.models import Vehicle, DistressCall
from resq_ai.annealing.state import ScheduleState


class GreedyAllocator:
    """Greedy dispatch baseline prioritizing highest-risk calls to nearest compatible vehicle."""

    @staticmethod
    def allocate(
        vehicles: Dict[str, Vehicle],
        calls: Dict[str, DistressCall],
        cost_matrix: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> Tuple[ScheduleState, float]:
        state = ScheduleState(
            vehicles=vehicles,
            calls=calls,
            routes={vid: [] for vid in vehicles},
            unassigned=[],
            cost_matrix=cost_matrix,
        )
        
        # Sort calls by risk score descending (P1 calls first)
        sorted_calls = sorted(calls.values(), key=lambda c: c.risk_score, reverse=True)

        for call in sorted_calls:
            cid = call.call_id
            best_vid = None
            min_travel_time = float("inf")

            for vid, vehicle in vehicles.items():
                if not state.is_vehicle_compatible(vid, cid):
                    continue

                # Check current vehicle load
                curr_load = sum(calls[c].occupant_count for c in state.routes[vid])
                if curr_load + call.occupant_count > vehicle.capacity:
                    continue

                # Compute time from current position or last call in route
                last_pos = state.routes[vid][-1] if state.routes[vid] else vid
                tt = state.get_travel_time(last_pos, cid, vehicle.vehicle_type)

                if tt < min_travel_time:
                    min_travel_time = tt
                    best_vid = vid

            if best_vid is not None:
                state.routes[best_vid].append(cid)
            else:
                state.unassigned.append(cid)

        cost = state.evaluate_cost()
        return state, cost


class ClarkeWrightSavingsAllocator:
    """
    Clarke-Wright Savings heuristic for vehicle routing:
    Computes pairwise savings s_{ij} = c(depot, i) + c(depot, j) - c(i, j),
    merging tours iteratively to minimize total dispatch cost.
    """

    @staticmethod
    def allocate(
        vehicles: Dict[str, Vehicle],
        calls: Dict[str, DistressCall],
        cost_matrix: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> Tuple[ScheduleState, float]:
        state = ScheduleState(
            vehicles=vehicles,
            calls=calls,
            routes={vid: [] for vid in vehicles},
            unassigned=[],
            cost_matrix=cost_matrix,
        )

        vids = list(vehicles.keys())
        call_ids = list(calls.keys())
        if not call_ids or not vids:
            return state, state.evaluate_cost()

        # Step 1: Initialize separate single-call routes where possible
        unassigned_calls = []
        for cid in call_ids:
            # Find best compatible vehicle
            comp_vids = [v for v in vids if state.is_vehicle_compatible(v, cid)]
            if not comp_vids:
                unassigned_calls.append(cid)
                continue
            # Assign initially to vehicle with minimum distance
            best_v = min(comp_vids, key=lambda v: state.get_travel_time(v, cid, vehicles[v].vehicle_type))
            state.routes[best_v].append(cid)

        # Step 2: Intra-vehicle Clarke-Wright tour merging / reordering
        for vid, route in state.routes.items():
            if len(route) <= 2:
                continue
            v = vehicles[vid]
            vtype = v.vehicle_type

            # Compute pairwise savings matrix for calls within this vehicle's cluster
            savings = []
            for i in range(len(route)):
                for j in range(len(route)):
                    if i != j:
                        c_i, c_j = route[i], route[j]
                        d_0i = state.get_travel_time(vid, c_i, vtype)
                        d_0j = state.get_travel_time(vid, c_j, vtype)
                        d_ij = state.get_travel_time(c_i, c_j, vtype)
                        s = d_0i + d_0j - d_ij
                        if s > 0:
                            savings.append((s, c_i, c_j))

            savings.sort(key=lambda x: x[0], reverse=True)

            # Reconstruct sequence greedily along highest savings
            if savings:
                ordered = [savings[0][1], savings[0][2]]
                remaining = set(route) - set(ordered)
                while remaining:
                    last = ordered[-1]
                    # Next closest
                    next_c = min(remaining, key=lambda c: state.get_travel_time(last, c, vtype))
                    ordered.append(next_c)
                    remaining.remove(next_c)
                state.routes[vid] = ordered

        state.unassigned = unassigned_calls
        cost = state.evaluate_cost()
        return state, cost


class RandomAllocator:
    """Random assignment baseline allocating calls to compatible vehicles with stochastic ordering."""

    @staticmethod
    def allocate(
        vehicles: Dict[str, Vehicle],
        calls: Dict[str, DistressCall],
        cost_matrix: Optional[Dict[Tuple[str, str], float]] = None,
        seed: int = 42,
    ) -> Tuple[ScheduleState, float]:
        rng = np.random.default_rng(seed)
        state = ScheduleState(
            vehicles=vehicles,
            calls=calls,
            routes={vid: [] for vid in vehicles},
            unassigned=[],
            cost_matrix=cost_matrix,
        )
        vids = list(vehicles.keys())

        call_list = list(calls.values())
        rng.shuffle(call_list)

        for call in call_list:
            cid = call.call_id
            compatible_vids = [v for v in vids if state.is_vehicle_compatible(v, cid)]
            if not compatible_vids:
                state.unassigned.append(cid)
                continue

            chosen_vid = rng.choice(compatible_vids)
            state.routes[chosen_vid].append(cid)

        cost = state.evaluate_cost()
        return state, cost


class ClarkeWrightSavingsAllocator:
    """
    Clarke-Wright Savings heuristic for vehicle routing:
    Computes pairwise savings s_{ij} = c(depot, i) + c(depot, j) - c(i, j),
    merging tours iteratively to minimize total dispatch cost.
    """

    @staticmethod
    def allocate(
        vehicles: Dict[str, Vehicle],
        calls: Dict[str, DistressCall],
        cost_matrix: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> Tuple[ScheduleState, float]:
        state = ScheduleState(
            vehicles=vehicles,
            calls=calls,
            routes={vid: [] for vid in vehicles},
            unassigned=[],
            cost_matrix=cost_matrix,
        )

        vids = list(vehicles.keys())
        call_ids = list(calls.keys())
        if not call_ids or not vids:
            return state, state.evaluate_cost()

        # Step 1: Initialize separate single-call routes where possible
        unassigned_calls = []
        for cid in call_ids:
            # Find best compatible vehicle
            comp_vids = [v for v in vids if state.is_vehicle_compatible(v, cid)]
            if not comp_vids:
                unassigned_calls.append(cid)
                continue
            # Assign initially to vehicle with minimum distance
            best_v = min(comp_vids, key=lambda v: state.get_travel_time(v, cid, vehicles[v].vehicle_type))
            state.routes[best_v].append(cid)

        # Step 2: Intra-vehicle Clarke-Wright tour merging / reordering
        for vid, route in state.routes.items():
            if len(route) <= 2:
                continue
            v = vehicles[vid]
            vtype = v.vehicle_type

            # Compute pairwise savings matrix for calls within this vehicle's cluster
            savings = []
            for i in range(len(route)):
                for j in range(len(route)):
                    if i != j:
                        c_i, c_j = route[i], route[j]
                        d_0i = state.get_travel_time(vid, c_i, vtype)
                        d_0j = state.get_travel_time(vid, c_j, vtype)
                        d_ij = state.get_travel_time(c_i, c_j, vtype)
                        s = d_0i + d_0j - d_ij
                        if s > 0:
                            savings.append((s, c_i, c_j))

            savings.sort(key=lambda x: x[0], reverse=True)

            # Reconstruct sequence greedily along highest savings
            if savings:
                ordered = [savings[0][1], savings[0][2]]
                remaining = set(route) - set(ordered)
                while remaining:
                    last = ordered[-1]
                    # Next closest
                    next_c = min(remaining, key=lambda c: state.get_travel_time(last, c, vtype))
                    ordered.append(next_c)
                    remaining.remove(next_c)
                state.routes[vid] = ordered

        state.unassigned = unassigned_calls
        cost = state.evaluate_cost()
        return state, cost
