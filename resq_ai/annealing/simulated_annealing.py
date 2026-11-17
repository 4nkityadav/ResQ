"""
resq_ai.annealing.simulated_annealing
====================================
Simulated Annealing fleet allocation solver from scratch.
Implements:
- 4 stochastic neighborhood operators (Shift, Swap, 2-Opt Reorder, Unassigned-Exchange)
- Geometric and Adaptive cooling schedules
- Reheat restarts on stagnation
- Warm-start re-optimization for dynamic event arrivals
"""

from __future__ import annotations
import math
import time
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from resq_ai.core.models import Vehicle, DistressCall
from resq_ai.annealing.state import ScheduleState


class SimulatedAnnealingOptimizer:
    """Simulated Annealing metaheuristic for fleet routing and dispatch optimization."""

    def __init__(
        self,
        initial_temp: float = 120.0,
        min_temp: float = 0.10,
        cooling_rate: float = 0.985,
        max_iterations: int = 1200,
        restart_threshold: int = 150,
        cooling_mode: str = "adaptive",  # "geometric" or "adaptive"
        seed: int = 42,
    ):
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.restart_threshold = restart_threshold
        self.cooling_mode = cooling_mode
        self.rng = np.random.default_rng(seed)

        self.history: List[Dict[str, Any]] = []

    def _propose_neighbor(self, current: ScheduleState) -> Optional[ScheduleState]:
        """
        Samples a new candidate state using one of 4 neighborhood operators:
        1. Shift: Move a call to another compatible vehicle
        2. Swap: Inter-route exchange between two vehicles
        3. 2-Opt Reorder: Intra-route order reversal
        4. Unassigned Exchange: Swap or insert from unassigned queue
        """
        candidate = current.copy()
        vids = list(candidate.vehicles.keys())
        active_vids = [vid for vid in vids if len(candidate.routes[vid]) > 0]

        move_type = self.rng.choice(["shift", "swap", "reorder", "unassigned"], p=[0.35, 0.25, 0.25, 0.15])

        # Move 1: Shift a call from vehicle v1 to vehicle v2
        if move_type == "shift" and active_vids:
            v1 = self.rng.choice(active_vids)
            v2 = self.rng.choice(vids)
            if v1 != v2:
                call_idx = int(self.rng.integers(0, len(candidate.routes[v1])))
                cid = candidate.routes[v1][call_idx]
                if candidate.is_vehicle_compatible(v2, cid):
                    candidate.routes[v1].pop(call_idx)
                    insert_pos = int(self.rng.integers(0, len(candidate.routes[v2]) + 1))
                    candidate.routes[v2].insert(insert_pos, cid)
                    return candidate

        # Move 2: Swap a call between vehicle v1 and vehicle v2
        elif move_type == "swap" and len(active_vids) >= 2:
            v1, v2 = self.rng.choice(active_vids, size=2, replace=False)
            idx1 = int(self.rng.integers(0, len(candidate.routes[v1])))
            idx2 = int(self.rng.integers(0, len(candidate.routes[v2])))
            cid1 = candidate.routes[v1][idx1]
            cid2 = candidate.routes[v2][idx2]

            if candidate.is_vehicle_compatible(v1, cid2) and candidate.is_vehicle_compatible(v2, cid1):
                candidate.routes[v1][idx1] = cid2
                candidate.routes[v2][idx2] = cid1
                return candidate

        # Move 3: Intra-route 2-Opt or reversal within a vehicle's route
        elif move_type == "reorder" and active_vids:
            v = self.rng.choice(active_vids)
            route = candidate.routes[v]
            if len(route) >= 2:
                i, j = sorted(self.rng.choice(len(route), size=2, replace=False))
                # 2-opt reversal of sub-route [i:j+1]
                route[i : j + 1] = reversed(route[i : j + 1])
                return candidate

        # Move 4: Unassigned pool exchange or insertion
        elif move_type == "unassigned" and candidate.unassigned:
            u_idx = int(self.rng.integers(0, len(candidate.unassigned)))
            ucid = candidate.unassigned[u_idx]
            v = self.rng.choice(vids)

            if candidate.is_vehicle_compatible(v, ucid):
                candidate.unassigned.pop(u_idx)
                if candidate.routes[v] and self.rng.random() < 0.5:
                    # Swap with existing call
                    swap_idx = int(self.rng.integers(0, len(candidate.routes[v])))
                    old_cid = candidate.routes[v][swap_idx]
                    candidate.routes[v][swap_idx] = ucid
                    candidate.unassigned.append(old_cid)
                else:
                    # Append
                    candidate.routes[v].append(ucid)
                return candidate

        return None

    def optimize(
        self,
        initial_state: ScheduleState,
        max_time_sec: float = 1.5,
    ) -> Tuple[ScheduleState, float, List[Dict[str, Any]]]:
        """
        Executes Simulated Annealing optimization loop.
        Returns: (best_state, best_cost, convergence_history)
        """
        t_start = time.perf_counter()
        self.history.clear()

        current_state = initial_state.copy()
        current_cost = current_state.evaluate_cost()
        best_state = current_state.copy()
        best_cost = current_cost

        T = self.initial_temp
        stagnation_counter = 0
        accepted_recent = []

        for iteration in range(1, self.max_iterations + 1):
            if (time.perf_counter() - t_start) > max_time_sec:
                break

            neighbor = self._propose_neighbor(current_state)
            if neighbor is None:
                continue

            neighbor_cost = neighbor.evaluate_cost()
            delta_E = neighbor_cost - current_cost

            # Metropolis-Hastings acceptance rule
            accepted = False
            if delta_E < 0:
                accepted = True
            else:
                prob = math.exp(-delta_E / max(0.01, T))
                if self.rng.random() < prob:
                    accepted = True

            if accepted:
                current_state = neighbor
                current_cost = neighbor_cost
                if current_cost < best_cost:
                    best_state = current_state.copy()
                    best_cost = current_cost
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
            else:
                stagnation_counter += 1

            accepted_recent.append(1 if accepted else 0)
            if len(accepted_recent) > 30:
                accepted_recent.pop(0)

            # Reheat restart if trapped in local minimum
            if stagnation_counter >= self.restart_threshold:
                T = max(self.initial_temp * 0.45, T * 1.5)
                current_state = best_state.copy()
                current_cost = best_cost
                stagnation_counter = 0

            # Cooling step
            if self.cooling_mode == "adaptive" and len(accepted_recent) >= 20:
                acc_rate = sum(accepted_recent) / len(accepted_recent)
                if acc_rate > 0.55:
                    gamma = self.cooling_rate * 0.99  # Cool slightly faster
                elif acc_rate < 0.15:
                    gamma = min(0.998, self.cooling_rate * 1.01)  # Cool slower
                else:
                    gamma = self.cooling_rate
            else:
                gamma = self.cooling_rate

            T = max(self.min_temp, T * gamma)

            if iteration % 5 == 0 or iteration == 1 or iteration == self.max_iterations:
                self.history.append({
                    "iteration": iteration,
                    "temperature": round(T, 3),
                    "current_cost": round(current_cost, 2),
                    "best_cost": round(best_cost, 2),
                })

        return best_state, best_cost, self.history

    def warm_start_replan(
        self,
        current_state: ScheduleState,
        new_calls: List[DistressCall],
        cost_matrix: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> Tuple[ScheduleState, float]:
        """
        Warm-start re-optimization when new emergency calls arrive.
        Preserves ongoing routes and uses greedy insertion followed by a fast reheat loop.
        Completes in under 100 milliseconds!
        """
        reseeded = current_state.copy()
        if cost_matrix:
            reseeded.cost_matrix.update(cost_matrix)

        # 1. Cheapest greedy insertion of new calls into compatible vehicles
        for call in new_calls:
            reseeded.calls[call.call_id] = call
            best_vid = None
            best_delta = float("inf")
            best_pos = 0

            for vid in reseeded.vehicles:
                if not reseeded.is_vehicle_compatible(vid, call.call_id):
                    continue
                # Evaluate delta cost of inserting at end
                route = reseeded.routes[vid]
                test_state = reseeded.copy()
                test_state.routes[vid].append(call.call_id)
                cost_after = test_state.evaluate_cost()
                delta = cost_after - reseeded.evaluate_cost()

                if delta < best_delta:
                    best_delta = delta
                    best_vid = vid
                    best_pos = len(route)

            if best_vid is not None:
                reseeded.routes[best_vid].insert(best_pos, call.call_id)
            else:
                reseeded.unassigned.append(call.call_id)

        # 2. Focused reheat simulated annealing search (short 100 iteration burst)
        quick_optimizer = SimulatedAnnealingOptimizer(
            initial_temp=35.0,
            min_temp=0.5,
            cooling_rate=0.96,
            max_iterations=100,
            restart_threshold=40,
            cooling_mode="geometric",
            seed=int(self.rng.integers(0, 10000)),
        )
        best_reseeded, best_cost, _ = quick_optimizer.optimize(reseeded, max_time_sec=0.05)
        return best_reseeded, best_cost
