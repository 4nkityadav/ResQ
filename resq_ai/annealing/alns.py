"""
resq_ai.annealing.alns
======================
Adaptive Large Neighborhood Search (ALNS) for Multi-Vehicle Emergency Dispatch.
Implemented from scratch in pure Python/NumPy.
Features:
- Destroy operators: Random Removal, Worst-Cost Removal, Shaw Relatedness Removal
- Repair operators: Greedy Best-Insertion, Regret-2 Insertion
- Adaptive score-based roulette-wheel operator selection
- Multi-depot pickup-and-delivery with hospital drop-offs
"""

from __future__ import annotations
import math
import time
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from resq_ai.core.models import Vehicle, DistressCall
from resq_ai.annealing.state import ScheduleState


class ALNSOptimizer:
    """
    Adaptive Large Neighborhood Search metaheuristic for civil defense fleet routing.
    """

    def __init__(
        self,
        max_iterations: int = 400,
        initial_temp: float = 100.0,
        cooling_rate: float = 0.985,
        decay_param: float = 0.8,
        min_destroy_pct: float = 0.15,
        max_destroy_pct: float = 0.40,
        seed: int = 42,
    ):
        self.max_iterations = max_iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.decay = decay_param
        self.min_destroy_pct = min_destroy_pct
        self.max_destroy_pct = max_destroy_pct
        self.rng = np.random.default_rng(seed)

        # Operator names
        self.destroy_ops = ["random", "worst_cost", "shaw"]
        self.repair_ops = ["greedy", "regret2"]

        # Adaptive weights and scores
        self.destroy_weights = np.ones(len(self.destroy_ops), dtype=float)
        self.repair_weights = np.ones(len(self.repair_ops), dtype=float)
        self.destroy_scores = np.zeros(len(self.destroy_ops), dtype=float)
        self.repair_scores = np.zeros(len(self.repair_ops), dtype=float)

    def _select_operator(self, weights: np.ndarray) -> int:
        probs = weights / np.sum(weights)
        return int(self.rng.choice(len(weights), p=probs))

    # --- Destroy Operators ---

    def _destroy_random(self, state: ScheduleState, q: int) -> Tuple[ScheduleState, List[str]]:
        candidate = state.copy()
        assigned_calls = [cid for r in candidate.routes.values() for cid in r]
        if not assigned_calls:
            return candidate, []

        q = min(q, len(assigned_calls))
        removed = list(self.rng.choice(assigned_calls, size=q, replace=False))
        
        for vid, route in candidate.routes.items():
            candidate.routes[vid] = [c for c in route if c not in removed]
        return candidate, removed

    def _destroy_worst_cost(self, state: ScheduleState, q: int) -> Tuple[ScheduleState, List[str]]:
        candidate = state.copy()
        assigned_calls = [cid for r in candidate.routes.values() for cid in r]
        if not assigned_calls:
            return candidate, []

        base_cost = candidate.evaluate_cost()
        savings: List[Tuple[str, float]] = []

        # Compute marginal saving of removing each call
        for cid in assigned_calls:
            temp = candidate.copy()
            for vid, route in temp.routes.items():
                if cid in route:
                    temp.routes[vid].remove(cid)
                    break
            saving = base_cost - temp.evaluate_cost()
            savings.append((cid, saving))

        savings.sort(key=lambda x: x[1], reverse=True)
        q = min(q, len(savings))
        removed = [x[0] for x in savings[:q]]

        for vid, route in candidate.routes.items():
            candidate.routes[vid] = [c for c in route if c not in removed]
        return candidate, removed

    def _destroy_shaw(self, state: ScheduleState, q: int) -> Tuple[ScheduleState, List[str]]:
        """Shaw relatedness removal: removes calls near each other in space and priority."""
        candidate = state.copy()
        assigned_calls = [cid for r in candidate.routes.values() for cid in r]
        if not assigned_calls:
            return candidate, []

        q = min(q, len(assigned_calls))
        first_call = self.rng.choice(assigned_calls)
        removed = [first_call]

        loc1 = state.calls[first_call].location
        r1 = state.calls[first_call].risk_score

        relatedness = []
        for cid in assigned_calls:
            if cid == first_call:
                continue
            loc2 = state.calls[cid].location
            r2 = state.calls[cid].risk_score
            dist = math.hypot(loc1[0] - loc2[0], loc1[1] - loc2[1])
            rel = dist + abs(r1 - r2) * 2.0
            relatedness.append((cid, rel))

        relatedness.sort(key=lambda x: x[1])
        for cid, _ in relatedness:
            if len(removed) >= q:
                break
            removed.append(cid)

        for vid, route in candidate.routes.items():
            candidate.routes[vid] = [c for c in route if c not in removed]
        return candidate, removed

    # --- Repair Operators ---

    def _repair_greedy(self, state: ScheduleState, removed_calls: List[str]) -> ScheduleState:
        candidate = state.copy()
        vids = list(candidate.vehicles.keys())

        for cid in removed_calls:
            best_vid = None
            best_pos = None
            best_cost = float("inf")

            for vid in vids:
                if not candidate.is_vehicle_compatible(vid, cid):
                    continue
                route = candidate.routes[vid]
                for pos in range(len(route) + 1):
                    route.insert(pos, cid)
                    cost = candidate.evaluate_cost()
                    route.pop(pos)

                    if cost < best_cost:
                        best_cost = cost
                        best_vid = vid
                        best_pos = pos

            if best_vid is not None and best_pos is not None:
                candidate.routes[best_vid].insert(best_pos, cid)
            else:
                candidate.unassigned.append(cid)

        return candidate

    def _repair_regret2(self, state: ScheduleState, removed_calls: List[str]) -> ScheduleState:
        candidate = state.copy()
        vids = list(candidate.vehicles.keys())
        unplaced = list(removed_calls)

        while unplaced:
            best_cid = None
            max_regret = -1.0
            best_insertion_for_best_cid = None

            for cid in unplaced:
                insertion_costs = []
                for vid in vids:
                    if not candidate.is_vehicle_compatible(vid, cid):
                        continue
                    route = candidate.routes[vid]
                    for pos in range(len(route) + 1):
                        route.insert(pos, cid)
                        cost = candidate.evaluate_cost()
                        route.pop(pos)
                        insertion_costs.append((cost, vid, pos))

                if not insertion_costs:
                    continue

                insertion_costs.sort(key=lambda x: x[0])
                c1, v1, p1 = insertion_costs[0]
                c2 = insertion_costs[1][0] if len(insertion_costs) > 1 else c1 + 150.0
                regret = c2 - c1

                if regret > max_regret:
                    max_regret = regret
                    best_cid = cid
                    best_insertion_for_best_cid = (v1, p1)

            if best_cid is not None and best_insertion_for_best_cid is not None:
                bv, bp = best_insertion_for_best_cid
                candidate.routes[bv].insert(bp, best_cid)
                unplaced.remove(best_cid)
            else:
                for c in unplaced:
                    candidate.unassigned.append(c)
                break

        return candidate

    # --- Optimization Engine ---

    def optimize(
        self,
        initial_state: ScheduleState,
        max_time_sec: float = 3.0,
    ) -> Tuple[ScheduleState, float, Dict[str, Any]]:
        t_start = time.perf_counter()
        current_state = initial_state.copy()
        current_cost = current_state.evaluate_cost()

        best_state = current_state.copy()
        best_cost = current_cost
        temp = self.initial_temp

        n_total_calls = len(initial_state.calls)
        history = [best_cost]

        for it in range(self.max_iterations):
            if (time.perf_counter() - t_start) >= max_time_sec:
                break

            # Adaptive operator selection
            d_idx = self._select_operator(self.destroy_weights)
            r_idx = self._select_operator(self.repair_weights)

            # Determine number of calls to destroy
            q = int(self.rng.uniform(
                max(1, int(self.min_destroy_pct * n_total_calls)),
                max(2, int(self.max_destroy_pct * n_total_calls))
            ))

            # Apply destroy
            if self.destroy_ops[d_idx] == "random":
                destroyed, removed = self._destroy_random(current_state, q)
            elif self.destroy_ops[d_idx] == "worst_cost":
                destroyed, removed = self._destroy_worst_cost(current_state, q)
            else:
                destroyed, removed = self._destroy_shaw(current_state, q)

            # Repair combines removed calls with any currently unassigned calls
            to_repair = list(removed) + list(destroyed.unassigned)
            destroyed.unassigned = []

            # Apply repair
            if self.repair_ops[r_idx] == "greedy":
                candidate = self._repair_greedy(destroyed, to_repair)
            else:
                candidate = self._repair_regret2(destroyed, to_repair)

            candidate_cost = candidate.evaluate_cost()
            delta = candidate_cost - current_cost

            # Acceptance criterion (Metropolis)
            accepted = False
            score = 0.0

            if candidate_cost < best_cost:
                best_state = candidate.copy()
                best_cost = candidate_cost
                current_state = candidate
                current_cost = candidate_cost
                accepted = True
                score = 5.0  # Global new best
            elif delta < 0:
                current_state = candidate
                current_cost = candidate_cost
                accepted = True
                score = 3.0  # Improving
            elif temp > 0.01 and self.rng.random() < math.exp(-delta / max(temp, 1e-4)):
                current_state = candidate
                current_cost = candidate_cost
                accepted = True
                score = 1.0  # Accepted uphill

            # Update scores
            self.destroy_scores[d_idx] += score
            self.repair_scores[r_idx] += score

            # Cooling
            temp *= self.cooling_rate
            history.append(best_cost)

            # Periodic weight update
            if (it + 1) % 50 == 0:
                self.destroy_weights = self.decay * self.destroy_weights + (1 - self.decay) * (self.destroy_scores + 0.1)
                self.repair_weights = self.decay * self.repair_weights + (1 - self.decay) * (self.repair_scores + 0.1)
                self.destroy_scores.fill(0.0)
                self.repair_scores.fill(0.0)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        stats = {
            "best_cost": best_cost,
            "iterations": it + 1,
            "elapsed_ms": round(elapsed_ms, 2),
            "convergence_history": history,
        }
        return best_state, best_cost, stats
