"""
resq_ai.dstar.dstar_lite
========================
Full D* Lite (Koenig & Likhachev, 2002) implementation from scratch.
Maintains incremental shortest paths from start to goal on dynamic graphs.
When road cells flood or collapse, repairs affected vertices locally in O(k log N)
instead of re-running global search from scratch.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Set, Optional, Any

from resq_ai.core.models import VehicleType
from resq_ai.sim.city_grid import CityGrid
from resq_ai.dstar.priority_queue import IndexedPriorityQueue, KeyType


Coord = Tuple[int, int]


class DStarLite:
    """
    Incremental heuristic search engine for vehicle navigation in dynamic environments.
    """

    def __init__(
        self,
        grid: CityGrid,
        start: Coord,
        goal: Coord,
        vehicle_type: VehicleType = VehicleType.AMBULANCE,
    ):
        self.grid = grid
        self.s_start: Coord = start
        self.s_last: Coord = start
        self.s_goal: Coord = goal
        self.vtype: VehicleType = vehicle_type

        self.g: Dict[Coord, float] = {}
        self.rhs: Dict[Coord, float] = {}
        self.km: float = 0.0
        self.U = IndexedPriorityQueue()

        self.last_expanded_nodes: Set[Coord] = set()
        self.initialize()

    def get_g(self, s: Coord) -> float:
        return self.g.get(s, float("inf"))

    def get_rhs(self, s: Coord) -> float:
        return self.rhs.get(s, float("inf"))

    def heuristic(self, a: Coord, b: Coord) -> float:
        """
        Consistent Octile distance heuristic for 8-connected grid.
        Min step cost = 0.5 (highway base cost * 0.5 or 0.6).
        """
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        # Octile distance formula: max(dx, dy) + (sqrt(2) - 1) * min(dx, dy)
        min_d = min(dx, dy)
        max_d = max(dx, dy)
        return 0.3 * (max_d + 0.41421356 * min_d)

    def calculate_key(self, s: Coord) -> KeyType:
        """Computes priority key: [min(g, rhs) + h(start, s) + km, min(g, rhs)]."""
        m = min(self.get_g(s), self.get_rhs(s))
        k1 = m + self.heuristic(self.s_start, s) + self.km
        k2 = m
        return (k1, k2)

    def initialize(self) -> None:
        """Resets the search state and seeds the priority queue with the goal node."""
        self.U = IndexedPriorityQueue()
        self.g.clear()
        self.rhs.clear()
        self.km = 0.0
        self.s_last = self.s_start

        self.rhs[self.s_goal] = 0.0
        self.U.insert(self.s_goal, self.calculate_key(self.s_goal))

    def update_vertex(self, u: Coord) -> None:
        """Maintains consistency invariant for vertex u."""
        if u != self.s_goal:
            # rhs(u) = min_{s' in Succ(u)} (c(u, s') + g(s'))
            min_val = float("inf")
            for sprime in self.grid.get_neighbors(u):
                cost_u_sprime = self.grid.get_step_cost(u, sprime, self.vtype)
                if not math.isinf(cost_u_sprime):
                    candidate = cost_u_sprime + self.get_g(sprime)
                    if candidate < min_val:
                        min_val = candidate
            self.rhs[u] = min_val

        g_u = self.get_g(u)
        rhs_u = self.get_rhs(u)

        if self.U.contains(u) and math.isclose(g_u, rhs_u, abs_tol=1e-7):
            self.U.remove(u)
        elif self.U.contains(u) and not math.isclose(g_u, rhs_u, abs_tol=1e-7):
            self.U.update(u, self.calculate_key(u))
        elif (not self.U.contains(u)) and not math.isclose(g_u, rhs_u, abs_tol=1e-7):
            self.U.insert(u, self.calculate_key(u))

    def compute_shortest_path(self, max_expansions: int = 50000) -> Tuple[bool, Set[Coord]]:
        """
        Executes core D* Lite graph repair loop until start vertex is consistent.
        """
        expanded: Set[Coord] = set()
        expansions = 0

        while (
            self.U.top_key() < self.calculate_key(self.s_start)
            or not math.isclose(self.get_rhs(self.s_start), self.get_g(self.s_start), abs_tol=1e-7)
        ):
            if self.U.is_empty() or expansions >= max_expansions:
                break

            u, k_old = self.U.pop_min()
            k_new = self.calculate_key(u)
            expansions += 1
            expanded.add(u)

            if k_old < k_new:
                # Key was outdated due to km accumulated or graph updates
                self.U.insert(u, k_new)
            elif self.get_g(u) > self.get_rhs(u):
                # Overconsistent: propagate decrease in cost
                self.g[u] = self.get_rhs(u)
                # For all predecessors (neighbors in undirected/symmetric graph)
                for s in self.grid.get_neighbors(u):
                    self.update_vertex(s)
            else:
                # Underconsistent: propagate increase in cost (or infinite edge)
                self.g[u] = float("inf")
                self.update_vertex(u)
                for s in self.grid.get_neighbors(u):
                    self.update_vertex(s)

        self.last_expanded_nodes = expanded
        has_path = self.get_g(self.s_start) < float("inf")
        return has_path, expanded

    def notify_cell_changed(self, coord: Coord) -> None:
        """
        Notifies D* Lite that a cell's traversability or cost has changed dynamically.
        Queues vertex updates for the cell and all its neighbors.
        """
        # In our city grid, edge cost from neighbor n into coord changed
        self.update_vertex(coord)
        for n in self.grid.get_neighbors(coord):
            self.update_vertex(n)

    def set_start(self, new_start: Coord) -> None:
        """Advances current vehicle location along the path, accumulating km."""
        if new_start != self.s_start:
            self.km += self.heuristic(self.s_last, new_start)
            self.s_last = new_start
            self.s_start = new_start

    def extract_path(self) -> Tuple[List[Coord], float]:
        """
        Reconstructs the optimal trajectory from current start to goal
        by following steepest gradient of (c(curr, s') + g(s')).
        """
        if self.get_g(self.s_start) == float("inf"):
            return [], float("inf")

        path: List[Coord] = [self.s_start]
        curr = self.s_start
        total_cost = 0.0
        visited: Set[Coord] = {curr}

        while curr != self.s_goal:
            best_neighbor = None
            min_cost = float("inf")
            step_cost = float("inf")

            for sprime in self.grid.get_neighbors(curr):
                c = self.grid.get_step_cost(curr, sprime, self.vtype)
                if not math.isinf(c):
                    candidate_cost = c + self.get_g(sprime)
                    if candidate_cost < min_cost:
                        min_cost = candidate_cost
                        best_neighbor = sprime
                        step_cost = c

            if best_neighbor is None or best_neighbor in visited or math.isinf(min_cost):
                # Trapped or cycle detected
                break

            curr = best_neighbor
            path.append(curr)
            visited.add(curr)
            total_cost += step_cost

        if curr != self.s_goal:
            return [], float("inf")

        return path, round(total_cost, 2)

    def plan(self) -> Tuple[List[Coord], float, Set[Coord]]:
        """Initial global planning call."""
        # Sanity check: start or goal untraversable
        start_cell = self.grid.get_cell(self.s_start[0], self.s_start[1])
        goal_cell = self.grid.get_cell(self.s_goal[0], self.s_goal[1])
        if (
            start_cell is None
            or not start_cell.traversable_for(self.vtype)
            or goal_cell is None
            or not goal_cell.traversable_for(self.vtype)
        ):
            return [], float("inf"), set()

        self.initialize()
        self.compute_shortest_path()
        path, cost = self.extract_path()
        return path, cost, self.last_expanded_nodes

    def replan(self) -> Tuple[List[Coord], float, Set[Coord]]:
        """Incremental replan call after dynamic graph updates."""
        self.compute_shortest_path()
        path, cost = self.extract_path()
        return path, cost, self.last_expanded_nodes
