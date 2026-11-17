"""
resq_ai.dstar.astar
===================
Standard static A* search on the CityGrid for benchmarking against D* Lite.
Runs from scratch on each invocation.
"""

from __future__ import annotations
import heapq
import math
from typing import Dict, List, Tuple, Set, Optional

from resq_ai.core.models import VehicleType
from resq_ai.sim.city_grid import CityGrid


Coord = Tuple[int, int]


class AStar:
    """Standard static A* search solver for single-shot path planning."""

    def __init__(self, grid: CityGrid, vehicle_type: VehicleType = VehicleType.AMBULANCE):
        self.grid = grid
        self.vtype = vehicle_type

    def heuristic(self, a: Coord, b: Coord) -> float:
        """Octile distance heuristic for 8-connected grid."""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        min_d = min(dx, dy)
        max_d = max(dx, dy)
        return 0.3 * (max_d + 0.41421356 * min_d)

    def search(self, start: Coord, goal: Coord) -> Tuple[List[Coord], float, Set[Coord]]:
        """
        Executes standard A* search from scratch.
        Returns: (path, total_cost, expanded_nodes_set)
        """
        start_cell = self.grid.get_cell(start[0], start[1])
        goal_cell = self.grid.get_cell(goal[0], goal[1])
        if (
            start_cell is None
            or not start_cell.traversable_for(self.vtype)
            or goal_cell is None
            or not goal_cell.traversable_for(self.vtype)
        ):
            return [], float("inf"), set()

        if start == goal:
            return [start], 0.0, set()

        # Priority queue stores (f_score, tie_breaker, current_node)
        open_set: List[Tuple[float, int, Coord]] = []
        counter = 0
        heapq.heappush(open_set, (self.heuristic(start, goal), counter, start))

        came_from: Dict[Coord, Coord] = {}
        g_score: Dict[Coord, float] = {start: 0.0}
        expanded: Set[Coord] = set()

        while open_set:
            current_f, _, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path, round(g_score[goal], 2), expanded

            expanded.add(current)

            for neighbor in self.grid.get_neighbors(current):
                step_cost = self.grid.get_step_cost(current, neighbor, self.vtype)
                if math.isinf(step_cost):
                    continue

                tentative_g = g_score[current] + step_cost
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))

        return [], float("inf"), expanded
