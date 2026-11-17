"""
resq_ai.dstar
=============
Dynamic path replanning using D* Lite from scratch in pure Python/NumPy.
Includes:
- Indexed Priority Queue with O(log N) key updates
- D* Lite incremental replanning engine with rhs/g value repair
- Standard A* baseline for comparative benchmarking
- Multi-vehicle cost matrix router for Simulated Annealing
"""

from resq_ai.dstar.priority_queue import IndexedPriorityQueue
from resq_ai.dstar.dstar_lite import DStarLite
from resq_ai.dstar.astar import AStar
from resq_ai.dstar.router import FleetRouter

__all__ = ["IndexedPriorityQueue", "DStarLite", "AStar", "FleetRouter"]
