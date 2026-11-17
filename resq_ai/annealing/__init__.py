"""
resq_ai.annealing
=================
Fleet allocation and rescue scheduling via Simulated Annealing from scratch.
Pure Python and NumPy:
- ScheduleState representation with hybrid constraint handling
- 4 neighborhood operators: Shift, Swap, 2-Opt Reorder, Unassigned-Exchange
- Geometric & Adaptive cooling with reheat restarts
- Warm-start re-optimization for dynamic event handling
- Greedy and Random dispatch baselines for benchmarking
"""

from resq_ai.annealing.state import ScheduleState
from resq_ai.annealing.simulated_annealing import SimulatedAnnealingOptimizer
from resq_ai.annealing.baselines import GreedyAllocator, RandomAllocator

__all__ = [
    "ScheduleState",
    "SimulatedAnnealingOptimizer",
    "GreedyAllocator",
    "RandomAllocator",
]
