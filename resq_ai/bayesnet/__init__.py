"""
resq_ai.bayesnet
================
Exact probabilistic inference in Bayesian Networks from scratch.
Pure Python and NumPy:
- Factor operations (product, marginalize, observe, normalize)
- Variable Elimination with min-neighbors heuristic ordering
- Triage & Casualty Risk inference engine with explainability
"""

from resq_ai.bayesnet.factor import DiscreteFactor
from resq_ai.bayesnet.exact_inference import BayesianNetwork
from resq_ai.bayesnet.triage import TriageEngine

__all__ = ["DiscreteFactor", "BayesianNetwork", "TriageEngine"]
