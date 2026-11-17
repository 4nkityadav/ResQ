"""
resq_ai.hmm
===========
Continuous Gaussian Hidden Markov Model (HMM) implemented from scratch in pure NumPy.
Features:
- Numerically stable Forward-Backward algorithm with scaling
- Viterbi decoding in log-space
- Expectation-Maximization (Baum-Welch) parameter estimation
- Real-time online zone hazard tracking & escalation alerting
"""

from resq_ai.hmm.continuous_hmm import GaussianHMM
from resq_ai.hmm.tracker import ZoneHazardTracker

__all__ = ["GaussianHMM", "ZoneHazardTracker"]
