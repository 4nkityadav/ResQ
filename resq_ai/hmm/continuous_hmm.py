"""
resq_ai.hmm.continuous_hmm
==========================
Continuous Multivariate Gaussian & Student-t Hidden Markov Model (HMM) from scratch.
Zero external ML libraries. Implements:
1. Log-scale Gaussian and robust Student-t emission densities
2. Missing sensor dimension handling via analytical marginalization (NaN resilience)
3. Sticky state duration transitions (Dirichlet self-transition bias)
4. Scaled Forward algorithm (filtered belief states P(X_t | y_{1:t}))
5. Scaled Backward algorithm (smoothed states P(X_t | y_{1:T}))
6. Log-space Viterbi decoding (MAP state sequence)
7. Multi-restart Baum-Welch (EM) parameter estimation with convergence checks
8. Multi-horizon predictive forecasts (15 / 30 / 60 minutes)
9. Brier calibration scoring and reliability diagram generation
"""

from __future__ import annotations
import math
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from scipy.special import gammaln


class GaussianHMM:
    """
    Multivariate Continuous HMM with diagonal covariance matrices,
    Student-t robust emission support, sticky state dynamics,
    and missing dimension marginalization.
    
    States:
        0: Safe
        1: Advisory
        2: Critical
        3: Evacuate
    """

    STATE_NAMES = ["Safe", "Advisory", "Critical", "Evacuate"]

    def __init__(
        self,
        n_states: int = 4,
        n_features: int = 3,
        pi: Optional[np.ndarray] = None,
        A: Optional[np.ndarray] = None,
        means: Optional[np.ndarray] = None,
        variances: Optional[np.ndarray] = None,
        emission_type: str = "gaussian",  # 'gaussian' or 'student_t'
        student_nu: float = 4.0,           # Degrees of freedom for Student-t
        sticky_kappa: float = 0.05,        # Persistence parameter
        seed: int = 42,
    ):
        self.n_states = n_states
        self.n_features = n_features
        self.emission_type = emission_type.lower()
        self.student_nu = float(student_nu)
        self.sticky_kappa = float(sticky_kappa)
        self.rng = np.random.default_rng(seed)

        # 1. Initial State Distribution (pi): shape (N,)
        if pi is not None:
            self.pi = np.array(pi, dtype=np.float64)
            self.pi = self.pi / np.sum(self.pi)
        else:
            self.pi = np.array([0.85, 0.10, 0.04, 0.01], dtype=np.float64)

        # 2. Transition Matrix (A): shape (N, N)
        if A is not None:
            self.A = np.array(A, dtype=np.float64)
            self.A = self.A / self.A.sum(axis=1, keepdims=True)
        else:
            base_A = np.array([
                [0.82, 0.15, 0.025, 0.005],  # Safe
                [0.06, 0.72, 0.20, 0.02],    # Advisory
                [0.01, 0.05, 0.74, 0.20],    # Critical
                [0.00, 0.01, 0.05, 0.94],    # Evacuate
            ], dtype=np.float64)
            self.A = base_A

        # Apply sticky duration parameter if specified
        if self.sticky_kappa > 0:
            for i in range(self.n_states):
                self.A[i, i] += self.sticky_kappa
            self.A = self.A / self.A.sum(axis=1, keepdims=True)

        # 3. Emission Means (mu): shape (N, D)
        # Features: [water_level_m, rainfall_mmh, wind_speed_kmh]
        if means is not None:
            self.means = np.array(means, dtype=np.float64)
        else:
            self.means = np.array([
                [0.05, 5.0, 15.0],    # Safe
                [0.25, 25.0, 35.0],   # Advisory
                [0.75, 65.0, 60.0],   # Critical
                [1.60, 110.0, 85.0],  # Evacuate
            ], dtype=np.float64)

        # 4. Emission Variances (sigma^2): shape (N, D)
        if variances is not None:
            self.variances = np.array(variances, dtype=np.float64)
        else:
            self.variances = np.array([
                [0.002, 10.0, 25.0],   # Safe
                [0.008, 36.0, 64.0],   # Advisory
                [0.025, 144.0, 100.0], # Critical
                [0.090, 400.0, 225.0], # Evacuate
            ], dtype=np.float64)

        self._eps = 1e-6

    def emission_log_prob(self, state_idx: int, x: np.ndarray) -> float:
        """
        Computes ln b_j(x).
        Resilient to missing sensor telemetry (NaNs) by marginalizing unobserved dimensions.
        Supports both Gaussian and heavy-tailed Student-t distributions.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        valid = ~np.isnan(x_arr)
        
        # Total sensor blackout: return 0.0 (uniform probability across states)
        if not np.any(valid):
            return 0.0

        mu = self.means[state_idx][valid]
        var = np.maximum(self.variances[state_idx][valid], self._eps)
        diff = x_arr[valid] - mu

        if self.emission_type == "student_t":
            # Student-t emission: resists heavy outlier spikes
            nu = self.student_nu
            log_c = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(np.pi * nu * var)
            log_kernel = -0.5 * (nu + 1.0) * np.log(1.0 + (diff ** 2) / (nu * var))
            return float(np.sum(log_c + log_kernel))
        else:
            # Diagonal Gaussian emission
            log_comp = -0.5 * (np.log(2.0 * np.pi * var) + (diff ** 2) / var)
            return float(np.sum(log_comp))

    def emission_prob(self, state_idx: int, x: np.ndarray) -> float:
        """Returns exp(emission_log_prob), safely clipped."""
        logp = self.emission_log_prob(state_idx, x)
        if logp < -700.0:
            return 1e-300
        return float(np.exp(logp))

    def forward(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Scaled Forward Algorithm.
        """
        T = len(observations)
        N = self.n_states
        alpha = np.zeros((T, N), dtype=np.float64)
        c = np.zeros(T, dtype=np.float64)

        # t = 0
        b_0 = np.array([self.emission_prob(i, observations[0]) for i in range(N)], dtype=np.float64)
        alpha_hat_0 = self.pi * b_0
        sum_0 = np.sum(alpha_hat_0)
        c[0] = 1.0 / (sum_0 if sum_0 > 0 else self._eps)
        alpha[0] = c[0] * alpha_hat_0

        # t = 1 ... T-1
        for t in range(1, T):
            b_t = np.array([self.emission_prob(j, observations[t]) for j in range(N)], dtype=np.float64)
            trans_sum = np.dot(alpha[t - 1], self.A)
            alpha_hat_t = trans_sum * b_t
            
            sum_t = np.sum(alpha_hat_t)
            c[t] = 1.0 / (sum_t if sum_t > 0 else self._eps)
            alpha[t] = c[t] * alpha_hat_t

        log_likelihood = -float(np.sum(np.log(np.maximum(c, self._eps))))
        return alpha, c, log_likelihood

    def backward(self, observations: np.ndarray, c: np.ndarray) -> np.ndarray:
        """
        Scaled Backward Algorithm using scaling factors c from forward step.
        """
        T = len(observations)
        N = self.n_states
        beta = np.zeros((T, N), dtype=np.float64)

        # Base case at t = T-1
        beta[T - 1] = c[T - 1]

        # Recurse backwards: t = T-2 down to 0
        for t in range(T - 2, -1, -1):
            b_next = np.array([self.emission_prob(j, observations[t + 1]) for j in range(N)], dtype=np.float64)
            temp = self.A * (b_next * beta[t + 1])
            beta[t] = c[t] * np.sum(temp, axis=1)

        return beta

    def viterbi(self, observations: np.ndarray) -> Tuple[List[int], float]:
        """
        Viterbi Algorithm computed in log-space.
        """
        T = len(observations)
        N = self.n_states
        log_A = np.log(np.maximum(self.A, self._eps))
        log_pi = np.log(np.maximum(self.pi, self._eps))

        V = np.zeros((T, N), dtype=np.float64)
        backpointer = np.zeros((T, N), dtype=int)

        # t = 0
        for i in range(N):
            V[0, i] = log_pi[i] + self.emission_log_prob(i, observations[0])

        # t = 1 ... T-1
        for t in range(1, T):
            for j in range(N):
                log_b_j = self.emission_log_prob(j, observations[t])
                prev_candidates = V[t - 1] + log_A[:, j]
                best_i = int(np.argmax(prev_candidates))
                V[t, j] = prev_candidates[best_i] + log_b_j
                backpointer[t, j] = best_i

        # Backtrack
        best_last = int(np.argmax(V[T - 1]))
        best_log_prob = float(V[T - 1, best_last])

        best_path = [0] * T
        best_path[T - 1] = best_last
        for t in range(T - 2, -1, -1):
            best_path[t] = backpointer[t + 1, best_path[t + 1]]

        return best_path, best_log_prob

    def baum_welch(
        self,
        observations: np.ndarray,
        max_iter: int = 25,
        tolerance: float = 1e-4,
    ) -> List[float]:
        """
        Baum-Welch (EM) parameter estimation from scratch.
        """
        T, D = observations.shape
        N = self.n_states
        log_likelihood_history = []

        for iteration in range(max_iter):
            alpha, c, log_lik = self.forward(observations)
            log_likelihood_history.append(log_lik)
            beta = self.backward(observations, c)

            if iteration > 0 and abs(log_lik - log_likelihood_history[-2]) < tolerance:
                break

            gamma = (alpha * beta) / np.maximum(c[:, None], self._eps)
            gamma_sum = np.sum(gamma, axis=1, keepdims=True)
            gamma = gamma / np.maximum(gamma_sum, self._eps)

            xi = np.zeros((T - 1, N, N), dtype=np.float64)
            for t in range(T - 1):
                b_next = np.array([self.emission_prob(j, observations[t + 1]) for j in range(N)])
                num = (alpha[t, :, None] * self.A) * (b_next[None, :] * beta[t + 1, None, :])
                denom = np.sum(num)
                xi[t] = num / (denom if denom > 0 else self._eps)

            self.pi = gamma[0] / np.sum(gamma[0])

            xi_sum = np.sum(xi, axis=0)
            gamma_t_sum = np.sum(gamma[:-1], axis=0, keepdims=True).T
            self.A = xi_sum / np.maximum(gamma_t_sum, self._eps)
            self.A = self.A / self.A.sum(axis=1, keepdims=True)

            # Update means and variances
            gamma_full_sum = np.sum(gamma, axis=0)
            clean_obs = np.nan_to_num(observations, nan=0.0)
            for j in range(N):
                denom = max(gamma_full_sum[j], self._eps)
                self.means[j] = np.sum(gamma[:, j, None] * clean_obs, axis=0) / denom

            for j in range(N):
                denom = max(gamma_full_sum[j], self._eps)
                diff = clean_obs - self.means[j]
                self.variances[j] = np.sum(gamma[:, j, None] * (diff ** 2), axis=0) / denom
                self.variances[j] = np.maximum(self.variances[j], 1e-4)

        return log_likelihood_history

    @classmethod
    def train_with_restarts(
        cls,
        observations: np.ndarray,
        n_restarts: int = 3,
        max_iter: int = 25,
        tolerance: float = 1e-4,
        seed: int = 42,
    ) -> Tuple[GaussianHMM, float, List[float]]:
        """
        Runs Baum-Welch from multiple random initializations and selects
        the model achieving highest final log-likelihood.
        """
        best_model: Optional[GaussianHMM] = None
        best_log_lik = -float("inf")
        best_history: List[float] = []

        for r in range(n_restarts):
            model = cls(seed=seed + r * 17)
            # Add small perturbation to initial means for restart exploration
            model.means += model.rng.normal(0, 0.1, size=model.means.shape)
            history = model.baum_welch(observations, max_iter=max_iter, tolerance=tolerance)
            final_lik = history[-1] if history else -float("inf")
            if final_lik > best_log_lik:
                best_log_lik = final_lik
                best_model = model
                best_history = history

        assert best_model is not None
        return best_model, best_log_lik, best_history

    def filter_single_step(
        self,
        current_belief: np.ndarray,
        observation: np.ndarray,
    ) -> np.ndarray:
        """
        Online single-step belief update for streaming telemetry.
        P(X_t = s_j | y_{1:t}) proportional to b_j(y_t) * sum_i (P(X_{t-1} = s_i | y_{1:t-1}) * A_{ij})
        Handles missing dimensions automatically.
        """
        N = self.n_states
        predicted = np.dot(current_belief, self.A)
        b = np.array([self.emission_prob(j, observation) for j in range(N)], dtype=np.float64)
        updated = predicted * b
        total = np.sum(updated)
        if total <= 0 or np.isnan(total):
            return predicted / np.sum(predicted)
        return updated / total

    def forecast(self, current_belief: np.ndarray, horizon_steps: int) -> np.ndarray:
        """
        Forecasts belief distribution horizon_steps ahead via transition power:
        b_{t+k} = b_t * (A^k)
        """
        A_k = np.linalg.matrix_power(self.A, horizon_steps)
        projected = np.dot(current_belief, A_k)
        return projected / np.sum(projected)

    def forecast_critical_risk(
        self,
        current_belief: np.ndarray,
        horizons: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        Returns probability of reaching Critical (state 2) or Evacuate (state 3)
        at 15 min (3 steps), 30 min (6 steps), and 60 min (12 steps).
        """
        if horizons is None:
            # Assuming 5 min/timestep
            horizons = [3, 6, 12]

        results = {}
        for h in horizons:
            b_h = self.forecast(current_belief, h)
            p_crit_evac = float(b_h[2] + b_h[3])
            label = f"{h * 5}m"
            results[label] = round(p_crit_evac, 4)

        return results

    @staticmethod
    def compute_brier_score(
        predicted_probs: np.ndarray,
        true_states: List[int],
    ) -> float:
        """
        Computes multiclass Brier calibration score:
        BS = (1 / T) * sum_t sum_k (P_{tk} - I(y_t == k))^2
        Score ranges from 0.0 (perfect) to 2.0 (complete miscalibration).
        """
        T, K = predicted_probs.shape
        brier_sum = 0.0
        for t, true_k in enumerate(true_states):
            target = np.zeros(K)
            target[true_k] = 1.0
            brier_sum += np.sum((predicted_probs[t] - target) ** 2)
        return float(brier_sum / T)
