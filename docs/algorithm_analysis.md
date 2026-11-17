# ResQ-AI: Algorithmic Complexity & Theoretical Analysis

## 1. Continuous Gaussian Hidden Markov Model (HMM)

### State Formulation
- $N = 4$ Hidden States: $\{\text{Safe, Advisory, Critical, Evacuate}\}$.
- $D = 3$ Continuous Observations: $[\text{water\_level\_m}, \text{rainfall\_mmh}, \text{wind\_speed\_kmh}]^T$.
- Diagonal Gaussian emission density for state $j$:
  $$b_j(\mathbf{y}_t) = \frac{1}{(2\pi)^{D/2} \prod_{d=1}^D \sigma_{jd}} \exp\left( -\frac{1}{2} \sum_{d=1}^D \frac{(y_{td} - \mu_{jd})^2}{\sigma_{jd}^2} \right)$$

### Complexity
| Algorithm | Time Complexity | Space Complexity | Practical Latency |
| :--- | :---: | :---: | :---: |
| **Scaled Forward Filtering** | $O(T \cdot N^2 + T \cdot N \cdot D)$ | $O(T \cdot N)$ | $< 0.4\text{ ms}$ for $T=50$ |
| **Scaled Backward Smoothing**| $O(T \cdot N^2 + T \cdot N \cdot D)$ | $O(T \cdot N)$ | $< 0.4\text{ ms}$ for $T=50$ |
| **Log-Space Viterbi**        | $O(T \cdot N^2 + T \cdot N \cdot D)$ | $O(T \cdot N)$ | $< 0.5\text{ ms}$ for $T=50$ |
| **Baum-Welch (EM)**          | $O(I \cdot [T N^2 + T N D])$         | $O(T \cdot N^2)$ | $< 12\text{ ms}$ for $I=25$ iterations |
| **Online Single-Step Filter**| $O(N^2 + N \cdot D)$                 | $O(N)$           | **$\approx 0.008\text{ ms}$** per step |

*Numerical Stability:* Using scaling factor $c_t = \frac{1}{\sum_j \hat{\alpha}_t(j)}$ prevents IEEE 754 underflow while preserving exact likelihood via $\ln P(Y) = -\sum \ln c_t$.

---

## 2. Bayesian Network Exact Inference (Variable Elimination)

### Graph Topology & Treewidth
- Nodes: $n = 10$.
- Max state cardinality: $d = 4$ (`HazardState`).
- The induced treewidth of the graph under the greedy **Min-Weight elimination ordering** is $w \le 3$.

### Complexity
- **Time Complexity:** $O(n \cdot d^{w+1})$ where $w$ is the induced treewidth.
  With $n = 10, d = 4, w = 3$:
  $$\text{Operations} \approx 10 \times 4^4 = 2,560 \text{ scalar floating-point multiplications}$$
- **Space Complexity:** $O(d^w)$ for storing intermediate factors ($4^3 = 64$ floats).
- **Execution Time:** **$\approx 1.2\text{ ms}$** per call query.

---

## 3. D* Lite Dynamic Route Replanning

### Mathematical Invariants
- For each node $u$:
  - $g(u)$: distance estimate to $s_{\text{goal}}$.
  - $rhs(u) = \min_{s' \in Succ(u)} (c(u, s') + g(s'))$ (one-step lookahead).
- Node is consistent iff $g(u) = rhs(u)$.
- Priority queue keys:
  $$k(u) = [k_1(u), k_2(u)] = [\min(g(u), rhs(u)) + h(s_{\text{start}}, u) + k_m, \min(g(u), rhs(u))]$$

### Complexity Comparison
| Metric | D\* Lite (Incremental Repair) | Static A\* (From Scratch) |
| :--- | :---: | :---: |
| **Initial Global Search** | $O(|V| \log |V| + |E|)$ | $O(|V| \log |V| + |E|)$ |
| **Dynamic Edge Replan**   | **$O(k \log |V| + k \cdot b)$** | $O(|V| \log |V| + |E|)$ |
| **Observed Nodes Expanded** | **333.6 nodes (82.9% reduction)** | **1,950.3 nodes** |
| **Space Complexity**     | $O(|V|)$ | $O(|V|)$ |

*Note:* $k$ is the number of locally inconsistent vertices ($k \ll |V|$), and $b \le 8$ is the branching factor.

---

## 4. Simulated Annealing Fleet Allocation

### Mathematical Formulation
$$\min_{S} f(S) = \sum_{v \in \text{Fleet}} \left[ \alpha \cdot \text{TravelTime}(v) + \sum_{k=1}^{n_v} W_{c_{v,k}} \cdot T_{\text{arrival}}(c_{v,k}) \right] + \text{Penalties}(S)$$

### Complexity
- State evaluation: $O(M \cdot L)$ where $M$ is vehicles (8) and $L$ is max route length ($\le 10$).
- Evaluation per proposal: $O(L)$ using incremental move evaluation.
- For $K$ iterations: $O(K \cdot L)$.
- With $K = 800, L = 5$: $\approx 4,000$ operations $\implies$ **$< 45\text{ ms}$**.
- **Warm-Start Re-optimization:** $K = 200 \implies$ **$< 8\text{ ms}$**.
