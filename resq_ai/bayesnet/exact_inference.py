"""
resq_ai.bayesnet.exact_inference
================================
Exact Bayesian Network inference via Variable Elimination from scratch.
Supports partial evidence, missing observations, topological DAG validation,
and heuristic elimination ordering (Min-Weight / Min-Fill).
"""

from __future__ import annotations
import math
import itertools
from typing import Dict, List, Set, Tuple, Optional, Any
import yaml
import numpy as np

from resq_ai.bayesnet.factor import DiscreteFactor


class BayesianNetwork:
    """
    Direct Acyclic Graph (DAG) Bayesian Network supporting exact inference.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.parents: Dict[str, List[str]] = {}
        self.children: Dict[str, List[str]] = {}
        self.factors: List[DiscreteFactor] = []
        
        if config_path:
            self.load_from_yaml(config_path)

    def add_node(
        self,
        name: str,
        states: List[str],
        parents: Optional[List[str]] = None,
        cpt_data: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Registers a variable node and its CPT."""
        parents = parents or []
        self.nodes[name] = {"states": states, "cardinality": len(states)}
        self.parents[name] = list(parents)
        if name not in self.children:
            self.children[name] = []
        for p in parents:
            if p not in self.children:
                self.children[p] = []
            self.children[p].append(name)

        if cpt_data is not None:
            factor = self._build_factor_from_cpt(name, states, parents, cpt_data)
            self.factors.append(factor)

    def load_from_yaml(self, path: str) -> None:
        """Parses network structure and CPT tables from a YAML configuration file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for n_def in data.get("nodes", []):
            self.add_node(
                name=n_def["name"],
                states=n_def["states"],
                parents=n_def.get("parents", []),
                cpt_data=n_def.get("cpt", []),
            )
        self.validate_dag()

    def validate_dag(self) -> None:
        """Verifies that the graph is strictly acyclic (DAG) using Kahn's algorithm."""
        in_degree = {n: len(self.parents[n]) for n in self.nodes}
        queue = [n for n, deg in in_degree.items() if deg == 0]
        visited_count = 0

        while queue:
            node = queue.pop(0)
            visited_count += 1
            for child in self.children.get(node, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if visited_count != len(self.nodes):
            raise ValueError("Cycle detected in Bayesian Network! Graph must be an acyclic DAG.")

    def _build_factor_from_cpt(
        self,
        node_name: str,
        node_states: List[str],
        parents: List[str],
        cpt_data: List[Dict[str, Any]],
    ) -> DiscreteFactor:
        """Constructs a multidimensional DiscreteFactor from declarative CPT rows."""
        # Variables in factor: [parents..., node]
        factor_vars = list(parents) + [node_name]
        factor_cards = [self.nodes[p]["cardinality"] for p in parents] + [len(node_states)]
        state_names = {p: self.nodes[p]["states"] for p in parents}
        state_names[node_name] = node_states

        factor_values = np.zeros(factor_cards, dtype=np.float64)

        if not parents:
            # Root node
            probs = cpt_data[0]["probs"]
            factor_values[:] = np.array(probs, dtype=np.float64)
        else:
            # Conditional distribution
            for row in cpt_data:
                parent_states = row["states"]
                probs = row["probs"]
                
                # Compute indexing tuple into multidimensional matrix
                idx = []
                for p_idx, p_name in enumerate(parents):
                    p_state = parent_states[p_idx]
                    idx.append(self.nodes[p_name]["states"].index(p_state))
                
                idx.append(slice(None))
                factor_values[tuple(idx)] = np.array(probs, dtype=np.float64)

        return DiscreteFactor(
            variables=factor_vars,
            cardinalities=factor_cards,
            values=factor_values,
            state_names=state_names,
        )

    def _select_elimination_order(
        self,
        variables_to_eliminate: Set[str],
        active_factors: List[DiscreteFactor],
    ) -> List[str]:
        """
        Greedy Min-Weight elimination ordering heuristic.
        Selects variable that produces the smallest intermediate factor size.
        """
        uneliminated = set(variables_to_eliminate)
        order = []

        while uneliminated:
            best_var = None
            min_weight = float("inf")

            for candidate in uneliminated:
                # Find all active factors that contain candidate
                containing = [f for f in active_factors if candidate in f.variables]
                # Compute union scope size if multiplied
                union_vars = set()
                for f in containing:
                    union_vars.update(f.variables)
                # Intermediate factor size
                weight = 1
                for v in union_vars:
                    weight *= self.nodes[v]["cardinality"]

                if weight < min_weight:
                    min_weight = weight
                    best_var = candidate

            order.append(best_var)
            uneliminated.remove(best_var)
            
            # Simulate elimination to update active factors for next heuristic iteration
            containing = [f for f in active_factors if best_var in f.variables]
            active_factors = [f for f in active_factors if best_var not in f.variables]
            union_vars = set()
            for f in containing:
                union_vars.update(f.variables)
            union_vars.discard(best_var)
            if union_vars:
                dummy_factor = DiscreteFactor(
                    variables=list(union_vars),
                    cardinalities=[self.nodes[v]["cardinality"] for v in union_vars],
                    values=np.ones([self.nodes[v]["cardinality"] for v in union_vars]),
                )
                active_factors.append(dummy_factor)

        return order

    def query(
        self,
        query_var: str,
        evidence: Optional[Dict[str, str]] = None,
        elimination_order: Optional[List[str]] = None,
    ) -> DiscreteFactor:
        """
        Exact inference via Variable Elimination:
        Computes P(query_var | evidence).
        Handles missing evidence naturally by marginalizing unobserved variables.
        """
        evidence = evidence or {}
        if query_var in evidence:
            # Query variable itself is observed
            obs_state = evidence[query_var]
            states = self.nodes[query_var]["states"]
            vals = np.zeros(len(states), dtype=np.float64)
            vals[states.index(obs_state)] = 1.0
            return DiscreteFactor(
                variables=[query_var],
                cardinalities=[len(states)],
                values=vals,
                state_names={query_var: states},
            )

        # 1. Instantiate copies of all network factors conditioned on evidence
        active_factors = [f.observe(evidence) for f in self.factors]
        # Filter out empty or scalar trivial factors
        active_factors = [f for f in active_factors if len(f.variables) > 0]

        # 2. Identify variables to eliminate
        all_vars = set(self.nodes.keys())
        observed_vars = set(evidence.keys())
        vars_to_eliminate = all_vars - observed_vars - {query_var}

        # 3. Determine elimination order
        if elimination_order is None:
            elimination_order = self._select_elimination_order(vars_to_eliminate, [f.copy() for f in active_factors])

        # 4. Eliminate variables sequentially
        for var in elimination_order:
            # Partition factors into those containing 'var' and those not
            factors_with_var = [f for f in active_factors if var in f.variables]
            active_factors = [f for f in active_factors if var not in f.variables]

            if not factors_with_var:
                continue

            # Take product of all factors containing 'var'
            product_factor = factors_with_var[0]
            for f in factors_with_var[1:]:
                product_factor = product_factor.product(f)

            # Marginalize out 'var'
            marginalized = product_factor.marginalize([var])
            if len(marginalized.variables) > 0:
                active_factors.append(marginalized)

        # 5. Multiply all remaining factors
        if not active_factors:
            raise RuntimeError(f"No factors remained when querying {query_var}")

        final_factor = active_factors[0]
        for f in active_factors[1:]:
            final_factor = final_factor.product(f)

        # 6. Normalize to obtain posterior P(query_var | evidence)
        final_factor.normalize()
        return final_factor

    def compute_entropy(self, factor: DiscreteFactor) -> float:
        """
        Computes Shannon entropy H(Y) in bits:
        H(Y) = - sum_y P(y) log2 P(y)
        """
        probs = factor.values.flatten()
        entropy = 0.0
        for p in probs:
            if p > 1e-12:
                entropy -= p * math.log2(p)
        return float(entropy)

    def value_of_information(
        self,
        target_var: str,
        candidate_vars: List[str],
        current_evidence: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """
        Computes Expected Value of Information (Information Gain / Mutual Information):
        VoI(X_k) = H(Y | e) - E_{x_k}[ H(Y | e, X_k = x_k) ]
        
        Returns dictionary mapping variable name -> information gain in bits.
        """
        current_evidence = current_evidence or {}
        
        # 1. Base entropy H(Y | e)
        base_posterior = self.query(target_var, evidence=current_evidence)
        h_base = self.compute_entropy(base_posterior)

        voi_results: Dict[str, float] = {}

        for var in candidate_vars:
            if var in current_evidence or var == target_var:
                continue
            if var not in self.nodes:
                continue

            # Query marginal distribution P(var | e)
            p_var_factor = self.query(var, evidence=current_evidence)
            states = self.nodes[var]["states"]

            expected_conditional_entropy = 0.0
            for s_idx, state_val in enumerate(states):
                p_s = float(p_var_factor.values.flatten()[s_idx])
                if p_s <= 1e-9:
                    continue
                # Conditioned evidence: e U {var: state_val}
                hypo_evidence = dict(current_evidence)
                hypo_evidence[var] = state_val
                hypo_posterior = self.query(target_var, evidence=hypo_evidence)
                h_hypo = self.compute_entropy(hypo_posterior)
                expected_conditional_entropy += p_s * h_hypo

            info_gain = max(0.0, h_base - expected_conditional_entropy)
            voi_results[var] = round(float(info_gain), 4)

        return dict(sorted(voi_results.items(), key=lambda item: item[1], reverse=True))

    def sensitivity_analysis(
        self,
        target_var: str,
        target_state: str,
        candidate_vars: List[str],
        current_evidence: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """
        Computes maximum probability swing in P(target_var = target_state)
        induced by varying each candidate variable across its states.
        """
        current_evidence = current_evidence or {}
        swings: Dict[str, float] = {}

        for var in candidate_vars:
            if var in current_evidence or var == target_var or var not in self.nodes:
                continue
            states = self.nodes[var]["states"]
            probs = []
            for s in states:
                hypo_ev = dict(current_evidence)
                hypo_ev[var] = s
                f = self.query(target_var, evidence=hypo_ev)
                p = f.get_probability({target_var: target_state})
                probs.append(p)
            swing = max(probs) - min(probs) if probs else 0.0
            swings[var] = round(float(swing), 4)

        return dict(sorted(swings.items(), key=lambda item: item[1], reverse=True))
