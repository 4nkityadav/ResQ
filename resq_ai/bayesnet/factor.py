"""
resq_ai.bayesnet.factor
=======================
Multidimensional Discrete Factor representation and factor algebra from scratch.
Supports factor product, marginalization, evidence reduction, and normalization
using vectorized NumPy tensor operations.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any, Sequence
import numpy as np


class DiscreteFactor:
    """
    Multidimensional potential / probability factor over discrete random variables.
    """

    def __init__(
        self,
        variables: Sequence[str],
        cardinalities: Sequence[int],
        values: np.ndarray,
        state_names: Optional[Dict[str, List[str]]] = None,
    ):
        self.variables: List[str] = list(variables)
        self.cardinalities: List[int] = list(cardinalities)
        
        # Ensure values shape matches cardinalities
        arr = np.array(values, dtype=np.float64)
        if arr.shape != tuple(self.cardinalities):
            arr = arr.reshape(tuple(self.cardinalities))
        self.values = arr

        self.state_names: Dict[str, List[str]] = state_names or {}
        # Fill default state names if not provided
        for idx, var in enumerate(self.variables):
            if var not in self.state_names:
                self.state_names[var] = [f"state_{s}" for s in range(self.cardinalities[idx])]

    @property
    def scope(self) -> List[str]:
        return list(self.variables)

    def copy(self) -> DiscreteFactor:
        return DiscreteFactor(
            variables=self.variables.copy(),
            cardinalities=self.cardinalities.copy(),
            values=self.values.copy(),
            state_names={k: v.copy() for k, v in self.state_names.items()},
        )

    def product(self, other: DiscreteFactor) -> DiscreteFactor:
        """
        Computes the factor product (phi1 * phi2) over the union of scopes.
        """
        # Determine union of variables preserving order
        union_vars = list(self.variables)
        for v in other.variables:
            if v not in union_vars:
                union_vars.append(v)

        # Build union cardinalities and state names
        union_cards = []
        union_state_names = {}
        for v in union_vars:
            if v in self.variables:
                idx = self.variables.index(v)
                union_cards.append(self.cardinalities[idx])
                union_state_names[v] = self.state_names[v]
            else:
                idx = other.variables.index(v)
                union_cards.append(other.cardinalities[idx])
                union_state_names[v] = other.state_names[v]

        # Reshape self to union shape
        self_slice = [slice(None)] * len(self.variables)
        self_reshaped = self.values
        for v in union_vars:
            if v not in self.variables:
                self_reshaped = np.expand_dims(self_reshaped, axis=-1)
        # Permute axes of self to match union_vars
        current_vars_self = self.variables + [v for v in union_vars if v not in self.variables]
        self_axis_order = [current_vars_self.index(v) for v in union_vars]
        self_aligned = np.transpose(self_reshaped, self_axis_order)

        # Reshape other to union shape
        other_reshaped = other.values
        for v in union_vars:
            if v not in other.variables:
                other_reshaped = np.expand_dims(other_reshaped, axis=-1)
        current_vars_other = other.variables + [v for v in union_vars if v not in other.variables]
        other_axis_order = [current_vars_other.index(v) for v in union_vars]
        other_aligned = np.transpose(other_reshaped, other_axis_order)

        # Pointwise product
        product_values = self_aligned * other_aligned

        return DiscreteFactor(
            variables=union_vars,
            cardinalities=union_cards,
            values=product_values,
            state_names=union_state_names,
        )

    def marginalize(self, variables_to_sum_out: Sequence[str]) -> DiscreteFactor:
        """
        Marginalizes (sums out) a subset of variables from this factor.
        """
        vars_to_remove = [v for v in variables_to_sum_out if v in self.variables]
        if not vars_to_remove:
            return self.copy()

        # Find axes to sum over
        axes_to_sum = tuple(self.variables.index(v) for v in vars_to_remove)
        new_values = np.sum(self.values, axis=axes_to_sum)

        remaining_vars = [v for v in self.variables if v not in vars_to_remove]
        remaining_cards = [self.cardinalities[self.variables.index(v)] for v in remaining_vars]
        remaining_state_names = {v: self.state_names[v] for v in remaining_vars}

        return DiscreteFactor(
            variables=remaining_vars,
            cardinalities=remaining_cards,
            values=new_values,
            state_names=remaining_state_names,
        )

    def observe(self, evidence: Dict[str, str]) -> DiscreteFactor:
        """
        Reduces factor by conditioning on observed variable assignments.
        Variables in evidence are sliced to their observed index and eliminated from scope.
        """
        active_evidence = {k: v for k, v in evidence.items() if k in self.variables}
        if not active_evidence:
            return self.copy()

        # Build indexer slice
        indexer = []
        new_vars = []
        new_cards = []
        new_state_names = {}

        for idx, var in enumerate(self.variables):
            if var in active_evidence:
                obs_val = active_evidence[var]
                state_list = self.state_names[var]
                if obs_val not in state_list:
                    raise ValueError(f"State '{obs_val}' invalid for variable '{var}'. Valid: {state_list}")
                state_idx = state_list.index(obs_val)
                indexer.append(state_idx)
            else:
                indexer.append(slice(None))
                new_vars.append(var)
                new_cards.append(self.cardinalities[idx])
                new_state_names[var] = self.state_names[var]

        sliced_values = self.values[tuple(indexer)]

        return DiscreteFactor(
            variables=new_vars,
            cardinalities=new_cards,
            values=sliced_values,
            state_names=new_state_names,
        )

    def normalize(self) -> DiscreteFactor:
        """Normalizes factor values so they sum to 1.0."""
        total = np.sum(self.values)
        if total > 0:
            self.values = self.values / total
        return self

    def get_probability(self, state_assignments: Dict[str, str]) -> float:
        """Extracts the scalar probability of a specific variable state assignment."""
        indexer = []
        for var in self.variables:
            val = state_assignments[var]
            state_idx = self.state_names[var].index(val)
            indexer.append(state_idx)
        return float(self.values[tuple(indexer)])

    def to_dict(self) -> Dict[str, float]:
        """For a 1-variable factor, returns {state: probability} mapping."""
        if len(self.variables) != 1:
            raise ValueError("to_dict() only supported for single-variable factors")
        var = self.variables[0]
        states = self.state_names[var]
        flat_vals = self.values.flatten()
        return {states[i]: float(flat_vals[i]) for i in range(len(states))}

    def __repr__(self) -> str:
        return f"DiscreteFactor(scope={self.variables}, shape={self.cardinalities})"
