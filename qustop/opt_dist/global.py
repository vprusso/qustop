
from typing import List

import cvxpy
import numpy as np

from toqito.channels import partial_transpose
from qustop.ensemble import Ensemble

def state_distinguishability(
    states: List[np.ndarray], probs: List[float] = None, dist_method: str = "min-error"
) -> float:
    r"""Compute probability of state distinguishability [ELD03]_."""
    obj_func = []
    measurements = []
    constraints = []

    __is_states_valid(states)
    if probs is None:
        probs = [1 / len(states)] * len(states)
    __is_probs_valid(probs)

    dim_x, dim_y = states[0].shape

    # The variable `states` is provided as a list of vectors. Transform them
    # into density matrices.
    if dim_y == 1:
        for i, state_ket in enumerate(states):
            states[i] = state_ket * state_ket.conj().T

    # Unambiguous state discrimination has an additional constraint on the states and measurements.
    if dist_method == "unambiguous":

        # Note we have one additional measurement operator in the unambiguous case.
        for i in range(len(states) + 1):
            measurements.append(cvxpy.Variable((dim_x, dim_x), PSD=True))

        # This is an extra condition required for the unambiguous case.
        for i, _ in enumerate(states):
            for j, _ in enumerate(states):
                if i != j:
                    constraints.append(cvxpy.trace(states[i].conj().T @ measurements[i]) == 0)

    if dist_method == "min-error":
        for i, _ in enumerate(states):
            measurements.append(cvxpy.Variable((dim_x, dim_x), PSD=True))

    # Objective function is the inner product between the states and measurements.
    for i, _ in enumerate(states):
        obj_func.append(probs[i] * cvxpy.trace(states[i].conj().T @ measurements[i]))

    constraints.append(sum(measurements) == np.identity(dim_x))

    objective = cvxpy.Maximize(sum(obj_func))
    problem = cvxpy.Problem(objective, constraints)
    sol_default = problem.solve()

    return sol_default