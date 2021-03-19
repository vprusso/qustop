"""PPT distinguishability."""
from typing import List

import cvxpy
import numpy as np

from toqito.channels import partial_transpose
from qustop.core.ensemble import Ensemble


class PPT:
    def __init__(self, ensemble, error, fast=False):
        self.ensemble = ensemble
        self.states = self.ensemble.density_matrices
        self.probs = self.ensemble.probs
        self.error = error

        self.dim_x, self.dim_y = self.ensemble[0].shape
        self.dim_list = self.ensemble[0].dims

        dim = int(np.log2(self.dim_x))
        self.sys_list = list(range(1, dim, 2))

        self.fast = fast

    def solve(self):
        # If just the optimal value is required, it is often less
        # computationally intensive to solve the dual problem.
        if self.fast:
            return self.dual_problem()
        # Otherwise, return the optimal value and the optimal measurements for
        # obtaining that value.
        return self.primal_problem()

    def primal_problem(self):
        r"""
        Calculate primal problem for PPT distinguishability.
        """
        obj_func = []
        meas = []
        constraints = []

        # Unambiguous consists of k + 1 operators, where the outcome of the
        # k+1^st corresponds to the inconclusive answer.
        if self.error == "unambiguous":
            for i in range(len(self.states) + 1):
                meas.append(cvxpy.Variable((self.dim_x, self.dim_x), PSD=True))
                constraints.append(partial_transpose(meas[i], self.sys_list, self.dim_list) >> 0)

            for i, _ in enumerate(self.states):
                for j, _ in enumerate(self.states):
                    if i != j:
                        constraints.append(self.probs[j] * cvxpy.trace(self.states[j].conj().T @ meas[i]) == 0)

        # Minimize error of distinguishing via PPT measurements.
        elif self.error == "min-error":
            for i, _ in enumerate(self.states):
                meas.append(cvxpy.Variable((self.dim_x, self.dim_x), PSD=True))
                constraints.append(partial_transpose(meas[i], self.sys_list, self.dim_list) >> 0)

        for i, _ in enumerate(self.states):
            obj_func.append(self.probs[i] * cvxpy.trace(self.states[i].conj().T @ meas[i]))

        # Valid collection of measurements need to sum to the identity
        # operator.
        constraints.append(sum(meas) == np.identity(self.dim_x))

        objective = cvxpy.Maximize(sum(obj_func))
        problem = cvxpy.Problem(objective, constraints)
        sol_default = problem.solve(solver="CVXOPT")

        return sol_default, meas

    def dual_problem(self):
        r"""
        Calculate dual problem for PPT distinguishability.
        """
        constraints = []
        dual_vars = []

        y_var = cvxpy.Variable((self.dim_x, self.dim_x), hermitian=True)
        objective = cvxpy.Minimize(cvxpy.trace(cvxpy.real(y_var)))

        if self.error == "min-error":
            for i, _ in enumerate(self.states):
                dual_vars.append(cvxpy.Variable((self.dim_x, self.dim_x), PSD=True))
                constraints.append(
                    cvxpy.real(y_var - self.probs[i] * self.states[i])
                    >> partial_transpose(dual_vars[i], sys=self.sys_list, dim=self.dim_list)
                )

        if self.error == "unambiguous":
            for j, _ in enumerate(self.states):
                sum_val = 0
                for i, _ in enumerate(self.states):
                    if i != j:
                        sum_val += cvxpy.real(cvxpy.Variable()) * self.probs[i] * self.states[i]
                dual_vars.append(cvxpy.Variable((self.dim_x, self.dim_x), PSD=True))
                constraints.append(
                    cvxpy.real(y_var - self.probs[j] * self.states[j] + sum_val)
                    >> partial_transpose(dual_vars[j], sys=self.sys_list, dim=self.dim_list)
                )

            dual_vars.append(cvxpy.Variable((self.dim_x, self.dim_x), PSD=True))
            constraints.append(
                cvxpy.real(y_var) >> partial_transpose(dual_vars[-1], sys=self.sys_list, dim=self.dim_list)
            )

        problem = cvxpy.Problem(objective, constraints)
        sol_default = problem.solve(solver="CVXOPT", verbose=True)

        return sol_default, dual_vars