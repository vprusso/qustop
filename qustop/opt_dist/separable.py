# Copyright (C) 2021 Vincent Russo
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import cvxpy
import numpy as np

from qustop.core import Ensemble
from toqito.channels import partial_trace, partial_transpose
from toqito.perms import symmetric_projection


class Separable:
    """Separable distinguishability."""

    def __init__(
        self, ensemble: Ensemble, dist_method: str, return_optimal_meas: bool = False
    ) -> None:
        self.ensemble = ensemble
        self.dist_method = dist_method
        self.return_optimal_meas = return_optimal_meas

        self.states = self.ensemble.density_matrices
        self.probs = self.ensemble.probs

        self.dim_x, self.dim_y = self.ensemble[0].shape
        self.dim_list = self.ensemble[0].dims

        self.level = 1

        # TODO: Something needs to be generalized here for sys_list.
        dim = int(np.log2(self.dim_x))
        self.sys_list = list(range(1, dim, 2))

        self.dim = int(np.log2(self.dim_x))

    def solve(self):
        # If just the optimal value is required, it is often less
        # computationally intensive to solve the dual problem.
        if self.return_optimal_meas:
            return self.dual_problem()
        # Otherwise, return the optimal value and the optimal measurements for
        # obtaining that value.
        return self.primal_problem()

    def primal_problem(self):
        r"""Compute optimal value of the symmetric extension hierarchy SDP."""
        obj_func = []
        meas = []
        x_var = []
        constraints = []

        dim = int(np.log2(self.dim_x))
        dim_list = [dim] * (self.level + 1)
        # The `sys_list` variable contains the numbering pertaining to the symmetrically extended
        # spaces.
        sys_list = list(range(3, 3 + self.level - 1))
        sym = symmetric_projection(dim, self.level)

        dim_xy = self.dim_x
        dim_xyy = np.prod(dim_list)
        for k, _ in enumerate(states):
            meas.append(cvxpy.Variable((dim_xy, dim_xy), PSD=True))
            x_var.append(cvxpy.Variable((dim_xyy, dim_xyy), PSD=True))
            constraints.append(partial_trace(x_var[k], sys_list, dim_list) == meas[k])
            constraints.append(
                np.kron(np.identity(dim), sym) @ x_var[k] @ np.kron(np.identity(dim), sym)
                == x_var[k]
            )
            constraints.append(partial_transpose(x_var[k], 1, dim_list) >> 0)
            for sys in range(level - 1):
                constraints.append(partial_transpose(x_var[k], sys + 3, dim_list) >> 0)

            obj_func.append(probs[k] * cvxpy.trace(states[k].conj().T @ meas[k]))

        constraints.append(sum(meas) == np.identity(dim_xy))

        objective = cvxpy.Maximize(sum(obj_func))
        problem = cvxpy.Problem(objective, constraints)
        sol_default = problem.solve()

        return sol_default

    def primal_problem(self):
        r"""
        Calculate primal problem for PPT distinguishability.
        """
        obj_func = []
        meas = []
        constraints = []

        if self.dist_method == "unambiguous":
            num_measurements = len(self.states) + 1
        else:
            num_measurements = len(self.states)

        for i in range(num_measurements):
            meas.append(cvxpy.Variable((self.dim_x, self.dim_x), PSD=True))
            constraints.append(partial_transpose(meas[i], self.sys_list, self.dim_list) >> 0)

        # Unambiguous consists of k + 1 operators, where the outcome of the
        # k+1^st corresponds to the inconclusive answer.
        if self.dist_method == "unambiguous":
            for i, _ in enumerate(self.states):
                for j, _ in enumerate(self.states):
                    if i != j:
                        constraints.append(
                            self.probs[j] * cvxpy.trace(self.states[j].conj().T @ meas[i]) == 0
                        )

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

        if self.dist_method == "min-error":
            for i, _ in enumerate(self.states):
                dual_vars.append(cvxpy.Variable((self.dim_x, self.dim_x), PSD=True))
                constraints.append(
                    cvxpy.real(y_var - self.probs[i] * self.states[i])
                    >> partial_transpose(dual_vars[i], sys=self.sys_list, dim=self.dim_list)
                )

        if self.dist_method == "unambiguous":
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
                cvxpy.real(y_var)
                >> partial_transpose(dual_vars[-1], sys=self.sys_list, dim=self.dim_list)
            )

        problem = cvxpy.Problem(objective, constraints)
        sol_default = problem.solve(solver="CVXOPT")

        return sol_default, dual_vars
