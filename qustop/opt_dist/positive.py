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


from typing import List

import cvxpy
import numpy as np


class Positive:
    """Global distinguishability."""
    def __init__(self, ensemble, dist_method, fast=False):
        self.ensemble = ensemble
        self.states = self.ensemble.density_matrices
        self.probs = self.ensemble.probs
        self.dist_method = dist_method

        self.dim_x, self.dim_y = self.ensemble[0].shape
        self.dim_list = self.ensemble[0].dims

        # TODO: Something needs to be generalized here for sys_list.
        dim = int(np.log2(self.dim_x))
        self.sys_list = list(range(1, dim, 2))

        self.fast = fast

        self.dim = int(np.log2(self.dim_x))

    def solve(self):

        if len(self.ensemble) == 2:
            # TODO: Don't need SDP for just two states
            pass

        # If just the optimal value is required, it is often less
        # computationally intensive to solve the dual problem.
        if self.fast:
            return self.dual_problem()
        # Otherwise, return the optimal value and the optimal measurements for
        # obtaining that value.
        return self.primal_problem()

    def primal_problem(self):
        r"""
        Calculate primal problem for positive distinguishability.
        """
        obj_func = []
        measurements = []
        constraints = []

        if self.dist_method == "unambiguous":
            num_measurements = len(self.states) + 1
        elif self.dist_method == "min-error":
            num_measurements = len(self.states)

        # Unambiguous state discrimination has an additional constraint on the states and
        # measurements.
        if self.dist_method == "unambiguous":
            # This is an extra condition required for the unambiguous case.
            for i, _ in enumerate(self.states):
                for j, _ in enumerate(self.states):
                    if i != j:
                        constraints.append(cvxpy.trace(self.states[i].conj().T @ measurements[i]) == 0)

        # Note we have one additional measurement operator in the unambiguous case.
        for i in range(num_measurements):
            measurements.append(cvxpy.Variable((self.dim_x, self.dim_x), PSD=True))

        # Objective function is the inner product between the states and measurements.
        for i, _ in enumerate(self.states):
            obj_func.append(self.probs[i] * cvxpy.trace(self.states[i].conj().T @ measurements[i]))

        constraints.append(sum(measurements) == np.identity(self.dim_x))

        objective = cvxpy.Maximize(sum(obj_func))
        problem = cvxpy.Problem(objective, constraints)
        sol_default = problem.solve(solver="CVXOPT")

        return sol_default, measurements

    def dual_problem(self):
        r"""
        Calculate dual problem for PPT distinguishability.
        """
        constraints = []
        dual_vars = []
        return 0, []
