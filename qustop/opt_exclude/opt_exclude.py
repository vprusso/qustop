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

from typing import Any

import cvxpy
import numpy as np
import picos

from qustop.core import Ensemble


class OptExclude:
    def __init__(
        self,
        ensemble: Ensemble,
        dist_method: str,
        **kwargs: Any,
    ):
        self._ensemble = ensemble
        self._dist_method = dist_method

        self._return_optimal_meas = kwargs.get("return_optimal_meas", True)
        self._solver = kwargs.get("solver", "SCS")
        self._verbose = kwargs.get("verbose", False)
        self._eps = kwargs.get("eps", 1e-8)

        self._states = self._ensemble.density_matrices
        self._probs = self._ensemble.probs

        self._dims = self._ensemble.dims

        self._optimal_value = None
        self._optimal_measurements: list[np.ndarray] = []

    @property
    def value(self) -> float:
        return self._optimal_value

    @property
    def measurements(self) -> list[np.ndarray]:
        if isinstance(self._optimal_measurements[0], cvxpy.Variable):
            self._optimal_measurements = self.convert_measurements(
                self._optimal_measurements
            )
        return self._optimal_measurements

    @staticmethod
    def convert_measurements(measurements) -> list[np.ndarray]:
        return [measurements[i].value for i in range(len(measurements))]

    def solve(self) -> None:
        """Solve either the primal or dual problem for the state exclusion SDP."""
        # Return the optimal value and the optimal measurements.
        if self._return_optimal_meas:
            self.primal_problem()

        # Otherwise, it is often less computationally intensive to just solve the dual problem.
        else:
            self.dual_problem()

    def primal_problem(self) -> None:
        """Calculate primal problem for the state exclusion SDP.

        The primal problem for the min-error case is defined in equation-3 from arXiv:1306.4683.
        The primal problem for the unambiguous case is defined in equation-37 from arXiv:1306.4683.
        """
        num_measurements = len(self._states)

        # Define each measurement variable to be a PSD variable of appropriate dimension.
        meas = [
            cvxpy.Variable(self._ensemble.shape, hermitian=True)
            for _ in range(num_measurements)
        ]

        # Unambiguous state discrimination has an additional constraint on the states and measurements.
        if self._dist_method == "unambiguous":
            # Objective function is the inner product between the states and measurements.
            obj_func = [
                self._probs[i] * cvxpy.trace(self._states[j].conj().T @ meas[i])
                for i, _ in enumerate(self._states)
                for j, _ in enumerate(self._states)
            ]
            # Valid collection of measurements need to sum to the identity operator and be positive semidefinite.
            constraints = [cvxpy.sum(meas) <= np.identity(self._ensemble.shape[0])]
            for i in range(num_measurements):
                constraints.append(meas[i] >> 0)

            for i in range(num_measurements):
                constraints.append(cvxpy.trace(self._states[i].conj().T @ meas[i]) == 0)

            obj_sum = cvxpy.sum(obj_func)
            objective = cvxpy.Maximize(cvxpy.real(obj_sum))

            problem = cvxpy.Problem(objective, constraints)
            opt_val = problem.solve(solver=self._solver, verbose=self._verbose, eps=self._eps)
            self._optimal_value = opt_val
            self._optimal_measurements = meas

        elif self._dist_method == "min-error":
            # Objective function is the inner product between the states and measurements.
            obj_func = [
                self._probs[i] * cvxpy.trace(self._states[i].conj().T @ meas[i])
                for i, _ in enumerate(self._states)
            ]

            # Valid collection of measurements need to sum to the identity operator and be positive semidefinite.
            constraints = [cvxpy.sum(meas) == np.identity(self._ensemble.shape[0])]
            for i in range(num_measurements):
                constraints.append(meas[i] >> 0)

            obj_sum = cvxpy.sum(obj_func)
            objective = cvxpy.Minimize(cvxpy.real(obj_sum))

            problem = cvxpy.Problem(objective, constraints)
            opt_val = problem.solve(solver=self._solver, verbose=self._verbose, eps=self._eps)
            self._optimal_value = opt_val
            self._optimal_measurements = meas
        elif self._dist_method == "worst-case":
            pass
        else:
            pass

    def dual_problem(self) -> None:
        if self._dist_method == "unambiguous":
            problem = picos.Problem()

            # States as rows:
            state_mtx = cvxpy.matrix(self._states)
            n = state_mtx.size[0]
            d = state_mtx.size[1]

            # Set up density matrices as problem parameters.
            density_matrices = []

            for k in range(n):
                mtx = (state_mtx[k, :].H * state_mtx[k, :])
                density_matrices.append(picos.new_param("P[{0}]".format(k), mtx))

            # Set up the Lagrange multiplier matrix (check the interp?).
            Y = problem.add_variable("Y", (d, d), "hermitian")

            # Add constraints:
            problem.add_list_of_constraints([Y << p for p in density_matrices])

            # Add objective:
            problem.set_objective("max", "I" | Y)

            # Solve the problem:
            solution = problem.solve(solver=self._solver, verbose=self._verbose)

            # Extract the optimal measurements:
            measurements = [problem.get_constraint(k).dual for k in range(n)]

            self._optimal_value = solution.value
            self._optimal_measurements = measurements
