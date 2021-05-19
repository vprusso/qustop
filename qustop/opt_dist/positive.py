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

from typing import List, Tuple

import cvxpy
import numpy as np


from qustop import Ensemble


class Positive:
    """Positive (global) distinguishability."""

    def __init__(
        self,
        ensemble: Ensemble,
        dist_method: str,
        return_optimal_meas: bool,
        solver: str,
        verbose: bool,
        eps: float,
    ) -> None:
        """Computes either the primal or dual problem of the positive (global) SDP.

        Args:
            ensemble:
            dist_method:
            return_optimal_meas: Whether the optimal measurements are to be returned.
            solver: The SDP solver to use.
            verbose: Overrides the default of hiding the solver output.
            eps: Convergence tolerance.
        """
        self._ensemble = ensemble
        self._dist_method = dist_method
        self._return_optimal_meas = return_optimal_meas
        self._solver = solver
        self._verbose = verbose
        self._eps = eps

        self._states = self._ensemble.density_matrices
        self._probs = self._ensemble.probs

    def solve(self):
        # Return the optimal value and the optimal measurements.
        if self._return_optimal_meas:
            return self.primal_problem()

        # Otherwise, it is often less computationally intensive to just solve the dual problem.
        return self.dual_problem()

    def primal_problem(self) -> Tuple[float, List[cvxpy.Variable]]:
        """Calculate primal problem for the pos (global) distinguishability SDP.

        The primal problem for the min-error case is defined in equation-20 from arXiv:1707.02571
        The primal problem for the unambiguous case is defined in equation- from arXiv:.
        """
        # Unambiguous consists of `len(self._states)` + 1 measurement operators, where the outcome
        # of the `len(self._states)`+1^st corresponds to the inconclusive answer.
        num_measurements = (
            len(self._states) + 1
            if self._dist_method == "unambiguous"
            else len(self._states)
        )

        # Define each measurement variable to be a PSD variable of appropriate dimension.
        meas = [
            cvxpy.Variable(self._ensemble.shape, hermitian=True)
            for _ in range(num_measurements)
        ]

        # Objective function is the inner product between the states and measurements.
        obj_func = [
            self._probs[i] * cvxpy.trace(self._states[i].conj().T @ meas[i])
            for i, _ in enumerate(self._states)
        ]

        # Valid collection of measurements need to sum to the identity operator and be
        # positive semidefinite.
        constraints = [cvxpy.sum(meas) == np.identity(self._ensemble.shape[0])]
        for i in range(num_measurements):
            constraints.append(meas[i] >> 0)

        # Unambiguous state discrimination has an additional constraint on the states and
        # measurements.
        if self._dist_method == "unambiguous":
            for i, _ in enumerate(self._states):
                for j, _ in enumerate(self._states):
                    if i != j:
                        constraints.append(
                            cvxpy.trace(self._states[j].conj().T @ meas[i])
                            == 0
                        )

        obj_sum = cvxpy.sum(obj_func)
        objective = cvxpy.Maximize(cvxpy.real(obj_sum))
        problem = cvxpy.Problem(objective, constraints)
        opt_val = problem.solve(
            solver=self._solver, verbose=self._verbose, eps=self._eps
        )
        return opt_val, meas

    def dual_problem(self) -> float:
        """Calculate dual problem for the positive (global) distinguishability SDP.

        The dual problem for the min-error case is defined in equation-21 from arXiv:1707.02571.
        The dual problem for the unambiguous case is defined in equation-4.73
        from https://uwspace.uwaterloo.ca/bitstream/handle/10012/9572/Cosentino_Alessandro.pdf.
        """
        num_measurements = (
            len(self._states) + 1
            if self._dist_method == "unambiguous"
            else len(self._states)
        )
        constraints = []
        y_var = cvxpy.Variable(self._ensemble.shape, hermitian=True)

        if self._dist_method == "min-error":
            constraints = [
                (y_var - self._probs[i] * self._states[i]) >> 0
                for i in range(num_measurements)
            ]

        # This implements the dual problem (equation-4.73) from
        # https://uwspace.uwaterloo.ca/bitstream/handle/10012/9572/Cosentino_Alessandro.pdf:
        if self._dist_method == "unambiguous":
            scalar_vars = [
                [cvxpy.Variable() for i, _ in enumerate(self._states)]
                for j, _ in enumerate(self._states)
            ]

            for j, _ in enumerate(self._states):
                sum_val = 0
                for i, _ in enumerate(self._states):
                    if i != j:
                        sum_val += (
                            scalar_vars[i][j]
                            * self._probs[i]
                            * self._states[i]
                        )
                constraints.append(
                    y_var - self._probs[j] * self._states[j] + sum_val >> 0
                )
            constraints.append(y_var >> 0)

        objective = cvxpy.Minimize(cvxpy.trace(cvxpy.real(y_var)))
        problem = cvxpy.Problem(objective, constraints)
        opt_val = problem.solve(
            solver=self._solver, verbose=self._verbose, eps=self._eps
        )

        return opt_val
