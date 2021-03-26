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
        abstol: float,
    ) -> None:
        self._ensemble = ensemble
        self._dist_method = dist_method
        self._return_optimal_meas = return_optimal_meas
        self._solver = solver
        self._verbose = verbose
        self._abstol = abstol

        self._states = self._ensemble.density_matrices
        self._probs = self._ensemble.probs

        self._dims = self._ensemble.dims
        self._sys = [i for i in self._ensemble.systems if i % 2 != 0]

    def solve(self):

        # If there is only one state in the ensemble, the optimal value is trivially equal to
        # one. The optimal measurement is simply the identity matrix.
        if len(self._ensemble) == 1:
            if self._return_optimal_meas:
                return 1.0, np.identity(self._ensemble.shape[0])
            else:
                return 1.0

        # # There is a closed-form expression for the distinguishability of two density matrices.
        # if len(self.ensemble) == 2:
        #     opt_val = (
        #         1 / 2
        #         + np.linalg.norm(
        #             self.probs[0] * self.states[0]
        #             - self.probs[1] * self.states[1]
        #         )
        #         / 2
        #     )
        #     D, V = np.linalg.eig(
        #         self.probs[0]
        #         * self.states[0][:, [0]]
        #         @ self.states[0][:, [0]].conj().T
        #         - self.probs[1]
        #         * self.states[1][:, [0]]
        #         @ self.states[1][:, [0]].conj().T
        #     )
        #     D = np.diag(D)
        #     pind = np.argwhere(np.asarray(D) >= 0)
        #     print(V[:, [pind]] @ V[:, [pind]].conj().T)
        #     # pind = (D >= 0).nonzero()
        #     #            print(V[:, pind])
        #     #            meas_1 = V[:, pind] @ V[:, pind].conj().T
        #     #            print(V[:, pind])
        #
        #     # Construct optimal measurements:
        #     return opt_val, []

        # Return the optimal value and the optimal measurements.
        if self._return_optimal_meas:
            return self.primal_problem()

        # Otherwise, it is often less computationally intensive to just solve the dual problem.
        return self.dual_problem()

    def primal_problem(self) -> tuple[float, list[cvxpy.Variable]]:
        """Calculate primal problem for the positive (global) distinguishability SDP.

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
            cvxpy.Variable(self._ensemble.shape, PSD=True)
            for _ in range(num_measurements)
        ]

        # Objective function is the inner product between the states and measurements.
        obj_func = [
            self._probs[i] * cvxpy.trace(self._states[i].conj().T @ meas[i])
            for i, _ in enumerate(self._states)
        ]

        # Valid collection of measurements need to sum to the identity operator.
        constraints = [sum(meas) == np.identity(self._ensemble.shape[0])]

        # Unambiguous state discrimination has an additional constraint on the states and
        # measurements.
        if self._dist_method == "unambiguous":
            # This is an extra condition required for the unambiguous case.
            for i, _ in enumerate(self._states):
                for j, _ in enumerate(self._states):
                    if i != j:
                        constraints.append(
                            cvxpy.trace(self._states[i].conj().T @ meas[i])
                            == 0
                        )

        objective = cvxpy.Maximize(sum(obj_func))
        problem = cvxpy.Problem(objective, constraints)
        opt_val = problem.solve(
            solver=self._solver, verbose=self._verbose, abstol=self._abstol
        )
        return opt_val, meas

    def dual_problem(self) -> float:
        """Calculate dual problem for the positive (global) distinguishability SDP.

        The dual problem for the min-error case is defined in equation-21 from arXiv:1707.02571.
        The dual problem for the unambiguous case is defined in equation- from arXiv:.
        """
        num_measurements = (
            len(self._states) + 1
            if self._dist_method == "unambiguous"
            else len(self._states)
        )

        y_var = cvxpy.Variable(self._ensemble.shape, hermitian=True)

        constraints = [
            cvxpy.real(y_var - self._probs[i] * self._states[i]) >> 0
            for i in range(num_measurements)
        ]

        objective = cvxpy.Minimize(cvxpy.trace(cvxpy.real(y_var)))
        problem = cvxpy.Problem(objective, constraints)
        opt_val = problem.solve(
            solver=self._solver, verbose=self._verbose, abstol=self._abstol
        )

        return opt_val
