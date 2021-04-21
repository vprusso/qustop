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

from typing import Any, List

import cvxpy
import numpy as np

from toqito.matrix_ops import tensor
from qustop.core import Ensemble


class OptClone:
    def __init__(
        self,
        ensemble: Ensemble,
        num_reps: int,
        **kwargs: Any,
    ):
        self._ensemble = ensemble
        self._num_reps = num_reps

        self._return_optimal_meas = kwargs.get("return_optimal_meas", True)
        self._solver = kwargs.get("solver", "SCS")
        self._verbose = kwargs.get("verbose", False)
        self._eps = kwargs.get("eps", 1e-8)

        self._optimal_value = None
        self._optimal_measurements: List[np.ndarray] = []

        self._states = self._ensemble.density_matrices
        self._probs = self._ensemble.probs

    @property
    def value(self) -> float:
        return self._optimal_value

    @property
    def measurements(self) -> List[np.ndarray]:
        if isinstance(self._optimal_measurements[0], cvxpy.Variable):
            self._optimal_measurements = self.convert_measurements(
                self._optimal_measurements
            )
        return self._optimal_measurements

    @staticmethod
    def convert_measurements(measurements) -> List[np.ndarray]:
        return [measurements[i].value for i in range(len(measurements))]

    def solve(self) -> None:
        """Depending on the measurement method selected, solve the appropriate optimization problem.
        """
        dim = len(self._states[0]) ** 3

        # Construct the following operator:
        #                                ___               ___
        # Q = ∑_{k=1}^N p_k |ψ_k ⊗ ψ_k ⊗ ψ_k> <ψ_k ⊗ ψ_k ⊗ ψ_k|
        q_a = np.zeros((dim, dim))
        for k, state in enumerate(self._states):
            q_a += (
                self._probs[k]
                * tensor(state, state, state.conj())
                * tensor(state, state, state.conj()).conj().T
            )

        # The system is over:
        # Y_1 ⊗ Z_1 ⊗ X_1, ... , Y_n ⊗ Z_n ⊗ X_n.
        num_spaces = 3

        # In the event of more than a single repetition, one needs to apply a
        # permutation operator to the variables in the SDP to properly align
        # the spaces.
        if num_reps == 1:
            pperm = np.array([1])
        else:
            # The permutation vector `perm` contains elements of the
            # sequence from: https://oeis.org/A023123
            q_a = tensor(q_a, num_reps)
            perm = []
            for i in range(1, num_spaces + 1):
                perm.append(i)
                var = i
                for j in range(1, num_reps):
                    perm.append(var + num_spaces * j)
            pperm = permutation_operator(2, perm)

        if strategy:
            return primal_problem(q_a, pperm, num_reps)
        return dual_problem(q_a, pperm, num_reps)
