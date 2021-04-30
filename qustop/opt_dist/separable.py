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

from typing import List, Tuple, Union

import cvxpy
import numpy as np

from qustop import Ensemble
from toqito.channels import partial_trace, partial_transpose
from toqito.perms import symmetric_projection
from toqito.helper import cvx_kron


class Separable:
    """Separable distinguishability."""

    def __init__(
        self,
        ensemble: Ensemble,
        dist_method: str,
        return_optimal_meas: bool,
        solver: str,
        verbose: bool,
        eps: float,
        level: int,
    ) -> None:
        """Computes either the primal or dual problem of the separable measurement SDP.

        Args:
            ensemble:
            dist_method:
            return_optimal_meas: Whether the optimal measurements are to be returned.
            solver: The SDP solver to use.
            verbose: Overrides the default of hiding the solver output.
            eps: Convergence tolerance.
            level: Level of the hierarchy to compute.
        """
        self._ensemble = ensemble
        self._dist_method = dist_method
        self._return_optimal_meas = return_optimal_meas
        self._solver = solver
        self._verbose = verbose
        self._eps = eps
        self._level = level

        self._states = self._ensemble.density_matrices
        self._probs = self._ensemble.probs

        self._dims = self._ensemble.dims
        self._sys = [i for i in self._ensemble.systems if i % 2 != 0]

        self.dim_x, self.dim_y = self._ensemble[0].shape
        self.dim_list = self._ensemble[0].dims

        # The symmetrically extended list of spaces based on the level. That is
        # Y_2 \otimes ... \otimes Y_{level}
        self._sym_ext_sys_list = list(range(3, 3 + self._level - 1))

        # The symmetrically extended list of dimensions based on the level. That is
        # (X_1 \otimes Y_1) \otimes Y_2 \otimes ... \otimes Y_{level}
        self._sym_ext_dim_list = [self._dims[0]] * (self._level + 1)

        # dim(X_1) * dim(Y_1) * dim(Y_2) * ... * dim(Y_{level})
        self._sym_ext_dims = self._ensemble.shape[0] * np.prod(self._sym_ext_dim_list)

    def solve(self) -> Union[float, Tuple[float, List[cvxpy.Variable]]]:
        """Solve either the primal or dual problem for the separable SDP."""

        # Return the optimal value and the optimal measurements.
        if self._return_optimal_meas:
            return self.primal_problem()

        # Otherwise, it is often less computationally intensive to just solve the dual problem.
        return self.dual_problem()

    def primal_problem(self):
        r"""Compute optimal value of the symmetric extension hierarchy SDP."""
        constraints = []

        # TODO: This can be done in a better and more intuitive manner.
        dim = int(np.log2(self.dim_x))
        dim_list = (2 + self._level - 1) * [dim]
        dim_xyy = np.prod(dim_list)

        sym = symmetric_projection(dim, self._level)

        meas = [
            cvxpy.Variable(self._ensemble.shape, hermitian=True)
            for i, _ in enumerate(self._states)
        ]
        x_var = [
            cvxpy.Variable((dim_xyy, dim_xyy), hermitian=True)
            for i, _ in enumerate(self._states)
        ]
        obj_func = [
            self._probs[i]
            * cvxpy.trace(self._states[i].conj().T @ meas[i])
            for i, _ in enumerate(self._states)
        ]

        for k, _ in enumerate(self._states):
            # Tr_{Y_2 \otimes ... \otimes Y_l}(X_k) = meas[k]:
            constraints.append(
                partial_trace(x_var[k], self._sym_ext_sys_list, dim_list) == meas[k]
            )
            # (I_X \otimes Pi) X_k (I_X \otimes Pi) = X_k
            # where "Pi" is the symmetric projection on (Y \ovee Y_2 \ovee ... \ovee Y_l)
            constraints.append(
                np.kron(np.identity(dim), sym)
                @ x_var[k]
                @ np.kron(np.identity(dim), sym)
                == x_var[k]
            )
            constraints.append(partial_transpose(x_var[k], 1, dim_list) >> 0)
            constraints.append(meas[k] >> 0)
            constraints.append(x_var[k] >> 0)
            for sys in range(3, self._level + 2):
                constraints.append(
                    partial_transpose(x_var[k], sys, dim_list) >> 0
                )

        constraints.append(cvxpy.sum(meas) == np.identity(self._ensemble.shape[0]))

        obj_sum = cvxpy.sum(obj_func)
        objective = cvxpy.Maximize(cvxpy.real(obj_sum))
        problem = cvxpy.Problem(objective, constraints)
        opt_val = problem.solve(
            solver=self._solver, verbose=self._verbose, eps=self._eps
        )

        return opt_val, meas

    def dual_problem(self) -> float:
        dim_x, dim_y = self._states[0].shape

        constraints = []
        Q = []
        R = []
        S = []
        Z = []

        dim = int(np.log2(dim_x))
        dim_list = [dim] * (self._level + 1)

        dim_xy = dim_x
        dim_xyy = np.prod(dim_list)
        sym = symmetric_projection(dim, self._level)
        print(f"DIM: {dim}")
        print(f"DIM_XY: {dim_xy}")
        print(f"DIM_XYY: {dim_xyy}")
        print(f"DIM_LIST: {dim_list}")

        if self._level == 1:
            dim_yp = 1
        else:
            dim_yp = dim * (self._level - 1)

        h_var = cvxpy.Variable((dim_xy, dim_xy), hermitian=True)
        for k, _ in enumerate(self._states):
            Q.append(cvxpy.Variable((dim_xy, dim_xy), hermitian=True))
            R.append(cvxpy.Variable((dim_xyy, dim_xyy), hermitian=True))
            S.append(cvxpy.Variable((dim_xyy, dim_xyy), hermitian=True))
            Z.append(cvxpy.Variable((dim_xyy, dim_xyy), hermitian=True))

            constraints.append(
                h_var - Q[k] >> self._probs[k] * self._states[k]
            )

            constraints.append(
                (
                    cvx_kron(Q[k], np.identity(dim_yp))
                    + (
                        np.kron(np.identity(dim), sym)
                        @ R[k]
                        @ np.kron(np.identity(dim), sym)
                    )
                    - R[k]
                    - partial_transpose(S[k], 1, dim_list)
                    - partial_transpose(Z[k], 2, dim_list)
                )
                >> 0
            )

            constraints.append(R[k] >> 0)
            constraints.append(S[k] >> 0)
            constraints.append(Z[k] >> 0)

        objective = cvxpy.Minimize(cvxpy.trace(cvxpy.real(h_var)))
        problem = cvxpy.Problem(objective, constraints)
        opt_val = problem.solve(
            solver=self._solver, verbose=self._verbose, eps=self._eps
        )

        return opt_val
