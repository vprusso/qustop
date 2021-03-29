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
from cvxpy.expressions.expression import Expression


def cvx_kron(
    expr_1: Union[np.ndarray, Expression], expr_2: Union[np.ndarray, Expression]
) -> Expression:
    """
    Compute Kronecker product between CVXPY objects.

    By default, CVXPY does not support taking the Kronecker product when the argument on the left is
    equal to a CVXPY object and the object on the right is equal to a numpy object.

    At most one of :code:`expr_1` and :code:`b` may be CVXPY Variable objects.

    Kudos to Riley J. Murray for this function:
    https://github.com/cvxgrp/cvxpy/issues/457

    :param expr_1: 2D numpy ndarray, or a CVXPY Variable with expr_1.ndim == 2
    :param expr_2: 2D numpy ndarray, or a CVXPY Variable with expr_2.ndim == 2
    :return: The tensor product of two CVXPY expressions.
    """
    expr = np.kron(expr_1, expr_2)
    num_rows = expr.shape[0]
    rows = [cvxpy.hstack(expr[i, :]) for i in range(num_rows)]
    full_expr = cvxpy.vstack(rows)
    return full_expr


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

        # TODO: Something needs to be generalized here for sys_list.
        dim = int(np.log2(self.dim_x))
        self.sys_list = list(range(1, dim, 2))

        self.dim = int(np.log2(self.dim_x))

    def solve(self) -> Union[float, Tuple[float, List[cvxpy.Variable]]]:
        """Solve either the primal or dual problem for the PPT SDP."""

        # Return the optimal value and the optimal measurements.
        if self._return_optimal_meas:
            return self.primal_problem()

        # Otherwise, it is often less computationally intensive to just solve the dual problem.
        return self.dual_problem()
        #return None

    def primal_problem(self):
        r"""Compute optimal value of the symmetric extension hierarchy SDP."""
        constraints = []

        dim = int(np.log2(self.dim_x))
        dim_list = [dim] * (self._level + 1)
        # The `sys_list` variable contains the numbering pertaining to the symmetrically extended
        # spaces.
        sys_list = list(range(3, 3 + self._level - 1))
        sym = symmetric_projection(dim, self._level)

        dim_xy = self.dim_x
        dim_xyy = np.prod(dim_list)

        meas = [cvxpy.Variable((dim_xy, dim_xy), PSD=True) for i, _ in enumerate(self._states)]
        x_var = [cvxpy.Variable((dim_xyy, dim_xyy), PSD=True) for i, _ in enumerate(self._states)]
        obj_func = [self._probs[i] * cvxpy.trace(self._states[i].conj().T @ meas[i]) for i, _ in enumerate(self._states)]

        for k, _ in enumerate(self._states):
            constraints.append(
                partial_trace(x_var[k], sys_list, dim_list) == meas[k]
            )
            constraints.append(
                np.kron(np.identity(dim), sym)
                @ x_var[k]
                @ np.kron(np.identity(dim), sym)
                == x_var[k]
            )
            constraints.append(partial_transpose(x_var[k], 1, dim_list) >> 0)
            for sys in range(self._level - 1):
                constraints.append(
                    partial_transpose(x_var[k], sys + 3, dim_list) >> 0
                )
        constraints.append(sum(meas) == np.identity(dim_xy))

        objective = cvxpy.Maximize(sum(obj_func))
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
        objective = cvxpy.Minimize(cvxpy.real(cvxpy.trace(h_var)))
        for k, _ in enumerate(self._states):
            Q.append(cvxpy.Variable((dim_xy, dim_xy), hermitian=True))
            R.append(cvxpy.Variable((dim_xyy, dim_xyy), hermitian=True))
            S.append(cvxpy.Variable((dim_xyy, dim_xyy), PSD=True))
            Z.append(cvxpy.Variable((dim_xyy, dim_xyy), PSD=True))

            constraints.append(h_var - Q[k] >> self._probs[k] * self._states[k])

            constraints.append((cvx_kron(Q[k], np.identity(dim_yp)) +
                               (np.kron(np.identity(dim), sym) @ R[k] @ np.kron(np.identity(dim), sym)) -
                               R[k] -
                               partial_transpose(S[k], 1, dim_list) -
                               partial_transpose(Z[k], 2, dim_list)) >> 0)

            constraints.append(R[k] >> 0)

        problem = cvxpy.Problem(objective, constraints)
        sol_default = problem.solve(solver=cvxpy.SCS)

        return sol_default
