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

from qustop.core import Ensemble
from toqito.channels import partial_trace, partial_transpose
from toqito.perms import symmetric_projection


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
    ) -> None:
        """Computes either the primal or dual problem of the PPT SDP.

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

        self._dims = self._ensemble.dims
        self._sys = [i for i in self._ensemble.systems if i % 2 != 0]

        self.dim_x, self.dim_y = self._ensemble[0].shape
        self.dim_list = self._ensemble[0].dims

        self._level = 2

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
        #return self.dual_problem()
        return None

    def primal_problem(self):
        r"""Compute optimal value of the symmetric extension hierarchy SDP."""
        obj_func = []
        meas = []
        x_var = []
        constraints = []

        dim = int(np.log2(self.dim_x))
        dim_list = [dim] * (self._level + 1)
        # The `sys_list` variable contains the numbering pertaining to the symmetrically extended
        # spaces.
        sys_list = list(range(3, 3 + self._level - 1))
        sym = symmetric_projection(dim, self._level)

        dim_xy = self.dim_x
        dim_xyy = np.prod(dim_list)
        for k, _ in enumerate(self._states):
            meas.append(cvxpy.Variable((dim_xy, dim_xy), PSD=True))
            x_var.append(cvxpy.Variable((dim_xyy, dim_xyy), PSD=True))
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

            obj_func.append(
                self._probs[k] * cvxpy.trace(self._states[k].conj().T @ meas[k])
            )

        constraints.append(sum(meas) == np.identity(dim_xy))

        objective = cvxpy.Maximize(sum(obj_func))
        problem = cvxpy.Problem(objective, constraints)
        sol_default = problem.solve(solver="SCS")

        return sol_default, meas
