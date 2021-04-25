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

from qustop.core import Ensemble
from qustop.opt_dist import Positive, PPT, Separable


class OptDist:
    def __init__(
        self,
        ensemble: Ensemble,
        dist_measurement: str,
        dist_method: str,
        **kwargs: Any,
    ):
        self.ensemble = ensemble
        self.dist_measurement = dist_measurement
        self.dist_method = dist_method

        self.return_optimal_meas = kwargs.get("return_optimal_meas", True)
        self.solver = kwargs.get("solver", "SCS")
        self.verbose = kwargs.get("verbose", False)
        self.eps = kwargs.get("eps", 1e-8)
        self.level = kwargs.get("level", 2)

        self._optimal_value = None
        self._optimal_measurements: List[np.ndarray] = []

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

    def pre_optimize(self):
        """For certain special cases of ensembles, we do not need to solve any SDP and can obtain
        a result analytically.
        """
        # If there is only one state in the ensemble, the optimal value is trivially equal to
        # one. The optimal measurement is simply the identity matrix.
        if len(self.ensemble) == 1:
            if self.return_optimal_meas:
                return 1.0, np.identity(self.ensemble.shape[0])
            else:
                return 1.0, []

        # There is a closed-form expression for the distinguishability of two density matrices.
        if len(self.ensemble) == 2:
            # TODO:
            pass

        # # If the states are mutually orthogonal, it is possible to perfectly distinguish.
        if self.ensemble.is_mutually_orthogonal:
            # TODO
            pass

        return None, []

    def solve(self) -> None:
        """Depending on the measurement method selected, solve the appropriate optimization problem.

        Raises:
            ValueError:
                * If the `dist_measurement` argument is not supported.
        """

        # If `value` and `meas` were solved in the pre-optimization step, there's no need to
        # perform any further calculations.
        value, meas = self.pre_optimize()
        if value is not None and self.return_optimal_meas is False:
            self._optimal_value = value
            return
        elif value is not None and self.return_optimal_meas:
            self._optimal_value, self._optimal_measurements = value, meas
            return

        if self.dist_measurement == "ppt":
            opt = PPT(
                self.ensemble,
                self.dist_method,
                self.return_optimal_meas,
                self.solver,
                self.verbose,
                self.eps,
            )
            if self.return_optimal_meas:
                self._optimal_value, self._optimal_measurements = opt.solve()
            else:
                self._optimal_value = opt.solve()

        elif self.dist_measurement == "pos":
            opt = Positive(
                self.ensemble,
                self.dist_method,
                self.return_optimal_meas,
                self.solver,
                self.verbose,
                self.eps,
            )
            if self.return_optimal_meas:
                self._optimal_value, self._optimal_measurements = opt.solve()
            else:
                self._optimal_value = opt.solve()
        elif self.dist_measurement == "sep":
            opt = Separable(
                self.ensemble,
                self.dist_method,
                self.return_optimal_meas,
                self.solver,
                self.verbose,
                self.eps,
                self.level,
            )
            if self.return_optimal_meas:
                self._optimal_value, self._optimal_measurements = opt.solve()
            else:
                self._optimal_value = opt.solve()
        else:
            raise ValueError(
                f"Measurement type {self.dist_method} not supported."
            )
