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

from qustop.core import Ensemble
from qustop.opt_dist import Positive, PPT, Separable


class OptDist:
    """Quantum state distinguishability via positive, PPT, or separable measurements."""
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
        """Depending on the measurement method selected, solve the appropriate optimization problem.

        Raises:
            ValueError:
                * If the `dist_measurement` argument is not supported.
        """
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
