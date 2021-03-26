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

import numpy as np

from qustop.core import Ensemble
from qustop.opt_dist import Positive, PPT


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

        self.return_optimal_meas = kwargs.get("return_optimal_meas", False)
        self.solver = kwargs.get("solver", "CVXOPT")
        self.verbose = kwargs.get("verbose", False)
        self.abstols = kwargs.get("abstol", 1e-5)

        self._optimal_value = None
        self._optimal_measurements = []

    @property
    def value(self) -> float:
        return self._optimal_value

    @property
    def measurements(self) -> list[np.ndarray]:
        return self._optimal_measurements

    @staticmethod
    def convert_measurements(measurements) -> list[np.ndarray]:
        return [measurements[i].value for i in range(len(measurements))]

    def solve(self):
        if self.dist_measurement == "ppt":
            opt = PPT(
                self.ensemble,
                self.dist_method,
                self.return_optimal_meas,
                self.solver,
                self.verbose,
                self.abstols,
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
                self.abstols,
            )
            if self.return_optimal_meas:
                self._optimal_value, self._optimal_measurements = opt.solve()
            else:
                self._optimal_value = opt.solve()
