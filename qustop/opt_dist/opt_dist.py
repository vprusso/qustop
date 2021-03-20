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

import numpy as np
from typing import List

from qustop.core import Ensemble
from qustop.opt_dist import Positive, PPT


class OptDist:
    def __init__(self, ensemble: Ensemble, dist_measurement: str, dist_method: str, fast: bool = False):
        self.ensemble = ensemble
        self.dist_measurement = dist_measurement
        self.dist_method = dist_method
        self.fast = fast

        # TODO: Specify solver and precision.

        self._optimal_value = None
        self._optimal_measurements = None

    @property
    def value(self) -> float:
        return self._optimal_value

    @property
    def measurements(self) -> List[np.ndarray]:
        return self._optimal_measurements
    
    def solve(self):
        if self.dist_measurement == "ppt":
            opt = PPT(self.ensemble, self.dist_method, self.fast)
            self._optimal_value, self._optimal_measurements = opt.solve()
        if self.dist_measurement == "positive":
            opt = Positive(self.ensemble, self.dist_method, self.fast)
            self._optimal_value, self._optimal_measurements = opt.solve()
