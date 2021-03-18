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

from qustop.core.ensemble import Ensemble
from qustop.opt_dist.ppt import PPT


class OptDist:
    def __init__(self, ensemble, measurement, error, **kwargs):
        self.ensemble = ensemble
        self.measurement = measurement
        self.error = error

        self._optimal_value = None
        self._optimal_measurements = None

    @property
    def value(self):
        return self._optimal_value

    @property
    def measurements(self):
        return self._optimal_measurements
    
    def solve(self):
        if self.measurement == "ppt":
            opt = PPT(self.ensemble, self.error)
            self._optimal_value, self._optimal_measurements = opt.solve()
            
