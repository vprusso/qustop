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
from toqito.states import bell

from qustop import Ensemble, OptDist, State


def test_invalid_ensemble():
    """Invalid distinguishability measurement specified."""
    with np.testing.assert_raises(ValueError):
        dims = [2, 2]
        rho_0 = State(bell(0) * bell(0).conj().T, dims)
        rho_1 = State(bell(1) * bell(1).conj().T, dims)
        rho_2 = State(bell(2) * bell(2).conj().T, dims)
        ensemble = Ensemble([rho_0, rho_1, rho_2])

        res = OptDist(
            ensemble=ensemble,
            dist_measurement="invalid",
            dist_method="min-error",
            return_optimal_meas=True,
        )
        res.solve()
