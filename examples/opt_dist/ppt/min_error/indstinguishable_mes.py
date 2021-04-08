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

from qustop import State, Ensemble, OptDist


# Define the maximally entangled states from arXiv1107.3224
dims = [2, 2, 2, 2]
rho_0 = np.kron(bell(0), bell(0)) * np.kron(bell(0), bell(0)).conj().T
rho_1 = np.kron(bell(2), bell(1)) * np.kron(bell(2), bell(1)).conj().T
rho_2 = np.kron(bell(3), bell(1)) * np.kron(bell(3), bell(1)).conj().T
rho_3 = np.kron(bell(1), bell(1)) * np.kron(bell(1), bell(1)).conj().T

ensemble = Ensemble([
    State(rho_0, dims), State(rho_1, dims),
    State(rho_2, dims), State(rho_3, dims)
])

sd = OptDist(ensemble=ensemble, 
             dist_measurement="ppt",
             dist_method="min-error")
sd.solve()

# The min-error probability of distinguishing via PPT
# is equal to 7/8.
print(sd.value)
