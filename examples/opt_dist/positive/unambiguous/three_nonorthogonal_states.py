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

from qustop import State, Ensemble, OptDist

# Define single-qubit |0> and |1> basis states.
e_0, e_1 = np.array([[1, 0]]).T, np.array([[0, 1]]).T
e_p, e_m = 1/np.sqrt(2) * np.array([[1, 1]]).T, 1/np.sqrt(2) * np.array([[1, -1]]).T

# rho_1 = 3/4 |+><+| + 1/4|-><-|
rho_1 = 3/4 * (e_p * e_p.conj().T) + 1/4 * (e_m * e_m.conj().T)

# rho_2 = 1/4 |+><+| + 3/4|-><-|
rho_2 = 1/4 * (e_p * e_p.conj().T) + 3/4 * (e_m * e_m.conj().T)

# rho_3 = 1/2 |+><+| + 1/2 |-><-|
rho_3 = 1/2 * (e_p * e_p.conj().T) + 1/2 * (e_m * e_m.conj().T)

dims = [2]
rho_1 = State(rho_1, dims)
rho_2 = State(rho_2, dims)
rho_3 = State(rho_3, dims)

ensemble = Ensemble([rho_1, rho_2, rho_3])
sd = OptDist(ensemble=ensemble,
             dist_measurement="pos",
             dist_method="unambiguous")

# Probability of distinguishing unambiguously is zero.
sd.solve()
print(sd.value)
