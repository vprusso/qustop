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

# Define v_1 = sqrt(3)/2|00> + 1/2|11>
v_1 = np.sqrt(3)/2 * np.kron(e_0, e_0) + 1/2 * np.kron(e_1, e_1)
# Define v_2 = 1/2|00> + sqrt(3)/2|11>
v_2 = 1/2 * np.kron(e_0, e_0) + np.sqrt(3)/2 * np.kron(e_1, e_1)

dims = [2, 2]
rho_1 = State(v_1, [2, 2])
rho_2 = State(v_2, [2, 2])

ensemble = Ensemble([rho_1, rho_2])

sd = OptDist(ensemble=ensemble,
             dist_measurement="pos",
             dist_method="min-error")

# 0.7499999977563431
sd.solve()
print(sd.value)

# The closed-form equation yields: 3/4 = 0.75
print(1/2 + 1/4 * np.linalg.norm(rho_1.value - rho_2.value, ord="nuc"))
