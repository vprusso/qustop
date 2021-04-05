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
e_p, e_m = 1/np.sqrt(2) * np.array([[1, 1]]).T, 1/np.sqrt(2) * np.array([[1, -1]]).T

# Define v_1 = sqrt(3/4)|+> + sqrt(1/4)|->
v_1 = np.sqrt(3/4) * e_p + np.sqrt(1/4) * e_m
# Define v_2 = sqrt(1/4)|+> - sqrt(3/4)|->
v_2 = np.sqrt(1/4) * e_p - np.sqrt(3/4) * e_m

dims = [2]
rho_1 = State(v_1, dims)
rho_2 = State(v_2, dims)

# Verify that the states `rho_1` and `rho_2` are pure:
print(f"Is rho_1 pure: {rho_1.is_pure}")
print(f"Is rho_2 pure: {rho_2.is_pure}")

ensemble = Ensemble([rho_1, rho_2])

sd = OptDist(ensemble=ensemble,
             dist_measurement="pos",
             dist_method="min-error")

# 0.999999998061445
sd.solve()
print(sd.value)

# The closed-form equation yields: 0.9999999999999999 = 1
print(1/2 + 1/4 * np.linalg.norm(rho_1.value - rho_2.value, ord="nuc"))
