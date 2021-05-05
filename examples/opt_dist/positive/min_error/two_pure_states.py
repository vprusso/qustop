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

# Define single-qubit |+> and |-> basis states.
e_p, e_m = (
    1 / np.sqrt(2) * np.array([[1, 1]]).T,
    1 / np.sqrt(2) * np.array([[1, -1]]).T,
)

dims = [2]
# sqrt(3/4)|+> + sqrt(1/4)|->
psi_1 = State(np.sqrt(3 / 4) * e_p + np.sqrt(1 / 4) * e_m, dims)
# sqrt(1/4)|+> - sqrt(3/4)|->
psi_2 = State(np.sqrt(1 / 4) * e_p - np.sqrt(3 / 4) * e_m, dims)

# Verify that the states `psi_1` and `psi_2` are pure:
print(f"Is psi_1 pure: {psi_1.is_pure}")
print(f"Is psi_2 pure: {psi_2.is_pure}")

ensemble = Ensemble([psi_1, psi_2])

res = OptDist(ensemble, "pos", "min-error")

# 0.999999998061445
res.solve()
print(res.value)

# The closed-form equation yields: 0.9999999999999999 = 1
print(1 / 2 + 1 / 4 * np.linalg.norm(psi_1.value - psi_2.value, ord="nuc"))
