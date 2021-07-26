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
ensemble = Ensemble([
    State(np.sqrt(3 / 4) * e_p + np.sqrt(1 / 4) * e_m, dims),
    State(np.sqrt(1 / 4) * e_p + np.sqrt(3 / 4) * e_m, dims),
    State(np.sqrt(1 / 2) * e_p + np.sqrt(1 / 2) * e_m, dims)
])

res = OptDist(ensemble, "pos", "unambiguous")

# Probability of distinguishing unambiguously is zero.
res.solve()
print(res.value)
