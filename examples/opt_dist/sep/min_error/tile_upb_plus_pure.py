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
from toqito.states import basis, tile

from qustop import Ensemble, OptDist, State

dims = [3, 3]
e_0, e_1, e_2 = basis(3, 0), basis(3, 1), basis(3, 2)

# Define a pure state to add to the ensemble.
psi = (
    1
    / 2
    * (
        np.kron(e_0, e_0)
        + np.kron(e_0, e_1)
        - np.kron(e_0, e_2)
        - np.kron(e_1, e_2)
    )
)

# Construct the corresponding density matrices of the Tiles UPB + pure state:
ensemble = Ensemble(
    [
        State(tile(0) * tile(0).conj().T, dims),
        State(tile(1) * tile(1).conj().T, dims),
        State(tile(2) * tile(2).conj().T, dims),
        State(tile(3) * tile(3).conj().T, dims),
        State(tile(4) * tile(4).conj().T, dims),
        State(psi * psi.conj().T, dims),
    ]
)

res = OptDist(ensemble, "sep", "min-error", level=2)
res.solve()

# 0.9860588510298623
print(res.value)
