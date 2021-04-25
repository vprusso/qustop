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
from toqito.states import basis
from qustop import Ensemble, State, OptDist


e_0, e_1, e_2, e_3 = basis(4, 0), basis(4, 1), basis(4, 2), basis(4, 3)

phi_1 = np.kron(e_0, e_0)
phi_2 = np.kron(e_1, (e_0 - e_2 + e_3) / np.sqrt(3))
phi_3 = np.kron(e_2, (e_0 + e_1 - e_3) / np.sqrt(3))
phi_4 = np.kron(e_3, e_3)
phi_5 = np.kron((e_1 + e_2 + e_3), (e_0 - e_1 + e_2) / 3)
phi_6 = np.kron((e_0 - e_2 + e_3), e_2 / np.sqrt(3))
phi_7 = np.kron((e_0 + e_1 - e_3), e_1 / np.sqrt(3))
phi_8 = np.kron((e_0 - e_1 + e_2), (e_1 + e_2 + e_3) / 3)

dims = [4, 4]
states = [
    State(phi_1 * phi_1.conj().T, dims),
    State(phi_2 * phi_2.conj().T, dims),
    State(phi_3 * phi_3.conj().T, dims),
    State(phi_4 * phi_4.conj().T, dims),
    State(phi_5 * phi_5.conj().T, dims),
    State(phi_6 * phi_6.conj().T, dims),
    State(phi_7 * phi_7.conj().T, dims),
    State(phi_8 * phi_8.conj().T, dims),
]
ensemble = Ensemble(states=states)
sd = OptDist(
    ensemble,
    "sep",
    "min-error",
    return_optimal_meas=True,
    solver="SCS",
    verbose=False,
    eps=1e-6,
    level=2,
)
sd.solve()

# 0.9967296337698935
print(sd.value)
