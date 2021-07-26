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

from toqito.states import basis, bell
from qustop import State, Ensemble, OptDist


e_0, e_1 = basis(2, 0), basis(2, 1)

eps = 0.5
tau = np.sqrt((1 + eps) / 2) * np.kron(e_0, e_0) + np.sqrt(
    (1 - eps) / 2
) * np.kron(e_1, e_1)

dims = [2, 2, 2, 2]
ensemble = Ensemble(
    [
        State(np.kron(bell(0), tau), dims),
        State(np.kron(bell(1), tau), dims),
        State(np.kron(bell(2), tau), dims),
        State(np.kron(bell(3), tau), dims)
    ],
    [1 / 4, 1 / 4, 1 / 4, 1 / 4]
)

ppt_res = OptDist(ensemble, "ppt", "min-error")
ppt_res.solve()

sep_res = OptDist(ensemble, "sep", "min-error")
sep_res.solve()

eq = 1 / 2 * (1 + np.sqrt(1 - eps ** 2))

# 0.9330127018922193
print(eq)
# 0.9330127016540999
print(ppt_res.value)
# 0.9330127016540999
print(sep_res.value)
