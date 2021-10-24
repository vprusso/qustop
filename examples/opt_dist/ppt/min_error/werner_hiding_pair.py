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
from toqito.perms import swap_operator

from qustop import Ensemble, OptDist, State

dim = 2
sigma_0 = (
    np.kron(np.identity(dim), np.identity(dim)) + swap_operator(dim)
) / (dim * (dim + 1))
sigma_1 = (
    np.kron(np.identity(dim), np.identity(dim)) - swap_operator(dim)
) / (dim * (dim - 1))

ensemble = Ensemble([State(sigma_0, [dim, dim]), State(sigma_1, [dim, dim])])

expected_val = 1 / 2 + 1 / (dim + 1)

res = OptDist(ensemble, "ppt", "min-error")
res.solve()

# opt_ppt \approx 0.8333333333668715
print(res.value)
# Closed-form expression is : 1/2 + 1/(dim+1) = 0.8333333333333333
print(expected_val)
