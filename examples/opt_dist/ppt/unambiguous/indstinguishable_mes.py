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

from qustop import Ensemble, OptDist, State

# Define the maximally entangled states from arXiv1107.3224
dims = [2, 2, 2, 2]
ensemble = Ensemble(
    [
        State(
            np.kron(bell(0), bell(0)) * np.kron(bell(0), bell(0)).conj().T,
            dims,
        ),
        State(
            np.kron(bell(2), bell(1)) * np.kron(bell(2), bell(1)).conj().T,
            dims,
        ),
        State(
            np.kron(bell(3), bell(1)) * np.kron(bell(3), bell(1)).conj().T,
            dims,
        ),
        State(
            np.kron(bell(1), bell(1)) * np.kron(bell(1), bell(1)).conj().T,
            dims,
        ),
    ]
)

# The unambiguous probability of distinguishing via PPT
# is equal to 3/4.
res = OptDist(ensemble, "ppt", "unambiguous")
res.solve()

# 0.7499999975754753
print(res.value)
