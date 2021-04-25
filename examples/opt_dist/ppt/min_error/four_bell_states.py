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
from toqito.states import bell
from qustop import Ensemble, State, OptDist


# Construct the corresponding density matrices of the Bell states.
dims = [2, 2]
states = [
    State(bell(0) * bell(0).conj().T, dims),
    State(bell(1) * bell(1).conj().T, dims),
    State(bell(2) * bell(2).conj().T, dims),
    State(bell(3) * bell(3).conj().T, dims),
]
ensemble = Ensemble(states=states, probs=[1 / 4, 1 / 4, 1 / 4, 1 / 4])
sd = OptDist(
    ensemble=ensemble, dist_measurement="ppt", dist_method="min-error"
)
sd.solve()
print(sd.value)
