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

from qustop import Ensemble, OptDist, State

dims = [2, 2]
ensemble = Ensemble(
    [
        State(bell(0), dims),
        State(bell(1), dims),
        State(bell(2), dims),
        State(bell(3), dims),
    ]
)

print(f"Are states mutually orthogonal: {ensemble.is_mutually_orthogonal}")

sd = OptDist(
    ensemble=ensemble,
    dist_measurement="pos",
    dist_method="unambiguous",
    return_optimal_meas=True,
)

# Mutually orthogonal states are optimally distinguishable;
# giving an optimal value of one.
sd.solve()
print(sd.value)
