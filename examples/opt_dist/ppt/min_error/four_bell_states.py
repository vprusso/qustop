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
ensemble = Ensemble(
    [
        State(bell(0), dims),
        State(bell(1), dims),
        State(bell(2), dims),
        State(bell(3), dims),
    ],
    [1 / 4, 1 / 4, 1 / 4, 1 / 4]
)
res = OptDist(ensemble, "ppt", "min-error")
res.solve()

# 0.5000000000530641
print(res.value)
