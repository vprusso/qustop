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

from qustop import State, Ensemble, OptDist

dims = [2, 2]
rho_1 = State(bell(0), dims)
rho_2 = State(bell(1), dims)
rho_3 = State(bell(2), dims)
rho_4 = State(bell(3), dims)

ensemble = Ensemble([rho_1, rho_2, rho_3, rho_4])

print(f"Are states mutually orthogonal: {ensemble.is_mutually_orthogonal}")

sd = OptDist(ensemble=ensemble,
             dist_measurement="pos",
             dist_method="min-error")

# Mutually orthogonal states are optimally distinguishable--giving an optimal value of one.
sd.solve()
print(sd.value)
