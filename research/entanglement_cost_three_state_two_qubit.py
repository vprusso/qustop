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

from qustop import Ensemble, OptDist, State

# Define the |0> and |1> basis states:
e_0, e_1 = basis(2, 0), basis(2, 1)

# Define constants "n" and "epsilon":
n = 0.0
eps = 0.75

# Parameters alpha and beta are defined in terms of "n"
alpha, beta = np.sqrt((1 + n) / 2), np.sqrt((1 - n) / 2)

# Define the two-qubit ensemble states:
psi_0 = alpha * np.kron(e_0, e_0) + beta * np.kron(e_1, e_1)
psi_1 = beta * np.kron(e_0, e_0) - alpha * np.kron(e_1, e_1)
psi_2 = alpha * np.kron(e_0, e_1) + beta * np.kron(e_1, e_0)

# Define the resource state:
tau_state = np.sqrt((1 + eps) / 2) * np.kron(e_0, e_0) + np.sqrt(
    (1 - eps) / 2
) * np.kron(e_1, e_1)
tau = tau_state * tau_state.conj().T

# Create the ensemble to distinguish:
dims = [2, 2, 2, 2]
rho_0 = State(np.kron(psi_0 * psi_0.conj().T, tau), dims)
rho_1 = State(np.kron(psi_1 * psi_1.conj().T, tau), dims)
rho_2 = State(np.kron(psi_2 * psi_2.conj().T, tau), dims)
ensemble = Ensemble([rho_0, rho_1, rho_2])

# Determine the optimal value of distinguishing the ensemble via PPT
# measurements:
ppt_res = OptDist(ensemble, "ppt", "min-error")
ppt_res.solve()

# Print value of "n", "eps", and the optimal value of distinguishing via PPT
# measurements:
print(f"For n = {n} and eps={eps}, the PPT value is {ppt_res.value}")
sep_eq = 1 / 3 * (2 + np.sqrt(1 - eps ** 2))
eq = 1 / 3 * (2 + np.sqrt(1.219 - eps ** 2))
print(f"EQ:{eq}")
print(f"SEP_EQ: {sep_eq}")
