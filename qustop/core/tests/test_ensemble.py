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

from qustop import State, Ensemble


# Define single-qubit |0> and |1> basis states.
e_0, e_1 = np.array([[1, 0]]).T, np.array([[0, 1]]).T

# Define two-qubit |00>, |01>, |10>, and |11> basis states.
e_00, e_01 = np.kron(e_0, e_0), np.kron(e_0, e_1)
e_10, e_11 = np.kron(e_1, e_0), np.kron(e_1, e_1)

# Define the Bell state vectors.
b_0 = 1 / np.sqrt(2) * (e_00 + e_11)
b_1 = 1 / np.sqrt(2) * (e_00 - e_11)
b_2 = 1 / np.sqrt(2) * (e_01 + e_10)
b_3 = 1 / np.sqrt(2) * (e_01 - e_10)


def test_invalid_ensemble():
    """Invalid number of states in ensemble."""
    with np.testing.assert_raises(ValueError):
        Ensemble([])


def test_invalid_state_distinguishability_probs():
    """Invalid probability vector for state distinguishability."""
    with np.testing.assert_raises(ValueError):
        rho1 = b_0 * b_0.conj().T
        rho2 = b_1 * b_1.conj().T
        dims = [2, 2]
        Ensemble([State(rho1, dims), State(rho2, dims)], [1, 2, 3])
