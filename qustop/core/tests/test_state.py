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
from qustop import State

e_0, e_1 = basis(2, 0), basis(2, 1)


def test_state_shape():
    """Test `shape` property of `State` object."""
    # Ensure the state shape is properly set when vector is provided as argument.
    bell_vec = 1 / np.sqrt(2) * (np.kron(e_0, e_0) + np.kron(e_1, e_1))
    state_vec = State(bell_vec, [2, 2])

    assert state_vec.shape == (4, 4)

    # Ensure the state shape is properly set when density matrix is provided as argument.
    bell_state = bell_vec * bell_vec.conj().T
    state_matrix = State(bell_state, [2, 2])

    assert state_matrix.shape == (4, 4)


def test_state_purity():
    """Ensure pure states are flagged as pure and non-pure are flagged as mixed states."""
    # Define single-qubit |0> and |1> basis states.
    e_p, e_m = (
        1 / np.sqrt(2) * np.array([[1, 1]]).T,
        1 / np.sqrt(2) * np.array([[1, -1]]).T,
    )

    # Define v_1 = sqrt(3/4)|+> + sqrt(1/4)|->
    v_1 = np.sqrt(3 / 4) * e_p + np.sqrt(1 / 4) * e_m
    # Define v_2 = sqrt(1/4)|+> - sqrt(3/4)|->
    v_2 = np.sqrt(1 / 4) * e_p - np.sqrt(3 / 4) * e_m

    dims = [2]
    rho_1 = State(v_1, dims)
    rho_2 = State(v_2, dims)

    assert rho_1.is_pure is True
    assert rho_2.is_pure is True

    # sigma_1 = 3/4 |+><+| + 1/4|-><-|
    sigma_1 = 3 / 4 * (e_p * e_p.conj().T) + 1 / 4 * (e_m * e_m.conj().T)

    # sigma_2 = 1/4 |+><+| - 3/4|-><-|
    sigma_2 = 3 / 4 * (e_p * e_p.conj().T) + 1 / 4 * (e_m * e_m.conj().T)

    dims = [2]

    sigma_1 = State(sigma_1, dims)
    sigma_2 = State(sigma_2, dims)

    assert sigma_1.is_pure is False
    assert sigma_2.is_pure is False


def test_state_equality():
    pass


def test_invalid_state():
    """Invalid input state provided as non-density operator."""
    with np.testing.assert_raises(ValueError):
        State(
            np.array(
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
            ),
            dims=[2, 2],
        )


def test_invalid_swap_vector_length():
    """Too many entries in swap vector."""
    with np.testing.assert_raises(ValueError):
        dims = [2, 2]
        psi = State(bell(0), dims)
        psi.swap([1, 2, 3])


def test_invalid_swap_vector_out_of_range():
    """Swap vector contains values out of range."""
    with np.testing.assert_raises(ValueError):
        dims = [2, 2]
        psi = State(bell(0), dims)
        psi.swap([6, 7])
