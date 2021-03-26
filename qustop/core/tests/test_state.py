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
from qustop import State

e_0, e_1 = basis(2, 0), basis(2, 1)


def test_state_shape():

    # Ensure
    bell_vec = 1 / np.sqrt(2) * (np.kron(e_0, e_0) + np.kron(e_1, e_1))
    state_vec = State(bell_vec, [2, 2])

    assert state_vec.shape == (4, 4)


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
