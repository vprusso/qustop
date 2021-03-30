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
from qustop import State, Ensemble


def test_is_mutually_orthogonal():
    """Check if the states in the ensemble are mutually orthogonal or not."""
    dims = [2, 2]
    orthogonal_states = [
        State(bell(0), dims),
        State(bell(1), dims),
        State(bell(2), dims),
        State(bell(3), dims),
    ]
    orthogonal_ensemble = Ensemble(orthogonal_states)

    assert orthogonal_ensemble.is_mutually_orthogonal is True

    nonorthogonal_states = [
        State(bell(0), dims),
        State(bell(0), dims),
        State(bell(2), dims),
        State(bell(3), dims),
    ]
    nonorthogonal_ensemble = Ensemble(nonorthogonal_states)

    assert nonorthogonal_ensemble.is_mutually_orthogonal is False


def test_invalid_ensemble():
    """Invalid number of states in ensemble."""
    with np.testing.assert_raises(ValueError):
        Ensemble([])

    with np.testing.assert_raises(TypeError):
        states = [np.identity(4), np.identity(4)]
        Ensemble(states)


def test_invalid_state_distinguishability_probs():
    """Invalid probability vector for state distinguishability."""
    with np.testing.assert_raises(ValueError):
        rho1 = bell(0) * bell(0).conj().T
        rho2 = bell(1) * bell(1).conj().T
        dims = [2, 2]
        Ensemble([State(rho1, dims), State(rho2, dims)], [1, 2, 3])
