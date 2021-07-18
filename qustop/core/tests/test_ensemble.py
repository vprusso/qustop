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
from qustop import State, Ensemble


def test_ensemble_str_repr():
    """Test overloaded __str__ method for `Ensemble`."""
    dims = [2, 2]
    ensemble = Ensemble([
        State(bell(0), dims),
    ])
    assert isinstance(str(ensemble), str) is True


def test_ensemble_states():
    """Check to ensure that `Ensemble` returns list of `State` objects."""
    dims = [2, 2]
    bell_ensemble = Ensemble([
        State(bell(0), dims),
        State(bell(1), dims),
        State(bell(2), dims),
        State(bell(3), dims)
    ])
    assert len(bell_ensemble.states) == 4
    assert isinstance(bell_ensemble.states[0], State) is True


def test_ensemble_systems():
    """Check to ensure that `Ensemble` returns list of systems."""
    sys_1_dims = [2]
    sys_1_ensemble = Ensemble([
        State(basis(2, 0), sys_1_dims),
        State(basis(2, 1), sys_1_dims)
    ])
    assert sys_1_ensemble.systems == [1]

    sys_2_dims = [2, 2]
    sys_2_ensemble = Ensemble([
        State(bell(0), sys_2_dims),
        State(bell(1), sys_2_dims),
        State(bell(2), sys_2_dims),
        State(bell(3), sys_2_dims)
    ])
    assert sys_2_ensemble.systems == [1, 2]


def test_is_linearly_independent():
    """Check if the states are linearly independent or not."""
    dims = [2]
    li_states = [
        State(basis(2, 0), dims),
        State(basis(2, 1), dims)
    ]
    li_ensemble = Ensemble(li_states)
    assert li_ensemble.is_linearly_independent is True

    ld_states = [
        State(basis(2, 0), dims),
        State(basis(2, 1), dims),
        State(basis(2, 0), dims)
    ]
    ld_ensemble = Ensemble(ld_states)
    assert ld_ensemble.is_linearly_independent is False


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
