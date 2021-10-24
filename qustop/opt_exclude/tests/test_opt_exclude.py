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

from qustop import Ensemble, OptExclude, State


def test_conclusive_state_exclusion_one_state():
    """Conclusive state exclusion for single state."""
    dims = [2, 2]
    ensemble = Ensemble([State(bell(0) * bell(0).conj().T, dims)])

    res = OptExclude(
        ensemble=ensemble,
        dist_method="min-error",
        return_optimal_meas=True,
    )
    res.solve()
    np.testing.assert_equal(np.isclose(res.value, 1), True)


def test_conclusive_state_exclusion_one_state_vec():
    """Conclusive state exclusion for single vector state."""
    dims = [2, 2]
    ensemble = Ensemble([State(bell(0), dims)])

    res = OptExclude(
        ensemble=ensemble,
        dist_method="min-error",
        return_optimal_meas=True,
    )
    res.solve()
    np.testing.assert_equal(np.isclose(res.value, 1), True)


def test_unambiguous_state_exclusion_one_state_dual():
    """Unambiguous state exclusion for single state (dual problem)."""
    dims = [2, 2]
    ensemble = Ensemble([State(bell(0) * bell(0).conj().T, dims)])

    res = OptExclude(
        ensemble=ensemble,
        solver="cvxopt",
        dist_method="unambiguous",
        return_optimal_meas=False,
    )
    res.solve()
    np.testing.assert_equal(np.isclose(res.value, 1), True)


def test_unambiguous_state_exclusion_two_states_dual():
    """Unambiguous state exclusion for two states (dual problem)."""
    dims = [2, 2]
    ensemble = Ensemble([State(bell(0), dims), State(bell(1), dims)])

    res = OptExclude(
        ensemble=ensemble,
        solver="cvxopt",
        dist_method="unambiguous",
        return_optimal_meas=False,
    )
    res.solve()
    np.testing.assert_equal(np.isclose(res.value, 0), True)