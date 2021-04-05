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
import pytest

from toqito.states import bell
from qustop import Ensemble, State, OptDist


def test_symmetric_extension_hierarchy_four_bell_density_matrices():
    """Symmetric extension hierarchy for four Bell density matrices."""
    dims = [2, 2]
    states = [
        State(bell(0) * bell(0).conj().T, dims),
        State(bell(1) * bell(1).conj().T, dims),
        State(bell(2) * bell(2).conj().T, dims),
        State(bell(3) * bell(3).conj().T, dims),
    ]
    ensemble = Ensemble(states)

    # Test primal and dual problems on minimum-error method:
    min_error_primal_res = OptDist(
        ensemble=ensemble,
        dist_measurement="sep",
        dist_method="min-error",
        return_optimal_meas=True,
        level=2,
    )
    min_error_primal_res.solve()
    np.testing.assert_equal(
        np.isclose(min_error_primal_res.value, 1 / 2), True
    )

    min_error_dual_res = OptDist(
        ensemble=ensemble,
        dist_measurement="sep",
        dist_method="min-error",
        return_optimal_meas=False,
        level=2,
    )
    min_error_dual_res.solve()
    np.testing.assert_equal(np.isclose(min_error_dual_res.value, 1 / 2), True)


def test_symmetric_extension_hierarchy_four_bell_states():
    """Symmetric extension hierarchy for four Bell states."""
    dims = [2, 2]
    states = [
        State(bell(0), dims),
        State(bell(1), dims),
        State(bell(2), dims),
        State(bell(3), dims),
    ]
    ensemble = Ensemble(states)

    # Test primal and dual problems on minimum-error method:
    min_error_primal_res = OptDist(
        ensemble=ensemble,
        dist_measurement="sep",
        dist_method="min-error",
        return_optimal_meas=True,
        level=2,
    )
    min_error_primal_res.solve()
    np.testing.assert_equal(
        np.isclose(min_error_primal_res.value, 1 / 2), True
    )

    min_error_dual_res = OptDist(
        ensemble=ensemble,
        dist_measurement="sep",
        dist_method="min-error",
        return_optimal_meas=False,
        level=2,
    )
    min_error_dual_res.solve()
    np.testing.assert_equal(np.isclose(min_error_dual_res.value, 1 / 2), True)


def test_symmetric_extension_hierarchy_four_bell_with_resource_state_lvl_1():
    """Level 1 of hierarchy for four Bell states and resource state."""
    e_0, e_1 = np.array([[1, 0]]).T, np.array([[0, 1]]).T
    e_00, e_11 = np.kron(e_0, e_0), np.kron(e_1, e_1)

    eps = 1 / 2
    eps_state = np.sqrt((1 + eps) / 2) * e_00 + np.sqrt((1 - eps) / 2) * e_11
    eps_dm = eps_state * eps_state.conj().T

    dims = [2, 2, 2, 2]
    states = [
        State(np.kron(bell(0) * bell(0).conj().T, eps_dm), dims),
        State(np.kron(bell(1) * bell(1).conj().T, eps_dm), dims),
        State(np.kron(bell(2) * bell(2).conj().T, eps_dm), dims),
        State(np.kron(bell(3) * bell(3).conj().T, eps_dm), dims),
    ]
    states[0].swap([2, 3])
    states[1].swap([2, 3])
    states[2].swap([2, 3])
    states[3].swap([2, 3])
    ensemble = Ensemble(states)

    # Known closed-form result:
    exp_res = 1 / 2 * (1 + np.sqrt(1 - eps ** 2))

    # Level 1 of the hierarchy should be identical to the known PPT value
    # for this case.
    min_error_primal_res = OptDist(
        ensemble=ensemble,
        dist_measurement="sep",
        dist_method="min-error",
        return_optimal_meas=True,
        level=1,
    )
    min_error_primal_res.solve()
    np.testing.assert_equal(
        np.isclose(min_error_primal_res.value, exp_res), True
    )

    min_error_dual_res = OptDist(
        ensemble=ensemble,
        dist_measurement="sep",
        dist_method="min-error",
        return_optimal_meas=False,
        level=1,
    )
    min_error_dual_res.solve()
    np.testing.assert_equal(
        np.isclose(min_error_dual_res.value, exp_res), True
    )


@pytest.mark.skip(
    reason="This test takes too much time."
)  # pylint: disable=not-callable
def test_symmetric_extension_hierarchy_four_bell_with_resource_state():
    """Symmetric extension hierarchy for four Bell states and resource state."""
    e_0, e_1 = np.array([[1, 0]]).T, np.array([[0, 1]]).T
    e_00, e_11 = np.kron(e_0, e_0), np.kron(e_1, e_1)

    eps = 1 / 2
    eps_state = np.sqrt((1 + eps) / 2) * e_00 + np.sqrt((1 - eps) / 2) * e_11
    eps_dm = eps_state * eps_state.conj().T

    dims = [2, 2, 2, 2]
    states = [
        State(np.kron(bell(0) * bell(0).conj().T, eps_dm), dims),
        State(np.kron(bell(1) * bell(1).conj().T, eps_dm), dims),
        State(np.kron(bell(2) * bell(2).conj().T, eps_dm), dims),
        State(np.kron(bell(3) * bell(3).conj().T, eps_dm), dims),
    ]

    # Ensure we are checking the correct partition of the states.
    states[0].swap([2, 3])
    states[1].swap([2, 3])
    states[2].swap([2, 3])
    states[3].swap([2, 3])
    ensemble = Ensemble(states)

    exp_res = 1 / 2 * (1 + np.sqrt(1 - eps ** 2))

    min_error_primal_res = OptDist(
        ensemble=ensemble,
        dist_measurement="sep",
        dist_method="min-error",
        return_optimal_meas=True,
        level=2,
    )
    min_error_primal_res.solve()
    np.testing.assert_equal(
        np.isclose(min_error_primal_res.value, exp_res), True
    )

    min_error_dual_res = OptDist(
        ensemble=ensemble,
        dist_measurement="sep",
        dist_method="min-error",
        return_optimal_meas=False,
        level=2,
    )
    min_error_dual_res.solve()
    np.testing.assert_equal(
        np.isclose(min_error_dual_res.value, exp_res), True
    )


@pytest.mark.skip(
    reason="This test takes too much time."
)  # pylint: disable=not-callable
def test_symmetric_extension_hierarchy_three_bell_with_resource_state():
    """Symmetric extension hierarchy for three Bell and resource state."""
    e_0, e_1 = np.array([[1, 0]]).T, np.array([[0, 1]]).T
    e_00, e_11 = np.kron(e_0, e_0), np.kron(e_1, e_1)

    eps = 1 / 2
    eps_state = np.sqrt((1 + eps) / 2) * e_00 + np.sqrt((1 - eps) / 2) * e_11
    eps_dm = eps_state * eps_state.conj().T

    dims = [2, 2, 2, 2]
    states = [
        State(np.kron(bell(0) * bell(0).conj().T, eps_dm), dims),
        State(np.kron(bell(1) * bell(1).conj().T, eps_dm), dims),
        State(np.kron(bell(2) * bell(2).conj().T, eps_dm), dims),
    ]

    # Ensure we are checking the correct partition of the states.
    states[0].swap([2, 3])
    states[1].swap([2, 3])
    states[2].swap([2, 3])
    states[3].swap([2, 3])
    ensemble = Ensemble(states)

    min_error_primal_res = OptDist(
        ensemble=ensemble,
        dist_measurement="sep",
        dist_method="min-error",
        return_optimal_meas=True,
        level=2,
    )
    min_error_primal_res.solve()
    np.testing.assert_equal(
        np.isclose(min_error_primal_res.value, 0.9583057), True
    )

    min_error_dual_res = OptDist(
        ensemble=ensemble,
        dist_measurement="sep",
        dist_method="min-error",
        return_optimal_meas=False,
        level=2,
    )
    min_error_dual_res.solve()
    np.testing.assert_equal(
        np.isclose(min_error_dual_res.value, 0.9583057), True
    )
