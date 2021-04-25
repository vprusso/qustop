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

from toqito.states import basis, bell, tile
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


def test_symmetric_extension_hierarchy_ydy_states():
    """The separable value of distinguishing amongst the YDY states is equal to 3/4."""
    dims = [2, 2, 2, 2]
    rho_0 = np.kron(bell(0), bell(0)) * np.kron(bell(0), bell(0)).conj().T
    rho_1 = np.kron(bell(2), bell(1)) * np.kron(bell(2), bell(1)).conj().T
    rho_2 = np.kron(bell(3), bell(1)) * np.kron(bell(3), bell(1)).conj().T
    rho_3 = np.kron(bell(1), bell(1)) * np.kron(bell(1), bell(1)).conj().T

    ensemble = Ensemble(
        [
            State(rho_0, dims),
            State(rho_1, dims),
            State(rho_2, dims),
            State(rho_3, dims),
        ]
    )
    ensemble.swap([2, 3])

    primal_res = OptDist(ensemble,
                         "sep",
                         "min-error",
                         return_optimal_meas=True,
                         solver="SCS",
                         verbose=False,
                         eps=1e-6,
                         level=2)
    primal_res.solve()

    # The min-error probability of distinguishing via
    # separable measurements is equal to 3/4.
    np.testing.assert_equal(np.isclose(primal_res.value, 3 / 4), True)

    dual_res = OptDist(ensemble,
                       "sep",
                       "min-error",
                       return_optimal_meas=True,
                       solver="SCS",
                       verbose=False,
                       eps=1e-6,
                       level=2)
    dual_res.solve()

    # The min-error probability of distinguishing via
    # separable measurements is equal to 3/4.
    np.testing.assert_equal(np.isclose(dual_res.value, 3 / 4), True)


def test_symmetric_extension_hierarchy_four_bell_with_resource_state_lvl_1():
    """Level 1 of hierarchy for four Bell states and resource state."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
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


def test_symmetric_extension_hierarchy_tile_upb():
    """Symmetric extension hierarchy on the "Tiles" set UPB."""
    # Construct the corresponding density matrices of the Bell states.
    dims = [3, 3]
    states = [
        State(tile(0) * tile(0).conj().T, dims),
        State(tile(1) * tile(1).conj().T, dims),
        State(tile(2) * tile(2).conj().T, dims),
        State(tile(3) * tile(3).conj().T, dims),
        State(tile(4) * tile(4).conj().T, dims),
    ]
    ensemble = Ensemble(states=states)
    sd = OptDist(
        ensemble,
        "sep",
        "min-error",
        return_optimal_meas=True,
        solver="SCS",
        verbose=False,
        eps=1e-6,
        level=2,
    )
    sd.solve()
    np.testing.assert_equal(np.isclose(sd.value, 1), True)


def test_symmetric_extension_hierarchy_tile_upb_plus_pure():
    """Symmetric extension hierarchy on the "Tiles" set UPB plus extra pure state."""
    # Construct the corresponding density matrices of the Bell states.
    dims = [3, 3]
    e_0, e_1, e_2 = basis(3, 0), basis(3, 1), basis(3, 2)
    psi = (
        1
        / 2
        * (
            np.kron(e_0, e_0)
            + np.kron(e_0, e_1)
            - np.kron(e_0, e_2)
            - np.kron(e_1, e_2)
        )
    )
    states = [
        State(tile(0) * tile(0).conj().T, dims),
        State(tile(1) * tile(1).conj().T, dims),
        State(tile(2) * tile(2).conj().T, dims),
        State(tile(3) * tile(3).conj().T, dims),
        State(tile(4) * tile(4).conj().T, dims),
        State(psi * psi.conj().T, dims),
    ]
    ensemble = Ensemble(states=states)
    sd = OptDist(
        ensemble,
        "sep",
        "min-error",
        return_optimal_meas=True,
        solver="SCS",
        verbose=False,
        eps=1e-6,
        level=2,
    )
    sd.solve()

    np.testing.assert_equal(np.isclose(sd.value, 0.9860588), True)


@pytest.mark.skip(
    reason="This test takes too much time."
)  # pylint: disable=not-callable
def test_symmetric_extension_hierarchy_four_bell_with_resource_state():
    """Symmetric extension hierarchy for four Bell states and resource state."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
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
    e_0, e_1 = basis(2, 0), basis(2, 1)
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


@pytest.mark.skip(
    reason="This test takes too much time."
)  # pylint: disable=not-callable
def test_symmetric_extension_hierarchy_feng_upb():
    """Symmetric extension hierarchy for Feng unextendable product basis."""
    e_0, e_1, e_2, e_3 = basis(4, 0), basis(4, 1), basis(4, 2), basis(4, 3)

    phi_1 = np.kron(e_0, e_0)
    phi_2 = np.kron(e_1, (e_0 - e_2 + e_3) / np.sqrt(3))
    phi_3 = np.kron(e_2, (e_0 + e_1 - e_3) / np.sqrt(3))
    phi_4 = np.kron(e_3, e_3)
    phi_5 = np.kron((e_1 + e_2 + e_3), (e_0 - e_1 + e_2) / 3)
    phi_6 = np.kron((e_0 - e_2 + e_3), e_2 / np.sqrt(3))
    phi_7 = np.kron((e_0 + e_1 - e_3), e_1 / np.sqrt(3))
    phi_8 = np.kron((e_0 - e_1 + e_2), (e_1 + e_2 + e_3) / 3)

    dims = [4, 4]
    states = [
        State(phi_1 * phi_1.conj().T, dims),
        State(phi_2 * phi_2.conj().T, dims),
        State(phi_3 * phi_3.conj().T, dims),
        State(phi_4 * phi_4.conj().T, dims),
        State(phi_5 * phi_5.conj().T, dims),
        State(phi_6 * phi_6.conj().T, dims),
        State(phi_7 * phi_7.conj().T, dims),
        State(phi_8 * phi_8.conj().T, dims),
    ]
    ensemble = Ensemble(states=states)
    sd = OptDist(
        ensemble,
        "sep",
        "min-error",
        return_optimal_meas=False,
        solver="SCS",
        verbose=False,
        eps=1e-6,
        level=2,
    )
    sd.solve()
    np.testing.assert_equal(np.isclose(sd.value, 0.99672963), True)
