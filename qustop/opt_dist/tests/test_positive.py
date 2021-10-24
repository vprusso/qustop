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

from qustop import Ensemble, OptDist, State


def test_state_distinguishability_one_state():
    """State distinguishability for single state."""
    dims = [2, 2]
    rho = [State(bell(0) * bell(0).conj().T, dims)]
    ensemble = Ensemble(rho)

    primal_res = OptDist(
        ensemble=ensemble,
        dist_measurement="pos",
        dist_method="min-error",
        return_optimal_meas=True,
    )
    primal_res.solve()
    np.testing.assert_equal(np.isclose(primal_res.value, 1), True)

    dual_res = OptDist(
        ensemble=ensemble,
        dist_measurement="pos",
        dist_method="min-error",
        return_optimal_meas=False,
    )
    dual_res.solve()
    np.testing.assert_equal(np.isclose(dual_res.value, 1), True)


def test_state_distinguishability_one_state_vec():
    """State distinguishability for single vector state."""
    dims = [2, 2]
    rho = [State(bell(0), dims)]
    ensemble = Ensemble(rho)

    primal_res = OptDist(
        ensemble=ensemble,
        dist_measurement="pos",
        dist_method="min-error",
        return_optimal_meas=True,
    )
    primal_res.solve()
    np.testing.assert_equal(np.isclose(primal_res.value, 1), True)

    dual_res = OptDist(
        ensemble=ensemble,
        dist_measurement="pos",
        dist_method="min-error",
        return_optimal_meas=False,
    )
    dual_res.solve()
    np.testing.assert_equal(np.isclose(dual_res.value, 1), True)


def test_state_distinguishability_two_states():
    """State distinguishability for two state density matrices."""
    dims = [2, 2]
    states = [State(bell(0), dims), State(bell(1), dims)]
    probs = [1 / 2, 1 / 2]
    ensemble = Ensemble(states, probs)

    primal_res = OptDist(
        ensemble=ensemble,
        dist_measurement="pos",
        dist_method="min-error",
        return_optimal_meas=True,
    )
    primal_res.solve()
    np.testing.assert_equal(np.isclose(primal_res.value, 1), True)

    dual_res = OptDist(
        ensemble=ensemble,
        dist_measurement="pos",
        dist_method="min-error",
        return_optimal_meas=False,
    )
    dual_res.solve()
    np.testing.assert_equal(np.isclose(dual_res.value, 1), True)


def test_unambiguous_state_distinguishability_two_states():
    """Unambiguous state distinguishability for two state density matrices."""
    dims = [2, 2]
    states = [State(bell(0), dims), State(bell(1), dims)]
    probs = [1 / 2, 1 / 2]
    ensemble = Ensemble(states, probs)

    primal_res = OptDist(
        ensemble=ensemble,
        dist_measurement="pos",
        dist_method="unambiguous",
        return_optimal_meas=True,
    )
    primal_res.solve()
    np.testing.assert_equal(np.isclose(primal_res.value, 1), True)

    dual_res = OptDist(
        ensemble=ensemble,
        dist_measurement="pos",
        dist_method="unambiguous",
        return_optimal_meas=False,
    )
    dual_res.solve()
    np.testing.assert_equal(np.isclose(dual_res.value, 1), True)


def test_state_distinguishability_two_mixed_states():
    """Known closed-form exists for the distinguishability of two pure states."""
    # Define single-qubit |0> and |1> basis states.
    e_p, e_m = (
        1 / np.sqrt(2) * np.array([[1, 1]]).T,
        1 / np.sqrt(2) * np.array([[1, -1]]).T,
    )

    # rho_1 = 3/4 |+><+| + 1/4|-><-|
    rho_1 = 3 / 4 * (e_p * e_p.conj().T) + 1 / 4 * (e_m * e_m.conj().T)

    # rho_2 = 1/4 |+><+| + 3/4|-><-|
    rho_2 = 1 / 4 * (e_p * e_p.conj().T) + 3 / 4 * (e_m * e_m.conj().T)

    dims = [2]
    rho_1 = State(rho_1, dims)
    rho_2 = State(rho_2, dims)

    ensemble = Ensemble([rho_1, rho_2])

    primal_res = OptDist(
        ensemble=ensemble, dist_measurement="pos", dist_method="min-error"
    )
    primal_res.solve()
    np.testing.assert_equal(
        np.isclose(primal_res.value, 3 / 4, atol=0.001), True
    )

    dual_res = OptDist(
        ensemble=ensemble, dist_measurement="pos", dist_method="min-error"
    )
    dual_res.solve()
    np.testing.assert_equal(
        np.isclose(dual_res.value, 3 / 4, atol=0.001), True
    )


def test_state_distinguishability_three_non_orthogonal_states():
    """No closed-form is known to exist for this case, so we resort to solving the SDP."""
    e_p, e_m = (
        1 / np.sqrt(2) * np.array([[1, 1]]).T,
        1 / np.sqrt(2) * np.array([[1, -1]]).T,
    )

    # rho_1 = 3/4 |+><+| + 1/4|-><-|
    rho_1 = 3 / 4 * (e_p * e_p.conj().T) + 1 / 4 * (e_m * e_m.conj().T)

    # rho_2 = 1/4 |+><+| + 3/4|-><-|
    rho_2 = 1 / 4 * (e_p * e_p.conj().T) + 3 / 4 * (e_m * e_m.conj().T)

    # rho_3 = 1/2 |+><+| + 1/2 |-><-|
    rho_3 = 1 / 2 * (e_p * e_p.conj().T) + 1 / 2 * (e_m * e_m.conj().T)

    dims = [2]
    rho_1 = State(rho_1, dims)
    rho_2 = State(rho_2, dims)
    rho_3 = State(rho_3, dims)

    ensemble = Ensemble([rho_1, rho_2, rho_3])

    # For any set of more than two states that are non-mutually orthogonal, no closed-form expression for optimal
    # distinguishability is known to exist. Therefore, we must resort to solving the SDP to determine what the optimal
    # probability is.

    # Primal min-error:
    primal_min_error_res = OptDist(
        ensemble=ensemble,
        dist_measurement="pos",
        dist_method="min-error",
        return_optimal_meas=True,
    )
    primal_min_error_res.solve()
    np.testing.assert_equal(
        np.isclose(primal_min_error_res.value, 1 / 2, atol=0.001), True
    )

    # Dual min-error:
    dual_min_error_res = OptDist(
        ensemble=ensemble,
        dist_measurement="pos",
        dist_method="min-error",
        return_optimal_meas=False,
    )
    dual_min_error_res.solve()
    np.testing.assert_equal(
        np.isclose(dual_min_error_res.value, 1 / 2, atol=0.001), True
    )

    # Primal unambiguous:
    primal_unambig_res = OptDist(
        ensemble=ensemble,
        dist_measurement="pos",
        dist_method="unambiguous",
        return_optimal_meas=True,
    )
    primal_unambig_res.solve()
    np.testing.assert_equal(
        np.isclose(primal_unambig_res.value, 0, atol=0.001), True
    )

    # Dual unambiguous
    dual_unambig_res = OptDist(
        ensemble=ensemble,
        dist_measurement="pos",
        dist_method="unambiguous",
        return_optimal_meas=False,
    )
    dual_unambig_res.solve()
    np.testing.assert_equal(
        np.isclose(dual_unambig_res.value, 0, atol=0.001), True
    )


def test_state_distinguishability_ydy_density_matrices():
    """Global distinguishability of the YDY states should yield 1."""
    psi_0 = bell(0) * bell(0).conj().T
    psi_1 = bell(1) * bell(1).conj().T
    psi_2 = bell(2) * bell(2).conj().T
    psi_3 = bell(3) * bell(3).conj().T

    dims = [2, 2, 2, 2]
    states = [
        State(np.kron(psi_0, psi_0), dims),
        State(np.kron(psi_2, psi_1), dims),
        State(np.kron(psi_3, psi_1), dims),
        State(np.kron(psi_1, psi_1), dims),
    ]
    probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
    ensemble = Ensemble(states, probs)

    primal_res = OptDist(
        ensemble=ensemble,
        dist_measurement="pos",
        dist_method="min-error",
        return_optimal_meas=True,
    )
    primal_res.solve()
    np.testing.assert_equal(np.isclose(primal_res.value, 1, atol=0.001), True)


def test_state_distinguishability_bell_states():
    """Global distinguishability of the Bell states should yield 1."""
    dims = [2, 2]
    states = [
        State(bell(0), dims),
        State(bell(1), dims),
        State(bell(2), dims),
        State(bell(3), dims),
    ]
    probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
    ensemble = Ensemble(states, probs)

    primal_min_error_res = OptDist(
        ensemble=ensemble,
        dist_measurement="pos",
        dist_method="min-error",
        return_optimal_meas=True,
    )
    primal_min_error_res.solve()
    np.testing.assert_equal(
        np.isclose(primal_min_error_res.value, 1, atol=0.001), True
    )

    dual_min_error_res = OptDist(
        ensemble=ensemble,
        dist_measurement="pos",
        dist_method="min-error",
        return_optimal_meas=False,
    )
    dual_min_error_res.solve()
    np.testing.assert_equal(
        np.isclose(dual_min_error_res.value, 1, atol=0.001), True
    )

    primal_unambig_res = OptDist(
        ensemble=ensemble,
        dist_measurement="pos",
        dist_method="unambiguous",
        return_optimal_meas=True,
    )
    primal_unambig_res.solve()
    np.testing.assert_equal(
        np.isclose(primal_unambig_res.value, 1, atol=0.001), True
    )

    dual_unambig_res = OptDist(
        ensemble=ensemble,
        dist_measurement="pos",
        dist_method="unambiguous",
        return_optimal_meas=False,
    )
    dual_unambig_res.solve()
    np.testing.assert_equal(
        np.isclose(dual_unambig_res.value, 1, atol=0.001), True
    )
