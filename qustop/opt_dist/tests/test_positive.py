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

from qustop import Ensemble, State, OptDist
from toqito.states import bell


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


# dual_res = OptDist(ensemble=ensemble,
#                    dist_measurement="pos",
#                    dist_method="unambiguous",
#                    return_optimal_meas=False)
# dual_res.solve()
# np.testing.assert_equal(np.isclose(dual_res.value, 0), True)


def test_state_distinguishability_yyd_density_matrices():
    """Global distinguishability of the YYD states should yield 1."""
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
