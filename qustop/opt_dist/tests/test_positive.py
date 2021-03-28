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


def test_state_distinguishability_one_state():
    """State distinguishability for single state."""
    dims = [2, 2]
    rho = [State(b_0 * b_0.conj().T, dims)]
    ensemble = Ensemble(rho)

    primal_res = OptDist(
        ensemble=ensemble,
        dist_measurement="pos",
        dist_method="min-error",
        return_optimal_meas=True,
    )
    primal_res.solve()
    np.testing.assert_equal(np.isclose(primal_res.value, 1), True)

    # dual_res = OptDist(
    #     ensemble=ensemble,
    #     dist_measurement="pos",
    #     dist_method="min-error",
    #     return_optimal_meas=False,
    # )
    # dual_res.solve()
    # np.testing.assert_equal(np.isclose(dual_res.value, 1), True)
#
#
# def test_state_distinguishability_one_state_vec():
#     """State distinguishability for single vector state."""
#     dims = [2, 2]
#     rho = [State(b_0, dims)]
#     ensemble = Ensemble(rho)
#
#     primal_res = OptDist(
#         ensemble=ensemble,
#         dist_measurement="pos",
#         dist_method="min-error",
#         return_optimal_meas=True,
#     )
#     primal_res.solve()
#     np.testing.assert_equal(np.isclose(primal_res.value, 1), True)
#
#     dual_res = OptDist(
#         ensemble=ensemble,
#         dist_measurement="pos",
#         dist_method="min-error",
#         return_optimal_meas=False,
#     )
#     dual_res.solve()
#     np.testing.assert_equal(np.isclose(dual_res.value, 1), True)
#
#
# def test_state_distinguishability_two_states():
#     """State distinguishability for two state density matrices."""
#     dims = [2, 2]
#     states = [State(e_00, dims), State(e_11, dims)]
#     probs = [1 / 2, 1 / 2]
#     ensemble = Ensemble(states, probs)
#
#     primal_res = OptDist(
#         ensemble=ensemble,
#         dist_measurement="pos",
#         dist_method="min-error",
#         return_optimal_meas=True,
#     )
#     primal_res.solve()
#     np.testing.assert_equal(np.isclose(primal_res.value, 1), True)
#
#     dual_res = OptDist(
#         ensemble=ensemble,
#         dist_measurement="pos",
#         dist_method="min-error",
#         return_optimal_meas=False,
#     )
#     dual_res.solve()
#     np.testing.assert_equal(np.isclose(dual_res.value, 1), True)
#
#
# def test_unambiguous_state_distinguishability_two_states():
#     """Unambiguous state distinguishability for two state density matrices."""
#     dims = [2, 2]
#     states = [State(e_00, dims), State(e_11, dims)]
#     probs = [1 / 2, 1 / 2]
#     ensemble = Ensemble(states, probs)
#
#     primal_res = OptDist(
#         ensemble=ensemble,
#         dist_measurement="pos",
#         dist_method="unambiguous",
#         return_optimal_meas=True,
#     )
#     primal_res.solve()
#     np.testing.assert_equal(np.isclose(primal_res.value, 0), True)

    # dual_res = OptDist(ensemble=ensemble,
    #                    dist_measurement="pos",
    #                    dist_method="unambiguous",
    #                    return_optimal_meas=False)
    # dual_res.solve()
    # np.testing.assert_equal(np.isclose(dual_res.value, 0), True)


def test_state_distinguishability_yyd_density_matrices():
    """Global distinguishability of the YYD states should yield 1."""
    psi0 = b_0 * b_0.conj().T
    psi1 = b_1 * b_1.conj().T
    psi2 = b_2 * b_2.conj().T
    psi3 = b_3 * b_3.conj().T

    dims = [2, 2, 2, 2]
    states = [
        State(np.kron(psi0, psi0), dims),
        State(np.kron(psi2, psi1), dims),
        State(np.kron(psi3, psi1), dims),
        State(np.kron(psi1, psi1), dims),
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
