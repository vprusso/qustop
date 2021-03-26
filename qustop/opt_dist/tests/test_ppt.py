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

from toqito.perms import swap_operator
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


def test_ppt_distinguishability_yyd_density_matrices():
    """
    PPT distinguishing the YYD states from [1] should yield `7/8 ~ 0.875`

    Feeding the input to the function as density matrices.

    References:
    [1]: Yu, Nengkun, Runyao Duan, and Mingsheng Ying.
    "Four locally indistinguishable ququad-ququad orthogonal
    maximally entangled states."
    Physical review letters 109.2 (2012): 020506.
    https://arxiv.org/abs/1107.3224
    """
    psi_0 = b_0
    psi_1 = b_2
    psi_2 = b_3
    psi_3 = b_1

    x_1 = np.kron(psi_0, psi_0)
    x_2 = np.kron(psi_1, psi_3)
    x_3 = np.kron(psi_2, psi_3)
    x_4 = np.kron(psi_3, psi_3)

    rho_1 = x_1 * x_1.conj().T
    rho_2 = x_2 * x_2.conj().T
    rho_3 = x_3 * x_3.conj().T
    rho_4 = x_4 * x_4.conj().T

    dims = [2, 2, 2, 2]
    states = [
        State(rho_1, dims),
        State(rho_2, dims),
        State(rho_3, dims),
        State(rho_4, dims)
    ]
    probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
    ensemble = Ensemble(states, probs)

    # Test primal and dual problems on minimum-error method:
    min_error_primal_res = OptDist(
        ensemble=ensemble,
        dist_measurement="ppt",
        dist_method="min-error",
        return_optimal_meas=True,
    )
    min_error_primal_res.solve()
    min_error_dual_res = OptDist(
        ensemble=ensemble,
        dist_measurement="ppt",
        dist_method="min-error",
        return_optimal_meas=False,
    )
    min_error_dual_res.solve()

    np.testing.assert_equal(
        np.isclose(min_error_primal_res.value, 7 / 8, atol=0.001), True
    )
    np.testing.assert_equal(
        np.isclose(min_error_dual_res.value, 7 / 8, atol=0.001), True
    )

    # Test primal and dual problems on unambiguous method:
    unambig_primal_res = OptDist(
        ensemble=ensemble,
        dist_measurement="ppt",
        dist_method="unambiguous",
        return_optimal_meas=True,
    )
    unambig_primal_res.solve()
    unambig_dual_res = OptDist(
        ensemble=ensemble,
        dist_measurement="ppt",
        dist_method="unambiguous",
        return_optimal_meas=False,
    )
    unambig_dual_res.solve()

    np.testing.assert_equal(
        np.isclose(unambig_primal_res.value, 3 / 4, atol=0.001), True
    )
    np.testing.assert_equal(
        np.isclose(unambig_dual_res.value, 3 / 4, atol=0.001), True
    )


def test_ppt_distinguishability_yyd_vectors():
    """
    PPT distinguishing the YYD states from [1] should yield `7/8 ~ 0.875`

    Feeding the input to the function as state vectors.

    References:
    [1]: Yu, Nengkun, Runyao Duan, and Mingsheng Ying.
    "Four locally indistinguishable ququad-ququad orthogonal
    maximally entangled states."
    Physical review letters 109.2 (2012): 020506.
    https://arxiv.org/abs/1107.3224
    """
    psi_0 = b_0
    psi_1 = b_2
    psi_2 = b_3
    psi_3 = b_1

    x_1 = np.kron(psi_0, psi_0)
    x_2 = np.kron(psi_1, psi_3)
    x_3 = np.kron(psi_2, psi_3)
    x_4 = np.kron(psi_3, psi_3)

    dims = [2, 2, 2, 2]
    states = [
        State(x_1, dims),
        State(x_2, dims),
        State(x_3, dims),
        State(x_4, dims)
    ]
    probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
    ensemble = Ensemble(states, probs)

    # Test primal and dual problems on minimum-error method:
    min_error_primal_res = OptDist(
        ensemble=ensemble,
        dist_measurement="ppt",
        dist_method="min-error",
        return_optimal_meas=True,
    )
    min_error_primal_res.solve()
    min_error_dual_res = OptDist(
        ensemble=ensemble,
        dist_measurement="ppt",
        dist_method="min-error",
        return_optimal_meas=False,
    )
    min_error_dual_res.solve()

    np.testing.assert_equal(
        np.isclose(min_error_primal_res.value, 7 / 8, atol=0.001), True
    )
    np.testing.assert_equal(
        np.isclose(min_error_dual_res.value, 7 / 8, atol=0.001), True
    )

    # Test primal and dual problems on unambiguous method:
    unambig_primal_res = OptDist(
        ensemble=ensemble,
        dist_measurement="ppt",
        dist_method="unambiguous",
        return_optimal_meas=True,
    )
    unambig_primal_res.solve()
    unambig_dual_res = OptDist(
        ensemble=ensemble,
        dist_measurement="ppt",
        dist_method="unambiguous",
        return_optimal_meas=False,
    )
    unambig_dual_res.solve()

    np.testing.assert_equal(
        np.isclose(unambig_primal_res.value, 3 / 4, atol=0.001), True
    )
    np.testing.assert_equal(
        np.isclose(unambig_dual_res.value, 3 / 4, atol=0.001), True
    )



def test_ppt_distinguishability_yyd_states_no_probs():
    """
    PPT distinguishing the YYD states from [1] should yield 7/8 ~ 0.875

    If no probability vector is explicitly given, assume uniform
    probabilities are given.

    References:
    [1]: Yu, Nengkun, Runyao Duan, and Mingsheng Ying.
    "Four locally indistinguishable ququad-ququad orthogonal
    maximally entangled states."
    Physical review letters 109.2 (2012): 020506.
    https://arxiv.org/abs/1107.3224
    """
    psi_0 = b_0
    psi_1 = b_2
    psi_2 = b_3
    psi_3 = b_1

    x_1 = np.kron(psi_0, psi_0)
    x_2 = np.kron(psi_1, psi_3)
    x_3 = np.kron(psi_2, psi_3)
    x_4 = np.kron(psi_3, psi_3)

    dims = [2, 2, 2, 2]
    rho_1 = State(x_1 * x_1.conj().T, dims)
    rho_2 = State(x_2 * x_2.conj().T, dims)
    rho_3 = State(x_3 * x_3.conj().T, dims)
    rho_4 = State(x_4 * x_4.conj().T, dims)

    ensemble = Ensemble([rho_1, rho_2, rho_3, rho_4])

    # Test primal and dual problems on minimum-error method:
    min_error_primal_res = OptDist(
        ensemble=ensemble,
        dist_measurement="ppt",
        dist_method="min-error",
        return_optimal_meas=True,
    )
    min_error_primal_res.solve()
    min_error_dual_res = OptDist(
        ensemble=ensemble,
        dist_measurement="ppt",
        dist_method="min-error",
        return_optimal_meas=False,
    )
    min_error_dual_res.solve()

    np.testing.assert_equal(
        np.isclose(min_error_primal_res.value, 7 / 8, atol=0.001), True
    )
    np.testing.assert_equal(
        np.isclose(min_error_dual_res.value, 7 / 8, atol=0.001), True
    )

    # Test primal and dual problems on unambiguous method:
    unambig_primal_res = OptDist(
        ensemble=ensemble,
        dist_measurement="ppt",
        dist_method="unambiguous",
        return_optimal_meas=True,
    )
    unambig_primal_res.solve()
    unambig_dual_res = OptDist(
        ensemble=ensemble,
        dist_measurement="ppt",
        dist_method="unambiguous",
        return_optimal_meas=False,
    )
    unambig_dual_res.solve()

    np.testing.assert_equal(
        np.isclose(unambig_primal_res.value, 3 / 4, atol=0.001), True
    )
    np.testing.assert_equal(
        np.isclose(unambig_dual_res.value, 3 / 4, atol=0.001), True
    )


def test_ppt_distinguishability_werner_hiding_pairs():
    r"""
    One quantum data hiding scheme involves the Werner hiding pair.

    A Werner hiding pair is defined by

    .. math::
    \begin{equation}
        \sigma_0^{(n)} = \frac{\mathbb{I} \otimes \mathbb{I} + W_n}{n(n+1)}
        \quad \text{and} \quad
        \sigma_1^{(n)} = \frac{\mathbb{I} \otimes \mathbb{I} - W_n}{n(n-1)}
    \end{equation}

    The optimal probability to distinguish the Werner hiding pair is known
    to be upper bounded by the following equation

    .. math::
    \begin{equation}
        \frac{1}{2} + \frac{1}{n+1}
    \end{equation}

    References:
    [1]: Terhal, Barbara M., David P. DiVincenzo, and Debbie W. Leung.
    "Hiding bits in Bell states."
    Physical review letters 86.25 (2001): 5807.
    https://arxiv.org/abs/quant-ph/0011042

    [2]: Cosentino, Alessandro
    "Quantum state local distinguishability via convex optimization".
    University of Waterloo, Thesis
    https://uwspace.uwaterloo.ca/handle/10012/9572
    """
    dim = 2
    sigma_0 = (
        np.kron(np.identity(dim), np.identity(dim)) + swap_operator(dim)
    ) / (dim * (dim + 1))
    sigma_1 = (
        np.kron(np.identity(dim), np.identity(dim)) - swap_operator(dim)
    ) / (dim * (dim - 1))

    states = [State(sigma_0, [2, 2]), State(sigma_1, [2, 2])]
    ensemble = Ensemble(states)

    expected_val = 1 / 2 + 1 / (dim + 1)

    # Test primal and dual problems on minimum-error method:
    min_error_primal_res = OptDist(
        ensemble=ensemble,
        dist_measurement="ppt",
        dist_method="min-error",
        return_optimal_meas=True,
    )
    min_error_primal_res.solve()
    min_error_dual_res = OptDist(
        ensemble=ensemble,
        dist_measurement="ppt",
        dist_method="min-error",
        return_optimal_meas=False,
    )
    min_error_dual_res.solve()

    np.testing.assert_equal(
        np.isclose(min_error_primal_res.value, expected_val, atol=0.001), True
    )
    np.testing.assert_equal(
        np.isclose(min_error_dual_res.value, expected_val, atol=0.001), True
    )

    # Test primal and dual problems on unambiguous method:
    unambig_primal_res = OptDist(
        ensemble=ensemble,
        dist_measurement="ppt",
        dist_method="unambiguous",
        return_optimal_meas=True,
    )
    unambig_primal_res.solve()
    unambig_dual_res = OptDist(
        ensemble=ensemble,
        dist_measurement="ppt",
        dist_method="unambiguous",
        return_optimal_meas=False,
    )
    unambig_dual_res.solve()

    np.testing.assert_equal(
        np.isclose(unambig_primal_res.value, 1 / 3, atol=0.001), True
    )
    np.testing.assert_equal(
        np.isclose(unambig_dual_res.value, 1 / 3, atol=0.001), True
    )


def test_ppt_min_error_four_bell_states():
    r"""
    PPT distinguishing the four Bell states.
    There exists a closed form formula for the probability with which one
    is able to distinguish one of the four Bell states given with equal
    probability when Alice and Bob have access to a resource state [1].

    The resource state is defined by

    ..math::
        |\tau_{\epsilon} \rangle = \sqrt{\frac{1+\epsilon}{2}} +
        |0\rangle | 0\rangle +
        \sqrt{\frac{1-\epsilon}{2}} |1 \rangle |1 \rangle

    The closed form probability with which Alice and Bob can distinguish via
    PPT measurements is given as follows

    .. math::
        \frac{1}{2} \left(1 + \sqrt{1 - \epsilon^2} \right).

    This formula happens to be equal to LOCC and SEP as well for this case.
    Refer to Theorem 5 in [1] for more details.

    References:
    [1]: Bandyopadhyay, Somshubhro, et al.
    "Limitations on separable measurements by convex optimization."
    IEEE Transactions on Information Theory 61.6 (2015): 3593-3604.
    https://arxiv.org/abs/1408.6981
    """
    rho_1 = b_0 * b_0.conj().T
    rho_2 = b_1 * b_1.conj().T
    rho_3 = b_2 * b_2.conj().T
    rho_4 = b_3 * b_3.conj().T

    eps = 0.5
    resource_state = (
        np.sqrt((1 + eps) / 2) * e_00 + np.sqrt((1 - eps) / 2) * e_11
    )
    resource_state = resource_state * resource_state.conj().T

    dims = [2, 2, 2, 2]
    states = [
        State(np.kron(rho_1, resource_state), dims),
        State(np.kron(rho_2, resource_state), dims),
        State(np.kron(rho_3, resource_state), dims),
        State(np.kron(rho_4, resource_state), dims),
    ]
    probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
    ensemble = Ensemble(states, probs)

    exp_res = 1 / 2 * (1 + np.sqrt(1 - eps ** 2))

    # Solve the primal problem.
    primal_res = OptDist(
        ensemble=ensemble,
        dist_measurement="ppt",
        dist_method="min-error",
        return_optimal_meas=True,
    )
    primal_res.solve()

    # Solve the dual problem.
    dual_res = OptDist(
        ensemble=ensemble,
        dist_measurement="ppt",
        dist_method="min-error",
        return_optimal_meas=False,
    )
    dual_res.solve()

    np.testing.assert_equal(
        np.isclose(primal_res.value, exp_res, atol=0.001), True
    )
    np.testing.assert_equal(
        np.isclose(dual_res.value, exp_res, atol=0.001), True
    )