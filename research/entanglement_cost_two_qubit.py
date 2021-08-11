import numpy as np

from toqito.states import basis
import matplotlib.pyplot as plt

from qustop import Ensemble, State, OptDist


def plot_param(n_list, eps_list):
    e_0, e_1 = basis(2, 0), basis(2, 1)

    for i, n in enumerate(n_list):
        ppt_list = []

        alpha, beta = np.sqrt((1 + n) / 2), np.sqrt((1 - n) / 2)
        psi_0 = alpha * np.kron(e_0, e_0) + beta * np.kron(e_1, e_1)
        psi_1 = beta * np.kron(e_0, e_0) - alpha * np.kron(e_1, e_1)
        psi_2 = alpha * np.kron(e_0, e_1) + beta * np.kron(e_1, e_0)
        psi_3 = beta * np.kron(e_0, e_1) - alpha * np.kron(e_1, e_0)

        for eps in eps_list:
            tau_state = np.sqrt((1 + eps) / 2) * np.kron(e_0, e_0) + np.sqrt((1 - eps) / 2) * np.kron(e_1, e_1)
            tau = tau_state * tau_state.conj().T

            dims = [2, 2, 2, 2]
            rho_0 = State(np.kron(psi_0 * psi_0.conj().T, tau), dims)
            rho_1 = State(np.kron(psi_1 * psi_1.conj().T, tau), dims)
            rho_2 = State(np.kron(psi_2 * psi_2.conj().T, tau), dims)
            rho_3 = State(np.kron(psi_3 * psi_3.conj().T, tau), dims)

            ensemble = Ensemble([rho_0, rho_1, rho_2, rho_3])

            ppt_res = OptDist(ensemble, "ppt", "min-error")
            ppt_res.solve()

            ppt_list.append(ppt_res.value)
            print(f"N: {n} -- EPS: {eps} -- PPT: {ppt_res.value}")

        plt.scatter(eps_list, ppt_list, label=f"n = {n}")
    plt.legend()
    plt.xlabel(r"$\epsilon$")
    plt.ylabel("PPT value")
    plt.title(f"n = {n_list} vs. PPT")

    plt.show()


n_list = [0.0, 0.1]
eps_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
plot_param(n_list, eps_list)
