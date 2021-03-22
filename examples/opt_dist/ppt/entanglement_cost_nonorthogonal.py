import numpy as np

from toqito.states import basis, bell
from qustop.core import Ensemble, State
from qustop.opt_dist import OptDist


e_0, e_1 = basis(2, 0), basis(2, 1)

phi_1 = bell(0) * bell(0).conj().T
phi_2 = bell(1) * bell(1).conj().T
phi_3 = bell(2) * bell(2).conj().T
phi_4 = bell(3) * bell(3).conj().T

lam = 0.75
rho_1 = lam * phi_1 + (1 - lam)/4 * np.identity(4)
rho_2 = lam * phi_2 + (1 - lam)/4 * np.identity(4)
rho_3 = lam * phi_3 + (1 - lam)/4 * np.identity(4)
rho_4 = lam * phi_4 + (1 - lam)/4 * np.identity(4)

eps = 0.215
tau_state = np.sqrt((1 + eps) / 2) * np.kron(e_0, e_0) + np.sqrt((1 - eps) / 2) * np.kron(e_1, e_1)
tau = tau_state * tau_state.conj().T

states = [
    State(np.kron(rho_1, tau), [2, 2, 2, 2]),
    State(np.kron(rho_2, tau), [2, 2, 2, 2]),
    State(np.kron(rho_3, tau), [2, 2, 2, 2]),
    State(np.kron(rho_4, tau), [2, 2, 2, 2])
]

ensemble = Ensemble(states)
ensemble.swap([2, 3])
sd = OptDist(ensemble=ensemble,
             dist_measurement="ppt",
             dist_method="min-error")

sd.solve()
print(sd.value)
print(lam/2 * (1 + np.sqrt(1 - eps**2)) + 1/4 * (1-lam))