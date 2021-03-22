import numpy as np
from toqito.states import bell

from qustop.core import State, Ensemble
from qustop.opt_dist import OptDist


psi_0 = bell(0)
psi_1 = bell(2)
psi_2 = bell(3)
psi_3 = bell(1)

x_1 = np.kron(psi_0, psi_0)
x_2 = np.kron(psi_1, psi_3)

rho_1 = State(x_1 * x_1.conj().T, [2, 2, 2, 2])
rho_2 = State(x_2 * x_2.conj().T, [2, 2, 2, 2])

ensemble = Ensemble([rho_1, rho_2])

sd = OptDist(ensemble=ensemble,
             dist_measurement="positive",
             dist_method="min-error")
sd.solve()
print(sd.value)
