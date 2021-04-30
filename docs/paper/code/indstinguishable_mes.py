import numpy as np
from toqito.states import bell
from qustop import State, Ensemble, OptDist


# Define the maximally entangled states
# from arXiv:1107.3224
phi_0 = np.kron(bell(0), bell(0))
phi_1 = np.kron(bell(2), bell(1))
phi_2 = np.kron(bell(3), bell(1))
phi_3 = np.kron(bell(1), bell(1))


dims = [2, 2, 2, 2]
rho_0 = phi_0 * phi_0.conj().T
rho_1 = phi_1 * phi_1.conj().T
rho_2 = phi_2 * phi_2.conj().T
rho_3 = phi_3 * phi_3.conj().T

ensemble = Ensemble(
    [
        State(rho_0, dims),
        State(rho_1, dims),
        State(rho_2, dims),
        State(rho_3, dims),
    ]
)

sd = OptDist(
    ensemble=ensemble,
    dist_measurement="ppt",
    dist_method="min-error"
)
sd.solve()

# The min-error probability of distinguishing
# via PPT is equal to 7/8.
print(sd.value)
