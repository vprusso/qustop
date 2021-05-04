import numpy as np
from toqito.states import basis
from qustop import Ensemble, OptDist, State

n, m = 0, 0
alpha, beta = np.sqrt((1 + n) / 2), np.sqrt((1 - n) / 2)
gamma, delta = np.sqrt((1 + m) / 2), np.sqrt((1 - m) / 2)

e_0, e_1 = basis(2, 0), basis(2, 1)

dims = [2, 2]
ensemble = Ensemble(
    [
        State(alpha * np.kron(e_0, e_0) + beta * np.kron(e_1, e_1), dims),
        State(beta * np.kron(e_0, e_0) - alpha * np.kron(e_1, e_1), dims),
        State(gamma * np.kron(e_0, e_1) + delta * np.kron(e_1, e_0), dims),
        State(delta * np.kron(e_0, e_1) - gamma * np.kron(e_1, e_0), dims),
    ]
)

# Determine the optimal probability of distinguishing
# via PPT measurements with minimum error:
ppt_min = OptDist(ensemble, "ppt", "min-error")
ppt_min.solve()

print(ppt_min.value)

print(1 / 2 * (1 + (n + m) / 2))

