import numpy as np

from toqito.states import basis, bell
from qustop import State, Ensemble, OptDist


e_0, e_1 = basis(2, 0), basis(2, 1)

eps = 0.5
tau = np.sqrt((1 + eps) / 2) * np.kron(e_0, e_0) + np.sqrt((1 - eps) / 2) * np.kron(e_1, e_1)

dims = [2, 2, 2, 2]
states = [
    State(np.kron(bell(0), tau), dims),
    State(np.kron(bell(1), tau), dims),
    State(np.kron(bell(2), tau), dims),
    State(np.kron(bell(3), tau), dims),
]
probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
ensemble = Ensemble(states, probs)
ensemble.swap([2, 3])

sep_res = OptDist(ensemble, "sep", "min-error")
sep_res.solve()

ppt_res = OptDist(ensemble, "ppt", "min-error")
ppt_res.solve()

eq = 1 / 2 * (1 + np.sqrt(1 - eps ** 2))

print(eq)
print(ppt_res.value)
print(sep_res.value)
