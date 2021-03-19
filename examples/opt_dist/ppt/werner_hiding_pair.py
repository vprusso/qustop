import numpy as np
from toqito.perms import swap_operator
from qustop.core import Ensemble, State
from qustop.opt_dist import OptDist

dim = 2
sigma_0 = (np.kron(np.identity(dim), np.identity(dim)) + swap_operator(dim)) / (dim * (dim + 1))
sigma_1 = (np.kron(np.identity(dim), np.identity(dim)) - swap_operator(dim)) / (dim * (dim - 1))

states = [State(sigma_0, [2, 2]), State(sigma_1, [2, 2])]
ensemble = Ensemble(states)

expected_val = 1 / 2 + 1 / (dim + 1)

sd = OptDist(ensemble=ensemble, 
             measurement="ppt", 
             error="min-error")

sd.solve()

print(sd.value)
print(1/2 + 1/(dim+1))