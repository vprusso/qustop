import numpy as np

from toqito.states import bell
from toqito.state_ops import pure_to_mixed
from toqito.state_opt import ppt_distinguishability

# YDY vectors:
x_0 = np.kron(bell(0), bell(0))
x_1 = np.kron(bell(2), bell(1))
x_2 = np.kron(bell(3), bell(1))
x_3 = np.kron(bell(1), bell(1))

# YDY density matrices:
rho_0 = pure_to_mixed(x_0)
rho_1 = pure_to_mixed(x_1)
rho_2 = pure_to_mixed(x_2)
rho_3 = pure_to_mixed(x_3)

# Calculate PPT optimal value:
states = [rho_0, rho_1, rho_2, rho_3]
probs = [1/4, 1/4, 1/4, 1/4]

ppt_distinguishability(states, probs)
