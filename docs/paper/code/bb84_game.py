import numpy as np
from toqito.states import basis
from toqito.state_ops import pure_to_mixed
from toqito.nonlocal_games.extended_nonlocal_game import ExtendedNonlocalGame

# The basis: {|0>, |1>}:
e_0 = basis(2, 0)
e_1 = basis(2, 1)

# The basis: {|+>, |->}:
e_p = (e_0 + e_1) / np.sqrt(2)
e_m = (e_0 - e_1) / np.sqrt(2)

# Dimension of referee's measurement:
dim = 2
# The number of outputs for Alice and Bob:
a_out, b_out = 2, 2
# The number of inputs for Alice and Bob:
a_in, b_in = 2, 2

# The probability matrix.
prob_mat = 1 / 2 * np.identity(2)

# Define the predicate matrix V(a,b|x,y):
pred_mat = np.zeros([dim, dim, a_out, b_out, a_in, b_in])

# V(0,0|0,0) = |0><0|
pred_mat[:, :, 0, 0, 0, 0] = pure_to_mixed(e0)
# V(1,1|0,0) = |1><1|
pred_mat[:, :, 1, 1, 0, 0] = pure_to_mixed(e1)
# V(0,0|1,1) = |+><+|
pred_mat[:, :, 0, 0, 1, 1] = pure_to_mixed(ep)
# V(1,1|1,1) = |-><-|
pred_mat[:, :, 1, 1, 1, 1] = pure_to_mixed(em)

# Extended nonlocal game based on BB84 game.
bb84_game = ExtendedNonlocalGame(prob_mat, pred_mat)
