import numpy as np

from toqito.states import basis
from toqito.state_ops import pure_to_mixed
from toqito.nonlocal_games.extended_nonlocal_game import ExtendedNonlocalGame

prob_mat = 1 / 3 * np.identity(3)

dim = 3
e_0, e_1, e_2 = basis(dim, 0), basis(dim, 1), basis(dim, 2)

z = np.exp((2 * np.pi * 1j) / dim)
mub_1 = [
    (e_0 + e_1 + e_2) / np.sqrt(3),
    (e_0 + z ** 2 * e_1 + z * e_2) / np.sqrt(3),
    (e_0 + z * e_1 + z ** 2 * e_2) / np.sqrt(3),
]
mub_2 = [
    (e_0 + e_1 + z * e_2) / np.sqrt(3),
    (e_0 + z ** 2 * e_1 + z ** 2 * e_2) / np.sqrt(3),
    (e_0 + z * e_1 + e_2) / np.sqrt(3),
]
mub_3 = [
    (e_0 + e_1 + z ** 2 * e_2) / np.sqrt(3),
    (e_0 + z ** 2 * e_1 + e_2) / np.sqrt(3),
    (e_0 + z * e_1 + z * e_2) / np.sqrt(3),
]

mubs = [mub_1, mub_2, mub_3]

num_in = 3
num_out = 3
pred_mat = np.zeros(
    [dim, dim, num_out, num_out, num_in, num_in], dtype=complex
)

for i in range(num_in):
    for o in range(num_out):
        pred_mat[:, :, o, o, i, i] = pure_to_mixed(mubs[i][o])
mub_3_3_game = ExtendedNonlocalGame(prob_mat, pred_mat)
