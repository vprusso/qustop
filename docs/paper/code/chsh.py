import numpy as np
from toqito.nonlocal_games import xor_game

# Probability matrix for each question pair:
prob_mat = np.array([[1/4, 1/4],
                     [1/4, 1/4]])

# Encode winning conditions for the CHSH game:
pred_mat = np.array([[0, 0],
                    [0, 1]])

# Define the CHSH XOR nonlocal game:
chsh = xor_game.XORGame(prob_mat, pred_mat)
