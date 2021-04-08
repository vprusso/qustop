>>> # 2-fold parallel repetition of CHSH:
>>> chsh_2 = xor_game.XORGame(prob_mat, 
>>>                           pred_mat, 
>>>                           reps=2)
>>> # (cos^2(pi/8))^2 \approx 0.72853
>>> chsh_2.quantum_value()
0.7285533905932593


>>> # 3-fold parallel repetition of CHSH:
>>> chsh_3 = xor_game.XORGame(prob_mat, 
>>>                           pred_mat,
>>>                           reps=3)
>>> # (cos^2(pi/8))^3 \approx 0.62186
>>> chsh_3.quantum_value()
0.6218592167690961
