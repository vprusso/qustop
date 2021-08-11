# Copyright (C) 2021 Vincent Russo
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import pickle

from qustop import Ensemble, State, OptDist
from scipy.stats import unitary_group

# Number of random states to generate.
num_tests = 5

ppt_vals = []
for test in range(num_tests):
    # Generate an ensemble of random mutually orthogonal and pure states.
    # In this case, the ensemble consists of four states, but we could
    # potentially have any n >= 1 number of states in the ensemble.
    states = unitary_group.rvs(4)
    rho_0 = np.atleast_2d(states[:, 0]).T
    rho_1 = np.atleast_2d(states[:, 1]).T
    rho_2 = np.atleast_2d(states[:, 2]).T
    rho_3 = np.atleast_2d(states[:, 3]).T

    # Create the two-copy ensemble.
    dims_2_copy = [2, 2, 2, 2]
    ensemble_2_copies = Ensemble([State(np.kron(rho_0, rho_0), dims_2_copy),
                                  State(np.kron(rho_1, rho_1), dims_2_copy),
                                  State(np.kron(rho_2, rho_2), dims_2_copy),
                                  State(np.kron(rho_3, rho_3), dims_2_copy)])

    # Solve the two-copy PPT distinguishability SDPs.
    ppt_2_copy = OptDist(ensemble_2_copies, "ppt", "min-error")
    ppt_2_copy.solve()

    # If the PPT value of the two-copy ensemble is below some threshold
    # of perfect distinguishability, such an example has been found, in
    # which case, we want to ensure we capture the values and states!
    if not np.isclose(ppt_2_copy.value, 1, atol=0.001):
        print(f"PPT 2-copy: {ppt_2_copy.value}")
        with open("FOUND.pickle", "wb") as f:
            pickle.dump([states], f)
        break

    # In any case, print out the two-copy values of the ensembles
    # as we progress through the trials.
    print(f"PPT 2-copy: {ppt_2_copy.value}")
    ppt_vals.append(ppt_2_copy.value)
