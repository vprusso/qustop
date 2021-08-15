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


def generate_random_two_copy_ensemble(num_states: int) -> Ensemble:
    # Generate an ensemble of random mutually orthogonal and pure states.
    # In this case, the ensemble consists of four states, but we could
    # potentially have any n >= 1 number of states in the ensemble.
    group = unitary_group.rvs(num_states)
    states = [np.atleast_2d(group[:, i]).T for i in range(num_states)]

    # Create the two-copy ensemble.
    dims = [2] * num_states
    ensemble = [State(np.kron(state, state), dims) for _, state in enumerate(states)]
    return Ensemble(ensemble)


def run(num_states: int, num_trials: int) -> None:
    for test in range(num_trials):
        ensemble_2_copies = generate_random_two_copy_ensemble(num_states)

        # Solve the two-copy PPT distinguishability SDPs.
        ppt_2_copy = OptDist(ensemble_2_copies, "ppt", "min-error")
        ppt_2_copy.solve()

        # If the PPT value of the two-copy ensemble is below some threshold
        # of perfect distinguishability, such an example has been found, in
        # which case, we want to ensure we capture the values and states!
        if not np.isclose(ppt_2_copy.value, 1, atol=0.001):
            print(f"PPT 2-copy: {ppt_2_copy.value}")
            with open("FOUND.pickle", "wb") as f:
                pickle.dump([ensemble_2_copies], f)
            break

        # In any case, print out the two-copy values of the ensembles
        # as we progress through the trials.
        print(f"PPT 2-copy: {ppt_2_copy.value}")
