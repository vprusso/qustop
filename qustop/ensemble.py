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

from typing import List, Tuple

import numpy as np

from toqito.matrix_props import is_density


class Ensemble:
    def __init__(self, states: List[np.ndarray], probs: List[float]):
        self._states = states
        self._probs = probs

    def __str__(self) -> str:
        return f"({self.probs}, {self.states})"

    def __getitem__(self, key: int) -> Tuple[float, np.ndarray]:
        return self.probs[key], self.states[key]

    @property
    def probs(self):
        return self._probs

    @property
    def states(self):
        return self._states

    @staticmethod
    def _is_quantum_states_valid(states: List[np.ndarray]) -> bool:
        # Assume at least one state is provided.
        if states is None or states == []:
            raise ValueError("InvalidStates: There must be at least one state provided.")
        for state in states:
            if not is_density(state):
                raise ValueError("InvalidStates: All states must be density operators.")
        return True

    @staticmethod
    def _is_probs_valid(probs: List[float]) -> bool:
        if not np.isclose(sum(probs), 1):
            raise ValueError("InvalidProbabilities: Probabilities must sum to 1.")
        return True



if __name__ == "__main__":
    np_state_1 = 1/2 * np.identity(2)
    np_state_2 = 1/2 * np.identity(2)

    ensemble = Ensemble([np_state_1, np_state_2], [1/2, 1/2])
    print(ensemble.probs)
