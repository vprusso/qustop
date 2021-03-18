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

from typing import List, Optional, Tuple

import numpy as np

from toqito.perms import swap
from toqito.matrix_props import is_density

from qustop.core.state import State


class Ensemble:
    def __init__(self, states: List[State], probs: Optional[List[float]] = None) -> None:
        """
        If no probability vector is specified in `probs`, the default assumes
        a uniform probability distribution for all states in the ensemble.
        """
        self._states = self._prepare_states(states)
        self._probs = self._prepare_probs(probs)

    def __len__(self) -> int:
        return len(self._states)

    def __str__(self) -> str:
        out_s = f"Ensemble: num_states = {len(self)}\n"
        for i, subsystem in enumerate(self._states):
            if i == len(self._states)-1:
                out_s += f"ρ_{subsystem}"
            else:
                out_s += f"ρ_{subsystem} ⊗ "
        return out_s

    def __repr__(self) -> str:
        return self.__str__()

    def __getitem__(self, key: int) -> State:
        return self.states[key]

    def swap(self, sub_sys_swap: List[int]):
        for state in self._states:
            state.swap(sub_sys_swap)

    @property
    def subsystems(self):
        return self._subsystems

    @property
    def probs(self) -> List[float]:
        return self._probs

    @property
    def states(self) -> List[State]:
        return self._states

    @property
    def density_matrices(self):
        states: List[np.ndarray] = []
        for state in self._states:
            states.append(state.value)
        return states

    @staticmethod
    def _prepare_states(states: List[np.ndarray]) -> Optional[List[np.ndarray]]:
        # Assume at least one state is provided.
        if states is None or states == []:
            raise ValueError("InvalidStates: There must be at least one state provided.")
        return states

    def _prepare_probs(self, probs: List[float]) -> Optional[List[float]]:
        # If probability vector is not explicitly provided, assume the ensemble
        # has a uniform distribution.
        if probs is None:
            probs = [1 / len(self)] * len(self)

        # Probability vector must sum to one to be valid.
        if not np.isclose(sum(probs), 1):
            raise ValueError("InvalidProbabilities: Probabilities must sum to 1.")

        return probs
