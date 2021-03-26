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

"""Ensemble of quantum states."""
from typing import Optional

import numpy as np

from qustop.core.state import State


class Ensemble:
    """A set of `State` objects denoting quantum states where each element has an associated
    probability of being selected from the set.
    """

    def __init__(
        self, states: list[State], probs: Optional[list[float]] = None
    ) -> None:
        """Initializes an Ensemble.

        Args:
            states: A collection of State objects representing the quantum states for the ensemble.
            probs: A vector of associated probabilities for the quantum states of the ensemble.

        Raises:
            TypeError:
                * If all elements of `state` are not instances of `State`.
        """
        if not all(isinstance(state, State) for state in states):
            raise TypeError("All elements of `state` must be of type `State`.")

        self._states = self._prepare_states(states)
        self._probs = self._prepare_probs(probs)

    def __len__(self) -> int:
        return len(self._states)

    def __str__(self) -> str:
        out_s = f"Ensemble: num_states = {len(self)}\n"
        for i, _ in enumerate(self._states):
            if i == len(self._states) - 1:
                out_s += f"ρ_{i}"
            else:
                out_s += f"ρ_{i} ⊗ "
        return out_s

    def __repr__(self) -> str:
        return self.__str__()

    def __getitem__(self, key: int) -> State:
        return self._states[key]

    @property
    def probs(self) -> list[float]:
        return self._probs

    @property
    def states(self) -> list[State]:
        return self._states

    @property
    def systems(self) -> list[int]:
        return self._states[0].systems

    @property
    def dims(self) -> list[int]:
        return self._states[0].dims

    @property
    def shape(self) -> tuple[int, int]:
        return self._states[0].shape

    @property
    def density_matrices(self) -> list[np.ndarray]:
        return [state.value for state in self._states]

    def swap(self, sub_sys_swap: list[int]) -> None:
        """Performs a swap between two subsystems of each state in the ensemble.

        Args:
            sub_sys_swap: A list containing two elements representing the spaces to swap.

        Raises:
            ValueError:
                * If length of `sub_sys_swap` is not equal to 2.
                * If either element of `sub_sys_swap` is greater than the number of elements in
                  the ensemble.
        """

        if len(sub_sys_swap) != 2:
            raise ValueError(
                f"The length of the swap vector is {len(sub_sys_swap)}, but must be "
                f"of length 2."
            )

        if sub_sys_swap[0] > len(self) + 1 or sub_sys_swap[1] > len(self) + 1:
            raise ValueError(
                f"Cannot swap {sub_sys_swap[0]} with {sub_sys_swap[1]} as one or both "
                f"of these values exceed the number of systems in the ensemble."
            )

        # Perform the swap operation on each state in the ensemble.
        [state.swap(sub_sys_swap) for state in self._states]

    @staticmethod
    def _prepare_states(states: list[State]) -> Optional[list[State]]:
        """Returns the validated list of quantum states to be used for Ensemble.

        Args:
            states: A collection of State objects denoting valid quantum states.

        Raises:
            ValueError:
                * If `states` is empty.
                * If `states` have different dimensions.
        """
        # Assume at least one state is provided.
        if states is None or states == []:
            raise ValueError("An ensemble must contain at least one state.")

        # Each state in the ensemble must have the same dimension.
        dims = states[0].shape
        for state in states:
            if state.shape != dims:
                raise ValueError(
                    "Each state in the ensemble must be of equal dimension."
                )

        return states

    def _prepare_probs(self, probs: list[float]) -> Optional[list[float]]:
        """Returns the validated list of probabilities to be used for Ensemble.

        Args:
            probs: A vector of probabilities denoting each probability per quantum state.

        Raises:
            ValueError:
                * If the length of `probs` is not equal to the number of quantum states.
                * If the sum of the `probs` vector does not equal 1.
        """
        # If probability vector is not explicitly provided, assume the ensemble
        # has a uniform distribution.
        if probs is None:
            probs = [1 / len(self)] * len(self)

        # The probability vector must be of the same length of the number of
        # states in the ensemble.
        if len(probs) != len(self):
            raise ValueError(
                f"The number of probabilities in vector ({len(probs)}) must be the same as the "
                f"number of states in the ensemble ({len(self)})."
            )

        # Probability vector must sum to one to be valid.
        if not np.isclose(sum(probs), 1):
            raise ValueError(
                f"The probability vector must sum to 1, but it currently sums to "
                f"{sum(probs)}."
            )

        return probs
