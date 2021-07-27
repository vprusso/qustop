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

"""Quantum state object."""
from __future__ import annotations
from typing import Optional

import numpy as np

from toqito.perms import swap
from toqito.matrix_props import is_density


class State:
    """A :code:`State` object representing a quantum state."""

    def __init__(self, state: np.ndarray, dims: list[int]) -> None:
        """Initializes a quantum state.

        Args:
            state: A `numpy` matrix representing a quantum state.
            dims: A list of integers representing the dimensions of the subsystems of the state.
        """
        self._state = self._prepare_state(state)
        self._dims = self._prepare_dims(dims)
        self._systems = list(range(1, len(self._dims) + 1))

    def __eq__(self, other: State) -> bool:
        if isinstance(other, State):
            return (
                np.allclose(self.value, other.value)
                and self.dims == other.dims
            )

    def __str__(self) -> str:
        labels, spaces = "", ""
        for i in range(len(self._systems)):
            party = "A" if self._systems[i] % 2 != 0 else "B"
            if i == len(self._systems) - 1:
                spaces += f"ℂ^{self._dims[i]}"
                labels += f"{party}_{self._systems[i]}"
            else:
                spaces += f"ℂ^{self._dims[i]} ⊗ "
                labels += f"{party}_{self._systems[i]} ⊗ "

        out_s = (
            f"State: \n "
            f"dimensions = {self._dims}, \n "
            f"spaces = {spaces}, \n "
            f"labels = {labels}, \n "
            f"pure = {self.is_pure}, \n "
            f"shape = {self.shape}, \n"
        )
        return out_s

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def shape(self) -> tuple[int, int]:
        return self._state.shape

    @property
    def dims(self) -> list[int]:
        return self._dims

    @property
    def systems(self) -> list[int]:
        return self._systems

    @property
    def alice_systems(self) -> list[int]:
        return [i for i in self._systems if i % 2 != 0]

    @property
    def bob_systems(self) -> list[int]:
        return [i for i in self._systems if i % 2 == 0]

    @property
    def value(self) -> np.ndarray:
        return self._state

    @property
    def is_pure(self) -> bool:
        eigs, _ = np.linalg.eig(self._state)
        return np.allclose(np.max(np.diag(eigs)), 1)

    @staticmethod
    def _prepare_state(state: np.ndarray) -> Optional[np.ndarray]:
        """Returns the validated quantum state.

        Args:
            state: A `numpy` matrix representing a quantum state.

        Raises:
            ValueError:
                * If `state` is not a valid density matrix.
        """
        # If `state` is provided as a vector, transform it into density a matrix.
        if state.shape[1] == 1:
            state = state * state.conj().T

        if not is_density(state):
            raise ValueError(
                "All states must be density operators (PSD and trace equal to 1)."
            )

        return state

    def _prepare_dims(self, dims: list[int]) -> Optional[list[int]]:
        """Returns the validated list of dimensions to be used for the quantum state.

        Args:
            dims: A vector of integers representing the dimensions of the subsystems of the state.

        Raises:
            ValueError:
                * If the product of the elements of `dims` is not equal to the dim of the state.
        """
        dim = np.prod(dims)
        if self.shape[0] != dim or self.shape[1] != dim:
            raise ValueError(
                f"The product of `dims` should be equal to {self.shape[0]} and {self.shape[1]}."
            )
        return dims

    def kron(self, r_state: State) -> State:
        """Performs the Kronecker (tensor) product between two states.

        Args:
            r_state: The state on the right-side of the tensor product.
        """
        new_state = np.kron(self._state, r_state.value)
        new_dims = self._dims + r_state.dims
        return State(new_state, new_dims)

    def swap(self, sub_sys_swap: list[int]) -> None:
        """Performs a swap between two subsystems of the state.

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

        if (
            sub_sys_swap[0] > len(self._state) + 1
            or sub_sys_swap[1] > len(self._state) + 1
        ):
            raise ValueError(
                f"Cannot swap {sub_sys_swap[0]} with {sub_sys_swap[1]} as one or both "
                f"of these values exceed the number of systems in the ensemble."
            )

        self._state = swap(self._state, sub_sys_swap, self._dims)

        # Once the swap operation is performed, ensure the information is
        # propagated to the associated state property class variables.
        idx_1 = self._systems.index(sub_sys_swap[0])
        idx_2 = self._systems.index(sub_sys_swap[1])

        self._systems[idx_1], self._systems[idx_2] = (
            self._systems[idx_2],
            self._systems[idx_1],
        )
        self._dims[idx_1], self._dims[idx_2] = (
            self._dims[idx_2],
            self._dims[idx_1],
        )
