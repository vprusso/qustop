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

class State:
    def __init__(self, state: np.ndarray, dims: List[int]) -> None:
        self._state = state
        self._dims = dims
        self._partitions = list(range(1, len(self._dims) + 1))

    def __str__(self) -> str:
        labels = ""
        for i, sub_sys in enumerate(self._partitions):
            party = "A" if sub_sys % 2 != 0 else "B"
            if i == len(self._partitions)-1:
                labels += f"{party}_{sub_sys}"
            else:
                labels += f"{party}_{sub_sys} ⊗ "
        out_s = f"State: \n dimensions = {self._dims}, \n partitions = {labels}\n shape = {self.shape}"
        return out_s

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def shape(self):
        return self._state.shape

    @property
    def dims(self):
        return self._dims

    @property
    def partitions(self):
        return self._partitions

    @property
    def value(self):
        return self._state

    def swap(self, sub_sys_swap: List[int]):
        self._state = swap(self._state, sub_sys_swap, self._dims)
        idx_1 = self._partitions.index(sub_sys_swap[0])
        idx_2 = self._partitions.index(sub_sys_swap[1])

        self._partitions[idx_1], self._partitions[idx_2] = self._partitions[idx_2], self._partitions[idx_1]
        self._dims[idx_1], self._dims[idx_2] = self._dims[idx_2], self._dims[idx_1]

    @staticmethod
    def _validate_state(state):
        # If `state` is provided as a vector. Transform it into density a
        # matrix.
        _, dim_y = state.shape
        if dim_y == 1:
            state = state * state.conj().T

        if not is_density(state):
            raise ValueError("InvalidStates: All states must be density operators.")

        return states