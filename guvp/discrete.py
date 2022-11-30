# <pep8-80 compliant>

# grid-uv-packer packs irregularly shaped UV islands efficiently.
# Copyright (C) 2022  Atamert Ölçgen
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


from __future__ import annotations
from typing import (
    Any,
    Iterator,
    NamedTuple,
    Tuple,
)

import numpy as np            # type: ignore


class CellCoord(NamedTuple):
    x: int
    y: int

    @classmethod
    def zero(cls):
        return cls(0, 0)

    def offset(self, dx: int, dy: int) -> CellCoord:
        return CellCoord(self.x + dx, self.y + dy)


class Grid:
    def __init__(self, cells: np.ndarray):
        assert(cells.ndim == 2)
        (height, width) = cells.shape
        self.cells = cells
        self.width = width
        self.height = height

    def __and__(self, other: Any) -> bool:
        if not isinstance(other, Grid):
            return NotImplemented
        if self.width != other.width or self.height != other.height:
            raise ValueError("Grids are not same sized.")
        return self.cells & other.cells

    def __contains__(self, key: CellCoord):
        (column, row) = key
        return 0 <= column < self.width and 0 <= row < self.height

    def __delitem__(self, key: CellCoord):
        raise TypeError("'Grid' object does not support item deletion.")

    def __getitem__(self, key: CellCoord) -> bool:
        (column, row) = key
        return self.cells[(row, column)]

    def __iter__(self) -> Iterator[CellCoord]:
        return iter([CellCoord(x, y)
                     for x in range(self.width)
                     for y in range(self.height)])

    def __len__(self) -> int:
        return self.width * self.height

    def __repr__(self) -> str:
        return "<Grid(width={0}, height={1})>".format(
            self.width,
            self.height
        )

    def __setitem__(self, key: CellCoord, value: bool) -> None:
        (column, row) = key
        self.cells[(row, column)] = value

    def copy(self, margins: Tuple[int, int, int, int] = (0, 0, 0, 0)) -> Grid:
        (before_x, before_y, after_x, after_y) = margins
        cells = np.pad(self.cells,
                       pad_width=[(before_y, after_y), (before_x, after_x)],
                       constant_values=(False,))
        return Grid(cells)

    def dilate(self, size: int) -> Grid:
        # TODO: This is applying a box kernel.  It's fast but wasteful.
        #       Replace this with a circular kernel.
        if size == 1:
            return Grid(cells=self.cells.copy())
        kernel = [True] * (2 * size + 1)
        dilated = np.apply_along_axis(
            np.convolve,
            1,                         # along x
            self.cells,
            kernel,
            'same'
        )
        dilated = np.apply_along_axis(
            np.convolve,
            0,                         # along y
            dilated,
            kernel,
            'same'
        )
        return Grid(cells=dilated)

    def draw_str(self) -> None:
        MAX_DRAW_DIMENSION: int = 100
        if self.width <= MAX_DRAW_DIMENSION and \
           self.height <= MAX_DRAW_DIMENSION:
            for row in range(self.height - 1, 0, -1):
                print('<' if row == 0 else ' ', end='')
                for column in range(self.width):
                    print('#' if self[CellCoord(column, row)] else '.', end='')
                print('>' if row == self.height - 1 else '')
        else:
            print("Grid is too large, cannot draw to console.")

    @classmethod
    def empty(cls, width: int, height: int):
        return cls(np.zeros((height, width), dtype=np.bool_))
