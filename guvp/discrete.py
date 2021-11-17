from __future__ import annotations
from typing import (
    Any,
    Iterator,
    NamedTuple,
)

import numpy as np            # type: ignore


class CellCoord(NamedTuple):
    x: int
    y: int

    @classmethod
    def zero(cls):
        return cls(0, 0)


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

    def draw_str(self) -> None:
        MAX_DRAW_DIMENSION: int = 60
        if self.width <= MAX_DRAW_DIMENSION and \
           self.height <= MAX_DRAW_DIMENSION:
            for row in range(self.height):
                print('<' if row == 0 else ' ', end='')
                for column in range(self.width):
                    print('#' if self[CellCoord(column, row)] else '.', end='')
                print('>' if row == self.height - 1 else '')
        else:
            print("Grid is too large, cannot draw to console.")

    @classmethod
    def empty(cls, width: int, height: int):
        return cls(np.zeros((height, width), dtype=np.bool_))


class Size(NamedTuple):
    width: int
    height: int

    def __repr__(self) -> str:
        return "<Size(width={0}, height={1})>".format(self.width, self.height)

    def __str__(self) -> str:
        return "{0}x{1}".format(self.width, self.height)

    @classmethod
    def zero(cls):
        return cls(0, 0)
