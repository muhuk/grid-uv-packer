from __future__ import annotations
from dataclasses import dataclass
import enum
from typing import List

import bmesh                  # type: ignore

from guvp import (continuous, discrete)


CollisionResult = enum.Enum('CollisionResult', 'NO YES OUT_OF_BOUNDS')


class GridPacker:
    def __init__(
            self,
            initial_size: int,
            islands: List[continuous.Island]
    ) -> None:
        self._utilized_area = discrete.Size.zero()
        self._mask = discrete.Grid.empty(
            width=initial_size,
            height=initial_size
        )
        self._islands: List[IslandPlacement] = [
            IslandPlacement(offset=discrete.CellCoord.zero(), island=island)
            for island in islands
        ]

    @property
    def fitness(self) -> float:
        # TODO: Calculate actual fitness.
        #       Return value between 0.0 - 1.0
        return 0.0

    def run(self) -> None:
        filled_x: int = 0
        for ip in self._islands:
            # using current offset try collision: `mask & island`
            # if collision:
            #    increment offset
            # else:
            #    give new offset to island
            #    update the mask (stamp island)
            #    update utilized area
            pass
        for ip in self._islands:
            ip.offset = discrete.CellCoord(filled_x, 0)
            filled_x += ip.island.mask.width
            (uw, uh) = self._utilized_area
            self.utilized_area = discrete.Size(
                width=filled_x,
                height=max(ip.island.mask.height, uh)
            )
        print("Utilized area: {0}".format(self.utilized_area))

    def write(self, bm: bmesh.types.BMesh) -> None:
        for ip in self._islands:
            ip.write_uvs(bm)

    def _check_collision(self, ip: IslandPlacement) -> CollisionResult:
        if ip.offset.x + ip.island.mask.width > self._mask.width:
            return CollisionResult.OUT_OF_BOUNDS
        if ip.offset.y + ip.island.mask.height > self._mask.height:
            return CollisionResult.OUT_OF_BOUNDS
        raise NotImplementedError()


@dataclass
class IslandPlacement:
    offset: discrete.CellCoord
    # rotation: Enum
    #
    #   see: https://docs.python.org/3/library/enum.html
    island: continuous.Island

    def write_uvs(self, bm: bmesh.types.BMesh) -> None:
        self.island.write_uvs(bm, self.offset)
