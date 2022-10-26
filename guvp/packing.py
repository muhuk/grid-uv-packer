from __future__ import annotations
from collections import deque
from dataclasses import (dataclass, field, InitVar)
import enum
from typing import (List, Tuple, Optional)
import weakref

import bmesh                  # type: ignore
import numpy as np            # type: ignore

from guvp import (continuous, discrete)


CollisionResult = enum.Enum('CollisionResult', 'NO YES OUT_OF_BOUNDS')


class Solution:
    MAX_GROW_COUNT = 2

    """Store a set of placements."""
    def __init__(self, initial_size: int) -> None:
        self.islands: List[IslandPlacement] = []
        self._initial_size = initial_size
        self._utilized_area = (0, 0)
        self._mask = discrete.Grid.empty(
            width=initial_size,
            height=initial_size
        )

    @property
    def fitness(self) -> float:
        """Calculate fitness.  Return value between 0.0 - 1.0."""
        size = max(*self._utilized_area)
        # Don't forget to add 1 to the end index.
        ones = np.count_nonzero(self._mask.cells[0:size + 1, 0:size + 1])
        # TODO: use a function on Grid instead of accessing cells directly.
        return float(ones) / (size * size)

    def grow(self) -> None:
        # Limit growing.
        if self._mask.width >= self._initial_size * (2 ** self.MAX_GROW_COUNT):
            raise RuntimeError("solution cannot grow more")
        # Create a mask with the new size & copy current mask's contents.
        new_size = self._mask.width * 2
        new_mask = self._mask.copy((0,
                                    0,
                                    new_size - self._mask.width,
                                    new_size - self._mask.height))
        self._mask = new_mask
        # No need to update utilized area.
        pass

    def offer(self, island: IslandPlacement) -> CollisionResult:
        collision_result = self._check_collision(island)
        if collision_result is CollisionResult.NO:
            self.islands.append(island)
            self._write_island_to_mask(island)
            self._update_utilized_area(island)
        elif collision_result is CollisionResult.YES:
            # nothing to do here, let the solver decide.
            pass
        elif collision_result is CollisionResult.OUT_OF_BOUNDS:
            # nothing to do here, let the solver decide.
            pass
        else:
            raise NotImplementedError(
                "Unhandled collision result case '{0!r}'".format(
                    collision_result
                )
            )
        return collision_result

    @property
    def scaling_factor(self) -> float:
        return float(max(self._utilized_area[0],
                         self._utilized_area[1])) / self._mask.width

    @property
    def utilized_area(self) -> Tuple[int, int]:
        return self._utilized_area

    def _check_collision(self, ip: IslandPlacement) -> CollisionResult:
        island_bounds = ip.get_bounds()
        if island_bounds[0] < 0:
            return CollisionResult.OUT_OF_BOUNDS
        if island_bounds[1] < 0:
            return CollisionResult.OUT_OF_BOUNDS
        if island_bounds[2] > self._mask.width:
            return CollisionResult.OUT_OF_BOUNDS
        if island_bounds[3] > self._mask.height:
            return CollisionResult.OUT_OF_BOUNDS

        island_mask = ip.get_mask((self._mask.width, self._mask.height))
        if (self._mask & island_mask).any():
            return CollisionResult.YES
        else:
            return CollisionResult.NO

    def _update_utilized_area(self, ip: IslandPlacement) -> None:
        (_, _, a, b) = ip.get_bounds()
        self._utilized_area = (
            max(self._utilized_area[0], a),
            max(self._utilized_area[1], b)
        )

    def _write_island_to_mask(self, ip: IslandPlacement) -> None:
        self._mask.cells |= ip.get_mask((self._mask.width,
                                         self._mask.height)).cells


class GridPacker:
    def __init__(
            self,
            initial_size: int,
            islands: List[continuous.Island]
    ) -> None:
        self._initial_size = initial_size
        self._islands = islands
        self._winner: Optional[Solution] = None

    @property
    def fitness(self) -> float:
        if self._winner is None:
            return 0.0
        else:
            return self._winner.fitness

    def run(self) -> None:
        solution = Solution(self._initial_size)
        islands = deque(self._islands)
        while len(islands) > 0:
            island = islands.popleft()
            (x, _) = solution.utilized_area
            ip = IslandPlacement(
                offset=discrete.CellCoord(x, 0),
                island=island
            )
            collision_result = solution.offer(ip)
            print("collision result is {}".format(collision_result))
            if collision_result is CollisionResult.NO:
                # no collision, continue from next island.
                pass
            elif collision_result is CollisionResult.YES:
                # put island back into the bag, we will try again.
                islands.append(island)
                pass
            elif collision_result is CollisionResult.OUT_OF_BOUNDS:
                # expand solution or try another offset
                print("growing solution {}".format(repr(solution)))
                solution.grow()
                # put island back into the bag, we will try again.
                islands.append(island)
                pass
            else:
                raise NotImplementedError(
                    "Unhandled collision result case '{0!r}'".format(
                        collision_result
                    )
                )
        self._winner = solution
        print("Utilized area: {0}".format(solution._utilized_area))
        print("Fitness: {0}".format(solution.fitness))

    def write(self, bm: bmesh.types.BMesh) -> None:
        if self._winner is None:
            raise RuntimeError("write is called before run.")
        for ip in self._winner.islands:
            scaling_factor = self._winner.scaling_factor
            print("scaling_factor is {}".format(scaling_factor))
            ip.write_uvs(bm, scaling_factor)


@dataclass(frozen=True)
class IslandPlacement:
    offset: discrete.CellCoord
    # TODO: Add rotation support.
    # rotation: Enum
    #
    #   see: https://docs.python.org/3/library/enum.html
    island: InitVar[continuous.Island]
    _island_ref: weakref.ReferenceType[continuous.Island] = field(init=False)

    def __post_init__(self, island) -> None:
        assert(island is not None)
        object.__setattr__(self, "_island_ref", weakref.ref(island))

    def get_bounds(self) -> Tuple[int, int, int, int]:
        return (self.offset.x,
                self.offset.y,
                self.offset.x + self._island.mask.width,
                self.offset.y + self._island.mask.height)

    def get_mask(self, bounds: Tuple[int, int]) -> discrete.Grid:
        return self._island.mask.copy(
            (self.offset.x,
             self.offset.y,
             bounds[0] - (self.offset.x + self._island.mask.width),
             bounds[1] - (self.offset.y + self._island.mask.height))
        )

    def write_uvs(self, bm: bmesh.types.BMesh, scaling_factor: float) -> None:
        self._island.write_uvs(bm, self.offset, scaling_factor)

    @property
    def _island(self) -> continuous.Island:
        island = self._island_ref()
        if island is None:
            raise RuntimeError("island cannot be dereferenced.")
        return island
