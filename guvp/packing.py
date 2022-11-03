from __future__ import annotations
from collections import deque
from dataclasses import (dataclass, field, InitVar)
import enum
import random
from typing import (List, Tuple, Optional)
import weakref

import bmesh                  # type: ignore
import numpy as np            # type: ignore

from guvp import (continuous, discrete)


CollisionResult = enum.Enum('CollisionResult', 'NO YES OUT_OF_BOUNDS')


class Solution:
    MAX_GROW_COUNT = 2
    MAX_TRIES_PER_OFFER = 100    # Hard limit for tries

    """Store a set of placements."""
    def __init__(
            self,
            initial_size: int,
            random_seed: int
    ) -> None:
        self.islands: List[IslandPlacement] = []
        self._initial_size = initial_size
        self._rng = random.Random(random_seed)
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

    def offer(self, island: continuous.Island) -> bool:
        tries_left: int = min(
            self._mask.width * self._mask.height / 3,
            self.MAX_TRIES_PER_OFFER
        )
        is_successful: bool = False
        x: int = 0
        y: int = 0
        while tries_left > 0 and is_successful is False:
            tries_left -= 1
            island_placement = IslandPlacement(
                offset=discrete.CellCoord(x, y),
                island=island
            )
            collision_result = self._check_collision(island_placement)
            if collision_result is CollisionResult.NO:
                self.islands.append(island_placement)
                self._write_island_to_mask(island_placement)
                self._update_utilized_area(island_placement)
                is_successful = True
            elif collision_result is CollisionResult.YES:
                # nothing to do here, let the solver decide.
                if x + island.mask.width > self._mask.width:
                    y += 1
                else:
                    x += 1
            elif collision_result is CollisionResult.OUT_OF_BOUNDS:
                # expand solution or try another offset
                print("growing solution {!r}".format(self))
                self._grow()
            else:
                raise NotImplementedError(
                    "Unhandled collision result case '{0!r}'".format(
                        collision_result
                    )
                )
        return is_successful

    @property
    def scaling_factor(self) -> float:
        return float(self._initial_size) / max(self._utilized_area[0],
                                               self._utilized_area[1])

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

    def _grow(self) -> None:
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
    SEED_MAX = 2 ** 31 - 1
    OFFERS_MULTIPLIER = 5.0

    def __init__(
            self,
            initial_size: int,
            islands: List[continuous.Island],
            random_seed: Optional[int] = None
    ) -> None:
        self._initial_size = initial_size
        self._islands = islands
        self._rng = random.Random(random_seed)
        self._winner: Optional[Solution] = None

    @property
    def fitness(self) -> float:
        if self._winner is None:
            return 0.0
        else:
            return self._winner.fitness

    def run(self) -> None:
        offers_left: int = int(len(self._islands) * self.OFFERS_MULTIPLIER)
        solution = Solution(self._initial_size,
                            self._rng.randint(0, self.SEED_MAX))
        islands = deque(self._islands)
        while len(islands) > 0 and offers_left > 0:
            offers_left -= 1
            island = islands.popleft()
            # offer returns False when the island
            # is not accepted to the solution
            if not solution.offer(island):
                islands.append(island)
        # FIXME: It's possible that the solution is given up and not
        #        all islands are placed.  Fail if that's the case.
        self._winner = solution
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
