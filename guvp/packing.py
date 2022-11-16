from __future__ import annotations
from collections import deque
from dataclasses import (dataclass, field, InitVar)
import enum
import math
import random
from typing import (List, Tuple, Optional)
import weakref

import bmesh                  # type: ignore
import numpy as np            # type: ignore

from guvp import (continuous, discrete)


CollisionResult = enum.Enum('CollisionResult', 'NO YES OUT_OF_BOUNDS')


class Solution:
    GROW_AREA_CHANCE = 0.5          # Grow chance if utilized area is too big.
    GROW_AREA_RATIO = 0.85          # Grow if utilized area is larger than this.
    GROW_BASE_CHANCE = 0.15         # Base grow chance without modifiers
    GROW_REGULARITY_CHANCE = -0.25  # Grow change if the utilized area is closer to a rectangle.
    GROW_REGULARITY_RATIO = 0.667   # What is the threshold to consider a rectangle-like fill.
    MAX_GROW_COUNT = 2
    MAX_PLACEMENT_RETRIES = 2500    # Hard limit for tries
    SEARCH_START_RESET_CHANCE = 0.333

    """Store a set of placements."""
    def __init__(
            self,
            initial_size: int,
            random_seed: int
    ) -> None:
        self.islands: List[IslandPlacement] = []
        self._initial_size = initial_size
        self._rng = random.Random(random_seed)
        self._search_start = discrete.CellCoord.zero()
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
        return float(ones) / (size * size) if size > 0 else 0.0

    def run(self, islands_to_place: List[continuous.Island]) -> bool:
        islands_remaining = deque(islands_to_place)
        del islands_to_place
        self._rng.shuffle(islands_remaining)
        island_retries_left: int = len(islands_remaining)
        while len(islands_remaining) > 0 and island_retries_left > 0:
            # self._mask.draw_str()
            island_retries_left -= 1
            placement_retries_left: int = self.MAX_PLACEMENT_RETRIES
            island: continuous.Island = islands_remaining.popleft()
            search_cell: discrete.CellCoord = self._search_start
            island_placement: Optional[IslandPlacement] = None
            while placement_retries_left > 0 and island_placement is None:
                placement_retries_left -= 1
                island_placement = IslandPlacement(
                    offset=search_cell,
                    island=island
                )
                collision_result = self._check_collision(island_placement)
                if collision_result is CollisionResult.NO:
                    pass
                elif collision_result is CollisionResult.YES:
                    island_placement = None
                elif collision_result is CollisionResult.OUT_OF_BOUNDS:
                    island_placement = None
                else:
                    raise NotImplementedError(
                        "Unhandled collision result case '{0!r}'".format(
                            collision_result
                        )
                    )
                if island_placement is None:
                    search_cell = self._advance_search_cell(search_cell)
            if island_placement is None:
                islands_remaining.append(island)
                if self._rng.random() <= self._calculate_grow_chance():
                    self._grow()
            else:
                # Note that `island` was removed from `islands_remaining`.
                island_retries_left = len(islands_remaining)
                self._search_start = search_cell
                self.islands.append(island_placement)
                self._write_island_to_mask(island_placement)
                self._update_utilized_area(island_placement)
            if self._rng.random() <= self.SEARCH_START_RESET_CHANCE:
                self._search_start = discrete.CellCoord.zero()
        print("Solution {} -- remaining islands # = {}".format(id(self), len(islands_remaining)))
        # Run is successful if all the islands are placed.
        return len(islands_remaining) == 0

    @property
    def scaling_factor(self) -> float:
        return float(self._initial_size) / max(self._utilized_area[0],
                                               self._utilized_area[1])

    def _advance_search_cell(self, search_cell: discrete.CellCoord) -> discrete.CellCoord:
        x: float = float(search_cell[0]) / (self._mask.width - 1)
        y: float = float(search_cell[1]) / (self._mask.height - 1)
        up_chance: float = math.sin(x ** 3 * math.pi * 0.5) \
                         * math.cos(y ** 2 * math.pi * 0.5)
        new_search_cell = search_cell.offset(1, 0)
        if self._rng.random() <= up_chance or new_search_cell not in self._mask:
            new_search_cell = discrete.CellCoord(x=0, y=search_cell.y+1)
        return new_search_cell


    def _calculate_grow_chance(self) -> float:
        grow_chance: float = self.GROW_BASE_CHANCE
        # Grow if utilized area gets too large compared to current mask size.
        if float(self._utilized_area[0]) / self._mask.width > self.GROW_AREA_RATIO or \
           float(self._utilized_area[1]) / self._mask.height > self.GROW_AREA_RATIO:
            grow_chance += self.GROW_AREA_CHANCE
        # Grow only if utilized area is rectangular.
        ratio: float = float(self._utilized_area[0]) / float(self._utilized_area[1])
        if ratio > 1.0:
            ratio = 1.0 / ratio
        if ratio <= self.GROW_REGULARITY_RATIO:
            grow_chance += self.GROW_REGULARITY_CHANCE
        del ratio
        return max(0.0, min(grow_chance, 1.0))

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
            print("Cannot grow more.")
            return None
        print("Growing")
        # Create a mask with the new size & copy current mask's contents.
        new_size = self._mask.width * 2
        new_mask = self._mask.copy((0,
                                    0,
                                    new_size - self._mask.width,
                                    new_size - self._mask.height))
        self._mask = new_mask
        # No need to update utilized area.
        return None

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
        solution = Solution(self._initial_size,
                            self._rng.randint(0, self.SEED_MAX))
        result = solution.run(list(self._islands))
        # FIXME: It's possible that the solution is given up and not
        #        all islands are placed.  Fail if that's the case.
        if result:
            self._winner = solution

    def write(self, bm: bmesh.types.BMesh) -> None:
        if self._winner is None:
            raise RuntimeError("write is called before run.")
        for ip in self._winner.islands:
            scaling_factor = self._winner.scaling_factor
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
