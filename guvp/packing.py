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
from collections import deque
from concurrent import futures
from dataclasses import (dataclass, field, InitVar)
import enum
import itertools
import math
import random
import statistics
from typing import (List, Tuple, Optional)

import bmesh                  # type: ignore
import numpy as np            # type: ignore

from guvp import (continuous, debug, discrete)


CollisionResult = enum.Enum('CollisionResult', 'NO YES OUT_OF_BOUNDS')


class Solution:
    GROW_AREA_CHANCE = 0.5           # Grow chance if utilized area is too big.
    GROW_AREA_RATIO = 0.85           # Grow if utilized area is larger than this.
    GROW_BASE_CHANCE = 0.15          # Base grow chance without modifiers
    GROW_REGULARITY_CHANCE = -0.25   # Grow change if the utilized area is closer to a rectangle.
    GROW_REGULARITY_RATIO = 0.667    # What is the threshold to consider a rectangle-like fill.
    MAX_GROW_COUNT = 2
    MAX_PLACEMENT_RETRIES = 100_000  # Hard limit for tries
    SEARCH_START_RESET_CHANCE = 0.333

    """Store a set of placements."""
    def __init__(
            self,
            initial_size: int,
            random_seed: int
    ) -> None:
        self.islands: List[IslandPlacement] = []
        self.random_seed = random_seed
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

    def pack(self, islands_to_place: List[continuous.Island]) -> bool:
        islands_remaining = deque(islands_to_place)
        del islands_to_place
        self._rng.shuffle(islands_remaining)
        # We need to reset search cell here since pack may be called several times.
        self._search_start = discrete.CellCoord.zero()
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
        # Run is successful if all the islands are placed.
        return len(islands_remaining) == 0

    def pack_grouped(self, *grouped_islands: List[continuous.Island]) -> bool:
        result: bool = True
        group_idx: int = 0
        while result and group_idx < len(grouped_islands):
            if not self.pack(grouped_islands[group_idx]):
                result = False
            group_idx += 1
        return result

    @property
    def scaling_factor(self) -> float:
        return float(self._initial_size) / max(self._utilized_area[0],
                                               self._utilized_area[1])

    def _advance_search_cell(
            self,
            search_cell: discrete.CellCoord
    ) -> discrete.CellCoord:
        x: float = float(search_cell[0]) / (self._mask.width - 1)
        y: float = float(search_cell[1]) / (self._mask.height - 1)
        up_chance: float = \
            math.sin(x ** 3 * math.pi * 0.5) \
            * math.cos(y ** 2 * math.pi * 0.5)
        new_search_cell = search_cell.offset(1, 0)
        if self._rng.random() <= up_chance \
           or new_search_cell not in self._mask:
            new_search_cell = discrete.CellCoord(x=0, y=search_cell.y + 1)
        return new_search_cell

    def _calculate_grow_chance(self) -> float:
        grow_chance: float = self.GROW_BASE_CHANCE
        utilized_x: float = float(self._utilized_area[0])
        utilized_y: float = float(self._utilized_area[1])
        # Grow if utilized area gets too large compared to current mask size.
        if utilized_x / self._mask.width > self.GROW_AREA_RATIO or \
           utilized_y / self._mask.height > self.GROW_AREA_RATIO:
            grow_chance += self.GROW_AREA_CHANCE
        # Grow only if utilized area is rectangular.
        ratio: float = utilized_x / utilized_y \
            if utilized_x != 0.0 and utilized_y != 0.0 else 1.0
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
            if debug.is_debug():
                print("Cannot grow more.")
            return None
        if debug.is_debug():
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
        (large_islands, small_islands) = self._categorize_islands()
        executor: futures.Executor = futures.ProcessPoolExecutor()
        # TODO: Handle exceptions raised in workers.
        # TODO: Add execution timeout.
        n: int = 10
        results = executor.map(
            self._run_solution,
            itertools.repeat(self._initial_size),
            [self._rng.randint(0, self.SEED_MAX) for _ in range(n)],
            itertools.repeat(large_islands),
            itertools.repeat(small_islands)
        )
        executor.shutdown()
        highest_fitness: float = 0.0
        for (result, solution) in results:
            if result and solution.fitness > highest_fitness:
                highest_fitness = solution.fitness
                self._winner = solution

    def write(self, bm: bmesh.types.BMesh) -> None:
        if self._winner is None:
            raise RuntimeError("write is called before run.")
        for ip in self._winner.islands:
            scaling_factor = self._winner.scaling_factor
            ip.write_uvs(bm, scaling_factor)

    def _categorize_islands(self) -> Tuple[List[continuous.Island], List[continuous.Island]]:
        island_sizes: List[int] = sorted([len(i.mask) for i in self._islands])
        median_size: int = statistics.median_low(island_sizes)
        large_islands = [i for i in self._islands if len(i.mask) > median_size]
        small_islands = [i for i in self._islands if len(i.mask) <= median_size]
        return (large_islands, small_islands)

    @staticmethod
    def _run_solution(initial_size: int, seed: int, *grouped_islands: List[continuous.Island]) -> Tuple[bool, Solution]:
        solution: Solution = Solution(initial_size, seed)
        return solution.pack_grouped(*grouped_islands), solution


@dataclass(frozen=True)
class IslandPlacement:
    offset: discrete.CellCoord
    # TODO: Add rotation support.
    # rotation: Enum
    #
    #   see: https://docs.python.org/3/library/enum.html
    island: continuous.Island

    def get_bounds(self) -> Tuple[int, int, int, int]:
        return (self.offset.x,
                self.offset.y,
                self.offset.x + self.island.mask.width,
                self.offset.y + self.island.mask.height)

    def get_mask(self, bounds: Tuple[int, int]) -> discrete.Grid:
        return self.island.mask.copy(
            (self.offset.x,
             self.offset.y,
             bounds[0] - (self.offset.x + self.island.mask.width),
             bounds[1] - (self.offset.y + self.island.mask.height))
        )

    def write_uvs(self, bm: bmesh.types.BMesh, scaling_factor: float) -> None:
        self.island.write_uvs(bm, self.offset, scaling_factor)
