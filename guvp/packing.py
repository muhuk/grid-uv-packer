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
from dataclasses import dataclass
import enum
import math
import multiprocessing
import random
from typing import (List, Sequence, Tuple, Optional)

import bmesh                  # type: ignore
from mathutils import Vector

from guvp import (constants, continuous, debug, discrete)


CollisionResult = enum.Enum('CollisionResult', 'NO YES OUT_OF_BOUNDS')


class Solution:
    """Store a set of placements."""
    def __init__(
            self,
            initial_size: int,
            rotations: Sequence[constants.Rotation],
            random_seed: int
    ) -> None:
        self.islands: List[IslandPlacement] = []
        self.random_seed = random_seed
        self._initial_size = initial_size
        self._rotations = rotations
        self._rng = random.Random(random_seed)
        self._search_start = discrete.CellCoord.zero()
        self._utilized_area = (0, 0)
        self._use_available_area: float = 1.0
        self._mask = discrete.Grid.empty(
            width=initial_size,
            height=initial_size
        )
        self._collision_mask = discrete.Grid.empty(
            width=initial_size,
            height=initial_size
        )

    def __repr__(self):
        return "<Solution(random_seed={})>".format(self.random_seed)

    @property
    def fitness(self) -> float:
        """Calculate fitness.  Return value between 0.0 - 1.0."""
        size = max(*self._utilized_area)
        active: float = float(self._mask.active_cells)
        return active / (size * size) if size > 0 else 0.0

    def pack(self, islands_to_place: List[continuous.Island]) -> bool:
        islands_remaining = deque(islands_to_place)
        del islands_to_place
        self._rng.shuffle(islands_remaining)
        # We need to reset search cell here since
        # pack may be called several times.
        self._search_start = discrete.CellCoord.zero()
        max_island_retries: int = int(len(islands_remaining) * 1.5)
        island_retries_left: int = max_island_retries
        while len(islands_remaining) > 0 and island_retries_left > 0:
            island_retries_left -= 1
            placement_retries_left: int = constants.MAX_PLACEMENT_RETRIES
            island: continuous.Island = islands_remaining.popleft()
            search_cell: discrete.CellCoord = self._search_start
            island_placement: Optional[IslandPlacement] = None
            while placement_retries_left > 0 and island_placement is None:
                placement_retries_left -= 1
                island_placement = IslandPlacement(
                    offset=search_cell,
                    rotation=self._rng.sample(self._rotations, 1)[0],
                    _island=island
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
                    mask = island.mask_with_margin or island.mask
                    search_cell = self._advance_search_cell(
                        search_cell,
                        (mask.width, mask.height)
                    )
            if island_placement is None:
                islands_remaining.append(island)
                if self._should_grow():
                    self._grow()
            else:
                # Note that `island` was removed from `islands_remaining`.
                island_retries_left = len(islands_remaining)
                self._search_start = search_cell
                self.islands.append(island_placement)
                self._write_island_to_mask(island_placement)
                self._update_utilized_area(island_placement)
            if self._rng.random() <= constants.SEARCH_START_RESET_CHANCE:
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
        return float(self._initial_size) / max(*self._utilized_area)

    def _advance_search_cell(
            self,
            search_cell: discrete.CellCoord,
            island_mask_dimensions: Tuple[int, int]
    ) -> discrete.CellCoord:
        max_width: int = int(
            self._use_available_area * self._collision_mask.width
        ) + int(
            # Note that we are using the height of utilized area
            # not its width.
            (1.0 - self._use_available_area) * self._utilized_area[1]
        )
        max_width = max(1, min(self._collision_mask.width, max_width))
        # Even though we are using the island mask dimensions
        # and not the location of farthest non-empty cell it
        # produces the correct result because empty cells in
        # the island mask represent the margin.
        (iw, ih) = island_mask_dimensions
        new_search_cell = search_cell.offset(1, 0)
        # If search cell's x is OOB.
        if new_search_cell[0] + iw >= max_width:
            new_search_cell = discrete.CellCoord(x=0, y=search_cell.y + 1)
        # This is the case where y of search cell is OOB.
        if new_search_cell[1] + ih >= self._collision_mask.height:
            new_search_cell = discrete.CellCoord.zero()
        return new_search_cell

    def _aspect_ratio(self) -> float:
        (x, y) = self._utilized_area
        ratio: float = float(x) / float(y) if x != 0 and y != 0 else 1.0
        return ratio if ratio <= 1.0 else 1 / ratio

    def _check_collision(self, ip: IslandPlacement) -> CollisionResult:
        island_bounds = ip.get_bounds()
        if island_bounds[0] < 0:
            return CollisionResult.OUT_OF_BOUNDS
        if island_bounds[1] < 0:
            return CollisionResult.OUT_OF_BOUNDS
        if island_bounds[2] > self._collision_mask.width:
            return CollisionResult.OUT_OF_BOUNDS
        if island_bounds[3] > self._collision_mask.height:
            return CollisionResult.OUT_OF_BOUNDS

        island_mask = ip.get_collision_mask(
            (self._collision_mask.width, self._collision_mask.height)
        )
        if (self._collision_mask & island_mask).any():
            return CollisionResult.YES
        else:
            return CollisionResult.NO

    def _grow(self) -> None:
        # Limit growing.
        max_size: int = self._initial_size * (2 ** constants.MAX_GROW_COUNT)
        if self._collision_mask.width >= max_size:
            debug.print_("{!r} Cannot grow more.", self)
            return None
        debug.print_("{!r} Growing", self)
        # Create a mask with the new size & copy current mask's contents.
        new_mask = self._mask.copy(
            (0, 0, self._mask.width, self._mask.height)
        )
        self._mask = new_mask
        new_collision_mask = self._collision_mask.copy(
            (0, 0, self._collision_mask.width, self._collision_mask.height)
        )
        self._collision_mask = new_collision_mask
        # No need to update utilized area.
        return None

    def _should_grow(self) -> bool:
        # Grow if utilized area gets too large compared to current mask size.
        utilized_x: float = float(self._utilized_area[0])
        utilized_y: float = float(self._utilized_area[1])
        w: int = self._collision_mask.width
        h: int = self._collision_mask.height
        return (utilized_x / w > constants.GROW_AREA_RATIO) or \
            (utilized_y / h > constants.GROW_AREA_RATIO)

    def _update_utilized_area(self, ip: IslandPlacement) -> None:
        # Update self._utilized_area
        (_, _, a, b) = ip.get_bounds()
        self._utilized_area = (
            max(self._utilized_area[0], a),
            max(self._utilized_area[1], b)
        )
        # Update self._use_available_area
        assert self._mask.width == self._mask.height
        max_d: int = self._mask.width / 2
        dx: int = min(self._mask.width - self._utilized_area[0], max_d)
        dy: int = min(self._mask.height - self._utilized_area[1], max_d)
        ratio: float
        if dx != 0 and dy != 0:
            ratio = float(dx) / float(dy)
            if ratio < 1.0:
                ratio = 1.0 / ratio
        else:
            ratio = max_d
        # 1 - (1/X - 1/NEAR) / (1/FAR - 1/NEAR)
        # NEAR = 1
        # FAR = max_d
        self._use_available_area = \
            1.0 - ((1.0 / ratio) - 1.0) / ((1.0 / max_d) - 1.0)

    def _write_island_to_mask(self, ip: IslandPlacement) -> None:
        self._mask.combine(ip.get_mask(
            (self._mask.width, self._mask.height)
        ))
        self._collision_mask.combine(ip.get_collision_mask(
            (self._collision_mask.width, self._collision_mask.height)
        ))


class GridPacker:
    def __init__(
            self,
            initial_size: int,
            islands: List[continuous.Island],
            rotate: bool = False,
            random_seed: Optional[int] = None
    ) -> None:
        self._initial_size = initial_size
        self._islands = islands
        self._rotate: bool = rotate
        self._rng = random.Random(random_seed)
        self._winner: Optional[Solution] = None
        self._rotations: Sequence[constants.Rotation] = \
            constants.ALL_ROTATIONS if self._rotate \
            else (constants.Rotation.NONE,)

    @property
    def fitness(self) -> float:
        if self._winner is None:
            return 0.0
        else:
            return self._winner.fitness

    @property
    def scaling_factor(self) -> float:
        if self._winner is None:
            return 1.0
        else:
            return self._winner.scaling_factor

    def write(self, bm: bmesh.types.BMesh, udim_offset: Vector) -> None:
        if self._winner is None:
            raise RuntimeError("write is called before run.")
        for ip in self._winner.islands:
            ip.write_uvs(bm, udim_offset, self.scaling_factor)

    def _group_islands(self) -> List[List[continuous.Island]]:
        if len(self._islands) <= 20:
            return [self._islands]
        result = []
        total_area: int = sum([len(i.mask) for i in self._islands])
        islands_sorted: List[continuous.Island] = list(sorted(
            self._islands,
            key=lambda i: len(i.mask),
            reverse=True
        ))
        group_count: int = min(int(math.log(len(self._islands), 2)) - 2, 10)
        assert group_count >= 2
        group_area_limit = total_area / group_count
        for _ in range(group_count - 1):
            group_area_total = 0
            group_islands = []
            while group_area_total < group_area_limit:
                island = islands_sorted.pop(0)
                group_islands.append(island)
                group_area_total += len(island.mask)
            result.append(group_islands)
        result.append(islands_sorted[:])
        debug.print_(
            "groups = {!r}",
            list(map(len, result))
        )
        assert sum(map(len, result)) == len(self._islands)
        return result


class GridPackerSingle(GridPacker):
    def run_single(self) -> None:
        seed = self._rng.randint(0, constants.SEED_MAX)
        solution: Solution = Solution(
            self._initial_size,
            self._rotations,
            seed
        )
        if solution.pack_grouped(*self._group_islands()):
            self._winner = solution


class GridPackerParallel(GridPacker):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._executor: Optional[futures.Executor] = None
        self._grouped_islands: List[List[continuous.Island]] = []
        self._tasks: List[futures.Future[Tuple[bool, Solution]]] = []
        self._max_parallel_tasks: int = max(2, multiprocessing.cpu_count() - 1)
        self.iterations_completed: int = 0

    def __del__(self):
        if self._executor is not None:
            self.stop()

    def run(self) -> None:
        # Set up executor.
        self._executor = futures.ProcessPoolExecutor()
        self._grouped_islands = self._group_islands()
        # Continuously queue new jobs.
        self._create_tasks()
        # Continuously dequeue finished jobs and update winner.
        pass

    def stop(self) -> None:
        assert self._executor is not None
        executor = self._executor
        self._executor = None
        executor.shutdown(wait=False, cancel_futures=True)
        self._tasks = []

    def _create_tasks(self) -> None:
        assert self._executor is not None
        for _ in range(self._max_parallel_tasks - len(self._tasks)):
            task = self._executor.submit(
                self._run_solution,
                self._initial_size,
                self._rotations,
                self._rng.randint(0, constants.SEED_MAX),
                *self._grouped_islands
            )
            task.add_done_callback(self._process_result)
            self._tasks.append(task)

    def _process_result(
            self, task:
            futures.Future[Tuple[bool, Solution]]
    ) -> None:
        if task.cancelled():
            return None
        exception = task.exception()
        if exception is not None:
            raise exception
        del exception

        self.iterations_completed += 1
        (result, solution) = task.result()
        if result and solution.fitness > self.fitness:
            self._winner = solution
        if self._executor is not None:
            self._tasks.remove(task)
            self._create_tasks()
        debug.print_(
            "Task completed. # of tasks is {}/{}",
            len(self._tasks),
            self._max_parallel_tasks
        )

    @staticmethod
    def _run_solution(
            initial_size: int,
            rotations: Sequence[constants.Rotation],
            seed: int,
            *grouped_islands: List[continuous.Island]
    ) -> Tuple[bool, Solution]:
        solution: Solution = Solution(initial_size, rotations, seed)
        return solution.pack_grouped(*grouped_islands), solution


@dataclass(frozen=True)
class IslandPlacement:
    offset: discrete.CellCoord
    rotation: constants.Rotation

    _island: continuous.Island

    def get_bounds(self) -> Tuple[int, int, int, int]:
        if self.rotation in (constants.Rotation.NONE,
                             constants.Rotation.DEGREES_180):
            return (self.offset.x,
                    self.offset.y,
                    self.offset.x + self._island.mask.width,
                    self.offset.y + self._island.mask.height)
        else:
            return (self.offset.x,
                    self.offset.y,
                    self.offset.x + self._island.mask.height,
                    self.offset.y + self._island.mask.width)

    def get_mask(self, bounds: Tuple[int, int]) -> discrete.Grid:
        mask = self._island.mask.rotate(self.rotation)
        return mask.copy(
            (self.offset.x,
             self.offset.y,
             bounds[0] - (self.offset.x + mask.width),
             bounds[1] - (self.offset.y + mask.height))
        )

    def get_collision_mask(self, bounds: Tuple[int, int]) -> discrete.Grid:
        if self._island.mask_with_margin is None:
            return self.get_mask(bounds)
        else:
            mask = self._island.mask_with_margin.rotate(self.rotation)
            return mask.copy(
                (self.offset.x,
                 self.offset.y,
                 bounds[0] - (self.offset.x + mask.width),
                 bounds[1] - (self.offset.y + mask.height))
            )

    def write_uvs(
            self,
            bm: bmesh.types.BMesh,
            udim_offset: Vector,
            scaling_factor: float
    ) -> None:
        self._island.write_uvs(
            bm,
            udim_offset,
            self.offset,
            self.rotation,
            scaling_factor
        )
