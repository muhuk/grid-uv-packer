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


import enum

ADDON_NAME = __package__

DEBUG_MODE_DEFAULT: bool = False

# Grow if utilized area is larger than this.
GROW_AREA_RATIO = 0.85

MAX_GROW_COUNT = 2

MAX_ITERATIONS_DEFAULT = 500
MAX_ITERATIONS_LIMIT = 10000

# Hard limit for tries
MAX_PLACEMENT_RETRIES = 100_000

MAX_RUNTIME_DEFAULT = 90
MAX_RUNTIME_LIMIT = 3600  # 1 hour

OPERATOR_TIMER_TIME_STEP = 0.5  # seconds

SEARCH_START_RESET_CHANCE = 0.05

SEED_MAX = 2 ** 31 - 1

# Rotation is CW, like in Blender.
Rotation = enum.Enum(
    'Rotation',
    {'NONE': 0.0,
     'DEGREES_90': 90.0,
     'DEGREES_180': 180.0,
     'DEGREES_270': 270.0})
ALL_ROTATIONS = (
    Rotation.NONE,
    Rotation.DEGREES_90,
    Rotation.DEGREES_180,
    Rotation.DEGREES_270
)
