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


ADDON_NAME = __package__

DEBUG_MODE_DEFAULT: bool = False

# Grow chance if utilized area is too big.
GROW_AREA_CHANCE = 0.5

# Grow if utilized area is larger than this.
GROW_AREA_RATIO = 0.85

# Base grow chance without modifiers
GROW_BASE_CHANCE = 0.15

# Grow change if the utilized area is closer to a rectangle.
GROW_REGULARITY_CHANCE = -0.25

# What is the threshold to consider a rectangle-like fill.
GROW_REGULARITY_RATIO = 0.667

MAX_GROW_COUNT = 2

MAX_ITERATIONS_DEFAULT = 500
MAX_ITERATIONS_LIMIT = 10000

# Hard limit for tries
MAX_PLACEMENT_RETRIES = 100_000

SEARCH_START_RESET_CHANCE = 0.333

SEED_MAX = 2 ** 31 - 1
