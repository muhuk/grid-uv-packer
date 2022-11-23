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


import bpy          # type: ignore

from guvp import constants


class GridUVPackerAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = constants.ADDON_NAME

    debug_mode: bpy.props.BoolProperty(
        name="Debug mode",
        description="Enable debug output.",
        default=constants.DEBUG_MODE_DEFAULT
    )

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True, heading="Developer Preferences")
        col.prop(self, "debug_mode")
