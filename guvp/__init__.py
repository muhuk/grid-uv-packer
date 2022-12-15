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

if "bpy" in locals():
    import importlib
    for mod in [                  # noqa: F821
            constants,            # noqa: F821
            continuous,           # noqa: F821
            debug,                # noqa: F821
            discrete,             # noqa: F821
            packing,              # noqa: F821
            props                 # noqa: F821
    ]:
        print("reloading {0}".format(mod))
        importlib.reload(mod)
else:
    # stdlib
    from contextlib import contextmanager
    from functools import reduce
    import random
    from typing import (Iterable, List, Optional, Set)
    # blender
    import bpy                                                  # type: ignore
    from bpy_extras.bmesh_utils import bmesh_linked_uv_islands  # type: ignore
    import bmesh                                                # type: ignore
    from mathutils import Vector                                # type: ignore
    import numpy as np                                          # type: ignore
    # addon
    from guvp import (                                          # noqa: F401
        constants,
        continuous,
        debug,
        discrete,
        packing,
        props
    )


bl_info = {
    "name": "Grid UV Packer",
    "description": "A pure-Python UV packer.",
    "author": "Atamert Ölçgen",
    "version": (0, 3),
    "blender": (3, 4, 0),
    "location": "TBD",
    "tracker_url": "https://github.com/muhuk/grid_uv_packer",
    "support": "COMMUNITY",
    "category": "UV"
}


class GridUVPackOperator(bpy.types.Operator):
    """Grid UV Pack Operator"""
    bl_idname = "uv.grid_pack"
    bl_label = "Grid UV Pack"
    bl_options = {'UNDO'}

    grid_size: bpy.props.EnumProperty(  # type: ignore
        name="Grid Size",               # noqa: F722
        default="128",
        items=(
            ("64", "64", "", 'NONE', 64),     # noqa: F722,F821
            ("128", "128", "", 'NONE', 128),  # noqa: F722,F821
            ("256", "256", "", 'NONE', 256),  # noqa: F722,F821
            ("512", "512", "", 'NONE', 512)   # noqa: F722,F821
        )
    )
    margin: bpy.props.FloatProperty(          # type: ignore
        name="Margin",                        # noqa: F821
        description="Space between islands.",
        default=0.01,
        min=0.0,
        max=1.0,
        precision=3
    )
    seed: bpy.props.IntProperty(              # type: ignore
        name="Random Seed",
        description="Seed of the random generator.",
        default=0,
        min=0,
        max=constants.SEED_MAX
    )

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and \
            context.active_object.type == 'MESH' and \
            context.mode == 'EDIT_MESH'

    def execute(self, context: bpy.types.Context) -> Set[str]:
        assert context.mode == 'EDIT_MESH'
        assert 0.0 <= self.margin <= 1.0
        # We ignore the type of self.grid_size for bpy reasons.
        # So let's cast it explicitly here.
        grid_size: int = int(self.grid_size)
        # Size of one grid cell (square) in UV coordinate system.
        cell_size: float = 1.0 / grid_size

        with self._mesh_context(context) as (mesh, bm):
            # Calculate fitness of the input UVs.
            island_face_ids: List[set[int]] = self._island_face_ids(bm)
            baseline_fitness: Optional[float]
            baseline_fitness = self._calculate_baseline_fitness(
                bm,
                grid_size,
                reduce(lambda a, b: set(a) | b, island_face_ids, set())
            )
            if baseline_fitness is None:
                self.report(
                    {'ERROR'}, "Island out of bounds in active UV map."
                )
                return {'CANCELLED'}
            # Use a random seed if seed prop is set to zero.
            random_seed: int = self.seed if self.seed != 0 \
                else random.randint(0, constants.SEED_MAX)
            if debug.is_debug():
                print("Seed being used is: {}".format(random_seed))
            packer = packing.GridPacker(
                initial_size=grid_size,
                islands=[
                    continuous.Island.from_faces(
                        bm,
                        face_ids,
                        cell_size,
                        self.margin
                    )
                    for face_ids in island_face_ids
                ],
                random_seed=random_seed
            )
            packer.run()
            if debug.is_debug():
                assert baseline_fitness is not None
                print(
                    "Baseline fitness is {0:0.2f}%".format(
                        baseline_fitness * 100
                    )
                )
                print(
                    "Grid packer fitness is {0:0.2f}%".format(
                        packer.fitness * 100
                    )
                )
            # TODO: Handle failure better.
            #       Ideally fitness should be better than the current
            #       UV configuration.
            if packer.fitness > 0.20:
                packer.write(bm)
                # Get out of EDIT mode.
                bpy.ops.object.editmode_toggle()
                bm.to_mesh(mesh)
                # Back into EDIT mode.
                bpy.ops.object.editmode_toggle()
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    @staticmethod
    @contextmanager
    def _mesh_context(context: bpy.types.Context):
        mesh = context.active_object.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        try:
            yield (mesh, bm)
        finally:
            bm.free()

    @staticmethod
    def _calculate_baseline_fitness(
            bm: bmesh.types.BMesh,
            grid_size: int,
            face_ids: Iterable[int]
    ) -> Optional[float]:
        mask = discrete.Grid.empty(width=grid_size, height=grid_size)
        offset = Vector((0, 0))
        cell_size = 1.0 / grid_size
        out_of_bounds = not continuous.fill_mask(
            bm,
            set(face_ids),
            offset,
            cell_size,
            mask
        )
        ones = np.count_nonzero(mask.cells)
        return None if out_of_bounds else float(ones) / (grid_size * grid_size)

    @staticmethod
    def _island_face_ids(bm: bpy.types.BMesh) -> List[set[int]]:
        """Calculate sets of faces that form a UV island."""
        uv_ident = bm.loops.layers.uv.verify()
        island_faces = bmesh_linked_uv_islands(bm, uv_ident)
        return [set([f.index for f in faces]) for faces in island_faces]


def menu_draw(self, _context):
    self.layout.separator()
    self.layout.operator_context = 'INVOKE_DEFAULT'
    self.layout.operator("uv.grid_pack")


def register():
    # Register props
    bpy.utils.register_class(props.GridUVPackerAddonPreferences)

    # Register operations
    bpy.utils.register_class(GridUVPackOperator)

    # Register UI
    bpy.types.VIEW3D_MT_uv_map.append(menu_draw)
    bpy.types.IMAGE_MT_uvs_unwrap.append(menu_draw)


def unregister():
    # Unregister UI
    bpy.types.VIEW3D_MT_uv_map.remove(menu_draw)
    bpy.types.IMAGE_MT_uvs_unwrap.remove(menu_draw)

    # Unregister operations
    bpy.utils.unregister_class(GridUVPackOperator)

    # Unregister props
    bpy.utils.unregister_class(props.GridUVPackerAddonPreferences)


if __name__ == "__main__":
    register()

    # test call
    bpy.ops.uv.grid_pack()
