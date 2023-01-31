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
    from functools import reduce
    import random
    import time
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
        ),
        options={'SKIP_SAVE'}                 # noqa: F821
    )
    rotate: bpy.props.BoolProperty(           # type: ignore
        name="Rotate",                        # noqa: F821
        description="Rotate islands for best fit.",
        default=True,
        options={'SKIP_SAVE'}                 # noqa: F821
    )
    margin: bpy.props.FloatProperty(          # type: ignore
        name="Margin",                        # noqa: F821
        description="Space between islands.",
        default=0.01,
        min=0.0,
        max=1.0,
        precision=3,
        options={'SKIP_SAVE'}                 # noqa: F821
    )
    max_iterations: bpy.props.IntProperty(    # type: ignore
        name="Max Iterations",
        description="Maximum number of iterations.",
        default=constants.MAX_ITERATIONS_DEFAULT,
        min=1,
        max=constants.MAX_ITERATIONS_LIMIT,
        options={'SKIP_SAVE'}                 # noqa: F821
    )
    max_runtime: bpy.props.IntProperty(       # type: ignore
        name="Max Runtime",
        description="Maximum time the calculation can take, in seconds.",
        default=constants.MAX_RUNTIME_DEFAULT,
        min=0,
        max=constants.MAX_RUNTIME_LIMIT,
        options={'SKIP_SAVE'}                 # noqa: F821
    )
    seed: bpy.props.IntProperty(              # type: ignore
        name="Random Seed",
        description="Seed of the random generator.",
        default=0,
        min=0,
        max=constants.SEED_MAX,
        options={'SKIP_SAVE'}                 # noqa: F821
    )

    def __init__(self):
        # Size of one grid cell (square) in UV coordinate system.
        self.cell_size: float
        # This is the actual seed used in calculations.
        # If 'seed' is set to 0, 'random_seed' takes a random value.
        self.random_seed: int
        self.start_time_ns: int
        self.end_time_ns: Optional[int]
        self.bm: bmesh.types.BMesh
        self.island_face_ids: List[set[int]]
        self.baseline_fitness: float
        self.packer: packing.GridPacker
        self._timer: bpy.types.Timer

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and \
            context.active_object.type == 'MESH' and \
            context.active_object.data.uv_layers.active is not None and \
            context.mode == 'EDIT_MESH'

    def execute(self, context: bpy.types.Context) -> Set[str]:
        assert context.active_object is not None
        assert context.active_object.type == 'MESH'
        assert context.active_object.data.uv_layers.active is not None
        assert context.mode == 'EDIT_MESH'
        assert 0.0 <= self.margin <= 1.0

        self.cell_size = 1.0 / int(self.grid_size)
        self.random_seed = self.seed if self.seed != 0 \
            else random.randint(0, constants.SEED_MAX)
        self.start_time_ns = time.time_ns()

        self.bm = bmesh.from_edit_mesh(context.active_object.data)
        self.bm.verts.ensure_lookup_table()
        self.bm.edges.ensure_lookup_table()
        self.bm.faces.ensure_lookup_table()

        self.island_face_ids = self._island_face_ids(self.bm)

        # Calculate fitness of the input UVs.
        baseline_fitness: Optional[float] = self._calculate_baseline_fitness(
            self.bm,
            int(self.grid_size),
            reduce(lambda a, b: set(a) | b, self.island_face_ids, set())
        )
        if baseline_fitness is None:
            self.report(
                {'ERROR'}, "Island out of bounds in active UV map."
            )
            return {'CANCELLED'}
        else:
            self.baseline_fitness = baseline_fitness
            debug.print_("Seed being used is: {}", self.random_seed)
            if self.max_iterations == 1:
                return self.execute_single(context)
            else:
                return self.execute_parallel(context)

    def execute_single(self, context: bpy.types.Context) -> Set[str]:
        debug.print_("Running a single iteration.")

        self.packer = packing.GridPackerSingle(
            initial_size=int(self.grid_size),
            islands=[
                continuous.Island.from_faces(
                    self.bm,
                    face_ids,
                    self.cell_size,
                    self.margin
                )
                for face_ids in self.island_face_ids
            ],
            rotate=self.rotate,
            random_seed=self.random_seed
        )
        self.packer.run_single()

        return self.finish(context)

    def execute_parallel(self, context: bpy.types.Context) -> Set[str]:
        if self.max_runtime > 0:
            self.end_time_ns = self.start_time_ns + \
                self.max_runtime * 1_000_000_000

        wm: bpy.types.WindowManager = context.window_manager
        wm.progress_begin(0, 10000)
        wm.progress_update(1)

        self.packer = packing.GridPackerParallel(
            initial_size=int(self.grid_size),
            islands=[
                continuous.Island.from_faces(
                    self.bm,
                    face_ids,
                    self.cell_size,
                    self.margin
                )
                for face_ids in self.island_face_ids
            ],
            rotate=self.rotate,
            random_seed=self.random_seed
        )
        self.packer.run()

        self._timer = wm.event_timer_add(
            constants.OPERATOR_TIMER_TIME_STEP,
            window=context.window
        )
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self,
              context: bpy.types.Context,
              event: bpy.types.Event) -> Set[str]:
        assert type(self.packer) is packing.GridPackerParallel
        wm: bpy.types.WindowManager = context.window_manager
        should_continue = True
        if event.type == 'TIMER':
            if self.packer.iterations_completed < self.max_iterations \
               and (self.end_time_ns is None
                    or time.time_ns() < self.end_time_ns):
                wm.progress_update(int(
                    float(self.packer.iterations_completed)
                    / self.max_iterations * 10000
                ))
                debug.print_(
                    "Iterations so far {}, fitness {:.2f}%",
                    self.packer.iterations_completed,
                    self.packer.fitness * 100.0
                )
            else:
                should_continue = False
        elif event.type == 'ESC':
            should_continue = False

        if should_continue:
            return {'RUNNING_MODAL'}
        else:
            debug.print_("Stopping packer.")
            self.packer.stop()
            wm.progress_end()
            wm.event_timer_remove(self._timer)
            return self.finish(context)

    def finish(self, context: bpy.types.Context) -> Set[str]:
        debug.print_(
            "Baseline fitness is {:.2f}%",
            self.baseline_fitness * 100
        )
        debug.print_(
            "Grid packer fitness is {:.2f}%",
            self.packer.fitness * 100
        )
        if self.packer.fitness > self.baseline_fitness:
            self.packer.write(self.bm)
            bmesh.update_edit_mesh(
                mesh=context.active_object.data,
                loop_triangles=False,
                destructive=False
            )
            self.report({'INFO'}, "UVs updated.")
        else:
            self.report({'INFO'}, "UVs are not changed.")

        debug.print_(
            "Total time: {:.3f}",
            (time.time_ns() - self.start_time_ns) / 1_000_000_000
        )
        return {'FINISHED'}

    def invoke(self, context, event) -> Set[str]:
        wm: bpy.types.WindowManager = context.window_manager
        return wm.invoke_props_dialog(self)

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
        uv_ident = bm.loops.layers.uv.active
        island_faces = bmesh_linked_uv_islands(bm, uv_ident)
        result = []
        island_selected: bool
        for faces in island_faces:
            island_selected = True
            for face in faces:
                for loop in face.loops:
                    if loop[uv_ident].select is False:
                        island_selected = False
                        break  # if one face-corner is unselected
                if island_selected is False:
                    break  # if one of the faces in island is unselected
            if island_selected:
                result.append(set([f.index for f in faces]))
        return result


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
    bpy.types.IMAGE_MT_uvs_unwrap.append(menu_draw)


def unregister():
    # Unregister UI
    bpy.types.IMAGE_MT_uvs_unwrap.remove(menu_draw)

    # Unregister operations
    bpy.utils.unregister_class(GridUVPackOperator)

    # Unregister props
    bpy.utils.unregister_class(props.GridUVPackerAddonPreferences)


if __name__ == "__main__":
    register()

    # test call
    bpy.ops.uv.grid_pack()
