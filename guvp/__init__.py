from __future__ import annotations

if "bpy" in locals():
    import importlib
    for mod in [data]:  # noqa: F821
        print("reloading {0}".format(mod))
        importlib.reload(mod)
else:
    # stdlib
    from dataclasses import dataclass
    import math
    from typing import Iterable, Type
    # blender
    import bpy         # type: ignore
    import bpy_extras  # type: ignore
    import bmesh       # type: ignore
    # addon
    from guvp import data


bl_info = {
    "name": "Grid UV Packer",
    "description": "TBD",
    "author": "Atamert Ölçgen",
    "version": (1, 2),
    "blender": (2, 93, 0),
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
            # TODO: Remove 16 option, it is for debugging.
            ("16", "16", "", 'NONE', 16),     # noqa: F722,F821
            ("64", "64", "", 'NONE', 64),     # noqa: F722,F821
            ("128", "128", "", 'NONE', 128),  # noqa: F722,F821
            ("256", "256", "", 'NONE', 256),  # noqa: F722,F821
            ("512", "512", "", 'NONE', 512)   # noqa: F722,F821
        )
    )

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and \
            context.active_object.type == 'MESH' and \
            context.mode == 'EDIT_MESH'

    def execute(self, context):
        assert(context.mode == 'EDIT_MESH')
        # Size of one grid cell (square) in UV coordinate system.
        cell_size: float = 1.0 / float(self.grid_size)
        mesh = context.active_object.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        packer = data.GridPacker(
            [data.Island.from_faces(bm, face_ids, cell_size)
             for face_ids in self._island_face_ids(context, mesh)]
        )
        packer.run()
        print("Grid packer fitness is {0:0.2f}%".format(packer.fitness * 100))
        packer.write(bm)
        # We cannot write UVs in edit mode.
        bpy.ops.object.editmode_toggle()
        bm.to_mesh(mesh)
        bpy.ops.object.editmode_toggle()
        bm.free()

        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    @staticmethod
    def _island_face_ids(
        context: bpy.types.Context,
        mesh: bpy.types.Mesh
    ) -> Iterable[set[int]]:
        """Calculate sets of faces that form a UV island."""
        called_in_edit_mode = (context.mode == 'EDIT_MESH')
        island_face_ids = []
        # mesh_linked_uv_islands does not work in edit mode.
        if called_in_edit_mode:
            bpy.ops.object.editmode_toggle()
        # Calculate islands
        island_faces = bpy_extras.mesh_utils.mesh_linked_uv_islands(mesh)
        for face_ids in island_faces:
            island_face_ids.append(set(face_ids))
        # Restore edit mode if function is called in edit mode.
        if called_in_edit_mode:
            bpy.ops.object.editmode_toggle()
        return island_face_ids


def menu_draw(self, _context):
    self.layout.separator()
    self.layout.operator_context = 'INVOKE_DEFAULT'
    self.layout.operator("uv.grid_pack")


def register():
    bpy.utils.register_class(GridUVPackOperator)
    bpy.types.VIEW3D_MT_uv_map.append(menu_draw)
    bpy.types.IMAGE_MT_uvs_unwrap.append(menu_draw)


def unregister():
    bpy.types.VIEW3D_MT_uv_map.remove(menu_draw)
    bpy.types.IMAGE_MT_uvs_unwrap.remove(menu_draw)
    bpy.utils.unregister_class(GridUVPackOperator)


if __name__ == "__main__":
    register()

    # test call
    bpy.ops.uv.grid_pack()
