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
    import bpy
    import bpy_extras
    import bmesh
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

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and \
            context.active_object.type == 'MESH' and \
            context.mode == 'EDIT_MESH'

    def execute(self, context):
        assert(context.mode == 'EDIT_MESH')
        mesh = context.active_object.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        # This is needed to be able to use face ids.
        bm.faces.ensure_lookup_table()
        for face_ids in self._island_face_ids(context, mesh):
            island = data.Island.from_faces(bm, face_ids)
            island.write_uvs(bm)
        # We cannot write UVs in edit mode.
        bpy.ops.object.editmode_toggle()
        bm.to_mesh(mesh)
        bpy.ops.object.editmode_toggle()
        bm.free()

        return {'FINISHED'}

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
