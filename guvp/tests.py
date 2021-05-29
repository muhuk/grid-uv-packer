import contextlib
import unittest

import bpy
import bmesh
from mathutils import Vector
from typing import Optional

from guvp import data


class IslandTest(unittest.TestCase):
    def assertEqualVector(
            self,
            first: Vector,
            second: Vector,
            msg: Optional[str] = None
    ) -> None:
        return self.assertAlmostEqual((first - second).length, 0.0, msg=msg)

    def test_island_creation_from_bmesh_faces(self):
        bpy.ops.mesh.primitive_plane_add()
        obj = bpy.context.selected_objects[0]
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        island = data.Island.from_faces(bm, {0})
        # Check that size is correct
        self.assertAlmostEqual(island.size[0], 1.0)
        self.assertAlmostEqual(island.size[1], 1.0)
        # Check that UVs are set right.
        self.assertEqual(len(island.uvs), 4)
        self.assertEqualVector(island.uvs[0], Vector((0.0, 0.0)))
        self.assertEqualVector(island.uvs[1], Vector((1.0, 0.0)))
        self.assertEqualVector(island.uvs[2], Vector((1.0, 1.0)))
        self.assertEqualVector(island.uvs[3], Vector((0.0, 1.0)))
        # bm.to_mesh(obj.data)
        bm.free()
        del(bm)
        bpy.ops.object.delete()
        del(obj)


def run():
    assert(bpy.context.mode == 'OBJECT')
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(IslandTest)
    runner = unittest.TextTestRunner()
    with saved_selection():
        result = runner.run(suite)
    return result


@contextlib.contextmanager
def saved_selection():
    # Save selection
    saved_selection = bpy.context.selected_objects
    saved_active = bpy.context.active_object
    bpy.ops.object.select_all(action='DESELECT')
    # Run suite
    try:
        yield
    finally:
        # Restore selection
        for obj in saved_selection:
            obj.select_set(True)
        if saved_active is not None:
            bpy.context.view_layer.objects.active = saved_active
