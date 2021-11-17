from __future__ import annotations
from dataclasses import dataclass
import math
from typing import (
    Dict,
    List,
    Set,
    Type
)

import bmesh                  # type: ignore
from mathutils import Vector  # type: ignore
import mathutils.geometry     # type: ignore

from guvp import discrete


FaceUVs = Dict[int, Vector]


# face id -> (face loop id -> UV)
IslandUVs = Dict[int, FaceUVs]


@dataclass(frozen=True)
class Island:
    face_ids: set[int]
    uvs: IslandUVs
    cell_size: float
    mask: discrete.Grid

    @classmethod
    def from_faces(
            cls: Type[Island],
            bm: bmesh.types.BMesh,
            face_ids: set[int],
            cell_size: float
            # TODO: add margin parameters
    ) -> Island:
        (uvs, size, offset) = cls._calculate_uvs_and_size(bm, face_ids)
        mask = discrete.Grid.empty(
            width=math.ceil(size.x / cell_size),
            height=math.ceil(size.y / cell_size)
        )
        cls._fill_mask(bm, face_ids, offset, cell_size, mask)
        print(mask)
        mask.draw_str()
        return cls(
            face_ids=face_ids,
            uvs=uvs,
            cell_size=cell_size,
            mask=mask
        )

    def write_uvs(
            self,
            bm: bmesh.types.BMesh,
            offset: discrete.CellCoord
    ) -> None:
        uv_ident = bm.loops.layers.uv.verify()
        offset_vec: Vector = Vector(offset) * self.cell_size
        for face_id in self.face_ids:
            for face_loop in bm.faces[face_id].loops:
                assert(face_loop.index in self.uvs[face_id])
                face_loop[uv_ident].uv = self.uvs[face_id][face_loop.index]
                face_loop[uv_ident].uv += offset_vec

    @staticmethod
    def _calculate_uvs_and_size(
            bm: bmesh.types.BMesh,
            face_ids: set[int]
    ) -> tuple[IslandUVs, Vector, Vector]:
        uvs: IslandUVs = {}
        uv_ident = bm.loops.layers.uv.verify()
        (u_min, v_min) = (math.inf, math.inf)
        (u_max, v_max) = (-math.inf, -math.inf)
        for face_id in face_ids:
            face_uvs: FaceUVs = {}
            for face_loop in bm.faces[face_id].loops:
                # Store UV
                face_uvs[face_loop.index] = face_loop[uv_ident].uv
                # Update UV bounds
                (u, v) = face_loop[uv_ident].uv
                u_min = min(u, u_min)
                v_min = min(v, v_min)
                u_max = max(u, u_max)
                v_max = max(v, v_max)
                del u, v
            uvs[face_id] = face_uvs
        size = Vector((u_max - u_min, v_max - v_min))
        # Offset everything by (-u_min, -v_min)
        offset = Vector((u_min, v_min))
        uvs = {k: {loop_id: uv - offset for loop_id, uv in fuv.items()}
               for k, fuv in uvs.items()}
        return (uvs, size, offset)

    @staticmethod
    def _fill_mask(
            bm: bmesh.types.BMesh,
            face_ids: set[int],
            offset: Vector,
            cell_size: float,
            mask: discrete.Grid
    ):
        open_cells: Set[discrete.CellCoord] = set(mask)
        uv_ident = bm.loops.layers.uv.verify()

        for face_id in face_ids:
            loop_uvs = [face_loop[uv_ident].uv
                        for face_loop in bm.faces[face_id].loops]
            # When we use triangulate, we are assuming the face,
            # when it is an n-gon, is convex.  There is basically no way
            # to triangulate a concave n-gon without triangulating it first.
            for face_tri in Triangle2D.triangulate(
                [(u, v) for (u, v) in loop_uvs]
            ):
                hit_cells: Set[discrete.CellCoord] = set()
                for open_cell_id in open_cells:
                    # We need to invert y-axis because UVs
                    # use vertically increasing y-axis.
                    cell_x: int = open_cell_id[0]
                    cell_y: int = mask.size.height - open_cell_id[1]
                    # triangulate the cell's quad.
                    x = float(offset.x + cell_x * cell_size)
                    y = float(offset.y + cell_y * cell_size)
                    grid_tris = Triangle2D.triangulate([
                        Vector((x, y)),
                        Vector((x, y + cell_size)),
                        Vector((x + cell_size, y + cell_size)),
                        Vector((x + cell_size, y))
                    ])
                    for grid_tri in grid_tris:
                        if grid_tri.intersect(face_tri):
                            hit_cells.add(open_cell_id)
                            mask[open_cell_id] = True
                            break
                open_cells -= hit_cells


class Triangle2D:
    def __init__(self, a: Vector, b: Vector, c: Vector):
        self.a = a
        self.b = b
        self.c = c

    def intersect(self, other: Triangle2D) -> bool:
        return mathutils.geometry.intersect_tri_tri_2d(
            self.a,
            self.b,
            self.c,
            other.a,
            other.b,
            other.c
        )

    @classmethod
    def triangulate(cls, verts: List[Vector]) -> List[Triangle2D]:
        n: int = len(verts)
        vert_coords = list(map(Vector, verts))
        edges = [(a, (a + 1) % n) for a in range(n)]
        faces = [list(range(n))]
        output_type: int = 0  # triangles with convex hull
        epsilon: float = 1.0 / 1_000_000.0
        delanuay_result = mathutils.geometry.delaunay_2d_cdt(
            vert_coords,
            edges,
            faces,
            output_type,
            epsilon
        )
        (t_verts, _, t_faces, _, _, _) = delanuay_result
        result: List[Triangle2D] = []
        for vert_ids in t_faces:
            assert(len(vert_ids) == 3)
            verts = [t_verts[vert_id] for vert_id in vert_ids]
            result.append(cls(verts[0], verts[1], verts[2]))
        return result