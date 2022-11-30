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
from dataclasses import dataclass
import math
from typing import (
    Dict,
    List,
    Tuple,
    Type
)

import bmesh                  # type: ignore
from mathutils import Vector  # type: ignore
import mathutils.geometry     # type: ignore

from guvp import discrete


FaceUVs = Dict[int, Tuple[float, float]]


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
            cell_size: float,
            margin: float
    ) -> Island:
        (uvs, size, offset) = cls._calculate_uvs_and_size(bm, face_ids, margin)
        mask = discrete.Grid.empty(
            width=math.ceil(size.x / cell_size),
            height=math.ceil(size.y / cell_size)
        )
        fill_mask(bm, face_ids, offset, cell_size, mask)
        dilated_mask: discrete.Grid = mask.dilate(
            size=int(math.ceil(margin / cell_size / 2.0))
        )
        return cls(
            face_ids=face_ids,
            uvs=uvs,
            cell_size=cell_size,
            mask=dilated_mask
        )

    def write_uvs(
            self,
            bm: bmesh.types.BMesh,
            offset: discrete.CellCoord,
            scaling_factor: float
    ) -> None:
        uv_ident = bm.loops.layers.uv.verify()
        offset_vec: Vector = Vector(offset) * self.cell_size
        for face_id in self.face_ids:
            for face_loop in bm.faces[face_id].loops:
                assert face_loop.index in self.uvs[face_id]
                face_loop[uv_ident].uv = Vector(
                    self.uvs[face_id][face_loop.index]
                )
                face_loop[uv_ident].uv += offset_vec
                face_loop[uv_ident].uv *= scaling_factor

    @staticmethod
    def _calculate_uvs_and_size(
            bm: bmesh.types.BMesh,
            face_ids: set[int],
            margin: float
    ) -> tuple[IslandUVs, Vector, Vector]:
        margin_vec: Vector = Vector((margin, margin))
        uvs: IslandUVs = {}
        uv_ident = bm.loops.layers.uv.verify()
        (u_min, v_min) = (math.inf, math.inf)
        (u_max, v_max) = (-math.inf, -math.inf)
        for face_id in face_ids:
            face_uvs: FaceUVs = {}
            for face_loop in bm.faces[face_id].loops:
                (u, v) = face_loop[uv_ident].uv
                # Store UV
                face_uvs[face_loop.index] = (u, v)
                # Update UV bounds
                u_min = min(u, u_min)
                v_min = min(v, v_min)
                u_max = max(u, u_max)
                v_max = max(v, v_max)
                del u, v
            uvs[face_id] = face_uvs
        size = Vector((u_max - u_min, v_max - v_min))
        size += margin_vec
        # Offset everything by (-u_min, -v_min)
        offset = Vector((u_min, v_min))
        offset -= margin_vec / 2.0
        for k, fuv in uvs.items():
            for loop_id, uv in fuv.items():
                (new_u, new_v) = Vector(uv) - offset
                uvs[k][loop_id] = (new_u, new_v)
        return (uvs, size, offset)


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

    def intersect_quad(
            self,
            quad: Tuple[Vector, Vector, Vector, Vector]
    ) -> bool:
        return mathutils.geometry.intersect_tri_tri_2d(
            self.a, self.b, self.c,
            quad[0], quad[1], quad[2]
        ) or mathutils.geometry.intersect_tri_tri_2d(
            self.a, self.b, self.c,
            quad[1], quad[2], quad[3]
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
            assert len(vert_ids) == 3
            verts = [t_verts[vert_id] for vert_id in vert_ids]
            result.append(cls(verts[0], verts[1], verts[2]))
        return result


def fill_mask(
        bm: bmesh.types.BMesh,
        face_ids: set[int],
        offset: Vector,
        cell_size: float,
        mask: discrete.Grid
) -> bool:
    uv_ident = bm.loops.layers.uv.verify()

    for face_id in face_ids:
        loop_uvs: List[Vector] = [
            face_loop[uv_ident].uv - offset
            for face_loop in bm.faces[face_id].loops
        ]
        # Out of bounds check
        u_min, u_max = math.inf, -math.inf
        v_min, v_max = math.inf, -math.inf
        for (u, v) in loop_uvs:
            if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
                return False
            u_min = min(u_min, u)
            u_max = max(u_max, u)
            v_min = min(v_min, v)
            v_max = max(v_max, v)
        # When we use triangulate, we are assuming the face,
        # when it is an n-gon, is convex.  There is basically no way
        # to triangulate a concave n-gon without triangulating it first.
        for face_tri in Triangle2D.triangulate(loop_uvs):
            x_min: int = int(math.floor(u_min / cell_size))
            x_max: int = int(math.ceil(u_max / cell_size))
            y_min: int = int(math.floor(v_min / cell_size))
            y_max: int = int(math.ceil(v_max / cell_size))
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    # a & b are cells's bottom left coords in UV space.
                    a = float(x * cell_size)
                    b = float(y * cell_size)
                    quad: Tuple[Vector, Vector, Vector, Vector] = (
                        Vector((a, b + cell_size)),
                        Vector((a + cell_size, b + cell_size)),
                        Vector((a, b)),
                        Vector((a + cell_size, b))
                    )
                    if not mask[discrete.CellCoord(x, y)] \
                       and face_tri.intersect_quad(quad):
                        mask[discrete.CellCoord(x, y)] = True
    return True
