from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Type

import bmesh
from mathutils import Vector

@dataclass(frozen=True)
class Island:
    size: tuple[float, float]
    uvs: dict[int, Vector]  # face loop id -> UV

    @classmethod
    def from_faces(
            cls: Type[Island],
            bm: bmesh.types.BMesh,
            face_ids: set[int]
            # TODO: add margin parameters
    ) -> Island:
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        uvs = {}
        uv_ident = bm.loops.layers.uv.verify()
        (u_min, v_min) = (math.inf, math.inf)
        (u_max, v_max) = (-math.inf, -math.inf)
        for face_id in face_ids:
            print(face_id)
            for face_loop in bm.faces[face_id].loops:
                print(face_loop)
                # Store UV
                uvs[face_loop.index] = face_loop[uv_ident].uv
                # Update UV bounds
                (u, v) = face_loop[uv_ident].uv
                u_min = min(u, u_min)
                v_min = min(v, v_min)
                u_max = max(u, u_max)
                v_max = max(v, v_max)
                del u, v
        size = (u_max - u_min, v_max - v_min)
        return cls(uvs=uvs, size=size)
