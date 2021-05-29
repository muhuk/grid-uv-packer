from __future__ import annotations
from dataclasses import dataclass
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
        # FIXME: WTH loop indices are not set properly
        uvs = {}
        uv_ident = bm.loops.layers.uv.verify()
        for face_id in face_ids:
            print(face_id)
            for face_loop in bm.faces[face_id].loops:
                print(face_loop)
                uvs[face_loop.index] = face_loop[uv_ident].uv
        size = (1.0, 1.0)
        return cls(uvs=uvs, size=size)
