from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Dict, Type

import bmesh                  # type: ignore
from mathutils import Vector  # type: ignore


FaceUVs = Dict[int, Vector]

# face id -> (face loop id -> UV)
IslandUVs = Dict[int, FaceUVs]


class Grid:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def __repr__(self) -> str:
        return "<Grid(width={0}, height={1})>".format(self.width, self.height)


@dataclass(frozen=True)
class Island:
    face_ids: set[int]
    size: tuple[float, float]
    uvs: IslandUVs
    mask: Grid

    @classmethod
    def from_faces(
            cls: Type[Island],
            bm: bmesh.types.BMesh,
            face_ids: set[int],
            cell_size: float
            # TODO: add margin parameters
    ) -> Island:
        (uvs, size) = cls._calculate_uvs_and_size(bm, face_ids)
        mask = cls._calculate_mask_grid(uvs, size, cell_size)
        print(mask)
        return cls(
            face_ids=face_ids,
            uvs=uvs,
            size=size,
            mask=mask
        )

    def write_uvs(self, bm: bmesh.types.BMesh) -> None:
        uv_ident = bm.loops.layers.uv.verify()
        for face_id in self.face_ids:
            for face_loop in bm.faces[face_id].loops:
                assert(face_loop.index in self.uvs[face_id])
                face_loop[uv_ident].uv = self.uvs[face_id][face_loop.index]

    @staticmethod
    def _calculate_mask_grid(
            uvs: IslandUVs,
            size: tuple[float, float],
            cell_size: float
    ) -> Grid:
        width = math.ceil(size[0] / cell_size)
        height = math.ceil(size[1] / cell_size)
        return Grid(width, height)

    @staticmethod
    def _calculate_uvs_and_size(
            bm: bmesh.types.BMesh,
            face_ids: set[int]
    ) -> tuple[IslandUVs, tuple[float, float]]:
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
        size = (u_max - u_min, v_max - v_min)
        # Offset everything by (-u_min, -v_min)
        offset = Vector((u_min, v_min))
        uvs = {k: {loop_id: uv - offset for loop_id, uv in fuv.items()}
               for k, fuv in uvs.items()}
        return (uvs, size)
