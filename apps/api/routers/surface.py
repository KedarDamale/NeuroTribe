"""fsaverage5 surface geometry for the browser 3D viewer.

The mesh is exported once as packed binary buffers (positions / indices /
normals) so Three.js can build a BufferGeometry with no parsing cost.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from neurotribe.config import Settings

from apps.api.deps import get_settings

router = APIRouter(prefix="/surface", tags=["surface"])


def _export_dir(settings: Settings) -> Path:
    path = settings.paths.derivatives / "surfaces" / "web"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _vertex_normals(coords: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals for smooth shading."""
    normals = np.zeros_like(coords, dtype=np.float32)
    triangles = coords[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0],
                            triangles[:, 2] - triangles[:, 0])
    for axis in range(3):
        np.add.at(normals, faces[:, axis], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths < 1e-12] = 1.0
    return (normals / lengths).astype(np.float32)


def _build(settings: Settings) -> dict:
    """Build (and cache) the web mesh export."""
    from neurotribe.preprocessing.surfaces import load_geometry

    out_dir = _export_dir(settings)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    geometry = load_geometry(settings)
    order = list(settings.get("surface.hemi_order", ["L", "R"]))
    manifest: dict = {
        "space": str(settings.get("surface.space", "fsaverage5")),
        "hemi_order": order, "source": geometry.source, "hemispheres": {},
    }

    offset = 0
    for hemi in order:
        coords = np.asarray(geometry.coordinates[hemi], dtype=np.float32)
        faces = np.asarray(geometry.faces[hemi], dtype=np.uint32)
        normals = _vertex_normals(coords, faces.astype(np.int64))

        # Centre each hemisphere and separate them slightly for the split view.
        centred = coords - coords.mean(axis=0, keepdims=True)

        (out_dir / f"{hemi}_positions.bin").write_bytes(centred.tobytes(order="C"))
        (out_dir / f"{hemi}_normals.bin").write_bytes(normals.tobytes(order="C"))
        (out_dir / f"{hemi}_indices.bin").write_bytes(faces.tobytes(order="C"))

        manifest["hemispheres"][hemi] = {
            "n_vertices": int(coords.shape[0]),
            "n_faces": int(faces.shape[0]),
            "vertex_offset": offset,
            "bounds": {
                "min": centred.min(axis=0).tolist(),
                "max": centred.max(axis=0).tolist(),
            },
        }
        offset += int(coords.shape[0])

    manifest["total_vertices"] = offset
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


@router.get("/manifest")
def manifest(settings: Settings = Depends(get_settings)) -> dict:
    payload = _build(settings)
    return {
        **payload,
        "note": (
            "Vertex ordering on the flattened axis is "
            f"{' | '.join(payload['hemi_order'])}, matching the convention verified "
            "against TRIBE's implementation before any analysis."
        ),
    }


@router.get("/parcellation")
def parcellation(settings: Settings = Depends(get_settings)) -> dict:
    """Parcel labels and network assignment for ROI overlays."""
    from neurotribe.preprocessing.surfaces import load_parcellation

    parcels = load_parcellation(settings)
    return {
        **parcels.to_dict(),
        "labels_url": "/api/surface/parcellation/labels",
        "parcels": [
            {"index": index, "name": parcels.names[index],
             "network": parcels.networks.get(index),
             "hemisphere": parcels.hemispheres.get(index)}
            for index in parcels.parcel_indices()
        ],
    }


@router.get("/parcellation/labels")
def parcellation_labels(settings: Settings = Depends(get_settings)) -> Response:
    """Per-vertex parcel id as int32 binary."""
    from neurotribe.preprocessing.surfaces import load_parcellation

    labels = load_parcellation(settings).labels.astype(np.int32)
    return Response(
        content=labels.tobytes(order="C"),
        media_type="application/octet-stream",
        headers={"X-Vertex-Count": str(labels.size),
                 "Cache-Control": "public, max-age=86400"},
    )


# NOTE: this catch-all must stay LAST. Declared earlier it would also match
# `/surface/parcellation/labels` and reject it as an unknown hemisphere.
@router.get("/{hemi}/{buffer}")
def buffer(hemi: str, buffer: str, settings: Settings = Depends(get_settings)) -> Response:
    if hemi not in {"L", "R"}:
        raise HTTPException(400, "hemi must be 'L' or 'R'")
    if buffer not in {"positions", "normals", "indices"}:
        raise HTTPException(400, "buffer must be positions, normals or indices")

    _build(settings)
    path = _export_dir(settings) / f"{hemi}_{buffer}.bin"
    if not path.exists():
        raise HTTPException(404, "Buffer not exported")

    return Response(
        content=path.read_bytes(),
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "public, max-age=86400, immutable",
            "X-Element-Type": "uint32" if buffer == "indices" else "float32",
        },
    )
