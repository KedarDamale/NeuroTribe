"""fsaverage5 surface handling: GIFTI I/O, hemisphere concatenation, atlases.

The single most dangerous silent failure in this project is a hemisphere or
vertex-ordering mismatch between the TRIBE prediction and the observed BOLD.
Everything here therefore carries an explicit, recorded convention:

    flattened vertex axis = [ left hemisphere (10242) | right hemisphere (10242) ]

The ordering is *configuration*, and :mod:`neurotribe.tribe.geometry` verifies
it against TRIBE's own implementation before any analysis is permitted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from neurotribe.config import Settings
from neurotribe.logging_setup import get_logger

log = get_logger(__name__)

Hemisphere = Literal["L", "R"]
FSAVERAGE5_VERTICES_PER_HEMI = 10242
FSAVERAGE5_TOTAL_VERTICES = 20484


class SurfaceError(RuntimeError):
    """Geometry or I/O failure that must abort an analysis rather than degrade it."""


# --------------------------------------------------------------------------
# GIFTI time series
# --------------------------------------------------------------------------

def load_gifti_timeseries(path: Path) -> np.ndarray:
    """Load a ``*.func.gii`` surface time series as (n_timepoints, n_vertices)."""
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover
        raise SurfaceError("nibabel is required to read GIFTI surfaces") from exc

    image = nib.load(str(path))
    arrays = [np.asarray(darray.data, dtype=np.float32) for darray in image.darrays]
    if not arrays:
        raise SurfaceError(f"GIFTI file contains no data arrays: {path}")

    if len(arrays) == 1 and arrays[0].ndim == 2:
        data = arrays[0]
        # A single 2-D array may be (vertices, time) or (time, vertices).
        if data.shape[0] == FSAVERAGE5_VERTICES_PER_HEMI:
            data = data.T
        return np.ascontiguousarray(data)

    # The usual fMRIPrep layout: one darray per timepoint, each n_vertices long.
    stacked = np.vstack([a.reshape(1, -1) for a in arrays])
    return np.ascontiguousarray(stacked)


def save_gifti_timeseries(data: np.ndarray, path: Path) -> Path:
    """Write (n_timepoints, n_vertices) back out as a GIFTI func file."""
    import nibabel as nib

    path.parent.mkdir(parents=True, exist_ok=True)
    darrays = [
        nib.gifti.GiftiDataArray(np.asarray(row, dtype=np.float32), intent="NIFTI_INTENT_TIME_SERIES")
        for row in np.atleast_2d(data)
    ]
    nib.save(nib.gifti.GiftiImage(darrays=darrays), str(path))
    return path


def concatenate_hemispheres(left: np.ndarray, right: np.ndarray,
                            order: list[str] | None = None) -> np.ndarray:
    """Join hemispheres into the flattened vertex axis using the recorded order."""
    order = order or ["L", "R"]
    if left.shape[0] != right.shape[0]:
        raise SurfaceError(
            f"Hemispheres have different timepoint counts: L={left.shape[0]}, R={right.shape[0]}"
        )
    if order == ["L", "R"]:
        return np.hstack([left, right])
    if order == ["R", "L"]:
        return np.hstack([right, left])
    raise SurfaceError(f"Unsupported hemisphere order: {order}")


def split_hemispheres(data: np.ndarray, order: list[str] | None = None,
                      per_hemi: int = FSAVERAGE5_VERTICES_PER_HEMI) -> dict[str, np.ndarray]:
    """Inverse of :func:`concatenate_hemispheres`."""
    order = order or ["L", "R"]
    if data.shape[-1] != 2 * per_hemi:
        raise SurfaceError(
            f"Expected {2 * per_hemi} vertices, got {data.shape[-1]}"
        )
    first, second = data[..., :per_hemi], data[..., per_hemi:]
    return {order[0]: first, order[1]: second}


def validate_shape(data: np.ndarray, settings: Settings, *, label: str) -> None:
    """Hard-fail when the vertex count does not match the configured surface."""
    expected = int(settings.get("surface.total_vertices", FSAVERAGE5_TOTAL_VERTICES))
    if data.ndim != 2:
        raise SurfaceError(f"{label}: expected 2-D (time x vertices), got shape {data.shape}")
    if data.shape[1] != expected:
        raise SurfaceError(
            f"{label}: vertex count {data.shape[1]} does not match configured "
            f"{settings.get('surface.space')} surface ({expected} vertices). "
            "Analysis aborted rather than silently reshaped."
        )


# --------------------------------------------------------------------------
# fsaverage5 geometry
# --------------------------------------------------------------------------

@dataclass
class SurfaceGeometry:
    """Vertex coordinates and faces per hemisphere, for rendering and atlases."""

    coordinates: dict[str, np.ndarray] = field(default_factory=dict)
    faces: dict[str, np.ndarray] = field(default_factory=dict)
    source: str = "unknown"

    @property
    def available(self) -> bool:
        return bool(self.coordinates)

    def n_vertices(self, hemi: str) -> int:
        return int(self.coordinates[hemi].shape[0]) if hemi in self.coordinates else 0


def _fetch_fsaverage5_via_nilearn(mesh: str = "pial") -> SurfaceGeometry | None:
    try:
        from nilearn import datasets, surface
    except ImportError:
        return None
    try:
        fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")
    except Exception as exc:  # noqa: BLE001 - network/cache failures are expected offline
        log.info("fsaverage5 mesh unavailable from nilearn", extra={"error": str(exc)})
        return None

    geometry = SurfaceGeometry(source=f"nilearn:fsaverage5:{mesh}")
    for hemi, key in (("L", f"{mesh}_left"), ("R", f"{mesh}_right")):
        try:
            coords, faces = surface.load_surf_mesh(fsaverage[key])
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not load fsaverage5 hemisphere",
                        extra={"hemi": hemi, "error": str(exc)})
            return None
        geometry.coordinates[hemi] = np.asarray(coords, dtype=np.float32)
        geometry.faces[hemi] = np.asarray(faces, dtype=np.int32)
    return geometry


def _icosphere(subdivisions: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Generate an icosphere.

    fsaverage5 is an order-5 subdivided icosahedron (10242 vertices per
    hemisphere), so this reproduces the correct vertex count and a
    topologically faithful mesh when the real geometry cannot be fetched.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=np.float64)
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)

    vertex_list = [tuple(v) for v in vertices]
    for _ in range(subdivisions):
        midpoint_cache: dict[tuple[int, int], int] = {}
        new_faces: list[list[int]] = []

        def midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key in midpoint_cache:
                return midpoint_cache[key]
            point = (np.array(vertex_list[a]) + np.array(vertex_list[b])) / 2.0
            point /= np.linalg.norm(point)
            vertex_list.append(tuple(point))
            index = len(vertex_list) - 1
            midpoint_cache[key] = index
            return index

        for tri in faces:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = np.array(new_faces, dtype=np.int64)

    return np.array(vertex_list, dtype=np.float32), faces.astype(np.int32)


def load_geometry(settings: Settings, *, mesh: str = "pial") -> SurfaceGeometry:
    """Load fsaverage5 geometry, caching it under ``data/derivatives/surfaces``."""
    cache_dir = settings.paths.derivatives / "surfaces"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"fsaverage5_{mesh}.npz"

    if cache_file.exists():
        with np.load(cache_file, allow_pickle=False) as payload:
            geometry = SurfaceGeometry(source=str(payload["source"]))
            for hemi in ("L", "R"):
                geometry.coordinates[hemi] = payload[f"coords_{hemi}"]
                geometry.faces[hemi] = payload[f"faces_{hemi}"]
        return geometry

    geometry = _fetch_fsaverage5_via_nilearn(mesh)
    if geometry is None:
        coords, faces = _icosphere(5)
        if coords.shape[0] != FSAVERAGE5_VERTICES_PER_HEMI:
            raise SurfaceError(
                f"Generated icosphere has {coords.shape[0]} vertices, expected "
                f"{FSAVERAGE5_VERTICES_PER_HEMI}"
            )
        geometry = SurfaceGeometry(source="generated:icosphere-order5")
        # Mirror across x so left and right are anatomically distinguishable.
        geometry.coordinates["L"] = coords * np.array([-1.0, 1.0, 1.0], dtype=np.float32) * 70.0
        geometry.coordinates["R"] = coords * 70.0
        geometry.faces["L"] = faces
        geometry.faces["R"] = faces
        log.warning(
            "Using a generated fsaverage5-topology sphere: the real mesh could not be "
            "fetched. Vertex counts are correct; anatomical coordinates are not.",
        )

    np.savez_compressed(
        cache_file, source=geometry.source,
        coords_L=geometry.coordinates["L"], coords_R=geometry.coordinates["R"],
        faces_L=geometry.faces["L"], faces_R=geometry.faces["R"],
    )
    return geometry


# --------------------------------------------------------------------------
# Parcellation
# --------------------------------------------------------------------------

@dataclass
class Parcellation:
    """Vertex -> parcel mapping over the flattened surface."""

    labels: np.ndarray                     # int, shape (total_vertices,), 0 = unassigned
    names: dict[int, str] = field(default_factory=dict)
    networks: dict[int, str] = field(default_factory=dict)
    hemispheres: dict[int, str] = field(default_factory=dict)
    source: str = "unknown"
    is_approximate: bool = False
    n_parcels: int = 0

    def to_dict(self) -> dict:
        return {
            "source": self.source, "is_approximate": self.is_approximate,
            "n_parcels": self.n_parcels,
            "networks": sorted(set(self.networks.values())),
        }

    def parcel_indices(self) -> list[int]:
        return sorted(i for i in self.names if i > 0)

    def medial_wall_mask(self) -> np.ndarray:
        """True where a vertex belongs to no parcel (medial wall / unknown)."""
        return self.labels <= 0


def _load_annot_parcellation(settings: Settings) -> Parcellation | None:
    """Load Schaefer ``.annot`` files if the operator supplied them."""
    atlas_dir = settings.paths.data / "atlas"
    if not atlas_dir.exists():
        return None
    n_parcels = int(settings.get("surface.atlas.n_parcels", 200))
    n_networks = int(settings.get("surface.atlas.n_networks", 7))

    patterns = [
        f"*h.Schaefer2018_{n_parcels}Parcels_{n_networks}Networks_order.annot",
        f"*Schaefer*{n_parcels}Parcels*{n_networks}Networks*.annot",
        "*.annot",
    ]
    found: dict[str, Path] = {}
    for pattern in patterns:
        for path in sorted(atlas_dir.rglob(pattern)):
            name = path.name.lower()
            hemi = "L" if name.startswith("lh") or ".lh." in name or "_lh" in name else (
                "R" if name.startswith("rh") or ".rh." in name or "_rh" in name else None
            )
            if hemi and hemi not in found:
                found[hemi] = path
        if len(found) == 2:
            break
    if len(found) != 2:
        return None

    try:
        import nibabel as nib
    except ImportError:
        return None

    per_hemi = int(settings.get("surface.vertices_per_hemi", FSAVERAGE5_VERTICES_PER_HEMI))
    order = list(settings.get("surface.hemi_order", ["L", "R"]))
    labels = np.zeros(2 * per_hemi, dtype=np.int32)
    names: dict[int, str] = {}
    networks: dict[int, str] = {}
    hemispheres: dict[int, str] = {}
    offset = 0

    for hemi in order:
        annot_labels, _ctab, annot_names = nib.freesurfer.read_annot(str(found[hemi]))
        if annot_labels.shape[0] != per_hemi:
            log.warning("Annot vertex count mismatch",
                        extra={"hemi": hemi, "found": int(annot_labels.shape[0]),
                               "expected": per_hemi})
            return None
        decoded = [n.decode() if isinstance(n, bytes) else str(n) for n in annot_names]
        start = offset * per_hemi
        for local_index, name in enumerate(decoded):
            if local_index == 0 or "unknown" in name.lower() or "medial_wall" in name.lower():
                continue
            # Global parcel ids must not collide across hemispheres.
            global_id = len(names) + 1
            names[global_id] = name
            hemispheres[global_id] = hemi
            networks[global_id] = _network_from_name(name, settings)
            labels[start:start + per_hemi][annot_labels == local_index] = global_id
        offset += 1

    return Parcellation(
        labels=labels, names=names, networks=networks, hemispheres=hemispheres,
        source=f"annot:{found['L'].name}", is_approximate=False, n_parcels=len(names),
    )


def _network_from_name(name: str, settings: Settings) -> str:
    """Extract the canonical network from a Schaefer parcel name."""
    lowered = name.lower()
    table = {
        "vis": "Visual", "somMot": "Somatomotor", "sommot": "Somatomotor",
        "dorsattn": "DorsalAttention", "salventattn": "SalienceVentralAttention",
        "limbic": "Limbic", "cont": "Control", "default": "Default",
    }
    for token, network in table.items():
        if token.lower() in lowered:
            return network
    for network in settings.get("surface.atlas.networks", []):
        if network.lower() in lowered:
            return network
    return "Unassigned"


def _geometric_parcellation(settings: Settings, geometry: SurfaceGeometry) -> Parcellation:
    """Deterministic spatial parcellation used when no real atlas is available.

    This is an explicit, reproducible fallback so the pipeline remains testable
    offline. It is flagged ``is_approximate=True`` and must never be presented
    as the Schaefer atlas.
    """
    per_hemi = int(settings.get("surface.vertices_per_hemi", FSAVERAGE5_VERTICES_PER_HEMI))
    n_parcels = int(settings.get("surface.atlas.n_parcels", 200))
    order = list(settings.get("surface.hemi_order", ["L", "R"]))
    per_hemi_parcels = max(1, n_parcels // 2)

    labels = np.zeros(2 * per_hemi, dtype=np.int32)
    names: dict[int, str] = {}
    networks: dict[int, str] = {}
    hemispheres: dict[int, str] = {}
    network_names = list(settings.get("surface.atlas.networks", [])) or ["Unassigned"]
    next_id = 1

    for slot, hemi in enumerate(order):
        coords = geometry.coordinates.get(hemi)
        if coords is None or coords.shape[0] != per_hemi:
            coords = _icosphere(5)[0]
        centroids = _farthest_point_sample(coords, per_hemi_parcels, seed=1234 + slot)
        assignment = _nearest_centroid(coords, centroids)

        start = slot * per_hemi
        for local in range(per_hemi_parcels):
            global_id = next_id
            next_id += 1
            network = network_names[local % len(network_names)]
            names[global_id] = f"{hemi}_geo_{local + 1:03d}_{network}"
            networks[global_id] = network
            hemispheres[global_id] = hemi
            labels[start:start + per_hemi][assignment == local] = global_id

    return Parcellation(
        labels=labels, names=names, networks=networks, hemispheres=hemispheres,
        source="generated:farthest-point-geometric", is_approximate=True,
        n_parcels=len(names),
    )


def _farthest_point_sample(points: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Deterministic farthest-point sampling producing evenly spread centroids."""
    rng = np.random.default_rng(seed)
    n = points.shape[0]
    k = min(k, n)
    chosen = np.empty(k, dtype=np.int64)
    chosen[0] = int(rng.integers(0, n))
    distances = np.linalg.norm(points - points[chosen[0]], axis=1)
    for index in range(1, k):
        chosen[index] = int(np.argmax(distances))
        distances = np.minimum(distances, np.linalg.norm(points - points[chosen[index]], axis=1))
    return points[chosen]


def _nearest_centroid(points: np.ndarray, centroids: np.ndarray,
                      block: int = 2048) -> np.ndarray:
    """Assign each point to its nearest centroid, blocked to bound memory."""
    out = np.empty(points.shape[0], dtype=np.int32)
    for start in range(0, points.shape[0], block):
        chunk = points[start:start + block]
        distances = np.linalg.norm(chunk[:, None, :] - centroids[None, :, :], axis=2)
        out[start:start + block] = np.argmin(distances, axis=1)
    return out


def load_parcellation(settings: Settings) -> Parcellation:
    """Resolve the configured atlas, caching the result."""
    cache_dir = settings.paths.derivatives / "surfaces"
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_parcels = int(settings.get("surface.atlas.n_parcels", 200))
    cache_file = cache_dir / f"parcellation_{settings.get('surface.atlas.name')}_{n_parcels}.npz"
    meta_file = cache_file.with_suffix(".json")

    if cache_file.exists() and meta_file.exists():
        with np.load(cache_file, allow_pickle=False) as payload:
            labels = payload["labels"]
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        return Parcellation(
            labels=labels,
            names={int(k): v for k, v in meta["names"].items()},
            networks={int(k): v for k, v in meta["networks"].items()},
            hemispheres={int(k): v for k, v in meta["hemispheres"].items()},
            source=meta["source"], is_approximate=meta["is_approximate"],
            n_parcels=meta["n_parcels"],
        )

    parcellation = _load_annot_parcellation(settings)
    if parcellation is None:
        geometry = load_geometry(settings)
        parcellation = _geometric_parcellation(settings, geometry)
        log.warning(
            "No Schaefer .annot atlas found under data/atlas; using a deterministic "
            "geometric parcellation flagged as approximate. ROI names are NOT "
            "anatomical labels.",
            extra={"n_parcels": parcellation.n_parcels},
        )

    np.savez_compressed(cache_file, labels=parcellation.labels)
    meta_file.write_text(json.dumps({
        "names": {str(k): v for k, v in parcellation.names.items()},
        "networks": {str(k): v for k, v in parcellation.networks.items()},
        "hemispheres": {str(k): v for k, v in parcellation.hemispheres.items()},
        "source": parcellation.source,
        "is_approximate": parcellation.is_approximate,
        "n_parcels": parcellation.n_parcels,
    }, indent=2), encoding="utf-8")
    return parcellation


def aggregate_by_parcel(values: np.ndarray, parcellation: Parcellation,
                        how: str = "mean") -> dict[int, float]:
    """Aggregate a per-vertex map to parcels, ignoring NaN."""
    if values.shape[-1] != parcellation.labels.size:
        raise SurfaceError(
            f"Value array has {values.shape[-1]} vertices, parcellation has "
            f"{parcellation.labels.size}"
        )
    out: dict[int, float] = {}
    reducer = {"mean": np.nanmean, "median": np.nanmedian, "max": np.nanmax}.get(how, np.nanmean)
    for parcel_id in parcellation.parcel_indices():
        selection = values[..., parcellation.labels == parcel_id]
        if selection.size == 0:
            continue
        with np.errstate(invalid="ignore"):
            value = float(reducer(selection))
        out[parcel_id] = value if np.isfinite(value) else float("nan")
    return out


def aggregate_by_network(values: np.ndarray, parcellation: Parcellation,
                         how: str = "mean") -> dict[str, float]:
    """Aggregate a per-vertex map to canonical large-scale networks."""
    reducer = {"mean": np.nanmean, "median": np.nanmedian}.get(how, np.nanmean)
    by_network: dict[str, list[np.ndarray]] = {}
    for parcel_id, network in parcellation.networks.items():
        selection = values[..., parcellation.labels == parcel_id]
        if selection.size:
            by_network.setdefault(network, []).append(selection.ravel())
    out: dict[str, float] = {}
    for network, chunks in by_network.items():
        combined = np.concatenate(chunks)
        with np.errstate(invalid="ignore"):
            value = float(reducer(combined))
        out[network] = value if np.isfinite(value) else float("nan")
    return out
