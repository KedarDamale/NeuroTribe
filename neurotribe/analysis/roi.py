"""ROI and network aggregation.

Vertex-wise maps are hard to interpret, so every subject metric is also
summarised onto a cortical parcellation and onto the canonical large-scale
networks.

Interpretation rule enforced by the reporting layer: a single network score is
never given a medical meaning on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neurotribe.config import Settings
from neurotribe.logging_setup import get_logger
from neurotribe.preprocessing.surfaces import Parcellation

log = get_logger(__name__)


@dataclass
class RoiSummary:
    roi_index: int
    roi_name: str
    network: str | None
    hemisphere: str | None
    n_vertices: int
    agreement_r: float | None = None
    mad: float | None = None
    residual_variance: float | None = None

    def to_dict(self) -> dict:
        return {
            "roi_index": self.roi_index, "roi_name": self.roi_name,
            "network": self.network, "hemisphere": self.hemisphere,
            "n_vertices": self.n_vertices, "agreement_r": self.agreement_r,
            "mad": self.mad, "residual_variance": self.residual_variance,
        }


@dataclass
class NetworkSummary:
    network: str
    n_vertices: int
    agreement_r: float | None = None
    mad: float | None = None
    residual_variance: float | None = None

    def to_dict(self) -> dict:
        return {
            "network": self.network, "n_vertices": self.n_vertices,
            "agreement_r": self.agreement_r, "mad": self.mad,
            "residual_variance": self.residual_variance,
        }


@dataclass
class Aggregation:
    rois: list[RoiSummary] = field(default_factory=list)
    networks: list[NetworkSummary] = field(default_factory=list)
    atlas: dict = field(default_factory=dict)

    def top_deviation_rois(self, n: int = 3) -> list[RoiSummary]:
        scored = [r for r in self.rois if r.mad is not None and np.isfinite(r.mad)]
        return sorted(scored, key=lambda r: -r.mad)[:n]

    def top_deviation_networks(self, n: int = 3) -> list[NetworkSummary]:
        scored = [x for x in self.networks if x.mad is not None and np.isfinite(x.mad)]
        return sorted(scored, key=lambda x: -x.mad)[:n]

    def lowest_agreement_rois(self, n: int = 3) -> list[RoiSummary]:
        scored = [r for r in self.rois
                  if r.agreement_r is not None and np.isfinite(r.agreement_r)]
        return sorted(scored, key=lambda r: r.agreement_r)[:n]

    def to_dict(self) -> dict:
        return {
            "atlas": self.atlas,
            "rois": [r.to_dict() for r in self.rois],
            "networks": [n.to_dict() for n in self.networks],
        }


def _reduce(values: np.ndarray, selection: np.ndarray, how: str) -> float | None:
    """Aggregate a vertex map over a parcel, ignoring invalid vertices."""
    chunk = values[selection]
    finite = chunk[np.isfinite(chunk)]
    if finite.size == 0:
        return None
    reducer = {"mean": np.mean, "median": np.median, "max": np.max}.get(how, np.mean)
    return float(reducer(finite))


def aggregate(vertex_r: np.ndarray, vertex_mad: np.ndarray,
              vertex_variance: np.ndarray, parcellation: Parcellation,
              settings: Settings) -> Aggregation:
    """Aggregate all three vertex maps to ROIs and networks in one pass."""
    how = str(settings.get("analysis.roi.aggregate", "mean"))
    if vertex_r.size != parcellation.labels.size:
        raise ValueError(
            f"Vertex map has {vertex_r.size} entries but the parcellation has "
            f"{parcellation.labels.size}."
        )

    aggregation = Aggregation(atlas=parcellation.to_dict())
    network_accumulator: dict[str, list[np.ndarray]] = {}

    for parcel_id in parcellation.parcel_indices():
        selection = parcellation.labels == parcel_id
        n_vertices = int(selection.sum())
        if n_vertices == 0:
            continue
        network = parcellation.networks.get(parcel_id)
        summary = RoiSummary(
            roi_index=parcel_id,
            roi_name=parcellation.names.get(parcel_id, f"parcel-{parcel_id}"),
            network=network,
            hemisphere=parcellation.hemispheres.get(parcel_id),
            n_vertices=n_vertices,
            agreement_r=_reduce(vertex_r, selection, how),
            mad=_reduce(vertex_mad, selection, how),
            residual_variance=_reduce(vertex_variance, selection, how),
        )
        aggregation.rois.append(summary)

        if network:
            network_accumulator.setdefault(network, []).append(selection)

    for network, selections in sorted(network_accumulator.items()):
        combined = np.zeros_like(parcellation.labels, dtype=bool)
        for selection in selections:
            combined |= selection
        aggregation.networks.append(NetworkSummary(
            network=network, n_vertices=int(combined.sum()),
            agreement_r=_reduce(vertex_r, combined, how),
            mad=_reduce(vertex_mad, combined, how),
            residual_variance=_reduce(vertex_variance, combined, how),
        ))

    if parcellation.is_approximate:
        aggregation.atlas["warning"] = (
            "This parcellation is a deterministic geometric fallback, not the "
            "Schaefer atlas. ROI names are NOT anatomical labels."
        )
    log.debug("ROI aggregation complete",
              extra={"n_rois": len(aggregation.rois), "n_networks": len(aggregation.networks)})
    return aggregation


def network_vertex_map(parcellation: Parcellation) -> dict[str, np.ndarray]:
    """Boolean vertex masks per network, used by the 3D viewer overlays."""
    masks: dict[str, np.ndarray] = {}
    for parcel_id, network in parcellation.networks.items():
        mask = masks.setdefault(network, np.zeros_like(parcellation.labels, dtype=bool))
        mask |= parcellation.labels == parcel_id
    return masks


def rolling_by_network(residual: np.ndarray, usable: np.ndarray,
                       parcellation: Parcellation) -> dict[str, np.ndarray]:
    """Per-network deviation time course, for the synchronised subject timeline."""
    out: dict[str, np.ndarray] = {}
    for network, mask in network_vertex_map(parcellation).items():
        if not mask.any():
            continue
        series = np.full(residual.shape[0], np.nan, dtype=np.float32)
        with np.errstate(invalid="ignore"):
            values = np.nanmean(np.abs(residual[:, mask]), axis=1)
        series[usable] = values[usable]
        out[network] = series
    return out
