"""Builders for SYNTHETIC test fixtures.

Everything produced here is fake. It exists so the whole pipeline - acquisition,
preprocessing hand-off, TRIBE, alignment, deviation, statistics, reporting - can
be exercised before the DUA-controlled phenotype, the licensed stimulus and the
FreeSurfer license are available.

Nothing in this module is importable from ``neurotribe``; synthetic data can
never leak into a real analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

N_VERTICES_PER_HEMI = 10242


def write_fmriprep_outputs(
    out_dir: Path, participant_label: str, task: str, *,
    n_timepoints: int = 250, tr: float = 0.8, seed: int = 0,
    shared_signal: np.ndarray | None = None, shared_weight: float = 0.6,
    motion_scale: float = 1.0, space: str = "fsaverage5",
) -> dict[str, Path]:
    """Write GIFTI surfaces + a confounds TSV shaped exactly like fMRIPrep's.

    ``shared_signal`` lets several synthetic participants share a common
    stimulus-driven component, so a group analysis has real structure to find
    rather than pure noise.
    """
    import nibabel as nib

    rng = np.random.default_rng(seed)
    func = out_dir / f"sub-{participant_label}" / "func"
    func.mkdir(parents=True, exist_ok=True)

    if shared_signal is None:
        shared_signal = rng.standard_normal((n_timepoints, 12)).astype(np.float32)

    paths: dict[str, Path] = {}
    for hemi in ("L", "R"):
        # Spatially smooth loadings so neighbouring vertices behave alike.
        weights = rng.standard_normal((shared_signal.shape[1], N_VERTICES_PER_HEMI))
        kernel = np.ones(64) / 64.0
        for component in range(weights.shape[0]):
            weights[component] = np.convolve(weights[component], kernel, mode="same")

        signal = shared_weight * (shared_signal @ weights)
        signal += (1.0 - shared_weight) * rng.standard_normal(
            (n_timepoints, N_VERTICES_PER_HEMI)
        )
        signal = signal.astype(np.float32)

        darrays = [
            nib.gifti.GiftiDataArray(row, intent="NIFTI_INTENT_TIME_SERIES")
            for row in signal
        ]
        path = func / (
            f"sub-{participant_label}_task-{task}_space-{space}_hemi-{hemi}_bold.func.gii"
        )
        nib.save(nib.gifti.GiftiImage(darrays=darrays), str(path))
        paths[f"surface_{hemi}"] = path

    confounds_path = func / f"sub-{participant_label}_task-{task}_desc-confounds_timeseries.tsv"
    build_confounds(n_timepoints, seed=seed, motion_scale=motion_scale).to_csv(
        confounds_path, sep="\t", index=False, na_rep="n/a",
    )
    paths["confounds"] = confounds_path

    confounds_path.with_suffix(".json").write_text(
        json.dumps({"RepetitionTime": tr}, indent=2), encoding="utf-8",
    )
    paths["confounds_json"] = confounds_path.with_suffix(".json")
    return paths


def build_confounds(n_timepoints: int, *, seed: int = 0,
                    motion_scale: float = 1.0) -> pd.DataFrame:
    """A confounds table with the column families fMRIPrep actually emits."""
    rng = np.random.default_rng(seed + 5000)
    frame = pd.DataFrame({
        "trans_x": rng.normal(0, 0.1 * motion_scale, n_timepoints),
        "trans_y": rng.normal(0, 0.1 * motion_scale, n_timepoints),
        "trans_z": rng.normal(0, 0.1 * motion_scale, n_timepoints),
        "rot_x": rng.normal(0, 0.01 * motion_scale, n_timepoints),
        "rot_y": rng.normal(0, 0.01 * motion_scale, n_timepoints),
        "rot_z": rng.normal(0, 0.01 * motion_scale, n_timepoints),
        "global_signal": rng.normal(0, 1, n_timepoints),
        "csf": rng.normal(0, 1, n_timepoints),
        "white_matter": rng.normal(0, 1, n_timepoints),
    })

    fd = np.abs(rng.normal(0.12 * motion_scale, 0.06 * motion_scale, n_timepoints))
    fd[0] = np.nan                       # fMRIPrep leaves the first frame n/a
    frame["framewise_displacement"] = fd
    frame["std_dvars"] = np.abs(rng.normal(1.0, 0.1, n_timepoints))
    frame.loc[0, "std_dvars"] = np.nan

    for base in ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"):
        frame[f"{base}_derivative1"] = frame[base].diff()
        frame[f"{base}_power2"] = frame[base] ** 2
        frame[f"{base}_derivative1_power2"] = frame[f"{base}_derivative1"] ** 2

    for index in range(6):
        frame[f"a_comp_cor_{index:02d}"] = rng.normal(0, 1, n_timepoints)

    # Two non-steady-state volumes at the start, as fMRIPrep flags them.
    for index in range(2):
        column = np.zeros(n_timepoints)
        column[index] = 1.0
        frame[f"non_steady_state_outlier{index:02d}"] = column

    return frame


def make_shared_stimulus_signal(n_timepoints: int, n_components: int = 12,
                                seed: int = 42) -> np.ndarray:
    """A stimulus-locked component set shared by every synthetic participant."""
    rng = np.random.default_rng(seed)
    signal = np.zeros((n_timepoints, n_components), dtype=np.float32)
    for component in range(n_components):
        noise = rng.standard_normal(n_timepoints)
        smoothed = np.zeros(n_timepoints)
        for t in range(1, n_timepoints):
            smoothed[t] = 0.8 * smoothed[t - 1] + 0.2 * noise[t]
        signal[:, component] = smoothed
    return signal
