"""Shared pytest fixtures.

Every fixture here is SYNTHETIC. Synthetic data lives only under
``tests/fixtures`` and is never mixed with a real analysis: the settings used
here carry ``profile='development'``, which is stamped into any provenance
manifest and disqualifies the run from being reported.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurotribe.config import Settings, load_settings, reset_settings_cache  # noqa: E402

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"

N_VERTICES_PER_HEMI = 10242
N_VERTICES = 2 * N_VERTICES_PER_HEMI


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    """An isolated project root with the real config files copied in."""
    root = tmp_path / "workspace"
    root.mkdir()
    shutil.copytree(Path(__file__).resolve().parent.parent / "config", root / "config")
    return root


@pytest.fixture()
def settings(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("NEUROTRIBE_ROOT", str(workspace))
    monkeypatch.setenv("NEUROTRIBE_PROFILE", "development")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(workspace / 'test.db').as_posix()}")
    reset_settings_cache()
    resolved = load_settings("development", workspace)
    resolved.paths.ensure()
    yield resolved
    reset_settings_cache()


@pytest.fixture()
def db(settings: Settings, monkeypatch: pytest.MonkeyPatch):
    """A fresh SQLite database bound to the isolated workspace."""
    from neurotribe.database.base import create_all, reset_engine, session_scope

    reset_engine()
    create_all()
    yield session_scope
    reset_engine()


@pytest.fixture()
def rng() -> np.random.Generator:
    """Deterministic RNG so every scientific test is reproducible."""
    return np.random.default_rng(20240101)


# --------------------------------------------------------------------------
# Synthetic signal builders
# --------------------------------------------------------------------------

def make_timeseries(n_timepoints: int, n_vertices: int, rng: np.random.Generator,
                    *, autocorr: float = 0.7) -> np.ndarray:
    """Temporally autocorrelated signal that behaves like BOLD."""
    noise = rng.standard_normal((n_timepoints, n_vertices)).astype(np.float32)
    out = np.zeros_like(noise)
    out[0] = noise[0]
    for t in range(1, n_timepoints):
        out[t] = autocorr * out[t - 1] + (1 - autocorr) * noise[t]
    return out


@pytest.fixture()
def synthetic_pair(rng: np.random.Generator):
    """A (predicted, observed) pair with a known ground-truth relationship."""
    n_timepoints, n_vertices = 200, 64
    predicted = make_timeseries(n_timepoints, n_vertices, rng)
    # Observed = predicted + independent noise, so correlation is high but < 1.
    observed = predicted + 0.5 * make_timeseries(n_timepoints, n_vertices, rng)
    return predicted, observed


@pytest.fixture()
def confounds_frame():
    """A minimal fMRIPrep-style confounds table."""
    import pandas as pd

    n = 120
    generator = np.random.default_rng(7)
    frame = pd.DataFrame({
        "trans_x": generator.normal(0, 0.1, n),
        "trans_y": generator.normal(0, 0.1, n),
        "trans_z": generator.normal(0, 0.1, n),
        "rot_x": generator.normal(0, 0.01, n),
        "rot_y": generator.normal(0, 0.01, n),
        "rot_z": generator.normal(0, 0.01, n),
        "framewise_displacement": np.abs(generator.normal(0.15, 0.08, n)),
        "std_dvars": np.abs(generator.normal(1.0, 0.15, n)),
        "global_signal": generator.normal(0, 1, n),
    })
    for base in ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"):
        frame[f"{base}_derivative1"] = frame[base].diff()
        frame[f"{base}_power2"] = frame[base] ** 2
        frame[f"{base}_derivative1_power2"] = frame[f"{base}_derivative1"] ** 2
    for index in range(6):
        frame[f"a_comp_cor_{index:02d}"] = generator.normal(0, 1, n)
    # fMRIPrep flags the first two volumes as non-steady-state.
    frame["non_steady_state_outlier00"] = [1.0] + [0.0] * (n - 1)
    frame["non_steady_state_outlier01"] = [0.0, 1.0] + [0.0] * (n - 2)
    # Inject a motion spike that censoring must catch.
    frame.loc[60, "framewise_displacement"] = 1.8
    return frame


@pytest.fixture()
def phenotype_csv(settings: Settings) -> Path:
    """A synthetic clinician-consensus export (LORIS-style column names)."""
    import csv

    path = settings.paths.phenotype_incoming / "synthetic_diagnosis.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for index in range(40):
        external_id = f"NDARSYN{index:05d}"
        if index < 15:
            rows.append({"Anonymized ID": external_id, "DX_01": "ADHD-Combined Type",
                         "DX_01_Conf": "Confirmed", "DX_02": "", "DX_02_Conf": ""})
        elif index < 22:
            rows.append({"Anonymized ID": external_id, "DX_01": "ADHD-Inattentive Type",
                         "DX_01_Conf": "Presumptive", "DX_02": "", "DX_02_Conf": ""})
        elif index < 34:
            rows.append({"Anonymized ID": external_id, "DX_01": "No Diagnosis Given",
                         "DX_01_Conf": "No Diagnosis Given", "DX_02": "", "DX_02_Conf": ""})
        else:
            rows.append({"Anonymized ID": external_id, "DX_01": "Specific Learning Disorder",
                         "DX_01_Conf": "Confirmed", "DX_02": "", "DX_02_Conf": ""})

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["Anonymized ID", "DX_01", "DX_01_Conf", "DX_02", "DX_02_Conf"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture()
def metadata_csv(settings: Settings) -> Path:
    """A synthetic HBN release metadata table."""
    import csv

    path = settings.paths.metadata / "Metadata_SYNTHETIC.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Anonymized ID", "Site", "Age", "Sex", "Release_Number",
                        "Commercial_Use", "MRI"],
        )
        writer.writeheader()
        for index in range(40):
            writer.writerow({
                "Anonymized ID": f"NDARSYN{index:05d}",
                "Site": ["RU", "CBIC", "CUNY"][index % 3],
                "Age": round(8 + (index % 10) + 0.5, 1),
                "Sex": "M" if index % 2 else "F",
                "Release_Number": "SYNTHETIC",
                "Commercial_Use": "1",
                "MRI": "1",
            })
    return path


@pytest.fixture()
def mriqc_csv(settings: Settings) -> Path:
    """A synthetic MRIQC functional IQM export."""
    import csv

    path = settings.paths.metadata / "IQM_functional_SYNTHETIC.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(11)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["bids_name", "fd_mean", "dvars_std", "tsnr", "efc", "fber",
                        "gsr_x", "some_future_iqm"],
        )
        writer.writeheader()
        for index in range(40):
            writer.writerow({
                "bids_name": f"sub-NDARSYN{index:05d}_task-movieDM_bold",
                "fd_mean": round(float(abs(generator.normal(0.18, 0.09))), 4),
                "dvars_std": round(float(abs(generator.normal(1.05, 0.12))), 4),
                "tsnr": round(float(abs(generator.normal(45, 6))), 3),
                "efc": 0.5, "fber": 1500.0, "gsr_x": 0.02,
                # An unknown column must be preserved, not dropped.
                "some_future_iqm": 42.0,
            })
    return path


def write_bids_dataset(root: Path, subjects: list[str], *, tr: float = 0.8,
                       n_volumes: int = 750, task: str = "movieDM") -> Path:
    """Create a minimal but valid BIDS tree with real (tiny) NIfTI files."""
    import nibabel as nib

    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "Synthetic HBN-like dataset", "BIDSVersion": "1.8.0",
                    "DatasetType": "raw"}, indent=2),
        encoding="utf-8",
    )

    generator = np.random.default_rng(3)
    for label in subjects:
        func = root / f"sub-{label}" / "func"
        anat = root / f"sub-{label}" / "anat"
        func.mkdir(parents=True, exist_ok=True)
        anat.mkdir(parents=True, exist_ok=True)

        # Tiny 4-D volume: the indexer only reads the header for the shape.
        bold = nib.Nifti1Image(
            generator.standard_normal((4, 4, 4, n_volumes)).astype(np.float32), np.eye(4),
        )
        bold_path = func / f"sub-{label}_task-{task}_bold.nii.gz"
        nib.save(bold, str(bold_path))
        bold_path.with_name(bold_path.name.replace(".nii.gz", ".json")).write_text(
            json.dumps({"RepetitionTime": tr, "EchoTime": 0.03, "TaskName": task,
                        "Manufacturer": "Siemens", "ManufacturersModelName": "Prisma",
                        "InstitutionName": "SYNTHETIC"}, indent=2),
            encoding="utf-8",
        )

        t1 = nib.Nifti1Image(generator.standard_normal((4, 4, 4)).astype(np.float32), np.eye(4))
        nib.save(t1, str(anat / f"sub-{label}_T1w.nii.gz"))

    return root
