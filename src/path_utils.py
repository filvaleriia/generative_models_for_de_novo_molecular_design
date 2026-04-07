from __future__ import annotations

from pathlib import Path

SCAFFOLD_RESULTS_DIRNAME = "results_scaffold_based"
PHARM_RESULTS_DIRNAME = "results_pharm_based"
COMPARISON_OUTPUTS_DIRNAME = "comparison_outputs"
BOOTSTRAP_OUTPUTS_DIRNAME = "bootstrap_threshold_analysis"


def resolve_data_folder(data_folder: str | Path | None = "") -> Path:
    """
    Resolve the user-provided project root that contains the ``data/`` directory.

    If ``data_folder`` is empty, the current working directory is treated as the
    project root. This matches notebook usage where users usually open the
    notebook in the repository root.
    """
    if data_folder in (None, ""):
        return Path.cwd().resolve()
    return Path(data_folder).expanduser().resolve()


def data_subdir(data_folder: str | Path | None, *parts: str) -> Path:
    """Build a path inside the project ``data/`` directory."""
    return resolve_data_folder(data_folder).joinpath("data", *parts)


def scaffold_results_dir(data_folder: str | Path | None, *parts: str) -> Path:
    """Build a path inside the canonical scaffold results directory."""
    return data_subdir(data_folder, SCAFFOLD_RESULTS_DIRNAME, *parts)


def pharm_results_dir(data_folder: str | Path | None, *parts: str) -> Path:
    """Build a path inside the canonical pharmacophore results directory."""
    return data_subdir(data_folder, PHARM_RESULTS_DIRNAME, *parts)


def comparison_outputs_dir(data_folder: str | Path | None, *parts: str) -> Path:
    """Build a path inside the canonical comparison outputs directory."""
    return data_subdir(data_folder, COMPARISON_OUTPUTS_DIRNAME, *parts)


def comparison_overlap_dir(
    data_folder: str | Path | None,
    receptor: str,
    *parts: str,
) -> Path:
    """Build a path for overlap/intermediate comparison outputs."""
    return comparison_outputs_dir(data_folder, "overlap", receptor, *parts)


def comparison_umap_dir(
    data_folder: str | Path | None,
    receptor: str,
    *parts: str,
) -> Path:
    """Build a path for UMAP data and figures."""
    return comparison_outputs_dir(data_folder, "umap", receptor, *parts)


def comparison_cross_modality_dir(
    data_folder: str | Path | None,
    *parts: str,
) -> Path:
    """Build a path for cross-modality summary outputs."""
    return comparison_outputs_dir(data_folder, "cross_modality", *parts)


def comparison_figure_dir(data_folder: str | Path | None, *parts: str) -> Path:
    """Build a path for comparison-level figures outside the overlap/UMAP subtrees."""
    return comparison_outputs_dir(data_folder, "figures", *parts)


def bootstrap_outputs_dir(data_folder: str | Path | None, *parts: str) -> Path:
    """Build a path inside the canonical bootstrap intermediate-output directory."""
    return data_subdir(data_folder, BOOTSTRAP_OUTPUTS_DIRNAME, *parts)
