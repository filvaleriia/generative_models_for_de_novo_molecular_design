from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


METRICS = ("RS", "SED", "ASER")
METRIC_RENAME = {"TUPOR": "RS", "SESY": "SED"}
RECEPTOR_ORDER = ("Glucocorticoid_receptor", "Leukocyte_elastase")
RECEPTOR_LABELS = {
    "Glucocorticoid_receptor": "Glucocorticoid receptor",
    "Leukocyte_elastase": "Leukocyte elastase",
}
SPLIT_ORDER = ("dis", "sim")
PANEL_ORDER = (("csk", "dis"), ("csk", "sim"), ("murcko", "dis"), ("murcko", "sim"))
STRUCTURAL_UNIT_LABELS = {
    "csk": "CSK",
    "murcko": "MURCKO",
    "rdkit": "pharmacophore fingerprint",
}
STRUCTURAL_UNIT_ORDER = ("CSK", "MURCKO", "pharmacophore fingerprint")
MODEL_ORDER = ("DrugEx_RNN", "DrugEx_GT", "REINVENT")
VARIANT_ORDER = {
    "DrugEx_RNN": ["without_fine_tuning", "epsilon_0.1_250k", "epsilon_0.6_250k"],
    "DrugEx_GT": ["without_fine_tuning", "epsilon_0.1_250k", "epsilon_0.6_250k"],
    "REINVENT": ["without_fine_tuning", "fine_tuned_250k"],
}
VARIANT_LABELS = {
    "without_fine_tuning": "without fine-tuning",
    "epsilon_0.1_250k": "fine-tuned, epsilon 0.1",
    "epsilon_0.6_250k": "fine-tuned, epsilon 0.6",
    "fine_tuned_250k": "fine-tuned",
    "fine_tuned_best": "best fine-tuned",
}
VARIANT_COLORS = {
    "without_fine_tuning": "#4c72b0",
    "epsilon_0.1_250k": "#dd8452",
    "epsilon_0.6_250k": "#55a868",
    "fine_tuned_250k": "#dd8452",
    "fine_tuned_best": "#8172b2",
}
WITHOUT_FINE_TUNING_PH4_GENERATORS = (
    "DrugEx_RNN_w_w",
    "DrugEx_GT_w_w",
    "REINVENT_w_w",
)
WITHOUT_FINE_TUNING_PH4_SPLITS = ("dis", "sim")
WITHOUT_FINE_TUNING_PH4_CLUSTERS = (0, 1, 2, 3, 4)
PH4_THRESHOLD_BY_SPLIT = {"dis": 0.7, "sim": 0.8}
METRIC_BASE_COLORS = {"RS": "#e97b32", "SED": "#97C2F0", "ASER": "#71ad48"}
SHORT_VARIANT_LABELS = {
    ("DrugEx_RNN", "epsilon_0.1_250k"): "DrugEx RNN (ε = 0.1)",
    ("DrugEx_RNN", "epsilon_0.6_250k"): "DrugEx RNN (ε = 0.6)",
    ("DrugEx_GT", "epsilon_0.1_250k"): "DrugEx GT (ε = 0.1)",
    ("DrugEx_GT", "epsilon_0.6_250k"): "DrugEx GT (ε = 0.6)",
    ("REINVENT", "fine_tuned_250k"): "REINVENT",
}


@dataclass
class FineTuningComparisonConfig:
    project_root: Path
    data_folder: Path
    fine_tuned_results_dir: Path
    without_fine_tuning_results_dir: Path
    output_dir: Path
    ph4_results_dir: Path
    without_fine_tuning_source_dir: Path
    without_fine_tuning_ph4_results_dir: Path
    canonical_output_sets_dir: Path
    ph4_output_sets_dir: Path
    metrics: tuple[str, ...] = METRICS
    receptor_order: tuple[str, ...] = RECEPTOR_ORDER
    split_order: tuple[str, ...] = SPLIT_ORDER
    structural_unit_order: tuple[str, ...] = STRUCTURAL_UNIT_ORDER
    model_order: tuple[str, ...] = MODEL_ORDER
    variant_order: dict[str, list[str]] = field(
        default_factory=lambda: {key: value[:] for key, value in VARIANT_ORDER.items()}
    )


def build_default_config(project_root: Path | None = None) -> FineTuningComparisonConfig:
    project_root = (project_root or Path.cwd()).resolve()
    data_folder = (project_root / ".." / "..").resolve()
    output_dir = data_folder / "data" / "fine_tuning_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    without_ft_source = data_folder / "data" / "generators_without_finetuning"
    canonical_output_sets_dir = data_folder / "data" / "output_sets"

    return FineTuningComparisonConfig(
        project_root=project_root,
        data_folder=data_folder,
        fine_tuned_results_dir=data_folder / "data" / "results_scaffold_based",
        without_fine_tuning_results_dir=without_ft_source / "results",
        output_dir=output_dir,
        ph4_results_dir=data_folder / "data" / "results_pharm_based",
        without_fine_tuning_source_dir=without_ft_source,
        without_fine_tuning_ph4_results_dir=without_ft_source / "results_ph4_based",
        canonical_output_sets_dir=canonical_output_sets_dir,
        ph4_output_sets_dir=canonical_output_sets_dir / "ph4",
    )


def print_config_summary(config: FineTuningComparisonConfig) -> None:
    print("Project root:", config.project_root)
    print("Data folder:", config.data_folder)
    print("Fine-tuned scaffold results:", config.fine_tuned_results_dir)
    print("Without fine-tuning scaffold results:", config.without_fine_tuning_results_dir)
    print("Canonical PH4 metric results:", config.ph4_results_dir)
    print("Without-fine-tuning PH4 metric results:", config.without_fine_tuning_ph4_results_dir)
    print("Output directory:", config.output_dir)


def stage_without_finetuning_output_sets(
    config: FineTuningComparisonConfig,
    overwrite: bool = False,
) -> list[Path]:
    """Copy output-set CSV files into the canonical output_sets tree."""
    copied: list[Path] = []

    for receptor in config.receptor_order:
        for generator in WITHOUT_FINE_TUNING_PH4_GENERATORS:
            source_dir = config.without_fine_tuning_source_dir / receptor / generator
            target_dir = config.canonical_output_sets_dir / receptor / generator
            target_dir.mkdir(parents=True, exist_ok=True)
            for split in WITHOUT_FINE_TUNING_PH4_SPLITS:
                for cluster in WITHOUT_FINE_TUNING_PH4_CLUSTERS:
                    filename = f"cOS_{generator}_{split}_{cluster}_one_column.csv"
                    source_path = source_dir / filename
                    target_path = target_dir / filename
                    if not source_path.exists():
                        print(f"Missing without-fine-tuning source file: {source_path}")
                        continue
                    if target_path.exists() and not overwrite:
                        print(f"Staging skipped (already exists): {target_path}")
                        continue
                    shutil.copy2(source_path, target_path)
                    copied.append(target_path)
                    print(f"Staged: {source_path} -> {target_path}")

    print(f"Staged {len(copied)} output-set CSV files.")
    return copied


def run_without_finetuning_ph4_precompute(
    config: FineTuningComparisonConfig,
    ncpus: int = 120,
    chunksize: int = 200,
    overwrite: bool = False,
) -> None:
    """Generate RDKit pharmacophore fingerprints for recall and output sets."""
    for receptor in config.receptor_order:
        for split in WITHOUT_FINE_TUNING_PH4_SPLITS:
            recall_command = [
                sys.executable,
                "-m",
                "src.compute_pharmacophore_fingerprints",
                "--receptor",
                receptor,
                "--split",
                split,
                "--type_phfp",
                "rdkit",
                "--dataset",
                "recall",
                "--ncpus",
                str(ncpus),
                "--chunksize",
                str(chunksize),
                "--data_folder",
                str(config.data_folder),
                "--clusters",
                *map(str, WITHOUT_FINE_TUNING_PH4_CLUSTERS),
            ]
            if overwrite:
                recall_command.append("--overwrite")
            print("Running recall PH4:", " ".join(recall_command))
            subprocess.run(recall_command, check=True)

            for generator in WITHOUT_FINE_TUNING_PH4_GENERATORS:
                command = [
                    sys.executable,
                    "-m",
                    "src.compute_pharmacophore_fingerprints",
                    "--receptor",
                    receptor,
                    "--generator",
                    generator,
                    "--split",
                    split,
                    "--type_phfp",
                    "rdkit",
                    "--dataset",
                    "output",
                    "--ncpus",
                    str(ncpus),
                    "--chunksize",
                    str(chunksize),
                    "--data_folder",
                    str(config.data_folder),
                    "--clusters",
                    *map(str, WITHOUT_FINE_TUNING_PH4_CLUSTERS),
                ]
                if overwrite:
                    command.append("--overwrite")
                print("Running output PH4:", " ".join(command))
                subprocess.run(command, check=True)


def sync_without_finetuning_ph4_results(
    config: FineTuningComparisonConfig,
    receptor: str,
    split: str,
    generator: str,
    threshold: float,
) -> Path | None:
    """Copy without-fine-tuning PH4 result files into the archive folder."""
    source_dir = (
        config.ph4_results_dir
        / receptor
        / "rdkit"
        / split
        / generator
        / f"threshold_{threshold}"
    )
    target_dir = (
        config.without_fine_tuning_ph4_results_dir
        / receptor
        / "rdkit"
        / split
        / generator
        / f"threshold_{threshold}"
    )
    if not source_dir.exists():
        print(f"Source PH4 result directory not found, skipping sync: {source_dir}")
        return None

    target_dir.mkdir(parents=True, exist_ok=True)
    copied_files = []
    for path in source_dir.glob("*.csv"):
        dest = target_dir / path.name
        shutil.copy2(path, dest)
        copied_files.append(dest)

    print(f"Synced {len(copied_files)} PH4 result files to: {target_dir}")
    return target_dir


def run_without_finetuning_ph4_metrics(
    config: FineTuningComparisonConfig,
    ncpus: int = 120,
    sync_results: bool = True,
) -> pd.DataFrame:
    """Calculate pharmacophore-based metrics for without-fine-tuning generators."""
    from src import metrics_ph4 as mph4

    results = []
    config.without_fine_tuning_ph4_results_dir.mkdir(parents=True, exist_ok=True)

    for receptor in config.receptor_order:
        for split in WITHOUT_FINE_TUNING_PH4_SPLITS:
            threshold = PH4_THRESHOLD_BY_SPLIT[split]
            for generator in WITHOUT_FINE_TUNING_PH4_GENERATORS:
                print("=" * 80)
                print(
                    f"PH4 metrics | receptor={receptor} | split={split} | "
                    f"generator={generator} | threshold={threshold}"
                )
                mt = mph4.MetricsPh4(
                    type_cluster=split,
                    type_phfp="rdkit",
                    generator_name=generator,
                    receptor=receptor,
                    threshold=threshold,
                    data_folder=str(config.data_folder),
                    ncpus=ncpus,
                )
                result = mt.calculate(cluster_range=range(5))
                if sync_results:
                    sync_without_finetuning_ph4_results(
                        config=config,
                        receptor=receptor,
                        split=split,
                        generator=generator,
                        threshold=threshold,
                    )
                if result is not None and not result.empty:
                    result = result.copy()
                    result["receptor"] = receptor
                    result["split"] = split
                    result["threshold"] = threshold
                    result["generator"] = generator
                    results.append(result)

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()


def parse_generator_name(generator: str) -> tuple[str | None, str | None]:
    if generator == "DrugEx_RNN_w_w":
        return "DrugEx_RNN", "without_fine_tuning"
    if generator == "DrugEx_GT_w_w":
        return "DrugEx_GT", "without_fine_tuning"
    if generator == "REINVENT_w_w":
        return "REINVENT", "without_fine_tuning"
    if generator.startswith("DrugEx_RNN_epsilon_0.1_250k"):
        return "DrugEx_RNN", "epsilon_0.1_250k"
    if generator.startswith("DrugEx_RNN_epsilon_0.6_250k"):
        return "DrugEx_RNN", "epsilon_0.6_250k"
    if generator.startswith("DrugEx_GT_epsilon_0.1_250k"):
        return "DrugEx_GT", "epsilon_0.1_250k"
    if generator.startswith("DrugEx_GT_epsilon_0.6_250k"):
        return "DrugEx_GT", "epsilon_0.6_250k"
    if generator == "REINVENT_250k":
        return "REINVENT", "fine_tuned_250k"
    return None, None


def infer_receptor_from_path(path: Path) -> str | None:
    for receptor in RECEPTOR_ORDER:
        if receptor in path.parts:
            return receptor
    return None


def keep_variant_for_training_group(variant: str, training_group: str) -> bool:
    if training_group == "fine_tuned":
        return variant != "without_fine_tuning"
    if training_group == "without_fine_tuning":
        return variant == "without_fine_tuning"
    return True


def load_scaffold_mean_tables(
    results_dir: Path,
    training_group: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    pattern = re.compile(r"_mean_(csk|murcko)_(dis|sim)\.csv$")

    for path in sorted(results_dir.rglob("*_mean_*_*.csv")):
        match = pattern.search(path.name)
        if not match:
            continue

        unit_code, split_from_name = match.groups()
        generator = path.parent.name
        model, variant = parse_generator_name(generator)
        if model is None or not keep_variant_for_training_group(variant, training_group):
            continue

        receptor = infer_receptor_from_path(path)
        if receptor is None:
            continue

        df = pd.read_csv(path).rename(columns=METRIC_RENAME)
        if df.empty:
            continue
        row = df.iloc[0]
        split = row.get("type_cluster", split_from_name)
        if split not in SPLIT_ORDER:
            continue

        structural_unit = STRUCTURAL_UNIT_LABELS[unit_code]
        for metric in METRICS:
            if metric not in df.columns:
                continue
            value = pd.to_numeric(row[metric], errors="coerce")
            if pd.isna(value):
                continue
            records.append(
                {
                    "representation_family": "scaffold",
                    "structural_unit": structural_unit,
                    "unit_code": unit_code,
                    "receptor": receptor,
                    "receptor_label": RECEPTOR_LABELS[receptor],
                    "model": model,
                    "variant": variant,
                    "variant_label": VARIANT_LABELS.get(variant, variant),
                    "generator": generator,
                    "split": split,
                    "metric": metric,
                    "value": float(value),
                    "training_group": training_group,
                    "source_file": str(path),
                }
            )

    return pd.DataFrame(records)


def load_ph4_mean_tables(
    results_dir: Path,
    training_group: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    pattern = re.compile(r"_mean_rdkit_(dis|sim)_threshold_([0-9.]+)\.csv$")

    for path in sorted(results_dir.rglob("*_mean_rdkit_*threshold_*.csv")):
        match = pattern.search(path.name)
        if not match:
            continue

        split_from_name, threshold_from_name = match.groups()
        generator = path.parent.parent.name
        model, variant = parse_generator_name(generator)
        if model is None or not keep_variant_for_training_group(variant, training_group):
            continue

        receptor = infer_receptor_from_path(path)
        if receptor is None:
            continue

        df = pd.read_csv(path).rename(columns=METRIC_RENAME)
        if df.empty:
            continue
        row = df.iloc[0]

        split = row.get("type_cluster", split_from_name)
        if split not in SPLIT_ORDER:
            continue

        threshold = pd.to_numeric(threshold_from_name, errors="coerce")
        expected_threshold = PH4_THRESHOLD_BY_SPLIT.get(split)
        if expected_threshold is not None and not pd.isna(threshold):
            if float(threshold) != float(expected_threshold):
                continue

        for metric in METRICS:
            if metric not in df.columns:
                continue
            value = pd.to_numeric(row[metric], errors="coerce")
            if pd.isna(value):
                continue
            records.append(
                {
                    "representation_family": "ph4",
                    "structural_unit": STRUCTURAL_UNIT_LABELS["rdkit"],
                    "unit_code": "rdkit",
                    "receptor": receptor,
                    "receptor_label": RECEPTOR_LABELS[receptor],
                    "model": model,
                    "variant": variant,
                    "variant_label": VARIANT_LABELS.get(variant, variant),
                    "generator": generator,
                    "split": split,
                    "metric": metric,
                    "value": float(value),
                    "threshold": float(threshold) if not pd.isna(threshold) else np.nan,
                    "training_group": training_group,
                    "source_file": str(path),
                }
            )

    return pd.DataFrame(records)


def load_comparison_tables(
    config: FineTuningComparisonConfig,
    save_long_table: bool = True,
) -> pd.DataFrame:
    fine_scaffold_df = load_scaffold_mean_tables(config.fine_tuned_results_dir, "fine_tuned")
    without_scaffold_df = load_scaffold_mean_tables(
        config.without_fine_tuning_results_dir,
        "without_fine_tuning",
    )
    fine_ph4_df = load_ph4_mean_tables(config.ph4_results_dir, "fine_tuned")
    without_ph4_df = load_ph4_mean_tables(
        config.without_fine_tuning_ph4_results_dir,
        "without_fine_tuning",
    )

    comparison_df = pd.concat(
        [fine_scaffold_df, without_scaffold_df, fine_ph4_df, without_ph4_df],
        ignore_index=True,
        sort=False,
    )

    if save_long_table:
        comparison_csv = config.output_dir / "fine_tuning_metric_values_long.csv"
        comparison_df.to_csv(comparison_csv, index=False)
        print(f"Saved long comparison table to: {comparison_csv}")

    print(f"Loaded {len(comparison_df)} metric rows.")
    return comparison_df


def validate_comparison_table(
    comparison_df: pd.DataFrame,
    config: FineTuningComparisonConfig,
) -> dict[str, Any]:
    expected = []
    for model in config.model_order:
        for variant in config.variant_order[model]:
            for receptor in config.receptor_order:
                for structural_unit in config.structural_unit_order:
                    for split in config.split_order:
                        for metric in config.metrics:
                            expected.append((model, variant, receptor, structural_unit, split, metric))

    observed = set(
        comparison_df[["model", "variant", "receptor", "structural_unit", "split", "metric"]]
        .itertuples(index=False, name=None)
    )
    missing = [row for row in expected if row not in observed]
    summary_counts = (
        comparison_df.groupby(["model", "variant", "receptor", "structural_unit", "split"])
        .size()
        .reset_index(name="n_metric_rows")
    )

    print("Expected combinations:", len(expected))
    print("Observed combinations:", len(observed))
    print("Missing combinations:", len(missing))
    if missing:
        print("Missing combinations detected.")
    else:
        print("All expected combinations are present.")

    return {
        "expected_count": len(expected),
        "observed_count": len(observed),
        "missing_df": pd.DataFrame(
            missing,
            columns=["model", "variant", "receptor", "structural_unit", "split", "metric"],
        ),
        "summary_counts": summary_counts,
    }


def make_pairwise_comparison(df: pd.DataFrame) -> pd.DataFrame:
    base_keys = [
        "model",
        "receptor",
        "receptor_label",
        "structural_unit",
        "unit_code",
        "representation_family",
        "split",
        "metric",
    ]
    without = (
        df[df["variant"] == "without_fine_tuning"][base_keys + ["value"]]
        .rename(columns={"value": "without_value"})
    )
    fine = (
        df[df["variant"] != "without_fine_tuning"][base_keys + ["variant", "variant_label", "value"]]
        .rename(columns={"value": "fine_tuned_value"})
    )
    paired = fine.merge(without, on=base_keys, how="inner")
    paired["delta_fine_minus_without"] = paired["fine_tuned_value"] - paired["without_value"]
    paired["winner"] = np.select(
        [paired["delta_fine_minus_without"] > 0, paired["delta_fine_minus_without"] < 0],
        ["fine_tuned", "without_fine_tuning"],
        default="tie",
    )
    return paired.sort_values(
        ["representation_family", "structural_unit", "model", "variant", "metric", "receptor", "split"]
    ).reset_index(drop=True)


def collapse_best_fine_tuned(paired: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "model",
        "receptor",
        "receptor_label",
        "structural_unit",
        "unit_code",
        "representation_family",
        "split",
        "metric",
    ]
    best_rows = []
    for _, sub in paired.groupby(group_cols, sort=False):
        best = sub.sort_values("fine_tuned_value", ascending=False).iloc[0].copy()
        best["variant"] = "fine_tuned_best"
        best["variant_label"] = VARIANT_LABELS["fine_tuned_best"]
        best_rows.append(best)

    best = pd.DataFrame(best_rows)
    best["delta_fine_minus_without"] = best["fine_tuned_value"] - best["without_value"]
    best["winner"] = np.select(
        [best["delta_fine_minus_without"] > 0, best["delta_fine_minus_without"] < 0],
        ["fine_tuned", "without_fine_tuning"],
        default="tie",
    )
    return best.sort_values(
        ["representation_family", "structural_unit", "model", "metric", "receptor", "split"]
    ).reset_index(drop=True)


def build_pairwise_comparison_tables(
    comparison_df: pd.DataFrame,
    config: FineTuningComparisonConfig,
    save_tables: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired_df = make_pairwise_comparison(comparison_df)
    best_ft_df = collapse_best_fine_tuned(paired_df)

    if save_tables:
        paired_csv = config.output_dir / "fine_tuning_pairwise_comparisons.csv"
        best_csv = config.output_dir / "fine_tuning_best_variant_pairwise_comparisons.csv"
        paired_df.to_csv(paired_csv, index=False)
        best_ft_df.to_csv(best_csv, index=False)
        print(f"Saved pairwise comparison table to: {paired_csv}")
        print(f"Saved best fine-tuned comparison table to: {best_csv}")

    return paired_df, best_ft_df


def summarize_pairwise_stats(paired: pd.DataFrame) -> pd.DataFrame:
    records = []
    group_cols = ["model", "variant", "metric", "structural_unit", "unit_code", "representation_family"]
    for keys, sub in paired.groupby(group_cols, sort=False):
        model, variant, metric, structural_unit, unit_code, representation_family = keys
        deltas = sub["delta_fine_minus_without"].dropna().to_numpy(dtype=float)
        n = len(deltas)
        n_positive = int((deltas > 0).sum())
        n_negative = int((deltas < 0).sum())
        n_ties = int((deltas == 0).sum())

        p_two_sided = np.nan
        p_fine_greater = np.nan
        statistic = np.nan
        if n > 0 and np.any(deltas != 0):
            try:
                res_two = stats.wilcoxon(deltas, alternative="two-sided", zero_method="wilcox")
                res_greater = stats.wilcoxon(deltas, alternative="greater", zero_method="wilcox")
                statistic = float(res_two.statistic)
                p_two_sided = float(res_two.pvalue)
                p_fine_greater = float(res_greater.pvalue)
            except ValueError:
                pass

        records.append(
            {
                "model": model,
                "variant": variant,
                "variant_label": VARIANT_LABELS.get(variant, variant),
                "metric": metric,
                "structural_unit": structural_unit,
                "unit_code": unit_code,
                "representation_family": representation_family,
                "n_pairs": n,
                "fine_tuned_better": n_positive,
                "without_better": n_negative,
                "ties": n_ties,
                "fine_tuned_win_rate": n_positive / n if n else np.nan,
                "mean_delta_fine_minus_without": float(np.mean(deltas)) if n else np.nan,
                "median_delta_fine_minus_without": float(np.median(deltas)) if n else np.nan,
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_two_sided": p_two_sided,
                "wilcoxon_p_fine_tuned_greater": p_fine_greater,
            }
        )

    return pd.DataFrame(records)


def build_summary_tables(
    paired_df: pd.DataFrame,
    best_ft_df: pd.DataFrame,
    config: FineTuningComparisonConfig,
    save_tables: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats_by_variant_df = summarize_pairwise_stats(paired_df)
    stats_best_df = summarize_pairwise_stats(best_ft_df)

    if save_tables:
        stats_variant_csv = config.output_dir / "fine_tuning_win_loss_counts_by_variant.csv"
        stats_best_csv = config.output_dir / "fine_tuning_win_loss_counts_best_variant.csv"
        stats_by_variant_df.to_csv(stats_variant_csv, index=False)
        stats_best_df.to_csv(stats_best_csv, index=False)
        print(f"Saved by-variant summary table to: {stats_variant_csv}")
        print(f"Saved best-variant summary table to: {stats_best_csv}")

    return stats_by_variant_df, stats_best_df


def _lighten_color(hex_color: str, factor: float = 0.55) -> tuple[float, float, float]:
    import matplotlib.colors as mc

    rgb = np.array(mc.to_rgb(hex_color))
    return tuple(rgb + (1 - rgb) * factor)


def _add_count_labels(
    ax: Any,
    without_vals: np.ndarray,
    fine_vals: np.ndarray,
    y_positions: np.ndarray,
    min_center_width: float = 0.7,
    fontsize: int = 18,
) -> None:
    for y, without_val, fine_val in zip(y_positions, without_vals, fine_vals):
        total = without_val + fine_val
        if without_val > 0:
            if without_val >= min_center_width:
                ax.text(without_val / 2, y, f"{int(without_val)}", ha="center", va="center", fontsize=fontsize)
            else:
                ax.text(without_val + 0.08, y, f"{int(without_val)}", ha="left", va="center", fontsize=fontsize)
        if fine_val > 0:
            if fine_val >= min_center_width:
                ax.text(
                    without_val + fine_val / 2,
                    y,
                    f"{int(fine_val)}",
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                )
            else:
                ax.text(total + 0.08, y, f"{int(fine_val)}", ha="left", va="center", fontsize=fontsize)


def _draw_left_row_labels(
    ax: Any,
    labels: list[str],
    y_positions: np.ndarray,
    x: float = -0.05,
    fontsize: int = 20,
) -> None:
    for y, label in zip(y_positions, labels):
        ax.text(
            x,
            y,
            label,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=fontsize,
            color="black",
            clip_on=False,
        )


def plot_model_metric_bars(
    df: pd.DataFrame,
    model: str,
    variant_order: list[str],
    output_dir: Path,
    aser_scale: float = 20.0,
    save: bool = True,
) -> plt.Figure | None:
    plot_df = df[(df["model"] == model) & (df["representation_family"] == "scaffold")].copy()
    if plot_df.empty:
        print(f"No scaffold-based data for model={model}")
        return None

    plot_df["plot_value"] = plot_df["value"]
    plot_df.loc[plot_df["metric"] == "ASER", "plot_value"] *= aser_scale

    sns.set_context("talk")
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(24, 9), sharey=False)
    x = np.arange(len(METRICS))
    n_variants = len(variant_order)
    bar_width = 0.75 / max(n_variants, 2)
    offsets = (np.arange(n_variants) - (n_variants - 1) / 2.0) * bar_width

    metric_labels = {"RS": "RS", "SED": "SED", "ASER": rf"ASER $\times {aser_scale:g}$"}

    for row_idx, receptor in enumerate(RECEPTOR_ORDER):
        for col_idx, (unit_code, split) in enumerate(PANEL_ORDER):
            ax = axes[row_idx, col_idx]
            sub = plot_df[
                (plot_df["receptor"] == receptor)
                & (plot_df["unit_code"] == unit_code)
                & (plot_df["split"] == split)
            ]
            if sub.empty:
                ax.set_axis_off()
                continue

            for variant_idx, variant in enumerate(variant_order):
                values = []
                for metric in METRICS:
                    vals = sub[(sub["variant"] == variant) & (sub["metric"] == metric)]["plot_value"]
                    values.append(float(vals.iloc[0]) if len(vals) else np.nan)
                ax.bar(
                    x + offsets[variant_idx],
                    values,
                    width=bar_width,
                    color=VARIANT_COLORS[variant],
                    label=VARIANT_LABELS[variant],
                )

            ax.set_xticks(x)
            ax.set_xticklabels([metric_labels[m] for m in METRICS])
            ax.set_title(f"{STRUCTURAL_UNIT_LABELS[unit_code]}, {split.upper()}", fontsize=15)
            ax.grid(axis="y", alpha=0.25)

            if col_idx == 0:
                ax.set_ylabel(RECEPTOR_LABELS[receptor], fontsize=15)

            if row_idx == 0 and col_idx == len(PANEL_ORDER) - 1:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.02), frameon=False)

    fig.suptitle(model.replace("_", " "), fontsize=20, y=1.02)
    plt.tight_layout()

    if save:
        out_png = output_dir / f"{model}_scaffold_metric_bars.png"
        out_svg = output_dir / f"{model}_scaffold_metric_bars.svg"
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_svg, bbox_inches="tight")
        print(f"Saved: {out_png}")
        print(f"Saved: {out_svg}")

    return fig


def plot_scaffold_metric_bar_grid(
    comparison_df: pd.DataFrame,
    config: FineTuningComparisonConfig,
    aser_scale: float = 20.0,
    save: bool = True,
) -> dict[str, plt.Figure]:
    figures: dict[str, plt.Figure] = {}
    for model in config.model_order:
        fig = plot_model_metric_bars(
            df=comparison_df,
            model=model,
            variant_order=config.variant_order[model],
            output_dir=config.output_dir,
            aser_scale=aser_scale,
            save=save,
        )
        if fig is not None:
            figures[model] = fig
    return figures


def plot_finetuning_win_loss_by_structural_unit(
    stats_df: pd.DataFrame,
    output_dir: Path,
    save: bool = True,
) -> plt.Figure | None:
    plot_df = stats_df[stats_df["variant"] != "without_fine_tuning"].copy()
    if plot_df.empty:
        print("No fine-tuning summary rows available.")
        return None

    row_order = [
        ("DrugEx_RNN", "epsilon_0.1_250k"),
        ("DrugEx_RNN", "epsilon_0.6_250k"),
        ("DrugEx_GT", "epsilon_0.1_250k"),
        ("DrugEx_GT", "epsilon_0.6_250k"),
        ("REINVENT", "fine_tuned_250k"),
    ]
    order_rank = {key: idx for idx, key in enumerate(row_order)}
    metric_order = ["RS", "SED", "ASER"]
    unit_order = ["CSK", "MURCKO", "pharmacophore fingerprint"]

    plot_df["row_key"] = list(zip(plot_df["model"], plot_df["variant"]))
    plot_df = plot_df[plot_df["row_key"].isin(order_rank)].copy()
    plot_df["row_rank"] = plot_df["row_key"].map(order_rank)
    plot_df["row_label"] = plot_df["row_key"].map(SHORT_VARIANT_LABELS)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 3, figsize=(22, 18), sharex=False, sharey=False)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.75, bottom=0.0, hspace=0.6, wspace=0.1)

    legend_handles: dict[str, tuple[Any, Any]] = {}

    for row_idx, structural_unit in enumerate(unit_order):
        for col_idx, metric in enumerate(metric_order):
            ax = axes[row_idx, col_idx]
            sub = plot_df[
                (plot_df["structural_unit"] == structural_unit) & (plot_df["metric"] == metric)
            ].copy()
            sub = sub.sort_values("row_rank").reset_index(drop=True)
            if sub.empty:
                ax.set_axis_off()
                continue

            y = np.arange(len(sub))
            base_color = METRIC_BASE_COLORS[metric]
            light_color = _lighten_color(base_color, factor=0.58)
            without_vals = sub["without_better"].to_numpy(dtype=float)
            fine_vals = sub["fine_tuned_better"].to_numpy(dtype=float)

            bars_without = ax.barh(
                y,
                without_vals,
                height=0.72,
                color=light_color,
                edgecolor="white",
                linewidth=1.5,
                label="without fine-tuning better",
            )
            bars_fine = ax.barh(
                y,
                fine_vals,
                left=without_vals,
                height=0.72,
                color=base_color,
                edgecolor="white",
                linewidth=1.5,
                label="fine-tuning better",
            )

            legend_handles[metric] = (bars_without[0], bars_fine[0])
            _add_count_labels(ax, without_vals, fine_vals, y)

            ax.set_yticks(y)
            ax.set_yticklabels([])
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.3)
            ax.grid(axis="y", visible=False)
            ax.tick_params(axis="x", labelbottom=False, bottom=False)
            ax.tick_params(axis="y", left=False)

            if col_idx == 0:
                _draw_left_row_labels(ax, sub["row_label"].tolist(), y)

            ax.set_title(metric, fontsize=20, pad=15, fontweight="semibold")
            ax.set_xlabel("Number of experimental settings", fontsize=18, labelpad=10)

    x_center = 0.1
    for _, metric in enumerate(metric_order):
        ax = axes[0, metric_order.index(metric)]
        pos = ax.get_position()
        y_top = pos.y1 + 0.085
        handles = legend_handles.get(metric)
        if handles is not None:
            fig.legend(
                handles,
                ["without fine-tuning better", "fine-tuning better"],
                ncol=2,
                loc="center",
                bbox_to_anchor=(x_center, y_top),
                frameon=False,
                fontsize=18,
            )
        x_center += 0.4

    for row_idx, structural_unit in enumerate(unit_order):
        left = axes[row_idx, 0].get_position().x0
        right = axes[row_idx, 2].get_position().x1
        top = max(axes[row_idx, col].get_position().y1 for col in range(3))
        fig.text(
            (left + right) / 2,
            top + 0.04,
            f"Structural units: {structural_unit}",
            ha="center",
            va="bottom",
            fontsize=22,
            fontweight="semibold",
        )

    if save:
        png_path = output_dir / "fine_tuning_win_loss_by_structural_unit_and_metric.png"
        svg_path = output_dir / "fine_tuning_win_loss_by_structural_unit_and_metric.svg"
        fig.savefig(png_path, dpi=500, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
        print(f"Saved: {png_path}")
        print(f"Saved: {svg_path}")

    return fig


def run_core_analysis(
    config: FineTuningComparisonConfig,
    save_tables: bool = True,
) -> dict[str, pd.DataFrame]:
    comparison_df = load_comparison_tables(config=config, save_long_table=save_tables)
    validation = validate_comparison_table(comparison_df=comparison_df, config=config)
    paired_df, best_ft_df = build_pairwise_comparison_tables(
        comparison_df=comparison_df,
        config=config,
        save_tables=save_tables,
    )
    stats_by_variant_df, stats_best_df = build_summary_tables(
        paired_df=paired_df,
        best_ft_df=best_ft_df,
        config=config,
        save_tables=save_tables,
    )
    return {
        "comparison_df": comparison_df,
        "missing_df": validation["missing_df"],
        "summary_counts": validation["summary_counts"],
        "paired_df": paired_df,
        "best_ft_df": best_ft_df,
        "stats_by_variant_df": stats_by_variant_df,
        "stats_best_df": stats_best_df,
    }
