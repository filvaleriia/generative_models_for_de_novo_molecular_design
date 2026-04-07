from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from src.metrics_ph4 import MetricsPh4, create_matching_dataframe as create_ph4_matching_dataframe, normalize_ncpus as normalize_ph4_ncpus
from src.metrics_scaffold import (
    MetricsScaffold,
    convert_to_scaffold,
    create_matching_dataframe as create_scaffold_matching_dataframe,
    load_scaffold_cache,
    normalize_ncpus as normalize_scaffold_ncpus,
    save_scaffold_cache,
)
from src.path_utils import pharm_results_dir, scaffold_results_dir


VALID_METRIC_FAMILIES = {"scaffold", "ph4"}
VALID_SCAFFOLDS = {"csk", "murcko"}


@dataclass(frozen=True)
class CustomMetricsConfig:
    metric_family: str
    unit_type: str
    generator_name: str
    receptor: str
    type_cluster: str
    data_folder: str = ""
    ncpus: int = 1
    threshold: Optional[float] = None
    use_cache: bool = True

    def normalized_family(self) -> str:
        family = self.metric_family.strip().lower()
        if family not in VALID_METRIC_FAMILIES:
            raise ValueError(f"Unsupported metric family '{self.metric_family}'. Use one of {sorted(VALID_METRIC_FAMILIES)}.")
        return family

    def normalized_unit(self) -> str:
        unit = self.unit_type.strip().lower()
        if self.normalized_family() == "scaffold" and unit not in VALID_SCAFFOLDS:
            raise ValueError(f"Unsupported scaffold type '{self.unit_type}'. Use one of {sorted(VALID_SCAFFOLDS)}.")
        return unit

    def resolved_threshold(self) -> float:
        if self.normalized_family() != "ph4":
            return 1.0
        if self.threshold is None:
            raise ValueError("threshold must be provided for pharmacophore-based custom metrics.")
        thr = float(self.threshold)
        if thr < 0.0 or thr > 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        return thr


def _ensure_parent_dir(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _format_cluster_id(cluster_id: int | str) -> str:
    return str(cluster_id)


def _scaffold_output_dir(cfg: CustomMetricsConfig) -> Path:
    return scaffold_results_dir(
        cfg.data_folder,
        cfg.receptor,
        f"{cfg.normalized_unit()}_scaffolds",
        cfg.type_cluster,
        cfg.generator_name,
    )


def _ph4_output_dir(cfg: CustomMetricsConfig) -> Path:
    return pharm_results_dir(
        cfg.data_folder,
        cfg.receptor,
        cfg.normalized_unit(),
        cfg.type_cluster,
        cfg.generator_name,
        f"threshold_{cfg.resolved_threshold()}",
    )


def _load_smiles_lines(path: str | Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as handle:
        return pd.DataFrame([line.strip() for line in handle if line.strip()])


def _load_fp_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=["fp"])


def _convert_scaffold_df(smiles_df: pd.DataFrame, source_path: str | Path, scaffold_type: str, ncpus: int, use_cache: bool) -> pd.DataFrame:
    if use_cache:
        cached = load_scaffold_cache(source_path, scaffold_type)
        if cached is not None:
            return cached

    if ncpus <= 1:
        results = [convert_to_scaffold(smiles, scaffold_type) for smiles in smiles_df[0]]
    else:
        from multiprocessing import Pool

        with Pool(processes=ncpus) as pool:
            results = pool.starmap(convert_to_scaffold, [(smiles, scaffold_type) for smiles in smiles_df[0]])

    scaffolds_df = pd.DataFrame(results).dropna()
    if use_cache:
        save_scaffold_cache(source_path, scaffold_type, scaffolds_df)
    return scaffolds_df


def calculate_scaffold_metrics_from_files(
    output_set_path: str | Path,
    recall_set_path: str | Path,
    cfg: CustomMetricsConfig,
    cluster_id: int | str,
) -> pd.DataFrame:
    ncpus = normalize_scaffold_ncpus(cfg.ncpus)
    scaffold_type = cfg.normalized_unit()

    output_set = _load_smiles_lines(output_set_path)
    recall_set = _load_smiles_lines(recall_set_path)

    output_scaffolds = _convert_scaffold_df(output_set, output_set_path, scaffold_type, ncpus, cfg.use_cache)
    recall_scaffolds = _convert_scaffold_df(recall_set, recall_set_path, scaffold_type, ncpus, cfg.use_cache)

    unique_output = output_scaffolds[0].unique() if not output_scaffolds.empty else np.asarray([], dtype=object)
    unique_recall = recall_scaffolds[0].unique() if not recall_scaffolds.empty else np.asarray([], dtype=object)

    count_metrics = create_scaffold_matching_dataframe(output_scaffolds, unique_recall)

    USo = len(unique_output)
    SSo = len(output_scaffolds)
    cASo = count_metrics["count_of_occurrence"].sum()
    UASo = int(count_metrics["unique_indicator"].sum()) if not count_metrics.empty else 0
    UASr = len(count_metrics)

    rs = UASo / UASr if UASr > 0 else 0.0
    rs_text = f"{UASo}/{UASr}"
    sed = USo / SSo if SSo > 0 else 0.0
    aser = cASo / SSo if SSo > 0 else 0.0

    cluster_label = _format_cluster_id(cluster_id)
    results = pd.DataFrame(
        {
            "name": [f"{cfg.generator_name}_{cluster_label}"],
            "type_cluster": [cfg.type_cluster],
            "scaffold": [scaffold_type],
            "SSo": [SSo],
            "RS_": [rs_text],
            "RS": [rs],
            "SED": [sed],
            "ASER": [aser],
        }
    )

    output_dir = _scaffold_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    count_metrics.to_csv(output_dir / f"count_of_occurrence_cluster_{cluster_label}_{cfg.type_cluster}_{cfg.generator_name}.csv", index=False)
    results.to_csv(output_dir / f"metrics_cluster_{cluster_label}_{cfg.type_cluster}_{cfg.generator_name}.csv", index=False)
    output_scaffolds.to_csv(output_dir / f"scaffolds_of_output_set_cluster_{cluster_label}_{cfg.type_cluster}_{cfg.generator_name}.csv", header=False, index=False)
    recall_scaffolds.to_csv(output_dir / f"scaffolds_of_recall_set_cluster_{cluster_label}_{cfg.type_cluster}_{cfg.generator_name}.csv", header=False, index=False)
    return results


def calculate_ph4_metrics_from_files(
    output_set_path: str | Path,
    recall_set_path: str | Path,
    cfg: CustomMetricsConfig,
    cluster_id: int | str,
) -> pd.DataFrame:
    threshold = cfg.resolved_threshold()
    ncpus = normalize_ph4_ncpus(cfg.ncpus)

    output_set = _load_fp_csv(output_set_path)
    recall_set = _load_fp_csv(recall_set_path).drop_duplicates(keep="first").reset_index(drop=True)

    count_metrics = pd.DataFrame(
        create_ph4_matching_dataframe(recall_set, output_set, threshold, ncpus)
    )

    UFo = len(output_set["fp"].drop_duplicates())
    SSo = len(output_set)
    CwAFo = count_metrics["CwAFo"].sum() if not count_metrics.empty else 0
    UAFo = count_metrics["UAFo"].sum() if not count_metrics.empty else 0
    UAFr = len(count_metrics)

    rs = UAFo / UAFr if UAFr > 0 else 0.0
    rs_text = f"{UAFo}/{UAFr}"
    sed = UFo / SSo if SSo > 0 else 0.0
    aser = CwAFo / SSo if SSo > 0 else 0.0

    cluster_label = _format_cluster_id(cluster_id)
    results = pd.DataFrame(
        {
            "name": [f"{cfg.generator_name}_{cluster_label}"],
            "type_cluster": [cfg.type_cluster],
            "phfp": [cfg.normalized_unit()],
            "UFo": [UFo],
            "SSo": [SSo],
            "RS_": [rs_text],
            "RS": [rs],
            "SED": [sed],
            "ASER": [aser],
            "CwAFo": [CwAFo],
        }
    )

    output_dir = _ph4_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    count_metrics.to_csv(output_dir / f"count_of_occurrence_cluster_{cluster_label}_{cfg.type_cluster}_{cfg.generator_name}_threshold_{threshold}.csv", index=False)
    results.to_csv(output_dir / f"metrics_cluster_{cluster_label}_{cfg.type_cluster}_{cfg.generator_name}_threshold_{threshold}.csv", index=False)
    return results


def calculate_metrics_from_files(
    output_set_path: str | Path,
    recall_set_path: str | Path,
    cfg: CustomMetricsConfig,
    cluster_id: int | str = 0,
) -> pd.DataFrame:
    if cfg.normalized_family() == "scaffold":
        return calculate_scaffold_metrics_from_files(output_set_path, recall_set_path, cfg, cluster_id)
    return calculate_ph4_metrics_from_files(output_set_path, recall_set_path, cfg, cluster_id)


def calculate_metrics_from_patterns(
    output_pattern: str,
    recall_pattern: str,
    cfg: CustomMetricsConfig,
    cluster_ids: Sequence[int | str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    used_clusters: list[int | str] = []

    for cluster_id in cluster_ids:
        cluster_label = _format_cluster_id(cluster_id)
        output_path = Path(output_pattern.format(cluster=cluster_label))
        recall_path = Path(recall_pattern.format(cluster=cluster_label))
        if not (output_path.exists() and recall_path.exists()):
            print(f"Skipping cluster {cluster_label}: missing input file(s)")
            continue
        rows.append(calculate_metrics_from_files(output_path, recall_path, cfg, cluster_label))
        used_clusters.append(cluster_id)

    if not rows:
        return pd.DataFrame()

    output_dir = _scaffold_output_dir(cfg) if cfg.normalized_family() == "scaffold" else _ph4_output_dir(cfg)
    combined_df = pd.concat([pd.read_csv(path) for path in _metric_file_paths(output_dir, cfg, used_clusters)], ignore_index=True)
    mean_df = _save_mean_tables(output_dir, cfg, combined_df)
    return mean_df


def _metric_file_paths(output_dir: Path, cfg: CustomMetricsConfig, cluster_ids: Iterable[int | str]) -> list[Path]:
    cluster_labels = [_format_cluster_id(cluster_id) for cluster_id in cluster_ids]
    if cfg.normalized_family() == "scaffold":
        return [
            output_dir / f"metrics_cluster_{cluster_label}_{cfg.type_cluster}_{cfg.generator_name}.csv"
            for cluster_label in cluster_labels
        ]
    threshold = cfg.resolved_threshold()
    return [
        output_dir / f"metrics_cluster_{cluster_label}_{cfg.type_cluster}_{cfg.generator_name}_threshold_{threshold}.csv"
        for cluster_label in cluster_labels
    ]


def _save_mean_tables(output_dir: Path, cfg: CustomMetricsConfig, combined_df: pd.DataFrame) -> pd.DataFrame:
    combined_df = combined_df.copy()
    mean_values = combined_df.mean(numeric_only=True)

    if cfg.normalized_family() == "scaffold":
        mean_row = {
            "name": f"{cfg.generator_name}_mean",
            "type_cluster": cfg.type_cluster,
            "scaffold": cfg.normalized_unit(),
            "SSo": mean_values.get("SSo", np.nan),
            "RS_": "-",
            "RS": mean_values.get("RS", np.nan),
            "SED": mean_values.get("SED", np.nan),
            "ASER": mean_values.get("ASER", np.nan),
        }
        output_path = output_dir / f"{cfg.generator_name}_mean_{cfg.normalized_unit()}_{cfg.type_cluster}.csv"
        with_comma = output_dir / "df_all_clusters_with_mean_with_coma.csv"
        raw_path = output_dir / "df_all_clusters_with_mean.csv"
    else:
        threshold = cfg.resolved_threshold()
        mean_row = {
            "name": f"{cfg.generator_name}_mean",
            "type_cluster": cfg.type_cluster,
            "phfp": cfg.normalized_unit(),
            "UFo": mean_values.get("UFo", np.nan),
            "SSo": mean_values.get("SSo", np.nan),
            "RS_": "-",
            "RS": mean_values.get("RS", np.nan),
            "SED": mean_values.get("SED", np.nan),
            "ASER": mean_values.get("ASER", np.nan),
            "CwAFo": mean_values.get("CwAFo", np.nan),
        }
        output_path = output_dir / f"{cfg.generator_name}_mean_{cfg.normalized_unit()}_{cfg.type_cluster}_threshold_{threshold}.csv"
        with_comma = output_dir / f"df_all_clusters_with_mean_with_coma_threshold_{threshold}.csv"
        raw_path = output_dir / f"df_all_clusters_with_mean_threshold_{threshold}.csv"

    combined_df = pd.concat([combined_df, pd.DataFrame([mean_row])], ignore_index=True).round(7)
    formatted_df = combined_df.copy()
    for column in [col for col in ("SSo", "UFo", "CwAFo") if col in formatted_df.columns]:
        formatted_df[column] = formatted_df[column].apply(lambda x: "{:,}".format(int(x)) if pd.notnull(x) and str(x) != "-" else x)

    formatted_df.to_csv(with_comma, index=False)
    combined_df.to_csv(raw_path, index=False)
    combined_df.tail(1).to_csv(output_path, index=False)
    return combined_df
