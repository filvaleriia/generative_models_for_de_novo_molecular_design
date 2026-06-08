from __future__ import annotations

import argparse
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool, shared_memory
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric, MurckoScaffoldSmiles
from tqdm import tqdm

try:
    from src.path_utils import BOOTSTRAP_OUTPUTS_DIRNAME, bootstrap_outputs_dir, resolve_data_folder
except ModuleNotFoundError:  # pragma: no cover - CLI execution from repo root
    from path_utils import BOOTSTRAP_OUTPUTS_DIRNAME, bootstrap_outputs_dir, resolve_data_folder

METRICS = ("RS", "SED", "ASER")
DEFAULT_PH4_THRESHOLDS = {"dis": 0.7, "sim": 0.8}
DEFAULT_SCAFFOLD_GENERATORS = [
    "Molpher",
    "DrugEx_GT_epsilon_0.1",
    "DrugEx_GT_epsilon_0.6",
    "DrugEx_RNN_epsilon_0.1",
    "DrugEx_RNN_epsilon_0.6",
    "GB_GA_log_p_mut_r_0.01",
    "GB_GA_log_p_mut_r_0.5",
    "GB_GA_mut_r_0.01",
    "GB_GA_mut_r_0.5",
    "REINVENT",
    "addcarbon",
]
DEFAULT_PH4_GENERATORS = [
    "Molpher",
    "DrugEx_GT_epsilon_0.1",
    "DrugEx_GT_epsilon_0.6",
    "DrugEx_RNN_epsilon_0.1",
    "DrugEx_RNN_epsilon_0.6",
    "GB_GA_log_p_mut_r_0.01",
    "GB_GA_log_p_mut_r_0.5",
    "GB_GA_mut_r_0.01",
    "GB_GA_mut_r_0.5",
    "REINVENT",
    "enamine",
]
FOREST_GENERATOR_ORDER = {
    "scaffold": [
        "Molpher",
        "REINVENT",
        "DrugEx_GT_epsilon_0.1",
        "DrugEx_GT_epsilon_0.6",
        "DrugEx_RNN_epsilon_0.1",
        "DrugEx_RNN_epsilon_0.6",
        "GB_GA_mut_r_0.01",
        "GB_GA_mut_r_0.5",
        "GB_GA_log_p_mut_r_0.01",
        "GB_GA_log_p_mut_r_0.5",
        "addcarbon",
    ],
    "ph4": [
        "Molpher",
        "REINVENT",
        "DrugEx_GT_epsilon_0.1",
        "DrugEx_GT_epsilon_0.6",
        "DrugEx_RNN_epsilon_0.1",
        "DrugEx_RNN_epsilon_0.6",
        "GB_GA_mut_r_0.01",
        "GB_GA_mut_r_0.5",
        "GB_GA_log_p_mut_r_0.01",
        "GB_GA_log_p_mut_r_0.5",
        "enamine",
    ],
}
FOREST_ROW_TITLES = {
    "csk_scaffolds": "STRUCTURAL REPRESENTATION: CSK",
    "murcko_scaffolds": "STRUCTURAL REPRESENTATION: MURCKO",
    "rdkit_ph4_fps": "STRUCTURAL REPRESENTATION: Pharmacophore fingerprints",
}
PAIRWISE_ROW_CONFIGS = [
    ("scaffold", "csk", "STRUCTURAL REPRESENTATION: CSK"),
    ("scaffold", "murcko", "STRUCTURAL REPRESENTATION: MURCKO"),
    ("ph4", "rdkit", "STRUCTURAL REPRESENTATION: Pharmacophore fingerprints"),
]


@dataclass(frozen=True)
class BootstrapThresholdConfig:
    metric_family: str
    units: list[str]
    clusters: list[int]
    n_subsample: int
    n_bootstrap: int
    alpha: float
    data_folder: str
    job_workers: int
    inner_workers: int
    chunksize: int = 200
    use_cache: bool = True
    show_progress: bool = True
    save_bootstrap_samples: bool = True

    def normalized_family(self) -> str:
        family = self.metric_family.strip().lower()
        if family not in {"scaffold", "ph4"}:
            raise ValueError("metric_family must be 'scaffold' or 'ph4'.")
        return family

    def intermediate_root(self) -> Path:
        return bootstrap_outputs_dir(self.data_folder, self.normalized_family())

    def repo_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    def default_final_csv(self) -> Path:
        if self.normalized_family() == "scaffold":
            return self.repo_root() / "effect_size_thresholds_scaffolds.csv"
        return self.repo_root() / "effect_size_thresholds_ph4_rdkit.csv"


def bootstrap_root_dir(data_folder: str | Path | None = "../../") -> Path:
    return resolve_data_folder(data_folder).joinpath("data", BOOTSTRAP_OUTPUTS_DIRNAME)


def _threshold_for_split(split: str) -> float:
    split_norm = split.strip().lower()
    if split_norm not in DEFAULT_PH4_THRESHOLDS:
        raise ValueError(f"Unknown split '{split}'. Expected one of {sorted(DEFAULT_PH4_THRESHOLDS)}.")
    return DEFAULT_PH4_THRESHOLDS[split_norm]


def _summary_filename(metric_family: str) -> str:
    if metric_family == "scaffold":
        return "bootstrap_ci_scaffold_metrics_summary.csv"
    return "bootstrap_ci_ph4_metrics_summary.csv"


def _pairwise_filename(metric_family: str) -> str:
    if metric_family == "scaffold":
        return "bootstrap_pairwise_tests_scaffolds.csv"
    return "bootstrap_pairwise_tests_ph4.csv"


def _unit_column(metric_family: str) -> str:
    return "Scaffold" if metric_family == "scaffold" else "PH4"


def _job_output_dir(root: Path, metric_family: str, receptor: str, split: str, unit_type: str, generator: str) -> Path:
    if metric_family == "scaffold":
        return root / receptor / f"{unit_type}_scaffolds" / split / generator
    return root / receptor / f"{unit_type}_ph4_fps" / split / generator


def _save_job_outputs(
    out_dir: Path,
    df_job: pd.DataFrame,
    bootstrap_samples: dict[str, np.ndarray] | None,
    meta_lines: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df_job.to_csv(out_dir / "bootstrap_ci.csv", index=False)

    if bootstrap_samples is not None:
        bs_dir = out_dir / "bootstrap_samples"
        bs_dir.mkdir(exist_ok=True)
        for metric, arr in bootstrap_samples.items():
            np.save(bs_dir / f"{metric}.npy", np.asarray(arr, dtype=float))

    with open(out_dir / "meta.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(meta_lines) + "\n")


def _parse_cluster_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_name_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _pairwise_from_vectors(
    vectors: dict[str, np.ndarray],
    alpha: float,
    extra_fields: dict[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for a, b in itertools.combinations(sorted(vectors.keys()), 2):
        va = vectors[a]
        vb = vectors[b]
        if va is None or vb is None:
            continue
        d = va - vb
        rows.append(
            {
                **extra_fields,
                "A": a,
                "B": b,
                "delta_mean": float(d.mean()),
                "delta_ci_low": float(np.quantile(d, alpha / 2)),
                "delta_ci_high": float(np.quantile(d, 1.0 - alpha / 2)),
                "p_value_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean())),
                "abs_delta_mean": float(abs(d.mean())),
            }
        )
    return pd.DataFrame(rows)


def compute_effect_size_thresholds_table(
    df_tests: pd.DataFrame,
    metric_family: str,
) -> pd.DataFrame:
    family = metric_family.strip().lower()
    unit_col = "unit_type"
    output_unit_col = _unit_column(family)

    if df_tests.empty or unit_col not in df_tests.columns:
        return pd.DataFrame(
            columns=[
                "Metric",
                output_unit_col,
                "Trivial Δ (≤25%)",
                "Small Δ (25–50%)",
                "Moderate Δ (50–75%)",
                "Large Δ (>75%)",
            ]
        )

    rows: list[dict[str, object]] = []

    for metric in METRICS:
        for unit_type in sorted(df_tests[unit_col].dropna().astype(str).unique()):
            subset = df_tests[(df_tests["metric"] == metric) & (df_tests[unit_col] == unit_type)]
            deltas = subset["abs_delta_mean"].dropna().to_numpy(dtype=float)
            if deltas.size == 0:
                t1 = t2 = t3 = np.nan
            else:
                t1, t2, t3 = np.quantile(deltas, [0.25, 0.5, 0.75])

            common = {
                "Metric": metric,
                "Trivial Δ (≤25%)": "NA" if np.isnan(t1) else f"Δ ≤ {t1:.3f}",
                "Small Δ (25–50%)": "NA" if np.isnan(t2) else f"{t1:.3f} < Δ ≤ {t2:.3f}",
                "Moderate Δ (50–75%)": "NA" if np.isnan(t3) else f"{t2:.3f} < Δ ≤ {t3:.3f}",
                "Large Δ (>75%)": "NA" if np.isnan(t3) else f"Δ > {t3:.3f}",
            }
            if family == "scaffold":
                rows.append({"Scaffold": unit_type.lower(), **common})
            else:
                rows.append({"PH4": unit_type.lower(), **common})

    columns = ["Metric", output_unit_col, "Trivial Δ (≤25%)", "Small Δ (25–50%)", "Moderate Δ (50–75%)", "Large Δ (>75%)"]
    return pd.DataFrame(rows, columns=columns)


def save_effect_size_thresholds_csv(
    df_thresholds: pd.DataFrame,
    out_csv: str | Path,
) -> Path:
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_thresholds.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Scaffold bootstrap
# ---------------------------------------------------------------------------


def _convert_to_scaffold(smiles: str, scaffold_type: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if scaffold_type == "csk":
            return MurckoScaffoldSmiles(Chem.MolToSmiles(MakeScaffoldGeneric(mol))) or None
        if scaffold_type == "murcko":
            return MurckoScaffoldSmiles(Chem.MolToSmiles(mol)) or None
        return None
    except Exception:
        return None


def _scaffold_cache_path(smiles_file: str | Path, scaffold_type: str) -> Path:
    p = Path(smiles_file)
    return p.with_suffix(p.suffix + f".{scaffold_type}.scaffolds.npy")


def _read_smiles(filepath: str | Path) -> list[str]:
    with open(filepath, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def smiles_to_scaffolds(
    filepath: str | Path,
    scaffold_type: str,
    ncpus: int,
    chunksize: int,
    use_cache: bool,
    show_progress: bool,
    label: str = "",
) -> np.ndarray:
    cache = _scaffold_cache_path(filepath, scaffold_type)
    if use_cache and cache.exists():
        return np.load(cache, allow_pickle=True)

    smiles = _read_smiles(filepath)
    if not smiles:
        return np.asarray([], dtype=object)

    worker = partial(_convert_to_scaffold, scaffold_type=scaffold_type)
    scaffolds: list[str] = []

    with Pool(processes=max(1, ncpus)) as pool:
        iterator = pool.imap_unordered(worker, smiles, chunksize=max(1, chunksize))
        if show_progress:
            iterator = tqdm(iterator, total=len(smiles), desc=label or f"scaffold {scaffold_type}", unit="mol")
        for scaffold in iterator:
            if scaffold:
                scaffolds.append(scaffold)

    arr = np.asarray(scaffolds, dtype=object)
    if use_cache:
        np.save(cache, arr, allow_pickle=True)
    return arr


def bootstrap_cluster_scaffold(
    output_scaffolds: np.ndarray,
    recall_unique: set[str],
    n_recall_unique: int,
    n_subsample: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    if len(output_scaffolds) == 0:
        z = np.zeros(n_bootstrap, dtype=float)
        return {"RS": z.copy(), "SED": z.copy(), "ASER": z.copy()}

    idx = rng.integers(0, len(output_scaffolds), size=(n_bootstrap, n_subsample))
    sampled = output_scaffolds[idx]

    rs = np.zeros(n_bootstrap, dtype=float)
    sed = np.zeros(n_bootstrap, dtype=float)
    aser = np.zeros(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        sample = sampled[b]
        uniq = set(sample.tolist())
        matches = sum(token in recall_unique for token in sample)
        recovered_unique = sum(token in recall_unique for token in uniq)

        sed[b] = len(uniq) / n_subsample
        aser[b] = matches / n_subsample
        rs[b] = recovered_unique / n_recall_unique if n_recall_unique > 0 else 0.0

    return {"RS": rs, "SED": sed, "ASER": aser}


def run_scaffold_job(
    receptor: str,
    split: str,
    unit_type: str,
    generator: str,
    cfg: BootstrapThresholdConfig,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    per_cluster: dict[str, list[np.ndarray]] = {metric: [] for metric in METRICS}
    used_clusters: list[int] = []

    for cluster in cfg.clusters:
        out_file = (
            resolve_data_folder(cfg.data_folder)
            / "data"
            / "output_sets"
            / receptor
            / generator
            / f"cOS_{generator}_{split}_{cluster}_one_column.csv"
        )
        rec_file = (
            resolve_data_folder(cfg.data_folder)
            / "data"
            / "input_recall_sets"
            / receptor
            / f"cRS_{receptor}_{split}_{cluster}.csv"
        )

        if not (out_file.exists() and rec_file.exists()):
            continue

        out_scaffolds = smiles_to_scaffolds(
            out_file,
            unit_type,
            ncpus=cfg.inner_workers,
            chunksize=cfg.chunksize,
            use_cache=cfg.use_cache,
            show_progress=cfg.show_progress,
            label=f"OS {generator} {split} c{cluster}",
        )
        rec_scaffolds = smiles_to_scaffolds(
            rec_file,
            unit_type,
            ncpus=cfg.inner_workers,
            chunksize=cfg.chunksize,
            use_cache=cfg.use_cache,
            show_progress=cfg.show_progress,
            label=f"RS {receptor} {split} c{cluster}",
        )

        rec_unique = set(np.unique(rec_scaffolds))
        if len(rec_unique) == 0 or len(out_scaffolds) == 0:
            continue

        cluster_res = bootstrap_cluster_scaffold(
            output_scaffolds=out_scaffolds,
            recall_unique=rec_unique,
            n_recall_unique=len(rec_unique),
            n_subsample=cfg.n_subsample,
            n_bootstrap=cfg.n_bootstrap,
            seed=int(rng.integers(1_000_000_000)),
        )
        for metric in METRICS:
            per_cluster[metric].append(cluster_res[metric])
        used_clusters.append(cluster)

    out_dir = _job_output_dir(cfg.intermediate_root(), "scaffold", receptor, split, unit_type, generator)

    if not used_clusters:
        df_err = pd.DataFrame(
            [
                {
                    "metric_family": "scaffold",
                    "receptor": receptor,
                    "split": split,
                    "unit_type": unit_type,
                    "scaffold": unit_type,
                    "generator": generator,
                    "metric": None,
                    "mean": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "clusters": "",
                    "error": "no clusters found (missing files or empty sets)",
                }
            ]
        )
        _save_job_outputs(
            out_dir,
            df_err,
            None,
            [
                f"metric_family=scaffold",
                f"receptor={receptor}",
                f"split={split}",
                f"unit_type={unit_type}",
                f"generator={generator}",
                "clusters_used=",
            ],
        )
        return df_err

    rows: list[dict[str, object]] = []
    bootstrap_samples: dict[str, np.ndarray] | None = {} if cfg.save_bootstrap_samples else None

    for metric in METRICS:
        avg = np.vstack(per_cluster[metric]).mean(axis=0)
        rows.append(
            {
                "metric_family": "scaffold",
                "receptor": receptor,
                "split": split,
                "unit_type": unit_type,
                "scaffold": unit_type,
                "generator": generator,
                "metric": metric,
                "mean": float(avg.mean()),
                "ci_low": float(np.quantile(avg, cfg.alpha / 2)),
                "ci_high": float(np.quantile(avg, 1.0 - cfg.alpha / 2)),
                "clusters": ",".join(map(str, used_clusters)),
                "error": None,
            }
        )
        if bootstrap_samples is not None:
            bootstrap_samples[metric] = avg.astype(float, copy=False)

    df_job = pd.DataFrame(rows)
    _save_job_outputs(
        out_dir,
        df_job,
        bootstrap_samples,
        [
            "metric_family=scaffold",
            f"receptor={receptor}",
            f"split={split}",
            f"unit_type={unit_type}",
            f"generator={generator}",
            f"clusters_used={used_clusters}",
            f"n_bootstrap={cfg.n_bootstrap}",
            f"n_subsample={cfg.n_subsample}",
            f"alpha={cfg.alpha}",
            f"job_workers={cfg.job_workers}",
            f"inner_workers={cfg.inner_workers}",
            f"chunksize={cfg.chunksize}",
            f"use_cache={cfg.use_cache}",
            f"save_bootstrap_samples={cfg.save_bootstrap_samples}",
        ],
    )
    return df_job


# ---------------------------------------------------------------------------
# PH4 bootstrap
# ---------------------------------------------------------------------------


def _ph4_paths_for_cluster(
    data_folder: str | Path,
    receptor: str,
    split: str,
    generator: str,
    cluster: int,
) -> tuple[Path, Path]:
    project_root = resolve_data_folder(data_folder)
    out_f = (
        project_root
        / "data"
        / "output_sets"
        / "ph4"
        / receptor
        / generator
        / f"phfp_of_output_set_cluster_{cluster}_{split}_{generator}_with_smiles.csv"
    )
    rec_f = (
        project_root
        / "data"
        / "output_sets"
        / "ph4"
        / receptor
        / "RS"
        / f"phfp_of_recall_set_cluster_{cluster}_{split}_with_smiles.csv"
    )
    return out_f, rec_f


def _fpbits_cache_path(fp_csv: str | Path) -> Path:
    p = Path(fp_csv)
    return p.with_suffix(p.suffix + ".fpbits.npy")


def load_fp_strings(fp_csv: str | Path) -> list[str]:
    df = pd.read_csv(fp_csv, usecols=["fp"])
    return df["fp"].dropna().astype(str).tolist()


def fp_string_to_array(fp: str) -> np.ndarray:
    return np.fromiter((1 if ch == "1" else 0 for ch in fp), dtype=np.int8)


def fp_strings_to_2d(fp_list: list[str]) -> np.ndarray:
    if not fp_list:
        return np.zeros((0, 0), dtype=np.int8)
    return np.stack([fp_string_to_array(x) for x in fp_list], axis=0)


def load_or_build_fpbits(
    fp_csv: str | Path,
    use_cache: bool,
    show_progress: bool,
    label: str = "",
) -> tuple[list[str], np.ndarray]:
    cache = _fpbits_cache_path(fp_csv)
    fp_list = load_fp_strings(fp_csv)

    if use_cache and cache.exists():
        return fp_list, np.load(cache)

    if not fp_list:
        return fp_list, np.zeros((0, 0), dtype=np.int8)

    if show_progress:
        print(f"Converting {label}: {len(fp_list):,} fingerprints -> bits")
    arr = fp_strings_to_2d(fp_list)
    if use_cache:
        np.save(cache, arr)
    return fp_list, arr


def tanimoto_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    inter = np.count_nonzero(np.logical_and(vec1, vec2))
    union = np.count_nonzero(np.logical_or(vec1, vec2))
    return 1.0 if union == 0 else inter / union


_shm = None
_out_shape = None
_out_dtype = None


def _init_match_worker(shm_name, shape, dtype) -> None:
    global _shm, _out_shape, _out_dtype
    _shm = shared_memory.SharedMemory(name=shm_name)
    _out_shape = shape
    _out_dtype = dtype


def _match_one_recall(args: tuple[int, np.ndarray, float]) -> tuple[int, list[int]]:
    i, recall_vec, threshold = args
    out_arr = np.ndarray(_out_shape, dtype=_out_dtype, buffer=_shm.buf)
    matched: list[int] = []
    for j, out_vec in enumerate(out_arr):
        if tanimoto_similarity(recall_vec, out_vec) >= threshold:
            matched.append(j)
    return i, matched


def build_output_to_recall_map(
    recall_bits: np.ndarray,
    output_unique_bits: np.ndarray,
    threshold: float,
    ncpus: int,
    chunksize: int,
    show_progress: bool,
) -> list[list[int]]:
    n_output = int(output_unique_bits.shape[0])
    match_lists: list[list[int]] = [[] for _ in range(n_output)]

    if n_output == 0 or recall_bits.shape[0] == 0:
        return match_lists

    if ncpus <= 1 or recall_bits.shape[0] < 10:
        iterator = range(recall_bits.shape[0])
        if show_progress:
            iterator = tqdm(iterator, total=recall_bits.shape[0], desc="Build match map", unit="recall")
        for i in iterator:
            for j, out_vec in enumerate(output_unique_bits):
                if tanimoto_similarity(recall_bits[i], out_vec) >= threshold:
                    match_lists[j].append(i)
        return match_lists

    shm = shared_memory.SharedMemory(create=True, size=output_unique_bits.nbytes)
    shared_out = np.ndarray(output_unique_bits.shape, dtype=output_unique_bits.dtype, buffer=shm.buf)
    shared_out[:] = output_unique_bits[:]

    try:
        args_list = [(i, recall_bits[i], threshold) for i in range(recall_bits.shape[0])]
        with Pool(
            processes=max(1, ncpus),
            initializer=_init_match_worker,
            initargs=(shm.name, output_unique_bits.shape, output_unique_bits.dtype),
        ) as pool:
            iterator = pool.imap_unordered(_match_one_recall, args_list, chunksize=max(1, chunksize))
            if show_progress:
                iterator = tqdm(iterator, total=len(args_list), desc="Build match map", unit="recall")
            for recall_idx, matched_js in iterator:
                for j in matched_js:
                    match_lists[j].append(recall_idx)
        return match_lists
    finally:
        shm.close()
        shm.unlink()


def bootstrap_cluster_ph4(
    output_fps_raw: np.ndarray,
    output_match_any_raw: np.ndarray,
    output_fp_to_recall: dict[str, list[int]],
    n_recall_unique: int,
    n_subsample: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    if len(output_fps_raw) == 0 or n_recall_unique == 0:
        z = np.zeros(n_bootstrap, dtype=float)
        return {"RS": z.copy(), "SED": z.copy(), "ASER": z.copy()}

    idx = rng.integers(0, len(output_fps_raw), size=(n_bootstrap, n_subsample))
    sampled_fps = output_fps_raw[idx]
    sampled_match_any = output_match_any_raw[idx]

    rs = np.zeros(n_bootstrap, dtype=float)
    sed = np.zeros(n_bootstrap, dtype=float)
    aser = np.zeros(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        sample = sampled_fps[b]
        uniq = set(sample.tolist())
        sed[b] = len(uniq) / n_subsample
        aser[b] = float(np.count_nonzero(sampled_match_any[b])) / n_subsample

        matched_recall: set[int] = set()
        for fp in uniq:
            for recall_idx in output_fp_to_recall.get(fp, []):
                matched_recall.add(recall_idx)
        rs[b] = len(matched_recall) / n_recall_unique

    return {"RS": rs, "SED": sed, "ASER": aser}


def run_ph4_job(
    receptor: str,
    split: str,
    unit_type: str,
    generator: str,
    cfg: BootstrapThresholdConfig,
    seed: int,
) -> pd.DataFrame:
    del unit_type  # current repo supports only the RDKit PH4 definition
    rng = np.random.default_rng(seed)
    threshold = _threshold_for_split(split)
    per_cluster: dict[str, list[np.ndarray]] = {metric: [] for metric in METRICS}
    used_clusters: list[int] = []

    for cluster in cfg.clusters:
        out_file, rec_file = _ph4_paths_for_cluster(cfg.data_folder, receptor, split, generator, cluster)
        if not (out_file.exists() and rec_file.exists()):
            continue

        out_fps, _ = load_or_build_fpbits(out_file, cfg.use_cache, cfg.show_progress, label=f"OS c{cluster}")
        rec_fps_raw, rec_bits_raw = load_or_build_fpbits(rec_file, cfg.use_cache, cfg.show_progress, label=f"RS c{cluster}")

        rec_unique = list(dict.fromkeys(rec_fps_raw))
        if len(rec_unique) == 0 or len(out_fps) == 0:
            continue

        rec_bits = fp_strings_to_2d(rec_unique) if len(rec_unique) != len(rec_fps_raw) else rec_bits_raw

        out_unique = list(dict.fromkeys(out_fps))
        out_unique_bits = fp_strings_to_2d(out_unique)

        map_cache = out_file.with_suffix(out_file.suffix + f".thr{threshold}.out2rec.npy")
        if cfg.use_cache and map_cache.exists():
            match_lists = np.load(map_cache, allow_pickle=True).tolist()
        else:
            match_lists = build_output_to_recall_map(
                recall_bits=rec_bits,
                output_unique_bits=out_unique_bits,
                threshold=threshold,
                ncpus=cfg.inner_workers,
                chunksize=cfg.chunksize,
                show_progress=cfg.show_progress,
            )
            if cfg.use_cache:
                np.save(map_cache, np.asarray(match_lists, dtype=object), allow_pickle=True)

        fp_to_recall = {fp: match_lists[i] for i, fp in enumerate(out_unique)}
        match_any_unique = {fp: (len(fp_to_recall[fp]) > 0) for fp in out_unique}
        match_any_raw = np.asarray([match_any_unique.get(fp, False) for fp in out_fps], dtype=bool)

        cluster_res = bootstrap_cluster_ph4(
            output_fps_raw=np.asarray(out_fps, dtype=object),
            output_match_any_raw=match_any_raw,
            output_fp_to_recall=fp_to_recall,
            n_recall_unique=len(rec_unique),
            n_subsample=cfg.n_subsample,
            n_bootstrap=cfg.n_bootstrap,
            seed=int(rng.integers(1_000_000_000)),
        )

        for metric in METRICS:
            per_cluster[metric].append(cluster_res[metric])
        used_clusters.append(cluster)

    out_dir = _job_output_dir(cfg.intermediate_root(), "ph4", receptor, split, "rdkit", generator)

    if not used_clusters:
        df_err = pd.DataFrame(
            [
                {
                    "metric_family": "ph4",
                    "receptor": receptor,
                    "split": split,
                    "unit_type": "rdkit",
                    "ph4_fp": "rdkit",
                    "generator": generator,
                    "threshold": threshold,
                    "metric": None,
                    "mean": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "clusters": "",
                    "error": "no clusters found (missing files or empty sets)",
                }
            ]
        )
        _save_job_outputs(
            out_dir,
            df_err,
            None,
            [
                "metric_family=ph4",
                f"receptor={receptor}",
                f"split={split}",
                "unit_type=rdkit",
                f"generator={generator}",
                f"threshold={threshold}",
                "clusters_used=",
            ],
        )
        return df_err

    rows: list[dict[str, object]] = []
    bootstrap_samples: dict[str, np.ndarray] | None = {} if cfg.save_bootstrap_samples else None

    for metric in METRICS:
        avg = np.vstack(per_cluster[metric]).mean(axis=0)
        rows.append(
            {
                "metric_family": "ph4",
                "receptor": receptor,
                "split": split,
                "unit_type": "rdkit",
                "ph4_fp": "rdkit",
                "generator": generator,
                "threshold": threshold,
                "metric": metric,
                "mean": float(avg.mean()),
                "ci_low": float(np.quantile(avg, cfg.alpha / 2)),
                "ci_high": float(np.quantile(avg, 1.0 - cfg.alpha / 2)),
                "clusters": ",".join(map(str, used_clusters)),
                "error": None,
            }
        )
        if bootstrap_samples is not None:
            bootstrap_samples[metric] = avg.astype(float, copy=False)

    df_job = pd.DataFrame(rows)
    _save_job_outputs(
        out_dir,
        df_job,
        bootstrap_samples,
        [
            "metric_family=ph4",
            f"receptor={receptor}",
            f"split={split}",
            "unit_type=rdkit",
            f"generator={generator}",
            f"threshold={threshold}",
            f"clusters_used={used_clusters}",
            f"n_bootstrap={cfg.n_bootstrap}",
            f"n_subsample={cfg.n_subsample}",
            f"alpha={cfg.alpha}",
            f"job_workers={cfg.job_workers}",
            f"inner_workers={cfg.inner_workers}",
            f"chunksize={cfg.chunksize}",
            f"use_cache={cfg.use_cache}",
            f"save_bootstrap_samples={cfg.save_bootstrap_samples}",
        ],
    )
    return df_job


# ---------------------------------------------------------------------------
# Unified orchestration
# ---------------------------------------------------------------------------


def run_all_bootstrap_jobs(
    receptors: Sequence[str],
    splits: Sequence[str],
    generators: Sequence[str],
    cfg: BootstrapThresholdConfig,
    base_seed: int = 42,
) -> pd.DataFrame:
    family = cfg.normalized_family()
    units = ["rdkit"] if family == "ph4" else list(cfg.units)
    jobs = [(receptor, split, unit_type, generator) for receptor in receptors for split in splits for unit_type in units for generator in generators]

    runner = run_scaffold_job if family == "scaffold" else run_ph4_job

    out: list[pd.DataFrame] = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=max(1, cfg.job_workers)) as executor:
        futures = [
            executor.submit(runner, receptor, split, unit_type, generator, cfg, base_seed + i * 1000)
            for i, (receptor, split, unit_type, generator) in enumerate(jobs)
        ]
        iterator = as_completed(futures)
        if cfg.show_progress:
            iterator = tqdm(iterator, total=len(futures), desc=f"{family} bootstrap jobs", unit="job")
        for future in iterator:
            out.append(future.result())

    df_summary = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    cfg.intermediate_root().mkdir(parents=True, exist_ok=True)
    summary_path = cfg.intermediate_root() / _summary_filename(family)
    df_summary.to_csv(summary_path, index=False)
    dt = time.time() - t0
    print(f"[DONE] Wrote summary to {summary_path} ({dt/60:.2f} min)")
    return df_summary


def load_saved_bootstrap_summary(cfg: BootstrapThresholdConfig) -> pd.DataFrame:
    summary_path = cfg.intermediate_root() / _summary_filename(cfg.normalized_family())
    return pd.read_csv(summary_path)


def run_pairwise_significance_tests(cfg: BootstrapThresholdConfig) -> pd.DataFrame:
    family = cfg.normalized_family()
    root = cfg.intermediate_root()
    rows: list[pd.DataFrame] = []
    summary_path = root / _summary_filename(family)
    allowed_generators: set[str] | None = None

    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        if "generator" in summary_df.columns:
            allowed_generators = set(summary_df["generator"].dropna().astype(str).unique())

    for receptor_dir in sorted(root.iterdir()):
        if not receptor_dir.is_dir():
            continue
        receptor = receptor_dir.name

        for unit_dir in sorted(receptor_dir.iterdir()):
            if not unit_dir.is_dir():
                continue
            unit_type = unit_dir.name.replace("_scaffolds", "").replace("_ph4_fps", "")

            for split_dir in sorted(unit_dir.iterdir()):
                if not split_dir.is_dir():
                    continue
                split = split_dir.name
                gen_dirs = [d for d in sorted(split_dir.iterdir()) if d.is_dir()]
                if allowed_generators is not None:
                    gen_dirs = [d for d in gen_dirs if d.name in allowed_generators]
                if len(gen_dirs) < 2:
                    continue

                for metric in METRICS:
                    vectors: dict[str, np.ndarray] = {}
                    for job_dir in gen_dirs:
                        sample_path = job_dir / "bootstrap_samples" / f"{metric}.npy"
                        if sample_path.exists():
                            vectors[job_dir.name] = np.load(sample_path)

                    if len(vectors) < 2:
                        continue

                    extra = {
                        "metric_family": family,
                        "receptor": receptor,
                        "split": split,
                        "unit_type": unit_type,
                        "metric": metric,
                    }
                    if family == "ph4":
                        extra["threshold"] = _threshold_for_split(split)
                    rows.append(_pairwise_from_vectors(vectors, cfg.alpha, extra))

    df_tests = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out_path = root / _pairwise_filename(family)
    df_tests.to_csv(out_path, index=False)
    print(f"[DONE] Wrote pairwise tests to {out_path}")
    return df_tests


def run_full_threshold_workflow(
    receptors: Sequence[str],
    splits: Sequence[str],
    generators: Sequence[str],
    cfg: BootstrapThresholdConfig,
    final_csv_path: str | Path | None = None,
    base_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_summary = run_all_bootstrap_jobs(receptors, splits, generators, cfg, base_seed=base_seed)
    df_tests = run_pairwise_significance_tests(cfg)
    df_thresholds = compute_effect_size_thresholds_table(df_tests, cfg.normalized_family())
    out_csv = cfg.default_final_csv() if final_csv_path is None else Path(final_csv_path)
    out_path = save_effect_size_thresholds_csv(df_thresholds, out_csv)
    print(f"[DONE] Wrote final threshold CSV to {out_path}")
    return df_summary, df_tests, df_thresholds


def build_default_config(metric_family: str, data_folder: str = "../../") -> BootstrapThresholdConfig:
    cpu = os.cpu_count() or 2
    family = metric_family.strip().lower()
    return BootstrapThresholdConfig(
        metric_family=family,
        units=["csk", "murcko"] if family == "scaffold" else ["rdkit"],
        clusters=[0, 1, 2, 3, 4],
        n_subsample=250_000,
        n_bootstrap=300,
        alpha=0.05,
        data_folder=data_folder,
        job_workers=max(1, cpu - 1),
        inner_workers=max(1, cpu - 1),
        chunksize=2000 if family == "scaffold" else 200,
        use_cache=True,
        show_progress=True,
        save_bootstrap_samples=True,
    )


def default_generators_for_family(metric_family: str) -> list[str]:
    family = metric_family.strip().lower()
    if family == "scaffold":
        return list(DEFAULT_SCAFFOLD_GENERATORS)
    if family == "ph4":
        return list(DEFAULT_PH4_GENERATORS)
    raise ValueError("metric_family must be 'scaffold' or 'ph4'.")


def _forest_pretty_generator_name(name: str) -> str:
    return (
        name.replace("_epsilon", "\n epsilon")
        .replace("_mut_r", "\n mut_r")
        .replace("addcarbon", "AddCarbon")
        .replace("enamine", "Enamine")
    )


def _forest_style_axis(ax, metric_df: pd.DataFrame, color_map: dict[str, tuple], metric: str, show_ylabels: bool = True) -> None:
    y = np.arange(len(metric_df))
    for i, row in metric_df.iterrows():
        color = color_map[row["generator"]]
        ax.hlines(i, row["q025"], row["q975"], color=color, linewidth=5.5, alpha=0.70)
        ax.scatter(row["median"], i, color=color, s=240, zorder=3)

    ax.set_yticks(y)
    if show_ylabels:
        ax.set_yticklabels(metric_df["generator_display"], fontsize=22)
    else:
        ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel(r"Value ($\times 10^{-2}$)" if metric == "ASER" else "Value", fontsize=24, fontweight="normal")
    ax.set_title(metric, fontsize=27, fontweight="normal", pad=10)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.grid(True, axis="x", alpha=0.20, linestyle="--")
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _build_forest_summary_from_arrays(
    arrays_by_generator_metric: dict[tuple[str, str], list[np.ndarray]],
    generator_order: Sequence[str],
    metrics: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for generator_name in generator_order:
        for metric in metrics:
            values_list = arrays_by_generator_metric.get((generator_name, metric), [])
            if not values_list:
                continue
            values = np.concatenate(values_list)
            scale = 100.0 if metric == "ASER" else 1.0
            rows.append(
                {
                    "generator": generator_name,
                    "generator_display": _forest_pretty_generator_name(generator_name),
                    "metric": metric,
                    "median": float(np.median(values) * scale),
                    "q025": float(np.quantile(values, 0.025) * scale),
                    "q975": float(np.quantile(values, 0.975) * scale),
                    "mean": float(np.mean(values) * scale),
                    "std": float(np.std(values, ddof=1) * scale) if len(values) > 1 else 0.0,
                }
            )

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        raise ValueError("No bootstrap sample arrays found for requested configuration.")
    return summary_df


def _draw_forest_plot(
    summary_df: pd.DataFrame,
    generator_order: Sequence[str],
    metrics: Sequence[str],
    output_path_prefix: Path,
    dpi: int = 300,
) -> None:
    palette = sns.color_palette("tab20", n_colors=max(len(generator_order), 1))
    color_map = {name: palette[idx] for idx, name in enumerate(generator_order)}

    fig, axes = plt.subplots(1, len(metrics), figsize=(11 * len(metrics), 15), sharey=False)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        metric_df = summary_df[summary_df["metric"] == metric].copy()
        metric_df["generator"] = pd.Categorical(metric_df["generator"], categories=generator_order, ordered=True)
        metric_df = metric_df.sort_values("generator").reset_index(drop=True)
        _forest_style_axis(ax, metric_df, color_map, metric, show_ylabels=True)

    plt.tight_layout()
    plt.savefig(f"{output_path_prefix}.png", dpi=dpi, bbox_inches="tight")
    plt.savefig(f"{output_path_prefix}.svg", bbox_inches="tight")
    plt.show()


def _load_split_summary(
    bootstrap_root: Path,
    metric_family: str,
    receptor: str,
    unit_type: str,
    split: str,
    metrics: Sequence[str],
    generator_order: Sequence[str],
) -> pd.DataFrame:
    family_root = bootstrap_root / metric_family / receptor / unit_type / split
    if not family_root.exists():
        raise FileNotFoundError(f"Missing bootstrap directory: {family_root}")

    arrays: dict[tuple[str, str], list[np.ndarray]] = {}
    for generator_name in generator_order:
        for metric in metrics:
            sample_path = family_root / generator_name / "bootstrap_samples" / f"{metric}.npy"
            if sample_path.exists():
                arrays.setdefault((generator_name, metric), []).append(np.load(sample_path))

    return _build_forest_summary_from_arrays(arrays, generator_order, metrics)


def plot_bootstrap_forest_all_receptors_pooled(
    receptors: Sequence[str] = ("Glucocorticoid_receptor", "Leukocyte_elastase"),
    splits: Sequence[str] = ("dis", "sim"),
    metrics: Sequence[str] = ("RS", "SED", "ASER"),
    output_dir: Path | None = None,
    data_folder: str | Path | None = "../../",
    dpi: int = 300,
) -> dict[str, pd.DataFrame]:
    bootstrap_root = bootstrap_root_dir(data_folder)
    if output_dir is None:
        output_dir = bootstrap_root
    output_dir.mkdir(parents=True, exist_ok=True)

    row_configs = [
        ("scaffold", "csk_scaffolds", FOREST_GENERATOR_ORDER["scaffold"]),
        ("scaffold", "murcko_scaffolds", FOREST_GENERATOR_ORDER["scaffold"]),
        ("ph4", "rdkit_ph4_fps", FOREST_GENERATOR_ORDER["ph4"]),
    ]

    row_summaries = []
    for metric_family, unit_type, generator_order in row_configs:
        arrays: dict[tuple[str, str], list[np.ndarray]] = {}
        for receptor in receptors:
            for split in splits:
                family_root = bootstrap_root / metric_family / receptor / unit_type / split
                if not family_root.exists():
                    continue
                for generator_name in generator_order:
                    for metric in metrics:
                        sample_path = family_root / generator_name / "bootstrap_samples" / f"{metric}.npy"
                        if sample_path.exists():
                            arrays.setdefault((generator_name, metric), []).append(np.load(sample_path))
        summary_df = _build_forest_summary_from_arrays(arrays, generator_order, metrics)
        row_summaries.append((metric_family, unit_type, generator_order, summary_df))

    max_generators = max(len(generator_order) for _, _, generator_order, _ in row_summaries)
    fig, axes = plt.subplots(
        len(row_summaries),
        len(metrics),
        figsize=(11 * len(metrics), 5 + 1.35 * max_generators * len(row_summaries)),
        sharey=False,
    )
    if len(row_summaries) == 1:
        axes = np.array([axes])

    for row_idx, (_, unit_type, generator_order, summary_df) in enumerate(row_summaries):
        palette = sns.color_palette("tab20", n_colors=max(len(generator_order), 1))
        color_map = {name: palette[idx] for idx, name in enumerate(generator_order)}

        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            metric_df = summary_df[summary_df["metric"] == metric].copy()
            metric_df["generator"] = pd.Categorical(metric_df["generator"], categories=generator_order, ordered=True)
            metric_df = metric_df.sort_values("generator").reset_index(drop=True)
            _forest_style_axis(ax, metric_df, color_map, metric, show_ylabels=(col_idx == 0))

    fig.subplots_adjust(top=0.95, bottom=0.05, hspace=0.23, wspace=0.18)

    for row_idx, (_, unit_type, _, _) in enumerate(row_summaries):
        row_axes = axes[row_idx, :]
        first_pos = row_axes[0].get_position()
        last_pos = row_axes[-1].get_position()
        x_center = (first_pos.x0 + last_pos.x1) / 2
        y_top = first_pos.y1 + 0.02
        fig.text(
            x_center,
            y_top,
            FOREST_ROW_TITLES[unit_type],
            ha="center",
            va="bottom",
            fontsize=27,
            fontweight="semibold",
        )

    filename = output_dir / "bootstrap_forest_all_receptors_pooled_dis_sim"
    plt.savefig(f"{filename}.png", dpi=dpi, bbox_inches="tight")
    plt.savefig(f"{filename}.svg", bbox_inches="tight")
    plt.show()
    return {unit_type: summary_df for _, unit_type, _, summary_df in row_summaries}


def _load_pairwise_effect_sizes(bootstrap_root: Path, metric_family: str) -> pd.DataFrame:
    if metric_family == "scaffold":
        path = bootstrap_root / "scaffold" / "bootstrap_pairwise_tests_scaffolds.csv"
    elif metric_family == "ph4":
        path = bootstrap_root / "ph4" / "bootstrap_pairwise_tests_ph4.csv"
    else:
        raise ValueError("metric_family must be 'scaffold' or 'ph4'.")

    if not path.exists():
        raise FileNotFoundError(f"Missing pairwise tests file: {path}")
    return pd.read_csv(path)


def plot_empirical_effect_size_distributions_pooled(
    output_dir: Path | None = None,
    metrics: Sequence[str] = ("RS", "SED", "ASER"),
    data_folder: str | Path | None = "../../",
    dpi: int = 300,
) -> pd.DataFrame:
    bootstrap_root = bootstrap_root_dir(data_folder)
    if output_dir is None:
        output_dir = bootstrap_root
    output_dir.mkdir(parents=True, exist_ok=True)

    scaffold_df = _load_pairwise_effect_sizes(bootstrap_root, "scaffold")
    ph4_df = _load_pairwise_effect_sizes(bootstrap_root, "ph4")
    family_map = {"scaffold": scaffold_df, "ph4": ph4_df}

    fig, axes = plt.subplots(3, len(metrics), figsize=(13 * len(metrics), 18), sharey=False)
    fig.subplots_adjust(top=0.95, bottom=0.06, hspace=0.6, wspace=0.18)

    distribution_summary: list[dict[str, object]] = []
    quartile_colors = {"q25": "#4C78A8", "q50": "#F58518", "q75": "#54A24B"}

    for row_idx, (metric_family, unit_type, row_title) in enumerate(PAIRWISE_ROW_CONFIGS):
        df = family_map[metric_family].copy()
        subset = df[df["unit_type"] == unit_type].copy()
        if subset.empty:
            raise ValueError(f"No pairwise effect-size rows found for family={metric_family}, unit_type={unit_type}")

        row_axes = []
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            metric_df = subset[subset["metric"] == metric].copy()
            values = metric_df["abs_delta_mean"].dropna().to_numpy(dtype=float)
            if values.size == 0:
                raise ValueError(
                    f"No abs_delta_mean values for family={metric_family}, unit_type={unit_type}, metric={metric}"
                )

            scale = 100.0 if metric == "ASER" else 1.0
            values_scaled = values * scale
            q25, q50, q75 = np.quantile(values_scaled, [0.25, 0.50, 0.75])

            distribution_summary.append(
                {
                    "metric_family": metric_family,
                    "unit_type": unit_type,
                    "metric": metric,
                    "q25": q25,
                    "q50": q50,
                    "q75": q75,
                }
            )

            sns.histplot(values_scaled, bins=18, stat="density", color="#9ECAE1", alpha=0.55, edgecolor="white", ax=ax)
            sns.kdeplot(values_scaled, color="#2C7FB8", linewidth=2.5, ax=ax, fill=False)
            ax.axvline(q25, color=quartile_colors["q25"], linestyle="--", linewidth=2.3)
            ax.axvline(q50, color=quartile_colors["q50"], linestyle="--", linewidth=2.3)
            ax.axvline(q75, color=quartile_colors["q75"], linestyle="--", linewidth=2.3)

            if metric == "ASER":
                ax.set_xlabel(r"Absolute pairwise mean difference ($\times 10^{-2}$)", fontsize=24, fontweight="normal")
            else:
                ax.set_xlabel("Absolute pairwise mean difference", fontsize=24, fontweight="normal")
            ax.set_ylabel("Density", fontsize=23, fontweight="normal")
            ax.set_title(metric, fontsize=28, fontweight="normal", pad=10)
            ax.tick_params(axis="both", labelsize=21)
            ax.grid(True, axis="y", alpha=0.18, linestyle="--")
            ax.grid(False, axis="x")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.text(
                0.98,
                0.96,
                f"25%={q25:.3f}\n50%={q50:.3f}\n75%={q75:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=20,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.82, edgecolor="none"),
            )
            row_axes.append(ax)

        first_pos = row_axes[0].get_position()
        last_pos = row_axes[-1].get_position()
        x_center = (first_pos.x0 + last_pos.x1) / 2
        y_top = first_pos.y1 + 0.04
        fig.text(x_center, y_top, row_title, ha="center", va="bottom", fontsize=27, fontweight="semibold")

    filename = output_dir / "empirical_effect_size_distribution_all_receptors_pooled_dis_sim"
    plt.savefig(f"{filename}.png", dpi=dpi, bbox_inches="tight")
    plt.savefig(f"{filename}.svg", bbox_inches="tight")
    plt.show()

    return pd.DataFrame(distribution_summary)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified bootstrap threshold analysis for scaffold and PH4 metrics.")
    parser.add_argument("--metric-family", required=True, choices=["scaffold", "ph4"])
    parser.add_argument("--data-folder", default="../../", help="Project root containing the data/ directory.")
    parser.add_argument("--receptors", default="Glucocorticoid_receptor,Leukocyte_elastase")
    parser.add_argument("--splits", default="dis,sim")
    parser.add_argument("--generators", default="", help="Comma-separated generator names. Empty uses family-specific defaults.")
    parser.add_argument("--units", default="", help="Comma-separated scaffold types or PH4 types. Empty uses defaults.")
    parser.add_argument("--clusters", default="0,1,2,3,4")
    parser.add_argument("--n-subsample", type=int, default=250_000)
    parser.add_argument("--n-bootstrap", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--job-workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--inner-workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--chunksize", type=int, default=200)
    parser.add_argument("--final-csv", default="", help="Optional output path for the final effect-size threshold CSV.")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-bootstrap-samples", action="store_true")
    return parser


def main_cli() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    family = args.metric_family.strip().lower()
    units = _parse_name_list(args.units)
    if not units:
        units = ["csk", "murcko"] if family == "scaffold" else ["rdkit"]

    cfg = BootstrapThresholdConfig(
        metric_family=family,
        units=units,
        clusters=_parse_cluster_list(args.clusters),
        n_subsample=int(args.n_subsample),
        n_bootstrap=int(args.n_bootstrap),
        alpha=float(args.alpha),
        data_folder=str(args.data_folder),
        job_workers=int(args.job_workers),
        inner_workers=int(args.inner_workers),
        chunksize=int(args.chunksize),
        use_cache=not bool(args.no_cache),
        show_progress=not bool(args.no_progress),
        save_bootstrap_samples=not bool(args.no_bootstrap_samples),
    )

    final_csv = args.final_csv or None
    generators = _parse_name_list(args.generators)
    if not generators:
        generators = default_generators_for_family(family)

    run_full_threshold_workflow(
        receptors=_parse_name_list(args.receptors),
        splits=_parse_name_list(args.splits),
        generators=generators,
        cfg=cfg,
        final_csv_path=final_csv,
    )


if __name__ == "__main__":
    main_cli()
