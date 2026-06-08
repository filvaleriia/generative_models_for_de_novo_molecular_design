#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
from multiprocessing import Pool
from pathlib import Path
import sys
from typing import Iterable

from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem.Pharm2D.SigFactory import SigFactory

# Allow both:
#   python -m src.compute_pharmacophore_fingerprints
# and:
#   python src/compute_pharmacophore_fingerprints.py
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.path_utils import data_subdir, resolve_data_folder


SIGFACTORY: SigFactory | None = None
LOG_FILE: Path | None = None
VALID_SPLITS = {"dis", "sim"}
DEFAULT_CLUSTERS = [0, 1, 2, 3, 4]


def log_print(*args) -> None:
    message = " ".join(map(str, args))
    timestamp = dt.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if LOG_FILE is not None:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} {message}\n")
    print(message, flush=True)


def normalize_ncpus(ncpus: int) -> int:
    available = os_cpu_count()
    return max(1, min(int(ncpus), available))


def os_cpu_count() -> int:
    import os

    return os.cpu_count() or 1


def find_basefeatures_fdef(explicit_path: str | None = None) -> str:
    candidates: list[Path] = []

    if explicit_path:
        candidates.append(Path(explicit_path).expanduser().resolve())

    candidates.append(Path.cwd() / "BaseFeatures.fdef")
    candidates.append(Path(__file__).resolve().parent / "BaseFeatures.fdef")
    candidates.append(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "Could not find 'BaseFeatures.fdef'. Provide --fdef_path or place the file "
        "in the working directory or next to the script."
    )


def prepare_sigfactory(fdef_path: str | None = None) -> SigFactory:
    feature_factory = ChemicalFeatures.BuildFeatureFactory(find_basefeatures_fdef(fdef_path))
    sig_factory = SigFactory(feature_factory, minPointCount=2, maxPointCount=3)
    sig_factory.SetBins([(0, 2), (2, 4), (4, 6), (6, 8)])
    sig_factory.Init()
    return sig_factory


def _init_worker(fdef_path: str | None) -> None:
    global SIGFACTORY
    SIGFACTORY = prepare_sigfactory(fdef_path)


def smiles_to_phfp(smiles: str) -> tuple[str, str]:
    global SIGFACTORY
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or SIGFACTORY is None:
            return smiles, ""

        fp = Generate.Gen2DFingerprint(mol, SIGFACTORY)
        if hasattr(fp, "ToBitString"):
            return smiles, fp.ToBitString()

        nonzero = fp.GetNonzeroElements()
        if not nonzero:
            return smiles, ""
        return smiles, ";".join(f"{idx}:{count}" for idx, count in nonzero.items())
    except Exception:
        return smiles, ""


def load_smiles_from_one_column_file(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def calculate_phfp_streaming(
    smiles_list: list[str],
    out_csv_path: Path,
    num_cpus: int,
    chunksize: int = 200,
    fdef_path: str | None = None,
) -> None:
    total = len(smiles_list)
    log_print("START PHFP streaming | N =", total, "| CPUs =", num_cpus, "| chunksize =", chunksize)
    log_print("Output CSV:", out_csv_path)

    processed = 0
    failed = 0
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with Pool(processes=num_cpus, initializer=_init_worker, initargs=(fdef_path,)) as pool, \
         out_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fp", "smiles"])

        for smiles, fp_str in pool.imap_unordered(smiles_to_phfp, smiles_list, chunksize=chunksize):
            if not fp_str:
                failed += 1
            writer.writerow([fp_str, smiles])
            processed += 1

            if processed % 50000 == 0:
                log_print("Processed", processed, "/", total, "| empty/failed fp:", failed)
                handle.flush()

    log_print("DONE | processed:", processed, "| empty/failed fp:", failed)
    gc.collect()


def build_output_source_path(base_dir: Path, receptor: str, generator: str, split: str, cluster: int) -> Path:
    return data_subdir(
        base_dir,
        "output_sets",
        receptor,
        generator,
    ) / f"cOS_{generator}_{split}_{cluster}_one_column.csv"


def build_recall_source_path(base_dir: Path, receptor: str, split: str, cluster: int) -> Path:
    return data_subdir(
        base_dir,
        "input_recall_sets",
        receptor,
    ) / f"cRS_{receptor}_{split}_{cluster}.csv"


def build_input_source_path(base_dir: Path, receptor: str, split: str, cluster: int) -> Path:
    return data_subdir(
        base_dir,
        "input_recall_sets",
        receptor,
    ) / f"cIS_{receptor}_{split}_{cluster}.csv"


def build_output_fp_path(base_dir: Path, receptor: str, generator: str, split: str, cluster: int) -> Path:
    return data_subdir(
        base_dir,
        "output_sets",
        "ph4",
        receptor,
        generator,
    ) / f"phfp_of_output_set_cluster_{cluster}_{split}_{generator}_with_smiles.csv"


def build_recall_fp_path(base_dir: Path, receptor: str, split: str, cluster: int) -> Path:
    return data_subdir(
        base_dir,
        "output_sets",
        "ph4",
        receptor,
        "RS",
    ) / f"phfp_of_recall_set_cluster_{cluster}_{split}_with_smiles.csv"


def build_input_fp_path(base_dir: Path, receptor: str, split: str, cluster: int) -> Path:
    return data_subdir(
        base_dir,
        "input_recall_sets",
        "ph4",
        receptor,
    ) / f"phfp_of_input_set_cluster_{cluster}_{split}_with_smiles.csv"


def iter_jobs(
    base_dir: Path,
    receptor: str,
    generator: str | None,
    split: str,
    clusters: Iterable[int],
    dataset: str,
) -> list[tuple[str, int, Path, Path]]:
    jobs: list[tuple[str, int, Path, Path]] = []
    for cluster in clusters:
        if dataset in {"output", "all"}:
            if not generator:
                raise ValueError("--generator is required when dataset includes output.")
            jobs.append(
                (
                    "output",
                    cluster,
                    build_output_source_path(base_dir, receptor, generator, split, cluster),
                    build_output_fp_path(base_dir, receptor, generator, split, cluster),
                )
            )
        if dataset in {"recall", "all"}:
            jobs.append(
                (
                    "recall",
                    cluster,
                    build_recall_source_path(base_dir, receptor, split, cluster),
                    build_recall_fp_path(base_dir, receptor, split, cluster),
                )
            )
        if dataset in {"input", "all"}:
            jobs.append(
                (
                    "input",
                    cluster,
                    build_input_source_path(base_dir, receptor, split, cluster),
                    build_input_fp_path(base_dir, receptor, split, cluster),
                )
            )
    return jobs


def run_job(
    label: str,
    source_path: Path,
    target_path: Path,
    ncpus: int,
    chunksize: int,
    overwrite: bool,
    fdef_path: str | None,
) -> None:
    if not source_path.exists():
        log_print(f"Skipping {label}: missing source file {source_path}")
        return

    if target_path.exists() and not overwrite:
        log_print(f"Skipping {label}: target already exists {target_path}")
        return

    smiles = load_smiles_from_one_column_file(source_path)
    log_print(f"Loaded {len(smiles):,} SMILES from {source_path}")
    calculate_phfp_streaming(
        smiles_list=smiles,
        out_csv_path=target_path,
        num_cpus=ncpus,
        chunksize=chunksize,
        fdef_path=fdef_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute RDKit Pharm2D pharmacophore fingerprints for output/recall/input sets."
    )
    parser.add_argument("--receptor", required=True, help="Receptor name")
    parser.add_argument("--split", required=True, choices=sorted(VALID_SPLITS), help="Cluster split: dis or sim")
    parser.add_argument("--generator", help="Generator name. Required for output/all dataset generation.")
    parser.add_argument("--type_phfp", default="rdkit", help="Pharmacophore fingerprint type. Only rdkit is supported.")
    parser.add_argument("--dataset", default="all", choices=["output", "recall", "input", "all"])
    parser.add_argument("--clusters", nargs="+", type=int, default=DEFAULT_CLUSTERS, help="Cluster numbers to process")
    parser.add_argument("--ncpus", type=int, default=1, help="Number of CPUs to use")
    parser.add_argument("--chunksize", type=int, default=200, help="Multiprocessing chunksize")
    parser.add_argument("--data_folder", default="", help="Project root containing the data/ directory")
    parser.add_argument("--fdef_path", help="Optional explicit path to BaseFeatures.fdef")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing fingerprint CSV files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.type_phfp.lower() != "rdkit":
        raise ValueError(f"Unsupported type_phfp '{args.type_phfp}'. Only 'rdkit' is currently supported.")

    base_dir = resolve_data_folder(args.data_folder)
    ncpus = normalize_ncpus(args.ncpus)

    global LOG_FILE
    LOG_FILE = data_subdir(base_dir, "output_sets", "ph4", args.receptor, "phfp_generation.log")

    log_print("ARGS:", vars(args))
    jobs = iter_jobs(
        base_dir=base_dir,
        receptor=args.receptor,
        generator=args.generator,
        split=args.split,
        clusters=args.clusters,
        dataset=args.dataset,
    )

    for dataset_label, cluster, source_path, target_path in jobs:
        log_print(f"=== {dataset_label.upper()} | cluster {cluster} ===")
        run_job(
            label=f"{dataset_label} cluster {cluster}",
            source_path=source_path,
            target_path=target_path,
            ncpus=ncpus,
            chunksize=args.chunksize,
            overwrite=args.overwrite,
            fdef_path=args.fdef_path,
        )


if __name__ == "__main__":
    main()
