from __future__ import annotations

import argparse

from src.metrics_custom_inputs import CustomMetricsConfig, calculate_metrics_from_patterns


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate scaffold-based or pharmacophore-based metrics from custom path patterns.",
    )
    parser.add_argument("--metric_family", required=True, choices=["scaffold", "ph4"])
    parser.add_argument("--unit_type", required=True, help="For scaffold: csk/murcko. For ph4: rdkit.")
    parser.add_argument("--generator_name", required=True, help="Label used in saved outputs.")
    parser.add_argument("--receptor", required=True, help="Folder label used in saved outputs.")
    parser.add_argument("--type_cluster", required=True, help="Split label used in saved outputs.")
    parser.add_argument("--output_pattern", required=True, help="Pattern with {cluster} placeholder for output files.")
    parser.add_argument("--recall_pattern", required=True, help="Pattern with {cluster} placeholder for recall files.")
    parser.add_argument("--clusters", default="0,1,2,3,4", help="Comma-separated cluster ids.")
    parser.add_argument("--threshold", type=float, default=None, help="Required for ph4 custom inputs.")
    parser.add_argument("--data_folder", default="", help="Project root containing the data/ directory.")
    parser.add_argument("--ncpus", type=int, default=1, help="Number of CPUs to use.")
    parser.add_argument("--no_cache", action="store_true", help="Disable scaffold caches.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    cluster_ids = [cluster.strip() for cluster in args.clusters.split(",") if cluster.strip()]

    cfg = CustomMetricsConfig(
        metric_family=args.metric_family,
        unit_type=args.unit_type,
        generator_name=args.generator_name,
        receptor=args.receptor,
        type_cluster=args.type_cluster,
        data_folder=args.data_folder,
        ncpus=args.ncpus,
        threshold=args.threshold,
        use_cache=not bool(args.no_cache),
    )

    result = calculate_metrics_from_patterns(
        output_pattern=args.output_pattern,
        recall_pattern=args.recall_pattern,
        cfg=cfg,
        cluster_ids=cluster_ids,
    )

    print("RESULTS:")
    if result.empty:
        print("No matching input files were found.")
    else:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
