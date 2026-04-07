from __future__ import annotations

import argparse

from src.metrics_custom_inputs import CustomMetricsConfig, calculate_metrics_from_files


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate scaffold-based or pharmacophore-based metrics from one custom recall/output pair.",
    )
    parser.add_argument("--metric_family", required=True, choices=["scaffold", "ph4"])
    parser.add_argument("--unit_type", required=True, help="For scaffold: csk/murcko. For ph4: rdkit.")
    parser.add_argument("--generator_name", required=True, help="Label used in saved outputs.")
    parser.add_argument("--receptor", default="custom", help="Folder label used in saved outputs.")
    parser.add_argument("--type_cluster", default="custom", help="Split/cluster label used in saved outputs.")
    parser.add_argument("--recall_set_path", required=True, help="Path to recall set file.")
    parser.add_argument("--output_set_path", required=True, help="Path to output set file.")
    parser.add_argument("--cluster_id", default="0", help="Cluster identifier used in saved filenames.")
    parser.add_argument("--threshold", type=float, default=None, help="Required for ph4 custom inputs.")
    parser.add_argument("--data_folder", default="", help="Project root containing the data/ directory.")
    parser.add_argument("--ncpus", type=int, default=1, help="Number of CPUs to use.")
    parser.add_argument("--no_cache", action="store_true", help="Disable scaffold caches.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

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

    result = calculate_metrics_from_files(
        output_set_path=args.output_set_path,
        recall_set_path=args.recall_set_path,
        cfg=cfg,
        cluster_id=args.cluster_id,
    )

    print("RESULTS:")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
