#!/bin/bash

# Scaffold-based metrics from one custom recall/output pair
python3 ../src/metrics_own_data.py \
    --metric_family scaffold \
    --unit_type csk \
    --generator_name custom_generator \
    --receptor custom_project \
    --type_cluster dis \
    --cluster_id 0 \
    --recall_set_path /absolute/path/to/recall_set.csv \
    --output_set_path /absolute/path/to/output_set.csv \
    --data_folder ../../ \
    --ncpus 4

# Pharmacophore-based metrics from one custom recall/output pair.
# Input CSV files must contain an fp column.
#
# python3 ../src/metrics_own_data.py \
#     --metric_family ph4 \
#     --unit_type rdkit \
#     --generator_name custom_generator \
#     --receptor custom_project \
#     --type_cluster sim \
#     --cluster_id 0 \
#     --recall_set_path /absolute/path/to/recall_fingerprints.csv \
#     --output_set_path /absolute/path/to/output_fingerprints.csv \
#     --threshold 0.8 \
#     --data_folder ../../ \
#     --ncpus 4
