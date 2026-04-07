#!/bin/bash

# Scaffold-based metrics from custom path patterns.
# The patterns must contain the {cluster} placeholder.
python3 ../src/metrics_define_path.py \
    --metric_family scaffold \
    --unit_type csk \
    --generator_name custom_generator \
    --receptor custom_project \
    --type_cluster dis \
    --recall_pattern "/absolute/path/to/recall/cRS_custom_project_dis_{cluster}.csv" \
    --output_pattern "/absolute/path/to/output/cOS_custom_generator_dis_{cluster}_one_column.csv" \
    --clusters 0,1,2,3,4 \
    --data_folder ../../ \
    --ncpus 4

# Pharmacophore-based metrics from custom path patterns.
# Input files must be CSV files containing an fp column.
#
# python3 ../src/metrics_define_path.py \
#     --metric_family ph4 \
#     --unit_type rdkit \
#     --generator_name custom_generator \
#     --receptor custom_project \
#     --type_cluster sim \
#     --recall_pattern "/absolute/path/to/recall/phfp_of_recall_set_cluster_{cluster}_sim_with_smiles.csv" \
#     --output_pattern "/absolute/path/to/output/phfp_of_output_set_cluster_{cluster}_sim_custom_generator_with_smiles.csv" \
#     --clusters 0,1,2,3,4 \
#     --threshold 0.8 \
#     --data_folder ../../ \
#     --ncpus 4

