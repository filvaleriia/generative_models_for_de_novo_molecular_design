#!/bin/bash

# Merge scaffold-based mean metric tables across generators
python3 ../src/metrics_connection.py \
    --type_cluster sim \
    --type_scaffold csk \
    --generator_list Molpher DrugEx_GT_epsilon_0.6 REINVENT DrugEx_GT_epsilon_0.1 \
    --receptor Glucocorticoid_receptor \
    --data_folder ../../

# Merge pharmacophore-based mean metric tables across generators
#
# python3 ../src/metrics_connection_phfp.py \
#     --type_cluster sim \
#     --type_phfp rdkit \
#     --generator_list Molpher DrugEx_GT_epsilon_0.6 REINVENT DrugEx_GT_epsilon_0.1 \
#     --receptor Glucocorticoid_receptor \
#     --threshold_sim 0.8 \
#     --threshold_dis 0.7 \
#     --data_folder ../../
