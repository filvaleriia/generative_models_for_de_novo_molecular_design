#!/bin/bash

# Scaffold-based metrics for one generator
python3 ../src/metrics_scaffold.py \
    --type_cluster sim \
    --type_scaffold csk \
    --generator Molpher \
    --receptor Glucocorticoid_receptor \
    --data_folder ../../ \
    --ncpus 4

# Pharmacophore-based metrics for one generator
# Requires precomputed fingerprint CSV files in data/output_sets/ph4/.
# Typical preprocessing step:
# python3 ../src/compute_pharmacophore_fingerprints.py \
#     --receptor Glucocorticoid_receptor \
#     --split sim \
#     --generator Molpher \
#     --dataset all \
#     --clusters 0 1 2 3 4 \
#     --type_phfp rdkit \
#     --data_folder ../../ \
#     --ncpus 4
#
# python3 ../src/metrics_ph4.py \
#     --type_cluster sim \
#     --type_phfp rdkit \
#     --generator Molpher \
#     --receptor Glucocorticoid_receptor \
#     --threshold 0.8 \
#     --data_folder ../../ \
#     --ncpus 4

