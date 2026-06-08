# Generative Models for de novo Molecular Design

This repository contains the evaluation workflow used in the dissertation project on molecular generators. The project compares generated molecules using two complementary metric families:

- scaffold-based metrics
- pharmacophore-based metrics

The repository is organized around a simple principle:

- reusable code lives in `src/`
- analyses are driven from Jupyter notebooks
- most notebooks now call helper functions from `src/` instead of keeping long plotting logic inline

## Scope

The repository supports:

- scaffold-based metric calculation
- pharmacophore-based metric calculation
- bootstrap-based effect-size threshold estimation
- scaffold vs pharmacophore comparison
- subset-size analysis
- pharmacophore threshold sensitivity analysis
- fine-tuning vs without-fine-tuning comparison
- heatmap and summary figure generation

The current case studies are:

- `Glucocorticoid_receptor`
- `Leukocyte_elastase`

## Metrics

Three metrics are used in both evaluation frameworks:

- `RS`
  Unit Recovery Score
- `SED`
  SEt unit Diversity
- `ASER`
  Absolute SEt unit Recall

The formulas are shared across both workflows. What changes is the definition of the structural unit:

- scaffold-based workflow: scaffold
- pharmacophore-based workflow: pharmacophore fingerprint match

### RS

$$
RS = \frac{\text{Unique active units in the OS}}{\text{Unique active units in the RS}}
$$

### SED

$$
SED = \frac{\text{Unique units in the OS}}{c_{OS}}
$$

### ASER

$$
ASER = \frac{\text{Count of active units in the OS}}{c_{OS}}
$$

Here, `OS` denotes the Output Set and `RS` denotes the Recall Set.

## Structural Representations

### Scaffold-based

The scaffold workflow currently supports:

- `csk`
- `murcko`

### Pharmacophore-based

The pharmacophore workflow currently uses:

- `rdkit` pharmacophore fingerprints

Pharmacophore matching is based on Tanimoto similarity. In the main workflow, the recommended thresholds are:

- `0.7` for the `dis` split
- `0.8` for the `sim` split

## Main Notebooks

The main notebook entry points are:

- [calculate_the_metrics_scaffold_based.ipynb](/home/filv/phd_projects/iga_2023/git_reccal/new/diseration_git/generative_models_for_de_novo_molecular_design/calculate_the_metrics_scaffold_based.ipynb)
- [calculate_the_metrics_ph4_based.ipynb](/home/filv/phd_projects/iga_2023/git_reccal/new/diseration_git/generative_models_for_de_novo_molecular_design/calculate_the_metrics_ph4_based.ipynb)
- [bootstrap_threshold_analysis.ipynb](/home/filv/phd_projects/iga_2023/git_reccal/new/diseration_git/generative_models_for_de_novo_molecular_design/bootstrap_threshold_analysis.ipynb)
- [comparison_of_metrics.ipynb](/home/filv/phd_projects/iga_2023/git_reccal/new/diseration_git/generative_models_for_de_novo_molecular_design/comparison_of_metrics.ipynb)
- [vizualizations_heat_maps.ipynb](/home/filv/phd_projects/iga_2023/git_reccal/new/diseration_git/generative_models_for_de_novo_molecular_design/vizualizations_heat_maps.ipynb)
- [select_subsets_for_analyze.ipynb](/home/filv/phd_projects/iga_2023/git_reccal/new/diseration_git/generative_models_for_de_novo_molecular_design/select_subsets_for_analyze.ipynb)
- [visualize_ph4_threshold_sensitivity.ipynb](/home/filv/phd_projects/iga_2023/git_reccal/new/diseration_git/generative_models_for_de_novo_molecular_design/visualize_ph4_threshold_sensitivity.ipynb)
- [compare_finetuning_vs_without_finetuning.ipynb](/home/filv/phd_projects/iga_2023/git_reccal/new/diseration_git/generative_models_for_de_novo_molecular_design/compare_finetuning_vs_without_finetuning.ipynb)

Additional helper notebooks in the repository:

- [generate_data_active.ipynb](/home/filv/phd_projects/iga_2023/git_reccal/new/diseration_git/generative_models_for_de_novo_molecular_design/generate_data_active.ipynb)
- [generated_set_convert_to_one_column.ipynb](/home/filv/phd_projects/iga_2023/git_reccal/new/diseration_git/generative_models_for_de_novo_molecular_design/generated_set_convert_to_one_column.ipynb)
- [statistic_about_clusters.ipynb](/home/filv/phd_projects/iga_2023/git_reccal/new/diseration_git/generative_models_for_de_novo_molecular_design/statistic_about_clusters.ipynb)

## Core Modules in `src/`

The most important modules are:

- [src/metrics_scaffold.py](/home/filv/phd_projects/iga_2023/git_reccal/new/diseration_git/generative_models_for_de_novo_molecular_design/src/metrics_scaffold.py)
  Scaffold-based metric calculation.
- [src/metrics_ph4.py](/home/filv/phd_projects/iga_2023/git_reccal/new/diseration_git/generative_models_for_de_novo_molecular_design/src/metrics_ph4.py)
  Pharmacophore-based metric calculation.
- [src/compute_pharmacophore_fingerprints.py](/home/filv/phd_projects/iga_2023/git_reccal/new/diseration_git/generative_models_for_de_novo_molecular_design/src/compute_pharmacophore_fingerprints.py)
  Precomputes RDKit pharmacophore fingerprints.
- [src/bootstrap_threshold_analysis.py](/home/filv/phd_projects/iga_2023/git_reccal/new/diseration_git/generative_models_for_de_novo_molecular_design/src/bootstrap_threshold_analysis.py)
  Unified bootstrap workflow and bootstrap visualizations.
- [src/metric_comparison_analysis.py](/home/filv/phd_projects/iga_2023/git_reccal/new/diseration_git/generative_models_for_de_novo_molecular_design/src/metric_comparison_analysis.py)
  Scaffold vs pharmacophore comparison workflow, overlap summaries, and chemical-space analysis.
- [src/heatmap_visualization.py](/home/filv/phd_projects/iga_2023/git_reccal/new/diseration_git/generative_models_for_de_novo_molecular_design/src/heatmap_visualization.py)
  Heatmap generation and figure export.
- [src/ph4_threshold_sensitivity.py](/home/filv/phd_projects/iga_2023/git_reccal/new/diseration_git/generative_models_for_de_novo_molecular_design/src/ph4_threshold_sensitivity.py)
  Threshold-sensitivity loading and plotting helpers.
- [src/fine_tuning_comparison.py](/home/filv/phd_projects/iga_2023/git_reccal/new/diseration_git/generative_models_for_de_novo_molecular_design/src/fine_tuning_comparison.py)
  Fine-tuning vs without-fine-tuning comparison helpers.
- [src/path_utils.py](/home/filv/phd_projects/iga_2023/git_reccal/new/diseration_git/generative_models_for_de_novo_molecular_design/src/path_utils.py)
  Canonical path helpers for `data/` subdirectories.

Other utility modules retained in the repository:

- `metrics_connection.py`
- `metrics_connection_phfp.py`
- `metrics_define_path.py`
- `metrics_custom_inputs.py`
- `metrics_own_data.py`
- `preprocesing.py`

## Repository Structure

```text
generative_models_for_de_novo_molecular_design/
├── src/
├── examples/
├── molecular_generators_scripts/
├── img/
├── calculate_the_metrics_scaffold_based.ipynb
├── calculate_the_metrics_ph4_based.ipynb
├── bootstrap_threshold_analysis.ipynb
├── comparison_of_metrics.ipynb
├── vizualizations_heat_maps.ipynb
├── select_subsets_for_analyze.ipynb
├── visualize_ph4_threshold_sensitivity.ipynb
├── compare_finetuning_vs_without_finetuning.ipynb
├── effect_size_thresholds_scaffolds.csv
├── effect_size_thresholds_ph4_rdkit.csv
└── environment.yml
```

## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate scaffold-based-metrics-env
```

Key dependencies include:

- `rdkit`
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `seaborn`
- `matplotlib`
- `umap-learn`
- `jupyterlab`

## Data Layout

The notebooks and modules expect a `data/` directory outside or above the repository, depending on the notebook working directory. In the current project layout, important subdirectories include:

```text
data/
├── input_recall_sets/
├── output_sets/
├── results_scaffold_based/
├── results_pharm_based/
├── thresholds_ph4/
├── fine_tuning_comparison/
├── bootstrap_threshold_analysis/
├── subset_analysis/
└── comparison_outputs/
```

Some workflows also use:

- `data/generators_without_finetuning/`
- `data/information_about_clusters/`

## Stored Threshold Tables

The repository currently keeps the final effect-size threshold CSV files in the repo root:

- [effect_size_thresholds_scaffolds.csv](/home/filv/phd_projects/iga_2023/git_reccal/new/diseration_git/generative_models_for_de_novo_molecular_design/effect_size_thresholds_scaffolds.csv)
- [effect_size_thresholds_ph4_rdkit.csv](/home/filv/phd_projects/iga_2023/git_reccal/new/diseration_git/generative_models_for_de_novo_molecular_design/effect_size_thresholds_ph4_rdkit.csv)

Intermediate bootstrap outputs are stored in:

- `data/bootstrap_threshold_analysis/`

## Archived Data Package

The archived data package is available on Zenodo:

[10.5281/zenodo.19466540](https://doi.org/10.5281/zenodo.19466540)

The archive is large, so downloading it requires sufficient local storage. Precomputed RDKit pharmacophore fingerprint files are not included and may need to be regenerated locally with [src/compute_pharmacophore_fingerprints.py](/home/filv/phd_projects/iga_2023/git_reccal/new/diseration_git/generative_models_for_de_novo_molecular_design/src/compute_pharmacophore_fingerprints.py).

## Figures

Repository-level overview figures:

- [img/project_workflow.png](/home/filv/phd_projects/iga_2023/git_reccal/new/diseration_git/generative_models_for_de_novo_molecular_design/img/project_workflow.png)
- [img/set_composition.png](/home/filv/phd_projects/iga_2023/git_reccal/new/diseration_git/generative_models_for_de_novo_molecular_design/img/set_composition.png)
- [img/sim-dis-with-background.png](/home/filv/phd_projects/iga_2023/git_reccal/new/diseration_git/generative_models_for_de_novo_molecular_design/img/sim-dis-with-background.png)

## Notes

- The repository still contains some older helper scripts and generator-specific folders used during model preparation.
- Recent refactoring moved several notebook plotting sections into `src/` modules to keep notebooks shorter and easier to maintain.
- The current README describes the evaluation workflow and main analysis entry points, not every archived training artifact under `molecular_generators_scripts/`.
