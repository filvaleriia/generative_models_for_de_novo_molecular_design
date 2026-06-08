# Generative Models for de novo Molecular Design

This repository contains the evaluation workflow used in the dissertation project on molecular generators. The project compares generated molecules using two complementary evaluation frameworks:

- scaffold-based evaluation
- pharmacophore-based evaluation

The repository follows a simple structure:

- reusable code lives in `src/`
- analyses are run from Jupyter notebooks
- notebooks are kept as thin runners whenever possible, while the main logic is stored in `src/`

## Scope

The repository currently supports:

- scaffold-based metric calculation
- pharmacophore-based metric calculation
- bootstrap-based effect-size threshold estimation
- scaffold-based versus pharmacophore-based comparison
- pairwise statistical testing of generators
- subset-size stability analysis
- pharmacophore threshold sensitivity analysis
- pretrained versus fine-tuned generator comparison
- figure and heatmap generation

The current case studies are:

- `Glucocorticoid_receptor`
- `Leukocyte_elastase`

## Metrics

Three metrics are used in both evaluation frameworks. The metric formula stays the same, while the definition of the structural unit depends on the framework:

- scaffold-based workflow: scaffold
- pharmacophore-based workflow: pharmacophore fingerprint match

`OS` denotes the Output Set and `RS` denotes the Recall Set.

### structural unit Recovery Score (RS)

$$
RS = \frac{\text{Unique Active structural units in the OS}}{\text{Unique Active structural units in the RS}}
$$

### SEt structural unit Diversity (SED)

$$
SED = \frac{\text{Unique structural units in the OS}}{c_{OS}}
$$

### Absolute SEt structural unit Recall (ASER)

$$
ASER = \frac{\text{Count of Active structural units in the OS}}{c_{OS}}
$$

## Structural Representations

### Scaffold-based

The scaffold-based workflow currently supports:

- `csk`
- `murcko`

### Pharmacophore-based

The pharmacophore-based workflow currently uses:

- `rdkit` pharmacophore fingerprints

Pharmacophore matching is based on Tanimoto similarity. In the main workflow, the selected thresholds are:

- `0.7` for the `dis` split
- `0.8` for the `sim` split

## Main Notebooks

The main notebook entry points are:

- `calculate_the_metrics_scaffold_based.ipynb`
- `calculate_the_metrics_ph4_based.ipynb`
- `bootstrap_threshold_analysis.ipynb`
- `comparison_of_metrics.ipynb`
- `generator_significance_testing.ipynb`
- `select_subsets_for_analyze.ipynb`
- `visualize_ph4_threshold_sensitivity.ipynb`
- `compare_finetuning_vs_without_finetuning.ipynb`
- `vizualizations_heat_maps.ipynb`

Additional helper notebooks in the repository:

- `generate_data_active.ipynb`
- `generated_set_convert_to_one_column.ipynb`
- `statistic_about_clusters.ipynb`

## Core Modules in `src/`

The main reusable modules are:

- `src/metrics_scaffold.py`
  Scaffold-based metric calculation.
- `src/metrics_ph4.py`
  Pharmacophore-based metric calculation.
- `src/compute_pharmacophore_fingerprints.py`
  Precomputes RDKit pharmacophore fingerprints.
- `src/bootstrap_threshold_analysis.py`
  Bootstrap workflow for effect-size threshold estimation and related visualizations.
- `src/metric_comparison_analysis.py`
  Scaffold-based versus pharmacophore-based comparison workflow, overlap summaries, and chemical-space analysis.
- `src/generator_pairwise_significance.py`
  Pairwise sign-flip permutation testing of generators and significance heatmap generation.
- `src/ph4_threshold_sensitivity.py`
  Pharmacophore threshold-sensitivity loading and plotting helpers.
- `src/fine_tuning_comparison.py`
  Pretrained versus fine-tuned generator comparison helpers.
- `src/heatmap_visualization.py`
  Heatmap generation and figure export.
- `src/path_utils.py`
  Canonical path helpers for `data/` subdirectories.

Additional legacy helper modules retained in the repository:

- `metrics_connection.py`
- `metrics_connection_phfp.py`
- `metrics_custom_inputs.py`
- `metrics_define_path.py`
- `metrics_own_data.py`
- `preprocesing.py`

## Repository Structure

```text
generative_models_for_de_novo_molecular_design/
├── src/
├── img/
├── molecular_generators_scripts/
├── bootstrap_threshold_analysis.ipynb
├── calculate_the_metrics_ph4_based.ipynb
├── calculate_the_metrics_scaffold_based.ipynb
├── compare_finetuning_vs_without_finetuning.ipynb
├── comparison_of_metrics.ipynb
├── generator_significance_testing.ipynb
├── select_subsets_for_analyze.ipynb
├── visualize_ph4_threshold_sensitivity.ipynb
├── vizualizations_heat_maps.ipynb
├── effect_size_thresholds_ph4_rdkit.csv
├── effect_size_thresholds_scaffolds.csv
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

The workflows expect a project-level `data/` directory. In practice, this directory is obtained by downloading and extracting the archived Zenodo data package, which is distributed as a large ZIP archive:

- `https://zenodo.org/records/20592102`

Downloading and extracting the archive requires sufficient local storage. After extraction, important subdirectories include:

```text
data/
├── bootstrap_threshold_analysis/
├── comparison_outputs/
├── fine_tuning_comparison/
├── generator_statistical_testing/
├── generators_without_finetuning/
├── information_about_clusters/
├── input_recall_sets/
├── nuclear_receptor/
├── output_sets/
├── protease/
├── results_pharm_based/
├── results_scaffold_based/
├── subset_analysis/
└── thresholds_ph4/
```

The file `data/enamine.smi` is used as the Enamine baseline/reference set. Precomputed RDKit pharmacophore fingerprint files are not included in the archive and may need to be regenerated locally with `src/compute_pharmacophore_fingerprints.py`.

## Stored Threshold Tables

The repository keeps the final effect-size threshold CSV files in the repo root:

- `effect_size_thresholds_scaffolds.csv`
- `effect_size_thresholds_ph4_rdkit.csv`

Intermediate bootstrap outputs are stored in:

- `data/bootstrap_threshold_analysis/`

## Figures

Repository-level overview figures:

- `img/project_workflow.png`
- `img/set_composition.png`
- `img/sim-dis-with-background.png`


