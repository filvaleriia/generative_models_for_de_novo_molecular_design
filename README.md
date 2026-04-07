# Generative Models for de novo Molecular Design

This repository contains the evaluation workflow for my dissertation project on molecular generators. The main goal is to evaluate generated molecules with two complementary metric families:

- scaffold-based metrics
- pharmacophore-based metrics

The repository is organized around a simple idea:

- all reusable code lives in `src/`
- analyses are executed from Jupyter notebooks
- users can choose how many CPU cores to use for metric calculations

## 📏 Metric Definitions

This repository evaluates molecular generators with three related metrics:

- `RS`
  Measures how well the generator recovers active motifs from the recall set.
- `SED`
  Measures how diverse the generated output set is.
- `ASER`
  Measures how strongly the generated set covers the target active region.

These three metrics are used in two different representations of molecular structure.

The metric formulas are the same in both workflows. What changes is the definition of the active motif:

- in scaffold-based analysis, the motif is a scaffold
- in pharmacophore-based analysis, the motif is a pharmacophore fingerprint match

### 🧮 Metric Formulas

#### unit Recovery Score (`RS`)

Unit Recovery Score quantifies how many unique active motifs from the Recall Set are recovered in the Output Set.

$$
RS = \frac{\text{Unique Active units in the OS}}{\text{Unique Active units in the RS}}
$$

where:

- `Unique Active units in the OS` means unique active units recovered in the Output Set
- `Unique Active units in the RS` means unique active units present in the Recall Set
- `units` means scaffolds in the scaffold-based workflow and pharmacophore fingerprints in the pharmacophore-based workflow

---

#### SEt unit Diversity (`SED`)

SEt unit Diversity reflects the diversity of the generated set.

$$
SED = \frac{\text{Unique units in the OS}}{c_{OS}}
$$

where:

- `Unique units in the OS` means the number of unique units in the Output Set
- `units` means scaffolds in the scaffold-based workflow and pharmacophore fingerprints in the pharmacophore-based workflow
- `cOS` is the total number of compounds in the Output Set

---

#### Absolute SEt unit Recall (`ASER`)

Absolute SEt unit Recall measures how many active motifs are present in the generated Output Set relative to the size of the Output Set.

$$
ASER = \frac{\text{Count of Active units in the OS}}{c_{OS}}
$$

where:

- `Count of Active units in the OS` means the total number of active unit occurrences in the Output Set
- `units` means scaffolds in the scaffold-based workflow and pharmacophore fingerprints in the pharmacophore-based workflow
- `cOS` is the total number of compounds in the Output Set

For `ASER`, the value can be greater than 1 because one molecule may contribute more than one active motif.

---

### 🧱 Scaffold-based metrics

In the scaffold-based workflow, the active motif is a scaffold derived from molecular structure.

The repository supports two scaffold definitions:

- `csk`
  CSK scaffold representation
- `murcko`
  Murcko scaffold representation from RDKit

For scaffold-based metrics, we compare scaffolds found in the generated Output Set against scaffolds present in the Recall Set.

The scheme below applies to both metric families. It uses the general term `units`, which means scaffolds in the scaffold-based workflow and pharmacophore fingerprints in the pharmacophore-based workflow.

![Set composition for both metric types](img/set_composition.png)

### 🧪 Pharmacophore-based metrics

In the pharmacophore-based workflow, the active motif is not a scaffold but a pharmacophore fingerprint match.

Here, generated molecules are compared to the Recall Set through pharmacophore fingerprints and Tanimoto similarity. A generated compound is considered a match if its pharmacophore fingerprint is similar enough to a recall fingerprint under a user-defined threshold.

In this repository, the current pharmacophore-based workflow uses:

- `rdkit`
  RDKit pharmacophore fingerprint definition

Recommended Tanimoto similarity thresholds are:

- `0.7` for `dis` split
- `0.8` for `sim` split

These thresholds reflect the fact that the dissimilarity split is structurally harder, while the similarity split is expected to contain closer analogs.

### ⚖️ What Is Compared in This Project

At a high level, the repository compares:

- different molecular generators
- different generator settings
- two receptors
- two split strategies: `dis` and `sim`
- two scaffold definitions: `csk` and `murcko`
- scaffold-based versus pharmacophore-based evaluation

For scaffold-based metrics, we evaluate how well generators recover relevant scaffolds.

For pharmacophore-based metrics, we evaluate how well generators recover relevant pharmacophore patterns, even in cases where the scaffold itself may differ.

![Similarity and dissimilarity split overview](img/sim-dis-with-background.png)

## 🗺️ Project Scope

The repository supports a full evaluation pipeline for generated molecular sets:

1. calculate scaffold-based metrics
2. calculate pharmacophore-based metrics
3. estimate effect-size thresholds with bootstrap analysis for both metric families
4. compare both metric families, including cross-modality miss analysis
5. visualize results with heatmaps and overlap analyses

The canonical notebook entry points are:

- `calculate_the_metrics_scaffold_based.ipynb`
- `calculate_the_metrics_ph4_based.ipynb`
- `bootstrap_threshold_analysis.ipynb`
- `comparison_of_metrics.ipynb`
- `vizualizations_heat_maps.ipynb`

Threshold tables currently stored in the repository:

- `effect_size_thresholds_scaffolds.csv`
- `effect_size_thresholds_ph4_rdkit.csv`

![Project workflow](img/project_workflow.png)

## 🗂️ Repository Structure

```text
generative_models_for_de_novo_molecular_design/
├── src/                                   # reusable Python code
├── examples/                              # CLI usage examples
├── calculate_the_metrics_scaffold_based.ipynb
├── calculate_the_metrics_ph4_based.ipynb
├── bootstrap_threshold_analysis.ipynb
├── comparison_of_metrics.ipynb
├── vizualizations_heat_maps.ipynb
├── effect_size_thresholds_scaffolds.csv
├── effect_size_thresholds_ph4_rdkit.csv
└── img/
```

## 🧩 Core Scripts in `src/`

The most important scripts are:

- `src/metrics_scaffold.py`
  Calculates scaffold-based recall metrics for CSK and Murcko scaffolds.
- `src/metrics_ph4.py`
  Calculates pharmacophore-based metrics from pharmacophore fingerprints.
- `src/metrics_connection.py`
  Merges scaffold metric outputs and prepares summary tables.
- `src/metrics_connection_phfp.py`
  Merges pharmacophore metric outputs and prepares summary tables.
- `src/compute_pharmacophore_fingerprints.py`
  Precomputes RDKit pharmacophore fingerprint CSV files required by the pharmacophore-based workflow.
- `src/bootstrap_threshold_analysis.py`
  Unified bootstrap workflow for scaffold-based and pharmacophore-based threshold estimation.
- `src/metric_comparison_analysis.py`
  Unified comparison workflow for scaffold vs pharmacophore analysis, overlap summaries, cross-modality misses, and stratified UMAP.
- `src/heatmap_visualization.py`
  Heatmap generation and figure export.
- `src/metrics_define_path.py`
  Calculates scaffold-based or pharmacophore-based metrics from user-defined file path patterns.
- `src/metrics_own_data.py`
  Calculates scaffold-based or pharmacophore-based metrics from one custom recall/output pair.

## 🛠️ Installation

Create the environment from the provided file:

```bash
conda env create -f environment.yml
conda activate scaffold-based-metrics-env
```

If needed, install the key packages manually:

- `rdkit`
- `pandas`
- `numpy`
- `jupyterlab`
- `scikit-learn`
- `seaborn`
- `matplotlib`
- `openpyxl`
- `umap-learn`

## 📁 Data Layout

The notebooks and scripts expect a `data/` directory in the repository root. In the current repository, the data tree contains receptor-level inputs, cluster metadata, generator outputs, metric results, comparison outputs, and bootstrap intermediates.

```text
data/
├── nuclear_receptor/
├── protease/
├── information_about_clusters/
├── input_recall_sets/
├── output_sets/
├── results_scaffold_based/
├── results_pharm_based/
├── comparison_outputs/
└── bootstrap_threshold_analysis/
```

The two receptors currently used in the repository are:

- `Glucocorticoid_receptor`
  nuclear receptor case study
- `Leukocyte_elastase`
  protease case study

Cluster metadata is stored in:

```text
data/
└── information_about_clusters/
   ├── Glucocorticoid_receptor/
   └── Leukocyte_elastase/
```

The raw receptor-level source sets are organized under:

```text
data/
├── nuclear_receptor/
├── protease/
├── input_recall_sets/{receptor}/
├── input_recall_sets/scaffold/{receptor}/
└── input_recall_sets/ph4/{receptor}/
```

Expected scaffold metric layout:

```text
data/
├── input_recall_sets/{receptor}/
├── output_sets/{receptor}/{generator}/
└── results_scaffold_based/{receptor}/{scaffold_type}_scaffolds/{split}/{generator}/
```

Expected pharmacophore metric layout:

```text
data/
├── output_sets/ph4/{receptor}/
│  ├── RS/
│  └── {generator}/
└── results_pharm_based/{receptor}/rdkit/{split}/{generator}/threshold_{value}/
```

Comparison and bootstrap outputs are stored in:

```text
data/
├── comparison_outputs/
│  ├── overlap/
│  ├── umap/
│  ├── cross_modality/
│  ├── figures/
│  └── metric_correlations/
└── bootstrap_threshold_analysis/
   ├── scaffold/
   └── ph4/
```

The `output_sets/` directory also contains historical subset runs such as `_10k`, `_62.5k`, `_100k`, `_125k`, `_250k`, `_300k`, and `_500k`. The main dissertation workflows described in this README use the canonical full-set generator folders unless a notebook or command explicitly selects a subset.

## ▶️ Running the Metric Calculations

The recommended way to run the project is through notebooks, with all heavy lifting delegated to `src/`.

### 1. 🧱 Scaffold-based metrics

Open `calculate_the_metrics_scaffold_based.ipynb` and call `src.metrics_scaffold`.

Python example:

```python
from src.metrics_scaffold import MetricsScaffold

mt = MetricsScaffold(
    type_cluster="dis",
    type_scaffold="csk",
    generator_name="Molpher",
    receptor="Glucocorticoid_receptor",
    data_folder="",
    ncpus=4,
)

result = mt.calculate()
```

CLI example:

```bash
python src/metrics_scaffold.py \
  --type_cluster dis \
  --type_scaffold csk \
  --generator Molpher \
  --receptor Glucocorticoid_receptor \
  --data_folder ../../ \
  --ncpus 4
```

### 2. 🧪 Pharmacophore-based metrics

Before calculating pharmacophore-based metrics, precompute pharmacophore fingerprints with `src.compute_pharmacophore_fingerprints`.

Precomputation example:

```bash
python src/compute_pharmacophore_fingerprints.py \
  --receptor Glucocorticoid_receptor \
  --split sim \
  --generator Molpher \
  --dataset all \
  --clusters 0 1 2 3 4 \
  --type_phfp rdkit \
  --data_folder ../../ \
  --ncpus 4
```

Then open `calculate_the_metrics_ph4_based.ipynb` and call `src.metrics_ph4`.

Python example:

```python
from src.metrics_ph4 import MetricsPh4

mt = MetricsPh4(
    type_cluster="dis",
    type_phfp="rdkit",
    generator_name="Molpher",
    receptor="Glucocorticoid_receptor",
    threshold=0.7,
    data_folder="",
    ncpus=4,
)

result = mt.calculate()
```

CLI example:

```bash
python src/metrics_ph4.py \
  --type_cluster dis \
  --type_phfp rdkit \
  --generator Molpher \
  --receptor Glucocorticoid_receptor \
  --threshold 0.7 \
  --data_folder ../../ \
  --ncpus 4
```

### 3. 🧭 Custom input workflows

Use `src.metrics_own_data.py` when you already have one recall/output pair and want to score it directly.

Scaffold-based custom example:

```bash
python src/metrics_own_data.py \
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
```

Use `src.metrics_define_path.py` when the same naming pattern is repeated across multiple clusters.

Pattern-based scaffold example:

```bash
python src/metrics_define_path.py \
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
```

The `examples/` directory contains ready-to-edit shell examples for:

- standard scaffold/ph4 metric calculation
- custom single-pair scoring
- custom pattern-based scoring
- merging per-generator mean tables

### 4. 🧾 Merging mean tables

Scaffold-based summary merge:

```bash
python src/metrics_connection.py \
  --type_cluster sim \
  --type_scaffold csk \
  --generator_list Molpher DrugEx_GT_epsilon_0.6 REINVENT DrugEx_GT_epsilon_0.1 \
  --receptor Glucocorticoid_receptor \
  --data_folder ../../
```

Pharmacophore-based summary merge:

```bash
python src/metrics_connection_phfp.py \
  --type_cluster sim \
  --type_phfp rdkit \
  --generator_list Molpher DrugEx_GT_epsilon_0.6 REINVENT DrugEx_GT_epsilon_0.1 \
  --receptor Glucocorticoid_receptor \
  --threshold_sim 0.8 \
  --threshold_dis 0.7 \
  --data_folder ../../
```

### 5. 🎯 Bootstrap threshold analysis

The unified bootstrap notebook is `bootstrap_threshold_analysis.ipynb`. The same workflow is also available from CLI.

Scaffold-based bootstrap example:

```bash
python src/bootstrap_threshold_analysis.py \
  --metric-family scaffold \
  --data-folder ../../ \
  --receptors Glucocorticoid_receptor,Leukocyte_elastase \
  --splits dis,sim \
  --n-subsample 250000 \
  --n-bootstrap 300
```

Pharmacophore-based bootstrap example:

```bash
python src/bootstrap_threshold_analysis.py \
  --metric-family ph4 \
  --data-folder ../../ \
  --receptors Glucocorticoid_receptor,Leukocyte_elastase \
  --splits dis,sim \
  --n-subsample 250000 \
  --n-bootstrap 300
```

## 🖥️ CPU Parallelization

Metric calculations support CPU parallelization. The user controls the number of worker processes with `ncpus`.

- in notebooks: set `ncpus=<number>`
- in CLI usage: pass `--ncpus <number>`

The scripts keep the requested value within the available CPU range of the machine, so the workflow is safer to run on both laptops and servers.

## ✅ Recommended Workflow

For a clean end-to-end run, use the repository in this order:

1. run scaffold-based metrics in `calculate_the_metrics_scaffold_based.ipynb`
2. run pharmacophore-based metrics in `calculate_the_metrics_ph4_based.ipynb`
3. estimate thresholds for both metric families with `bootstrap_threshold_analysis.ipynb`
4. compare both metric families in `comparison_of_metrics.ipynb`, including cross-modality miss analysis
5. generate final figures in `vizualizations_heat_maps.ipynb`

## 🖼️ Results

This section summarizes the final heatmap-based results for both metric families.

### 🧱 Scaffold-based metrics

The figure below shows the scaffold-based results across generators, receptors, scaffold definitions, and split strategies. The visualization uses effect-size-threshold binning to highlight performance relative to the best observed values within each metric.

![Scaffold-based heatmap results](img/heat_map/all_res_scaffold_based_with_thresholds.png)

### 🧪 Pharmacophore-based metrics

The figure below shows the pharmacophore-based results across generators, receptors, and split strategies using the RDKit pharmacophore fingerprint representation. As in the scaffold-based figure, the heatmap uses effect-size-threshold binning.

![Pharmacophore-based heatmap results](img/heat_map/all_res_pharm_based_with_thresholds.png)
