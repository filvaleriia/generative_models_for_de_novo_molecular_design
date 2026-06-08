from __future__ import annotations

from itertools import product
from pathlib import Path
import math
import re

import numpy as np
import pandas as pd


METRICS = ["RS", "SED", "ASER"]
METRIC_BASE_COLORS = {
    "RS": "#e97b32",
    "SED": "#97C2F0",
    "ASER": "#71ad48",
    "Combined": "#D1BDCF",
}

SCAFFOLD_GENERATORS = [
    "Molpher",
    "REINVENT",
    "DrugEx_GT_epsilon_0.1",
    "DrugEx_GT_epsilon_0.6",
    "DrugEx_RNN_epsilon_0.1",
    "DrugEx_RNN_epsilon_0.6",
    "GB_GA_mut_r_0.01",
    "GB_GA_mut_r_0.5",
    "GB_GA_log_p_mut_r_0.01",
    "GB_GA_log_p_mut_r_0.5",
    "addcarbon",
]

PH4_GENERATORS = [
    "Molpher",
    "REINVENT",
    "DrugEx_GT_epsilon_0.1",
    "DrugEx_GT_epsilon_0.6",
    "DrugEx_RNN_epsilon_0.1",
    "DrugEx_RNN_epsilon_0.6",
    "GB_GA_mut_r_0.01",
    "GB_GA_mut_r_0.5",
    "GB_GA_log_p_mut_r_0.01",
    "GB_GA_log_p_mut_r_0.5",
    "enamine",
]

PH4_THRESHOLD_BY_SPLIT = {"dis": "0.7", "sim": "0.8"}
DOMAIN_TITLES = {
    "scaffold": "Scaffold-based metrics",
    "ph4": "Pharmacophore-based metrics",
}


def _pretty_label(name: str) -> str:
    return (
        name.replace("addcarbon", "AddCarbon")
        .replace("enamine", "Enamine")
    )


def _read_single_row_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty metrics file: {path}")
    return df.iloc[[0]].copy()


def load_scaffold_cluster_table(root: Path) -> pd.DataFrame:
    rows = []
    pattern = re.compile(r"metrics_cluster_(\d+)_(dis|sim)_(.+)\.csv$")

    for csv_path in sorted(root.rglob("metrics_cluster_*.csv")):
        match = pattern.match(csv_path.name)
        if not match:
            continue

        cluster = int(match.group(1))
        split = match.group(2)
        generator = match.group(3)
        if generator not in SCAFFOLD_GENERATORS:
            continue

        scaffold_dir = csv_path.parents[2].name
        receptor = csv_path.parents[3].name
        scaffold = scaffold_dir.replace("_scaffolds", "")

        row = _read_single_row_csv(csv_path)
        row["domain"] = "scaffold"
        row["receptor"] = receptor
        row["representation"] = scaffold
        row["split"] = split
        row["generator"] = generator
        row["cluster"] = cluster
        rows.append(row)

    if not rows:
        raise RuntimeError("No scaffold cluster files found.")

    return pd.concat(rows, ignore_index=True)[
        ["domain", "receptor", "representation", "split", "generator", "cluster"] + METRICS
    ]


def load_ph4_cluster_table(root: Path) -> pd.DataFrame:
    rows = []
    pattern = re.compile(r"metrics_cluster_(\d+)_(dis|sim)_(.+)_threshold_(0\.7|0\.8)\.csv$")

    for csv_path in sorted(root.rglob("metrics_cluster_*.csv")):
        match = pattern.match(csv_path.name)
        if not match:
            continue

        cluster = int(match.group(1))
        split = match.group(2)
        generator = match.group(3)
        threshold = match.group(4)
        if PH4_THRESHOLD_BY_SPLIT.get(split) != threshold:
            continue
        if generator not in PH4_GENERATORS:
            continue

        receptor = csv_path.parents[4].name
        phfp = csv_path.parents[3].name

        row = _read_single_row_csv(csv_path)
        row["domain"] = "ph4"
        row["receptor"] = receptor
        row["representation"] = phfp
        row["split"] = split
        row["generator"] = generator
        row["cluster"] = cluster
        rows.append(row)

    if not rows:
        raise RuntimeError("No pharmacophore cluster files found.")

    return pd.concat(rows, ignore_index=True)[
        ["domain", "receptor", "representation", "split", "generator", "cluster"] + METRICS
    ]


def aggregate_repetitions_by_setting(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["domain", "receptor", "representation", "split", "generator"]
    aggregated = df.groupby(group_cols, as_index=False)[METRICS].mean()
    aggregated["setting_id"] = (
        aggregated["receptor"] + " | " + aggregated["representation"] + " | " + aggregated["split"]
    )
    return aggregated


def keep_repetitions_as_observations(df: pd.DataFrame) -> pd.DataFrame:
    repeated = df.copy()
    repeated["setting_id"] = (
        repeated["receptor"]
        + " | "
        + repeated["representation"]
        + " | "
        + repeated["split"]
        + " | rep "
        + repeated["cluster"].astype(str)
    )
    return repeated


def metric_scales_from_domain(df: pd.DataFrame) -> dict[str, float]:
    scales = df[METRICS].std(ddof=1).replace(0, 1.0)
    return scales.to_dict()


def build_pair_table(df: pd.DataFrame, generator_a: str, generator_b: str) -> pd.DataFrame:
    key_cols = ["receptor", "representation", "split", "setting_id"]
    if "cluster" in df.columns:
        key_cols.append("cluster")
    left = (
        df[df["generator"] == generator_a][key_cols + METRICS]
        .rename(columns={metric: f"{metric}_A" for metric in METRICS})
        .copy()
    )
    right = (
        df[df["generator"] == generator_b][key_cols + METRICS]
        .rename(columns={metric: f"{metric}_B" for metric in METRICS})
        .copy()
    )
    paired = left.merge(right, on=key_cols, how="inner")
    for metric in METRICS:
        paired[f"delta_{metric}"] = paired[f"{metric}_A"] - paired[f"{metric}_B"]
    return paired.sort_values(key_cols).reset_index(drop=True)


def _exact_or_random_signs(n: int, n_permutations: int, random_state: int, vector: bool = False) -> np.ndarray:
    if n <= 15:
        signs = np.array(list(product([-1.0, 1.0], repeat=n)), dtype=float)
    else:
        rng = np.random.default_rng(random_state)
        signs = rng.choice([-1.0, 1.0], size=(n_permutations, n))
    if vector:
        return signs[:, :, None]
    return signs


def sign_flip_test_1d(values: np.ndarray, n_permutations: int = 20000, random_state: int = 42) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return {"n": 0, "observed_stat": np.nan, "p_value": np.nan}

    observed = abs(values.mean())
    signs = _exact_or_random_signs(n, n_permutations, random_state, vector=False)
    permuted = np.abs((signs * values).mean(axis=1))
    p_value = (np.sum(permuted >= observed) + 1) / (len(permuted) + 1)
    return {"n": n, "observed_stat": observed, "p_value": p_value}


def sign_flip_test_multivariate(
    delta_matrix: np.ndarray,
    metric_scales: dict[str, float],
    n_permutations: int = 20000,
    random_state: int = 42,
) -> dict[str, float]:
    delta_matrix = np.asarray(delta_matrix, dtype=float)
    mask = np.all(np.isfinite(delta_matrix), axis=1)
    delta_matrix = delta_matrix[mask]
    n = delta_matrix.shape[0]
    if n == 0:
        return {"n": 0, "observed_stat": np.nan, "p_value": np.nan}

    scale_vector = np.array([metric_scales[metric] for metric in METRICS], dtype=float)
    scale_vector = np.where(scale_vector == 0, 1.0, scale_vector)
    standardized = delta_matrix / scale_vector

    observed_vector = standardized.mean(axis=0)
    observed_stat = float(np.linalg.norm(observed_vector, ord=2))

    signs = _exact_or_random_signs(n, n_permutations, random_state, vector=True)
    permuted_means = (signs * standardized[None, :, :]).mean(axis=1)
    permuted_stats = np.linalg.norm(permuted_means, axis=1)
    p_value = (np.sum(permuted_stats >= observed_stat) + 1) / (len(permuted_stats) + 1)

    return {
        "n": n,
        "observed_stat": observed_stat,
        "p_value": p_value,
        "mean_delta_z_RS": observed_vector[0],
        "mean_delta_z_SED": observed_vector[1],
        "mean_delta_z_ASER": observed_vector[2],
    }


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    valid = p_values.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=p_values.index)

    order = np.argsort(valid.to_numpy())
    sorted_p = valid.to_numpy()[order]
    n = len(sorted_p)
    adjusted = np.empty(n, dtype=float)

    running = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        candidate = sorted_p[i] * n / rank
        running = min(running, candidate)
        adjusted[i] = min(running, 1.0)

    restored = np.empty(n, dtype=float)
    restored[order] = adjusted
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    result.loc[valid.index] = restored
    return result


def compare_generators(
    df: pd.DataFrame,
    generator_a: str,
    generator_b: str,
    n_permutations: int = 20000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, float]]:
    paired = build_pair_table(df, generator_a, generator_b)
    result: dict[str, float] = {
        "generator_A": generator_a,
        "generator_B": generator_b,
        "n_pairs": len(paired),
    }

    for metric in METRICS:
        values = paired[f"delta_{metric}"].to_numpy(dtype=float)
        metric_test = sign_flip_test_1d(values, n_permutations=n_permutations, random_state=random_state)
        result[f"mean_delta_{metric}"] = float(np.mean(values))
        result[f"median_delta_{metric}"] = float(np.median(values))
        result[f"p_value_{metric}"] = metric_test["p_value"]

    delta_matrix = paired[[f"delta_{metric}" for metric in METRICS]].to_numpy(dtype=float)
    combined_test = sign_flip_test_multivariate(
        delta_matrix,
        metric_scales=metric_scales_from_domain(df),
        n_permutations=n_permutations,
        random_state=random_state,
    )
    result["combined_test_stat"] = combined_test["observed_stat"]
    result["combined_p_value"] = combined_test["p_value"]
    result["mean_delta_z_RS"] = combined_test["mean_delta_z_RS"]
    result["mean_delta_z_SED"] = combined_test["mean_delta_z_SED"]
    result["mean_delta_z_ASER"] = combined_test["mean_delta_z_ASER"]

    return paired, result


def run_all_pairwise_tests(
    df: pd.DataFrame,
    generators: list[str],
    domain_label: str,
    n_permutations: int = 20000,
    random_state: int = 42,
) -> pd.DataFrame:
    results = []
    for i, generator_a in enumerate(generators):
        for generator_b in generators[i + 1 :]:
            _, result = compare_generators(
                df,
                generator_a=generator_a,
                generator_b=generator_b,
                n_permutations=n_permutations,
                random_state=random_state,
            )
            result["domain"] = domain_label
            results.append(result)

    result_df = pd.DataFrame(results)
    for col in ["p_value_RS", "p_value_SED", "p_value_ASER", "combined_p_value"]:
        q_col = col.replace("p_value", "q_value")
        result_df[q_col] = benjamini_hochberg(result_df[col])

    result_df["combined_significant_fdr_0.05"] = result_df["combined_q_value"] < 0.05
    return result_df.sort_values(["domain", "combined_q_value", "generator_A", "generator_B"]).reset_index(drop=True)


def _matrix_from_results(
    result_df: pd.DataFrame,
    generators: list[str],
    value_col: str,
    diagonal_value,
) -> pd.DataFrame:
    if isinstance(diagonal_value, float) and math.isnan(diagonal_value):
        matrix = pd.DataFrame(np.nan, index=generators, columns=generators, dtype=float)
    else:
        matrix = pd.DataFrame(diagonal_value, index=generators, columns=generators, dtype=object)
    for _, row in result_df.iterrows():
        matrix.loc[row["generator_A"], row["generator_B"]] = row[value_col]
        matrix.loc[row["generator_B"], row["generator_A"]] = row[value_col]
    return matrix


def _cell_color_binary(value, diagonal=False) -> str:
    if diagonal:
        return "#f2f2f2"
    return "#4daf4a" if bool(value) else "#d9d9d9"


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _interpolate_color(start_hex: str, end_hex: str, t: float) -> str:
    start = _hex_to_rgb(start_hex)
    end = _hex_to_rgb(end_hex)
    t = max(0.0, min(1.0, float(t)))
    rgb = tuple(round(s + (e - s) * t) for s, e in zip(start, end))
    return _rgb_to_hex(rgb)


def _cell_color_from_qvalue(
    q_value,
    diagonal=False,
    q_cutoff: float = 0.05,
    base_hex: str = "#4daf4a",
) -> str:
    if diagonal:
        return "#f2f2f2"
    if pd.isna(q_value):
        return "#f2f2f2"
    if q_value >= q_cutoff:
        return "#d9d9d9"

    # Smaller q-values get a stronger green; values close to cutoff stay light.
    intensity = 1.0 - (float(q_value) / q_cutoff)
    return _interpolate_color("#d9d9d9", base_hex, intensity)


def _wrap_label(text: str) -> list[str]:
    return (
        _pretty_label(text)
        .replace("_epsilon", "\n epsilon")
        .replace("_mut_r", "\n mut_r")
        .split("\n")
    )


def resolve_repo_root(cwd: Path | None = None) -> Path:
    cwd = (cwd or Path.cwd()).resolve()
    if cwd.name == "generative_models_for_de_novo_molecular_design":
        return cwd

    candidate = cwd / "diseration_git" / "generative_models_for_de_novo_molecular_design"
    if candidate.exists():
        return candidate

    for parent in [cwd] + list(cwd.parents):
        candidate = parent / "diseration_git" / "generative_models_for_de_novo_molecular_design"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate the generative_models_for_de_novo_molecular_design repository root."
    )


def get_default_output_dir(repo_root: Path) -> Path:
    return repo_root.parent.parent / "data" / "generator_statistical_testing" / "all_pairs"


def load_q_matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def q_to_score(q_df: pd.DataFrame, cutoff: float = 0.05) -> pd.DataFrame:
    return (1 - (q_df / cutoff)).clip(lower=0, upper=1)


def q_to_annot(q_df: pd.DataFrame) -> pd.DataFrame:
    annot = q_df.copy().astype(object)
    for i in range(q_df.shape[0]):
        for j in range(q_df.shape[1]):
            val = q_df.iat[i, j]
            if pd.isna(val) or i == j:
                annot.iat[i, j] = ""
            elif val < 0.001:
                annot.iat[i, j] = "<0.001"
            else:
                annot.iat[i, j] = f"{val:.3f}"
    return annot


def _make_cmap(base_hex: str):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("custom_qmap", ["#d9d9d9", base_hex])


def load_domain_q_matrices(base_dir: Path, domain_key: str) -> dict[str, pd.DataFrame]:
    domain_prefix = {
        "scaffold": "scaffold_repetition",
        "ph4": "ph4_repetition",
    }[domain_key]
    file_map = {
        "RS": base_dir / f"{domain_prefix}_rs_qvalue_matrix.csv",
        "SED": base_dir / f"{domain_prefix}_sed_qvalue_matrix.csv",
        "ASER": base_dir / f"{domain_prefix}_aser_qvalue_matrix.csv",
        "Combined": base_dir / f"{domain_prefix}_combined_qvalue_matrix.csv",
    }
    return {metric: load_q_matrix(path) for metric, path in file_map.items()}


def plot_metric_panels_matplotlib(
    domain_key: str,
    base_dir: Path,
    figsize: tuple[int, int] = (24, 24),
    save_path: Path | None = None,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    q_mats = load_domain_q_matrices(base_dir, domain_key)
    metric_grid = [
        ("RS", (0, 0)),
        ("SED", (0, 1)),
        ("ASER", (1, 0)),
        ("Combined", (1, 1)),
    ]

    sns.set_style("white")
    plt.rcParams["font.family"] = "DejaVu Sans"

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    plt.subplots_adjust(left=0.14, right=0.98, top=0.90, bottom=0.14, wspace=0.12, hspace=0.34)

    generator_order = list(q_mats["RS"].index)

    for metric, (r, c) in metric_grid:
        ax = axes[r, c]
        q_df = q_mats[metric].loc[generator_order, generator_order]
        score_df = q_to_score(q_df)
        annot_df = q_to_annot(q_df)

        sns.heatmap(
            score_df,
            ax=ax,
            cmap=_make_cmap(METRIC_BASE_COLORS[metric]),
            vmin=0,
            vmax=1,
            cbar=False,
            square=False,
            annot=annot_df,
            fmt="",
            linewidths=0.5,
            linecolor="white",
            annot_kws={"size": 16, "color": "black"},
        )

        ax.set_title(metric, fontsize=26, fontweight="bold", pad=14)
        ax.set_xticklabels(
            [_pretty_label(x).replace("_epsilon", "\n epsilon").replace("_mut_r", "\n mut_r") for x in q_df.columns],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=20,
        )
        ax.set_yticklabels(
            [_pretty_label(x).replace("_epsilon", "\n epsilon").replace("_mut_r", "\n mut_r") for x in q_df.index],
            rotation=0,
            ha="right",
            fontsize=20,
        )
        ax.tick_params(axis="x", length=0, pad=2)
        ax.tick_params(axis="y", length=0, pad=2)
        for spine in ax.spines.values():
            spine.set_visible(False)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes


def render_dissertation_metric_heatmaps(output_dir: Path) -> None:
    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns

    base = output_dir

    files = {
        'scaffold': {
            'RS': base / 'scaffold_repetition_rs_qvalue_matrix.csv',
            'SED': base / 'scaffold_repetition_sed_qvalue_matrix.csv',
            'ASER': base / 'scaffold_repetition_aser_qvalue_matrix.csv',
            'Combined': base / 'scaffold_repetition_combined_qvalue_matrix.csv',
        },
        'ph4': {
            'RS': base / 'ph4_repetition_rs_qvalue_matrix.csv',
            'SED': base / 'ph4_repetition_sed_qvalue_matrix.csv',
            'ASER': base / 'ph4_repetition_aser_qvalue_matrix.csv',
            'Combined': base / 'ph4_repetition_combined_qvalue_matrix.csv',
        },
    }

    metric_base_colors = {
        'RS': '#e97b32',
        'SED': '#97C2F0',
        'ASER': '#71ad48',
        'Combined': '#D1BDCF',
    }

    def make_cmap(base_hex: str):
        return LinearSegmentedColormap.from_list('custom_qmap', ['#d9d9d9', base_hex])

    def format_generator_label(label: str) -> str:
        return (
            label.replace('addcarbon', 'AddCarbon')
                 .replace('enamine', 'Enamine')
                 .replace('_epsilon', '\n epsilon')
                 .replace('_mut_r', '\n mut_r')
        )

    def load_q_matrix(path: Path) -> pd.DataFrame:
        return pd.read_csv(path, index_col=0)

    def q_to_score(q_df: pd.DataFrame, cutoff: float = 0.05) -> pd.DataFrame:
        return (1 - (q_df / cutoff)).clip(lower=0, upper=1)

    def q_to_annot(q_df: pd.DataFrame) -> pd.DataFrame:
        annot = q_df.copy().astype(object)
        for i in range(q_df.shape[0]):
            for j in range(q_df.shape[1]):
                val = q_df.iat[i, j]
                if pd.isna(val) or i == j:
                    annot.iat[i, j] = ''
                elif val < 0.001:
                    annot.iat[i, j] = '<0.001'
                else:
                    annot.iat[i, j] = f'{val:.3f}'
        return annot

    q_mats = {
        domain: {metric: load_q_matrix(path) for metric, path in metric_dict.items()}
        for domain, metric_dict in files.items()
    }

    sns.set_style('white')
    plt.rcParams['font.family'] = 'DejaVu Sans'

    domain_titles = {
        'scaffold': 'Scaffold-based metrics',
        'ph4': 'Pharmacophore-based metrics',
    }

    domain_save_names = {
        'scaffold': 'scaffold_cluster_metric_panel_matplotlib',
        'ph4': 'ph4_cluster_metric_panel_matplotlib',
    }

    metric_grid = [
        ('RS', (0, 0)),
        ('SED', (0, 1)),
        ('ASER', (1, 0)),
        ('Combined', (1, 1)),
    ]

    for domain_key in ['scaffold', 'ph4']:
        fig, axes = plt.subplots(2, 2, figsize=(24, 24))
        plt.subplots_adjust(left=0.14, right=0.98, top=0.90, bottom=0.14, wspace=0.25, hspace=0.34)

        generator_order = list(q_mats[domain_key]['RS'].index)

        for metric, (r, c) in metric_grid:
            ax = axes[r, c]
            q_df = q_mats[domain_key][metric].loc[generator_order, generator_order]
            score_df = q_to_score(q_df)
            annot_df = q_to_annot(q_df)

            sns.heatmap(
                score_df,
                ax=ax,
                cmap=make_cmap(metric_base_colors[metric]),
                vmin=0,
                vmax=1,
                cbar=False,
                square=False,
                annot=annot_df,
                fmt='',
                linewidths=0.5,
                linecolor='white',
                annot_kws={'size': 16, 'color': 'black'},
            )

            ax.set_title(metric, fontsize=26, fontweight='bold', pad=14)
            ax.set_xticklabels(
                [format_generator_label(x) for x in q_df.columns],
                rotation=45,
                ha='right',
                rotation_mode='anchor',
                fontsize=20,
            )
            ax.set_yticklabels(
                [format_generator_label(x) for x in q_df.index],
                rotation=0,
                ha='right',
                fontsize=20,
            )
            ax.tick_params(axis='x', length=0, pad=2)
            ax.tick_params(axis='y', length=0, pad=2)
            ax.set_xlabel('')
            ax.set_ylabel('')

        fig.suptitle(domain_titles[domain_key], fontsize=30, fontweight='bold', y=0.96, x = 0.55)
        fig.savefig(output_dir / f"{domain_save_names[domain_key]}.png", dpi=300, bbox_inches='tight')
        fig.savefig(output_dir / f"{domain_save_names[domain_key]}.svg", bbox_inches='tight')
        plt.show()


def save_significance_heatmap_svg(
    matrix: pd.DataFrame,
    q_matrix: pd.DataFrame,
    output_path: Path,
    title: str,
    base_hex: str = "#5f9ea0",
) -> None:
    generators = list(matrix.index)
    n = len(generators)
    cell = 72
    left_margin = 210
    top_margin = 190
    right_margin = 40
    bottom_margin = 40
    width = left_margin + n * cell + right_margin
    height = top_margin + n * cell + bottom_margin

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        'text { font-family: "DejaVu Sans", sans-serif; fill: #222; }',
        '.title { font-size: 22px; font-weight: 700; }',
        '.label { font-size: 14px; }',
        '.celltext { font-size: 13px; text-anchor: middle; dominant-baseline: middle; }',
        '.legend { font-size: 14px; }',
        "</style>",
        f'<text class="title" x="{left_margin}" y="32">{title}</text>',
        f'<text class="legend" x="{left_margin}" y="56">Green = significant after FDR correction (q &lt; 0.05), gray = not significant</text>',
    ]

    for idx, generator in enumerate(generators):
        x = left_margin + idx * cell + cell / 2
        y = top_margin + idx * cell + cell / 2
        lines = _wrap_label(generator)
        for line_idx, line in enumerate(lines):
            dy = (line_idx - (len(lines) - 1) / 2) * 13
            parts.append(f'<text class="label" x="{x}" y="{top_margin - 22 + dy}" text-anchor="middle">{line}</text>')
            parts.append(f'<text class="label" x="{left_margin - 12}" y="{y + dy}" text-anchor="end">{line}</text>')

    for row_idx, row_name in enumerate(generators):
        for col_idx, col_name in enumerate(generators):
            x = left_margin + col_idx * cell
            y = top_margin + row_idx * cell
            diagonal = row_idx == col_idx
            q_value = q_matrix.loc[row_name, col_name]
            color = _cell_color_from_qvalue(q_value, diagonal=diagonal, base_hex=base_hex)
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>'
            )
            if diagonal:
                parts.append(f'<text class="celltext" x="{x + cell/2}" y="{y + cell/2}">—</text>')
            else:
                is_sig = bool(matrix.loc[row_name, col_name])
                label = "sig" if is_sig else "ns"
                q_text = "q<0.001" if q_value < 0.001 else f"q={q_value:.3f}"
                parts.append(f'<text class="celltext" x="{x + cell/2}" y="{y + cell/2 - 8}">{label}</text>')
                parts.append(f'<text class="celltext" x="{x + cell/2}" y="{y + cell/2 + 8}">{q_text}</text>')

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def save_metric_panel_heatmap_svg(
    generators: list[str],
    matrices: dict[str, pd.DataFrame],
    q_matrices: dict[str, pd.DataFrame],
    output_path: Path,
    title: str,
) -> None:
    panel_titles = ["RS", "SED", "ASER", "Combined"]
    n = len(generators)
    cell = 38
    panel_gap_x = 6
    panel_gap_y = 10
    panel_left = 110
    panel_top = 44
    panel_bottom = 96
    panel_width = panel_left + n * cell + 20
    panel_height = panel_top + n * cell + panel_bottom
    width = 20 + panel_width * 2 + panel_gap_x + 20
    height = 20 + panel_height * 2 + panel_gap_y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        'text { font-family: "DejaVu Sans", sans-serif; fill: #222; }',
        ".panel { font-size: 21px; font-weight: 700; text-anchor: middle; }",
        ".label { font-size: 13px; }",
        ".celltext { font-size: 10px; text-anchor: middle; dominant-baseline: middle; }",
        "</style>",
    ]

    for panel_idx, panel_name in enumerate(panel_titles):
        matrix = matrices[panel_name]
        q_matrix = q_matrices[panel_name]
        base_hex = METRIC_BASE_COLORS.get(panel_name, "#5f9ea0")
        grid_col = panel_idx % 2
        grid_row = panel_idx // 2
        x_offset = 20 + grid_col * (panel_width + panel_gap_x)
        y_offset = 20 + grid_row * (panel_height + panel_gap_y)

        panel_center_x = x_offset + panel_left + (n * cell) / 2
        parts.append(f'<text class="panel" x="{panel_center_x}" y="{y_offset + 20}">{panel_name}</text>')

        for idx, generator in enumerate(generators):
            x = x_offset + panel_left + idx * cell + cell / 2
            y = y_offset + panel_top + idx * cell + cell / 2
            lines = _wrap_label(generator)
            for line_idx, line in enumerate(lines):
                dy = (line_idx - (len(lines) - 1) / 2) * 12
                parts.append(f'<text class="label" x="{x_offset + panel_left - 10}" y="{y + dy}" text-anchor="end">{line}</text>')
            bottom_y = y_offset + panel_top + n * cell + 12
            for line_idx, line in enumerate(lines):
                line_x = x + (line_idx - (len(lines) - 1) / 2) * 10
                parts.append(
                    f'<text class="label" x="{line_x}" y="{bottom_y}" text-anchor="end" transform="rotate(-90 {line_x} {bottom_y})">{line}</text>'
                )

        for row_idx, row_name in enumerate(generators):
            for col_idx, col_name in enumerate(generators):
                x = x_offset + panel_left + col_idx * cell
                y = y_offset + panel_top + row_idx * cell
                diagonal = row_idx == col_idx
                q_value = q_matrix.loc[row_name, col_name]
                color = _cell_color_from_qvalue(q_value, diagonal=diagonal, base_hex=base_hex)
                parts.append(
                    f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="#ffffff" stroke-width="1.3"/>'
                )
                if diagonal:
                    continue
                if pd.notnull(q_value):
                    q_text = "&lt;0.001" if q_value < 0.001 else f"{q_value:.3f}"
                    parts.append(f'<text class="celltext" x="{x + cell/2}" y="{y + cell/2}">{q_text}</text>')

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def generate_all_pairwise_outputs(
    repo_root: Path,
    output_dir: Path,
    n_permutations: int = 20000,
    random_state: int = 42,
    write_svg_heatmaps: bool = False,
) -> dict[str, pd.DataFrame]:
    scaffold_root = repo_root.parent.parent / "data" / "results_scaffold_based"
    ph4_root = repo_root.parent.parent / "data" / "results_pharm_based"

    output_dir.mkdir(parents=True, exist_ok=True)

    scaffold_cluster_df = load_scaffold_cluster_table(scaffold_root)
    scaffold_df = aggregate_repetitions_by_setting(scaffold_cluster_df)
    scaffold_repetition_df = keep_repetitions_as_observations(scaffold_cluster_df)
    ph4_cluster_df = load_ph4_cluster_table(ph4_root)
    ph4_df = aggregate_repetitions_by_setting(ph4_cluster_df)
    ph4_repetition_df = keep_repetitions_as_observations(ph4_cluster_df)

    scaffold_results = run_all_pairwise_tests(
        scaffold_df,
        generators=SCAFFOLD_GENERATORS,
        domain_label="scaffold",
        n_permutations=n_permutations,
        random_state=random_state,
    )
    scaffold_repetition_results = run_all_pairwise_tests(
        scaffold_repetition_df,
        generators=SCAFFOLD_GENERATORS,
        domain_label="scaffold_repetition",
        n_permutations=n_permutations,
        random_state=random_state,
    )
    ph4_results = run_all_pairwise_tests(
        ph4_df,
        generators=PH4_GENERATORS,
        domain_label="ph4",
        n_permutations=n_permutations,
        random_state=random_state,
    )
    ph4_repetition_results = run_all_pairwise_tests(
        ph4_repetition_df,
        generators=PH4_GENERATORS,
        domain_label="ph4_repetition",
        n_permutations=n_permutations,
        random_state=random_state,
    )

    scaffold_results.to_csv(output_dir / "scaffold_pairwise_significance.csv", index=False)
    scaffold_repetition_results.to_csv(output_dir / "scaffold_repetition_pairwise_significance.csv", index=False)
    ph4_results.to_csv(output_dir / "ph4_pairwise_significance.csv", index=False)
    ph4_repetition_results.to_csv(output_dir / "ph4_repetition_pairwise_significance.csv", index=False)

    for domain, results, generators in [
        ("scaffold", scaffold_results, SCAFFOLD_GENERATORS),
        ("scaffold_repetition", scaffold_repetition_results, SCAFFOLD_GENERATORS),
        ("ph4", ph4_results, PH4_GENERATORS),
        ("ph4_repetition", ph4_repetition_results, PH4_GENERATORS),
    ]:
        metric_panels = {
            "RS": (
                _matrix_from_results(results.assign(significant=results["q_value_RS"] < 0.05), generators, "significant", diagonal_value=None),
                _matrix_from_results(results, generators, "q_value_RS", diagonal_value=np.nan),
            ),
            "SED": (
                _matrix_from_results(results.assign(significant=results["q_value_SED"] < 0.05), generators, "significant", diagonal_value=None),
                _matrix_from_results(results, generators, "q_value_SED", diagonal_value=np.nan),
            ),
            "ASER": (
                _matrix_from_results(results.assign(significant=results["q_value_ASER"] < 0.05), generators, "significant", diagonal_value=None),
                _matrix_from_results(results, generators, "q_value_ASER", diagonal_value=np.nan),
            ),
            "Combined": (
                _matrix_from_results(results, generators, "combined_significant_fdr_0.05", diagonal_value=None),
                _matrix_from_results(results, generators, "combined_q_value", diagonal_value=np.nan),
            ),
        }

        sig_matrix = metric_panels["Combined"][0]
        q_matrix = metric_panels["Combined"][1]
        sig_matrix.to_csv(output_dir / f"{domain}_combined_significance_matrix.csv")
        q_matrix.to_csv(output_dir / f"{domain}_combined_qvalue_matrix.csv")
        for metric_name, (metric_sig_matrix, metric_q_matrix) in metric_panels.items():
            metric_sig_matrix.to_csv(output_dir / f"{domain}_{metric_name.lower()}_significance_matrix.csv")
            metric_q_matrix.to_csv(output_dir / f"{domain}_{metric_name.lower()}_qvalue_matrix.csv")
        if write_svg_heatmaps:
            save_significance_heatmap_svg(
                sig_matrix,
                q_matrix,
                output_dir / f"{domain}_combined_significance_heatmap.svg",
                title=f"{domain.capitalize()} pairwise generator differences",
            )
            save_metric_panel_heatmap_svg(
                generators=generators,
                matrices={name: pair[0] for name, pair in metric_panels.items()},
                q_matrices={name: pair[1] for name, pair in metric_panels.items()},
                output_path=output_dir / f"{domain}_metric_panel_significance_heatmap.svg",
                title=f"{domain.capitalize()} pairwise generator significance by metric",
            )

    return {
        "scaffold": scaffold_results,
        "scaffold_repetition": scaffold_repetition_results,
        "ph4": ph4_results,
        "ph4_repetition": ph4_repetition_results,
    }


def run_significance_workflow(
    repo_root: Path | None = None,
    output_dir: Path | None = None,
    n_permutations: int = 20000,
    random_state: int = 42,
    write_svg_heatmaps: bool = False,
) -> tuple[dict[str, pd.DataFrame], Path]:
    repo_root = resolve_repo_root(repo_root)
    output_dir = output_dir or get_default_output_dir(repo_root)
    results = generate_all_pairwise_outputs(
        repo_root=repo_root,
        output_dir=output_dir,
        n_permutations=n_permutations,
        random_state=random_state,
        write_svg_heatmaps=write_svg_heatmaps,
    )
    return results, output_dir
