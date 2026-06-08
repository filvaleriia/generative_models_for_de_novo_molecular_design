from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METRICS = ["RS", "SED", "ASER"]
METRIC_RENAME = {"TUPOR": "RS", "SESY": "SED"}
RECEPTOR_DISPLAY_NAMES = {
    "Glucocorticoid_receptor": "Glucocorticoid receptor",
    "Leukocyte_elastase": "Leukocyte elastase",
}
SPLIT_DISPLAY_NAMES = {"dis": "DIS", "sim": "SIM"}
GENERATOR_ORDER = [
    "Molpher",
    "DrugEx_GT_epsilon_0.1",
    "DrugEx_GT_epsilon_0.6",
    "DrugEx_RNN_epsilon_0.1",
    "DrugEx_RNN_epsilon_0.6",
    "GB_GA_log_p_mut_r_0.01",
    "GB_GA_log_p_mut_r_0.5",
    "GB_GA_mut_r_0.01",
    "GB_GA_mut_r_0.5",
    "REINVENT",
    "enamine",
]


@dataclass(frozen=True)
class Ph4ThresholdPaths:
    data_folder: Path
    threshold_root: Path
    output_dir: Path


def build_default_paths(data_folder: str | Path = "../../") -> Ph4ThresholdPaths:
    data_folder_path = Path(data_folder).resolve()
    threshold_root = data_folder_path / "data" / "thresholds_ph4"
    output_dir = threshold_root / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    return Ph4ThresholdPaths(
        data_folder=data_folder_path,
        threshold_root=threshold_root,
        output_dir=output_dir,
    )


def _threshold_from_folder(name: str) -> float:
    raw = name.replace("threshold_", "")
    return float(raw)


def _clean_generator_name(name: str) -> str:
    return name.removesuffix("_250k")


def load_threshold_mean_tables(root: Path) -> pd.DataFrame:
    records = []
    mean_files = sorted(root.rglob("*mean*threshold*.csv"))
    print(f"Found {len(mean_files)} mean files")

    for path in mean_files:
        rel = path.relative_to(root)
        parts = rel.parts

        if len(parts) < 6:
            continue
        receptor, phfp, split, generator, threshold_folder = parts[:5]
        if path.name.startswith("df_all_clusters"):
            continue
        if not path.name.startswith(f"{generator}_mean_"):
            continue
        if phfp != "rdkit":
            continue
        if split not in {"dis", "sim"}:
            continue
        if not threshold_folder.startswith("threshold_"):
            continue

        try:
            threshold = _threshold_from_folder(threshold_folder)
        except ValueError:
            continue

        df = pd.read_csv(path).rename(columns=METRIC_RENAME)
        missing = [metric for metric in METRICS if metric not in df.columns]
        if missing:
            print(f"Skipping {path}: missing {missing}")
            continue

        row = df.iloc[0].copy()
        record = {
            "receptor": receptor,
            "phfp": phfp,
            "split": split,
            "generator": generator,
            "generator_clean": _clean_generator_name(generator),
            "threshold": threshold,
            "source_file": str(path),
        }
        for metric in METRICS:
            record[metric] = float(row[metric])
        records.append(record)

    out = pd.DataFrame(records)
    if not out.empty:
        out = out.sort_values(["receptor", "split", "generator", "threshold"]).reset_index(drop=True)
    return out


def summarize_threshold_table(threshold_df: pd.DataFrame) -> pd.DataFrame:
    return (
        threshold_df.groupby(["receptor", "split", "phfp"], as_index=False)
        .agg(
            n_generators=("generator", "nunique"),
            min_threshold=("threshold", "min"),
            max_threshold=("threshold", "max"),
            n_thresholds=("threshold", "nunique"),
            n_rows=("threshold", "size"),
        )
    )


def plot_threshold_curves(
    df: pd.DataFrame,
    receptor: str,
    split: str,
    output_dir: Path,
    phfp: str = "rdkit",
    generators: list[str] | None = None,
    save: bool = True,
):
    subset = df[(df["receptor"] == receptor) & (df["split"] == split) & (df["phfp"] == phfp)].copy()
    if generators is not None:
        subset = subset[subset["generator_clean"].isin(generators)]
    if subset.empty:
        raise ValueError(f"No data for receptor={receptor}, split={split}, phfp={phfp}")

    long_df = subset.melt(
        id_vars=["receptor", "split", "phfp", "generator", "generator_clean", "threshold"],
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    )

    fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharex=True)
    palette = sns.color_palette("tab20", n_colors=long_df["generator_clean"].nunique())

    for ax, metric in zip(axes, METRICS):
        metric_df = long_df[long_df["metric"] == metric]
        sns.lineplot(
            data=metric_df,
            x="threshold",
            y="value",
            hue="generator_clean",
            marker="o",
            linewidth=2,
            palette=palette,
            ax=ax,
        )
        ax.set_title(metric)
        ax.set_xlabel("Tanimoto similarity threshold")
        ax.set_ylabel(metric)
        ax.set_xlim(metric_df["threshold"].min(), metric_df["threshold"].max())
        ax.grid(True, alpha=0.3)
        if metric != METRICS[-1]:
            ax.get_legend().remove()

    handles, labels = axes[-1].get_legend_handles_labels()
    axes[-1].legend(
        handles,
        labels,
        title="Generator",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=10,
        title_fontsize=11,
    )

    fig.suptitle(f"PH4 threshold sensitivity: {receptor}, {split}, {phfp}", fontsize=18, fontweight="bold")
    plt.tight_layout()

    if save:
        safe_receptor = receptor.replace("/", "_")
        out_base = output_dir / f"ph4_threshold_sensitivity_{safe_receptor}_{split}_{phfp}"
        fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
        print("Saved:", out_base.with_suffix(".png"))
        print("Saved:", out_base.with_suffix(".svg"))

    return fig, axes


def plot_average_threshold_trend(
    df: pd.DataFrame,
    receptor: str,
    split: str,
    output_dir: Path,
    phfp: str = "rdkit",
    save: bool = True,
):
    subset = df[(df["receptor"] == receptor) & (df["split"] == split) & (df["phfp"] == phfp)].copy()
    if subset.empty:
        raise ValueError(f"No data for receptor={receptor}, split={split}, phfp={phfp}")

    long_df = subset.melt(
        id_vars=["receptor", "split", "phfp", "generator_clean", "threshold"],
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    for ax, metric in zip(axes, METRICS):
        metric_df = long_df[long_df["metric"] == metric]
        sns.lineplot(
            data=metric_df,
            x="threshold",
            y="value",
            estimator="mean",
            errorbar="sd",
            marker="o",
            linewidth=2.5,
            color="#2f6f9f",
            ax=ax,
        )
        ax.set_title(metric)
        ax.set_xlabel("Tanimoto similarity threshold")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Average PH4 threshold trend: {receptor}, {split}, {phfp}", fontsize=17, fontweight="bold")
    plt.tight_layout()

    if save:
        safe_receptor = receptor.replace("/", "_")
        out_base = output_dir / f"ph4_threshold_average_trend_{safe_receptor}_{split}_{phfp}"
        fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
        print("Saved:", out_base.with_suffix(".png"))
        print("Saved:", out_base.with_suffix(".svg"))

    return fig, axes


def make_threshold_long_table(df: pd.DataFrame, phfp: str = "rdkit") -> pd.DataFrame:
    subset = df[df["phfp"] == phfp].copy()
    long_df = subset.melt(
        id_vars=["receptor", "split", "phfp", "generator", "generator_clean", "threshold"],
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    )
    long_df["receptor_label"] = long_df["receptor"].map(RECEPTOR_DISPLAY_NAMES).fillna(long_df["receptor"])
    long_df["split_label"] = long_df["split"].map(SPLIT_DISPLAY_NAMES).fillna(long_df["split"].str.upper())
    long_df["panel_label"] = "receptor: " + long_df["receptor_label"] + ", split: " + long_df["split_label"]
    long_df["generator_clean"] = pd.Categorical(
        long_df["generator_clean"],
        categories=[g for g in GENERATOR_ORDER if g in set(long_df["generator_clean"])],
        ordered=True,
    )
    return long_df.sort_values(["receptor", "split", "metric", "generator_clean", "threshold"]).reset_index(drop=True)


def save_threshold_long_table(long_df: pd.DataFrame, output_dir: Path) -> Path:
    long_csv = output_dir / "ph4_threshold_sensitivity_long_table.csv"
    long_df.to_csv(long_csv, index=False)
    print("Saved:", long_csv)
    return long_csv


def plot_combined_threshold_sensitivity(
    long_df: pd.DataFrame,
    output_dir: Path,
    save: bool = True,
):
    row_order = [
        ("Glucocorticoid_receptor", "dis"),
        ("Glucocorticoid_receptor", "sim"),
        ("Leukocyte_elastase", "dis"),
        ("Leukocyte_elastase", "sim"),
    ]
    panel_labels = []
    for receptor, split in row_order:
        label = (
            f"receptor: {RECEPTOR_DISPLAY_NAMES.get(receptor, receptor)}, "
            f"split: {SPLIT_DISPLAY_NAMES.get(split, split.upper())}"
        )
        panel_labels.append(label)

    sns.set_context("talk")
    fig, axes = plt.subplots(nrows=len(row_order), ncols=len(METRICS), figsize=(32, 24), sharex=True)

    generator_values = [g for g in GENERATOR_ORDER if g in set(long_df["generator_clean"].dropna().astype(str))]
    palette = dict(zip(generator_values, sns.color_palette("tab20", n_colors=len(generator_values))))
    legend_rows = {0, 2}

    for row_idx, ((receptor, split), panel_label) in enumerate(zip(row_order, panel_labels)):
        for col_idx, metric in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            plot_df = long_df[
                (long_df["receptor"] == receptor)
                & (long_df["split"] == split)
                & (long_df["metric"] == metric)
            ].copy()
            if plot_df.empty:
                ax.set_axis_off()
                continue

            sns.lineplot(
                data=plot_df,
                x="threshold",
                y="value",
                hue="generator_clean",
                hue_order=generator_values,
                palette=palette,
                marker="o",
                linewidth=2.6,
                markersize=8,
                ax=ax,
                legend=(row_idx in legend_rows and col_idx == len(METRICS) - 1),
            )

            ax.set_title("")
            ax.set_xlabel("Tanimoto similarity threshold", fontsize=20, labelpad=8)
            ax.set_ylabel(f"metric value {metric}", fontsize=20)
            ax.tick_params(axis="both", labelsize=17)
            ax.tick_params(axis="x", labelbottom=True)
            ax.grid(True, alpha=0.25)
            ax.set_xlim(plot_df["threshold"].min(), plot_df["threshold"].max())

            legend = ax.get_legend()
            if legend is not None:
                if row_idx in legend_rows and col_idx == len(METRICS) - 1:
                    legend.set_title("Generator")
                    legend.set_bbox_to_anchor((1.03, 1.0))
                    legend._loc = 2
                    for text in legend.get_texts():
                        text.set_fontsize(17)
                    legend.get_title().set_fontsize(18)
                else:
                    legend.remove()

        axes[row_idx, 1].text(
            0.5,
            1.05,
            panel_label,
            transform=axes[row_idx, 1].transAxes,
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight="bold",
        )

    fig.subplots_adjust(left=0.06, right=0.85, top=0.96, bottom=0.06, hspace=0.36, wspace=0.22)

    if save:
        out_base = output_dir / "ph4_threshold_sensitivity_all_receptors_splits"
        fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
        print("Saved:", out_base.with_suffix(".png"))
        print("Saved:", out_base.with_suffix(".svg"))

    return fig, axes
