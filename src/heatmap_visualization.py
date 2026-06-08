import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import os
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from src.path_utils import pharm_results_dir, scaffold_results_dir


def resolve_plot_output_dir(save_folder: str = '', receptor: str = '') -> Path:
    output_dir = Path(save_folder) if save_folder else Path("img/heat_map") / receptor if receptor else Path("img/heat_map")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_current_figure(
    name: str,
    save_folder: str = '',
    receptor: str = '',
    bbox_inches: str | None = None,
    include_pdf: bool = False,
) -> None:
    output_dir = resolve_plot_output_dir(save_folder, receptor)
    common_kwargs = {}
    if bbox_inches is not None:
        common_kwargs["bbox_inches"] = bbox_inches

    plt.savefig(output_dir / f"{name}.svg", format="svg", **common_kwargs)
    plt.savefig(output_dir / f"{name}.png", format="png", dpi=300, **common_kwargs)
    if include_pdf:
        plt.savefig(output_dir / f"{name}.pdf", **common_kwargs)
        plt.savefig(output_dir / f"{name}.tiff", dpi=150, **common_kwargs)

def preprocesing(type_cluster, type_unit, generators_name_list, receptor, data_folder, ph4 = False, user_threshold = 1):
    '''
    Function for connection all data set with normalization
    '''
    # Define path to data

    if ph4:
        link = pharm_results_dir(data_folder, receptor, type_unit, type_cluster)
        link_mean = [
            link / generator / f"threshold_{user_threshold}" / f"{generator}_mean_{type_unit}_{type_cluster}_threshold_{user_threshold}.csv"
            for generator in generators_name_list
        ]
    else:
        link = scaffold_results_dir(data_folder, receptor, f"{type_unit}_scaffolds", type_cluster)
        link_mean = [
            link / generator / f"{generator}_mean_{type_unit}_{type_cluster}.csv"
            for generator in generators_name_list
        ]
    
    # Load data
    df_list = [pd.read_csv(f) for f in link_mean]
    df = pd.concat(df_list, axis=0, ignore_index=True)

    scaler = MinMaxScaler()
    numeric_columns = df.select_dtypes(include=['number']).columns  # Select only numeric columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])  # Apply normalization

    if 'name' not in df.columns:
        df['name'] = df['generator']
    else:
        df["name"] = df["name"].str.replace("_mean", "", regex=False)

    
    return df



def preprocesing_org(type_cluster, type_unit, generators_name_list, receptor, data_folder, ph4 = False, user_threshold = 1):
    '''
    Function for connection all data set with normalization
    '''
    # Define path to data

    if ph4:
        link = pharm_results_dir(data_folder, receptor, type_unit, type_cluster)
        link_mean = [
            link / generator / f"threshold_{user_threshold}" / f"{generator}_mean_{type_unit}_{type_cluster}_threshold_{user_threshold}.csv"
            for generator in generators_name_list
        ]
    else:
        link = scaffold_results_dir(data_folder, receptor, f"{type_unit}_scaffolds", type_cluster)
        link_mean = [
            link / generator / f"{generator}_mean_{type_unit}_{type_cluster}.csv"
            for generator in generators_name_list
        ]
    
    # Load data
    df_list = [pd.read_csv(f) for f in link_mean]
    df = pd.concat(df_list, axis=0, ignore_index=True)

    if 'name' not in df.columns:
        df['name'] = df['generator']
    else:
        df["name"] = df["name"].str.replace("_mean", "", regex=False)

    
    return df



def plot_heatmap(
    type_cluster,
    type_unit,
    generators_name_list,
    receptor,
    title='',
    name_save='',
    cmap='viridis',
    annotate=True,
    using_norm_values=True,
    data_folder='',
    save_folder='',
    ph4=False,
    user_threshold=1,
):
    ''' 
    Plots a single heatmap for the given data split.
    
    Args:
    - data (pd.DataFrame): The input data containing the columns to be visualized.
    - title (str): The title of the heatmap (default is empty).
    - cmap (str): The color map to be used for the heatmap (default is 'viridis').
    - annotate (bool): Whether to annotate the cells with their values (default is True).
    - using_norm_values: whether to use normalized values
    - ph4: if True, read pharmacophore-based results instead of scaffold-based results
    - user_threshold: Tanimoto threshold used for pharmacophore-based results
    '''

    # Extract relevant columns (RS, SED, ASER) for visualization
    if using_norm_values:
        data = preprocesing(
            type_cluster,
            type_unit,
            generators_name_list,
            receptor,
            data_folder,
            ph4=ph4,
            user_threshold=user_threshold,
        )
    else:
        data = preprocesing_org(
            type_cluster,
            type_unit,
            generators_name_list,
            receptor,
            data_folder,
            ph4=ph4,
            user_threshold=user_threshold,
        )

    df = data[['RS', 'SED', 'ASER']]
    # Set the index of the dataframe to the 'name' attribute of the data
    df.index = data.name.tolist()

    # Create a figure for the heatmap with a specific size
    plt.figure(figsize=(10, 0.7 * len(generators_name_list)))
    
    # Plot the heatmap using seaborn with optional annotations and custom color map
    sns.heatmap(df, annot=annotate, cmap=cmap,   annot_kws={"size": 17})
    
    # Set the title for the heatmap
    if not title:
        if ph4:
            title = f"{receptor.replace('_', ' ')} | {type_unit.upper()} | {type_cluster} | threshold {user_threshold}"
        else:
            title = f"{receptor.replace('_', ' ')} | {type_unit.upper()} | {type_cluster}"
    plt.title(title, fontsize=17 , pad=10)

    new_labels = [label.replace('_epsilon', '\n epsilon').replace('_mut_r', '\n mut_r').replace('addcarbon', 'AddCarbon') for label in df.index]
    #new_labels = [label.replace('_62.5k', '').replace('_125k', '').replace('_250k', '').replace('_500k', '') for label in new_labels]
    plt.xticks(fontsize=17)
    plt.yticks(ticks=np.arange(len(df.index)) + 0.5,labels=new_labels,fontsize=17)
    plt.tight_layout()
    # Save the plot as an SVG file
    save_current_figure(name_save, save_folder=save_folder, receptor=receptor)
    # Display the heatmap
    plt.show()



def plot_all_subsets(subset_dict, title='', receptor='', name_save='', cmap='viridis', annotate=True, numering='', save_folder=''):
    '''
    Plots heatmaps for multiple subsets in a single figure in paper style,
    with proper spacing and colorbar handling as in plot_heatmaps_with_diff_from_baseline.
    '''

    num_subsets = len(subset_dict)
    num_gen = len(subset_dict[next(iter(subset_dict))])

    # Dynamically determine columns and rows
    max_cols = 5
    cols = min(num_subsets, max_cols)
    rows = math.ceil(num_subsets / cols)

    # Adjust layout to avoid sparse last row
    if num_subsets > 1:
        cols_last_row = num_subsets % cols
        if cols_last_row != 0 and cols_last_row < math.ceil(cols / 2):
            cols = math.ceil(num_subsets / 2)
            rows = math.ceil(num_subsets / cols)

    # Figure size in paper style
    fig_width = cols * 12
    fig_height = rows * num_gen * 1.3
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(rows, cols, figure=fig)

    axes = []
    for i in range(num_subsets):
        row = i // cols
        col = i % cols
        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)

    # Plot each heatmap
    for ax, (subset_name, data) in zip(axes, subset_dict.items()):
        df = data[['RS', 'SED', 'ASER']]
        df.index = data.name.tolist()

        hm = sns.heatmap(
            df,
            annot=annotate,
            cmap=cmap,
            ax=ax,
            annot_kws={"size": 30},  # paper style
            cbar=False
        )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes(
            "right",
            size="2.0%",
            pad=0.15
        )
        cbar = plt.colorbar(hm.collections[0], cax=cax)
        cbar.ax.tick_params(labelsize=15)
        cbar.outline.set_visible(False)
        ax.figure.axes[-1].yaxis.label.set_size(15)

        # Format subset name
        if subset_name == '':
            subset_name_disp = 'Full OS'
        elif subset_name == '_62.5k':
            subset_name_disp = '62,500'
        else:
            subset_name_disp = subset_name.replace('_', '').replace('k', ',000')

        ax.set_title(f"{subset_name_disp}" if subset_name_disp == 'Full OS' else f"{subset_name_disp} subset",
                     fontsize=35, wrap=True, pad=12)

        # Format y-axis labels
        new_labels = [
            label.get_text()
            .replace('_epsilon', '\n epsilon')
            .replace('_mut_r', '\n mut_r')
            .replace('addcarbon', 'AddCarbon')
            .replace('enamine', 'Enamine')
            .replace('_mean', '')
            for label in ax.get_yticklabels()
        ]
        new_labels = [
            label.replace('_62.5k', '').replace('_125k', '')
                 .replace('_250k', '').replace('_500k', '')
                 .replace('_10k', '')
            for label in new_labels
        ]
        ax.set_yticklabels(new_labels, rotation=0, ha="right", fontsize=30)
        ax.set_xticklabels(ax.get_xticklabels(), ha="center", fontsize=30)
        ax.tick_params(axis='y', pad=12)
        ax.set_facecolor('white')

    # Hide unused subplot slots
    total_slots = rows * cols
    for j in range(len(axes), total_slots):
        row = j // cols
        col = j % cols
        ax_empty = fig.add_subplot(gs[row, col])
        ax_empty.axis('off')

    # Posun druhého řádku pro větší vertikální mezeru
    if rows == 2:
        for i, ax in enumerate(axes):
            if i >= cols:  # druhý řádek
                pos = ax.get_position()
                ax.set_position([pos.x0, pos.y0 + 0.03, pos.width + 0.01, pos.height])


    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Numbering and global title
    fig.text(0.005, 0.95, numering, ha='left', va='top', fontsize=40)
    if title:
        fig.suptitle(title, fontsize=40)

    # Save
    save_current_figure(name_save, save_folder=save_folder, receptor=receptor, bbox_inches='tight', include_pdf=True)

    plt.show()



def plot_heatmap_base(subset_dict, subset_dict_data, title='', receptor = '', name_save = '', cmap='viridis', annotate=True, save_folder = ''):
    '''
    Plots heatmaps for different subsets in a 2x2 grid, with each subset visualized in a separate subplot.
    
    Args:
    - subset_dict (dict): A dictionary containing subset names as keys (e.g., '0,0', '0,1', etc.) and corresponding DataFrames as values.
    - subset_dict_data (dict): A dictionary containing descriptive titles for each subset, used to label each subplot.
    - title (str): Title for the entire figure (default is empty).
    - cmap (str): Color map for the heatmaps (default is 'viridis').
    - annotate (bool): Whether to annotate the cells with their values (default is True).
    '''
    
    # Create a 2x2 grid of subplots with a specified figure size
    fig, axes = plt.subplots(2, 2, figsize=(14, 1.3 * len((subset_dict[next(iter(subset_dict))]))))
   
    # Iterate through the subset dictionary to plot each subset
    for axses, data in subset_dict.items():
        # Extract the relevant columns (RS, SED, ASER) for the heatmap
        df = data[['RS', 'SED', 'ASER']]
        
        # Set the index of the dataframe to the 'name' attribute of the data
        df.index = data.name.tolist()  # Using names as index
        
        # Determine the position of the subplot based on the key (e.g., '0,0', '0,1')
        i = int(axses.split(',')[0])  # Row index
        j = int(axses.split(',')[1])  # Column index
        ax = axes[i, j]

        # Plot the heatmap for the current subset on the specified subplot
        sns.heatmap(df, annot=annotate, cmap=cmap, ax=ax)

        # Modify the y-axis labels for better readability by inserting line breaks
        new_labels = [label.get_text().replace('_epsilon', '\n epsilon').replace('_mut_r', '\n mut_r').replace('addcarbon', 'AddCarbon') for label in ax.get_yticklabels()]
        ax.set_yticklabels(new_labels, rotation=0, ha="right", fontsize=11)

        # Set the title for the current subplot to indicate the subset
        ax.set_title(f"{subset_dict_data[axses]}")
    
    # Set the overall title for the figure
    fig.suptitle(f'{title}', fontsize=16)
    
    # Adjust layout to ensure titles and labels are well placed
    plt.tight_layout()


    save_current_figure(name_save, save_folder=save_folder, receptor=receptor)
    # Display the heatmap figure
    plt.show()



def plot_heatmaps_with_diff_from_baseline(baseline_df_all, data_dict, type_split, scaf, receptor='', name_save='', numering='', save_folder=''):
    '''
    Plots heatmaps showing differences from the baseline for multiple subsets,
    dynamically arranging subplots based on the number of subsets.
    '''
    num_subsets = len(data_dict)
    num_gen = len(data_dict[next(iter(data_dict))])

    # Dynamically determine number of rows and columns (max 5 columns)
    max_cols = 5
    cols = min(num_subsets, max_cols)
    rows = math.ceil(num_subsets / cols)

    # Adjust layout so that the last row is not too sparse
    if num_subsets > 1:
        cols_last_row = num_subsets % cols
        if cols_last_row != 0 and cols_last_row < math.ceil(cols / 2):
            cols = math.ceil(num_subsets / 2)
            rows = math.ceil(num_subsets / cols)

    # Figure size in paper style
    fig_width = cols * 12
    fig_height = rows * num_gen * 1.3
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(rows, cols, figure=fig)

    axes = []
    for i in range(num_subsets):
        row = i // cols
        col = i % cols
        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)

    # Prepare baseline values
    baseline_df = baseline_df_all[['RS', 'SED', 'ASER']]
    baseline_df.index = baseline_df_all.name.tolist()

    # Plot differences
    for ax, (subset, df) in zip(axes, data_dict.items()):
        normalized_df = df[['RS', 'SED', 'ASER']]
        normalized_df.index = df.name.tolist()
        
        # Align indexes with baseline
        baseline_df = baseline_df.set_index(normalized_df.index)

        # Compute differences; empty subset uses direct baseline values
        if subset == '':
            diff_df = baseline_df
        else:
            diff_df = normalized_df - baseline_df
            mask = (diff_df.abs() <= 0.1)
            diff_df[mask] = np.nan

        hm = sns.heatmap(
            diff_df,
            annot=True,
            cmap='coolwarm',
            cbar=False,
            ax=ax,
            annot_kws={"size": 30}  # paper style
        )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes(
            "right",
            size="2.0%",
            pad=0.15
        )
        cbar = plt.colorbar(hm.collections[0], cax=cax)
        cbar.ax.tick_params(labelsize=15)
        cbar.outline.set_visible(False)
        ax.figure.axes[-1].yaxis.label.set_size(15)

        # Format subset name for title
        if subset == '':
            subset_name = 'Full OS'
        elif subset == '_62.5k':
            subset_name = '62,500'
        else:
            subset_name = subset.replace('_', '').replace('k', ',000')

        title_text = f"{subset_name}" if subset_name == 'Full OS' else f"{subset_name} subset"
        ax.set_title(title_text, fontsize=35, wrap=True, pad=12)

        # Format Y-axis labels
        new_labels = [
            label.get_text()
            .replace('_epsilon', '\n epsilon')
            .replace('_mut_r', '\n mut_r')
            .replace('addcarbon', 'AddCarbon')
            .replace('enamine', 'Enamine')
            for label in ax.get_yticklabels()
        ]
        new_labels = [
            label.replace('_62.5k', '').replace('_125k', '')
                .replace('_250k', '').replace('_500k', '')
                .replace('_10k', '')
            for label in new_labels
        ]
        ax.set_yticklabels(new_labels, rotation=0, ha="right", fontsize=30)
        ax.tick_params(axis='y', pad=12)
        ax.set_xticklabels(ax.get_xticklabels(), ha="center",  fontsize=30)
        ax.set_facecolor('white')

    # Hide unused subplot slots
    total_slots = rows * cols
    for j in range(len(axes), total_slots):
        row = j // cols
        col = j % cols
        ax_empty = fig.add_subplot(gs[row, col])
        ax_empty.axis('off')


    if rows == 2:
        for i, ax in enumerate(axes):
            if i >= cols: 
                pos = ax.get_position()
                ax.set_position([pos.x0, pos.y0 + 0.03, pos.width + 0.01, pos.height])

    # Add numbering text
    fig.text(0.005, 0.95, numering, ha='left', va='top', fontsize=40)

    # Format scaffold label
    scaf_str = 'CSK' if scaf == 'csk' else scaf
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    save_current_figure(name_save, save_folder=save_folder, receptor=receptor, bbox_inches='tight', include_pdf=True)

    plt.show()



def make_cmap_to_white(base_hex_color):
    # Convert a base hex color to RGB
    base_rgb = to_rgb(base_hex_color)
    # Define white color in RGB
    white_rgb = to_rgb('#f0f0f0')
    # Create a gradient from white to the base color
    colors = [white_rgb, base_rgb]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    return cmap




def plot_combined_heatmap_with_single_column_for_each_metric_rotated(
        generators, receptors, scaffolds, splits,
        metrics=['RS', 'SED', 'ASER'], 
        title=None, save_name=None, using_norm_values=False,
        data_folder='', save_folder='',
        inter_metric_wspace=0.15,
        intra_metric_wspace=0.05
    ):

    # Base colors for each metric
    metric_base_colors = {
        'RS': "#e97b32",
        'SED': "#97C2F0",
        'ASER': "#71ad48",
        'RS_inp': "#c86428",
        'ASER_new_1': "#588a37",

    }

    # --- Build a DataFrame with all values ---
    data = []
    for gen in generators:
        for receptor in receptors:
            for type_scaffold in scaffolds:
                for type_cluster in splits:

                    df = (preprocesing_org if not using_norm_values else preprocesing)(
                        type_cluster,
                        type_scaffold,
                        generators,
                        receptor,
                        data_folder
                    )

                    for met in metrics:
                        value = df[df.name.str.startswith(gen)][met].iloc[0]
                        data.append([
                            gen,
                            receptor,
                            type_scaffold,
                            type_cluster,
                            met,
                            value
                        ])

    df = pd.DataFrame(
        data,
        columns=['Generator', 'Receptor', 'Scaffold', 'Split', 'Metric', 'Value']
    )

    # -----------------------------------------
    # ROTATED LAYOUT
    # -----------------------------------------

    nrows = len(metrics)      # rows = metrics
    ncols = len(receptors)   # cols = receptors

    fig_width = 1.7 * 4 * ncols + 2
    fig_height = 6 * nrows * 1.3

    fig = plt.figure(figsize=(fig_width, fig_height))

    outer_gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols,
        wspace=inter_metric_wspace,
        hspace=0.1
    )

    # MAIN LOOP
    for met_idx, metric in enumerate(metrics):

        metric_df = df[df['Metric'] == metric].copy()
        cmap_custom = make_cmap_to_white(metric_base_colors[metric])

        for rec_idx, receptor in enumerate(receptors):

            inner = outer_gs[met_idx, rec_idx].subgridspec(
                nrows=1,
                ncols=4,
                wspace=intra_metric_wspace,
                hspace=0.0
            )

            group_axes = []

            for sc_idx, scaffold_type in enumerate(["csk", "murcko"]):

                block_df = metric_df[
                    (metric_df['Receptor'] == receptor) &
                    (metric_df['Scaffold'] == scaffold_type)
                ]

                for split_idx, split in enumerate(["dis", "sim"]):

                    col = sc_idx * 2 + split_idx
                    ax = fig.add_subplot(inner[0, col])
                    group_axes.append(ax)

                    sub_df = block_df[block_df['Split'] == split].copy()
                    sub_df = sub_df.set_index('Generator').reindex(generators)

                    heatmap_array = sub_df['Value'].to_numpy().reshape(-1, 1)

                    vmin = heatmap_array.min()
                    vmax = heatmap_array.max()

                    heatmap_flat = heatmap_array.flatten()
                    max_idx = np.argmax(heatmap_flat)

                    annot_array = []
                    for i, val in enumerate(heatmap_flat):
                        txt = f"{val:.4f}" if not using_norm_values else f"{val:.3f}"
                        if i == max_idx:
                            txt = r"$\bf{" + txt + "}$"
                        annot_array.append(txt)

                    annot_array = np.array(annot_array).reshape(heatmap_array.shape)

                    show_colorbar = (sc_idx == 1 and split_idx == 1)

                    if show_colorbar:
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes(
                            "right",
                            size="5%",
                            pad=0.05
                        )
                    else:
                        cax = None

                    sns.heatmap(
                        heatmap_array,
                        annot=annot_array,
                        fmt="",
                        cmap=cmap_custom,
                        ax=ax,
                        cbar=show_colorbar,
                        cbar_ax=cax,
                        annot_kws={
                            "size": 16,
                            "color": "black"
                        },
                        vmin=vmin,
                        vmax=vmax
                    )

                    ax.set_aspect("auto")

                    # X-axis = split labels
                    ax.set_xticks([0.5])
                    ax.set_xticklabels([split], rotation=0, ha="center", fontsize=14)

                    # -----------------------------------------
                    # Y LABELS  (GENS only in first column)
                    # -----------------------------------------
                    if rec_idx == 0 and sc_idx == 0 and split_idx == 0:
                        new_labels = [
                            g.replace('_epsilon', '\n epsilon')
                             .replace('_mut_r', '\n mut_r')
                             .replace('addcarbon', 'AddCarbon')
                            for g in generators
                        ]

                        ax.set_yticks(np.arange(len(generators)) + 0.5)
                        ax.set_yticklabels(new_labels, rotation=0, fontsize=15)
                        ax.set_ylabel(metric, fontsize=16, fontweight="bold", labelpad=8)

                    else:
                        ax.set_yticks([])
                        ax.set_ylabel("")

            # -----------------------------------------
            # TOP TITLE = RECEPTOR
            # -----------------------------------------
            if met_idx == 0:

                p0 = group_axes[0].get_position()
                p1 = group_axes[-1].get_position()

                x_mid = (p0.x0 + p1.x1) / 2
                y_top = p0.y1 + 0.018

                fig.text(
                    x_mid,
                    y_top,
                    receptor.replace('_', ' '),
                    ha="center",
                    va="bottom",
                    fontsize=16,
                    fontweight="bold"
                )

                # CSK = axes 0,1
                p0 = group_axes[0].get_position()
                p1 = group_axes[1].get_position()

                x_mid = (p0.x0 + p1.x1) / 2
                y_top = p0.y1 + 0.004

                fig.text(
                    x_mid,
                    y_top,
                    "CSK",
                    ha="center",
                    va="bottom",
                    fontsize=14
                )

                # MURCKO = axes 2,3
                p2 = group_axes[2].get_position()
                p3 = group_axes[3].get_position()

                x_mid = (p2.x0 + p3.x1) / 2
                y_top = p2.y1 + 0.004

                fig.text(
                    x_mid,
                    y_top,
                    "MURCKO",
                    ha="center",
                    va="bottom",
                    fontsize=14
                )

    # -----------------------------------------
    # FINAL TOUCHES
    # -----------------------------------------
    if title:
        fig.suptitle(title, fontsize=14, y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    if save_name:
        save_current_figure(save_name, save_folder=save_folder, bbox_inches='tight', include_pdf=True)

    plt.show()



def plot_combined_heatmap_with_single_column_for_each_metric_pharm(
        generators, receptors, type_fps, splits,
        metrics=['RS', 'SED', 'ASER'], 
        title=None, save_name=None, using_norm_values=False,
        inter_metric_wspace=0.15,   # Larger spacing between metrics
        intra_metric_wspace=0.05,    # Smaller spacing within a metric block
        ph4 = True, 
        user_threshold = 0.7,
        threshold_sim = None,  # Threshold for 'sim' split
        threshold_dis = None,  # Threshold for 'dis' split
        data_folder = '../', save_folder=''
    ):

    # Base colors for each metric
    metric_base_colors = {
        'RS': "#e97b32",
        'SED': "#97C2F0",
        'ASER': "#71ad48"
    }

    # --- Build a DataFrame with all values ---
    data = []
    for gen in generators:
        for receptor in receptors:
            for type_fp in type_fps:
                for type_cluster in splits:
                    
                    # Determine threshold based on split type
                    if threshold_sim is not None and 'sim' in type_cluster.lower():
                        current_threshold = threshold_sim
                    elif threshold_dis is not None and 'dis' in type_cluster.lower():
                        current_threshold = threshold_dis
                    else:
                        current_threshold = user_threshold

                    df = (preprocesing_org if not using_norm_values else preprocesing)(
                        type_cluster,
                        type_fp,
                        generators,
                        receptor,
                        data_folder, ph4, current_threshold
                    )

                    for met in metrics:
                        value = df[df.name.str.startswith(gen)][met].iloc[0]
                        data.append([
                            gen,
                            receptor,
                            type_fp,
                            type_cluster,
                            met,
                            value
                        ])

    df = pd.DataFrame(
        data,
        columns=['Generator', 'Receptor', 'FP', 'Split', 'Metric', 'Value']
    )

    # -----------------------------------------
    # DYNAMIC LAYOUT BASED ON type_fps LENGTH
    # -----------------------------------------
    num_fps = len(type_fps)
    num_splits = len(splits)
    cols_per_receptor = num_fps * num_splits  # Total columns per receptor
    
    nrows = len(metrics)      # rows = metrics
    ncols = len(receptors)   # cols = receptors

    # Adjust figure width based on number of fingerprint types
    fig_width = 1.7 * cols_per_receptor * ncols + 2
    fig_height = 6 * nrows * 1.3

    fig = plt.figure(figsize=(fig_width, fig_height))

    outer_gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols,
        wspace=inter_metric_wspace,
        hspace=0.1
    )

    # MAIN LOOP
    for met_idx, metric in enumerate(metrics):

        metric_df = df[df['Metric'] == metric].copy()
        cmap_custom = make_cmap_to_white(metric_base_colors[metric])

        for rec_idx, receptor in enumerate(receptors):

            inner = outer_gs[met_idx, rec_idx].subgridspec(
                nrows=1,
                ncols=cols_per_receptor,
                wspace=intra_metric_wspace,
                hspace=0.0
            )

            group_axes = []

            for sc_idx, fp_type in enumerate(type_fps):

                block_df = metric_df[
                    (metric_df['Receptor'] == receptor) &
                    (metric_df['FP'] == fp_type)
                ]

                for split_idx, split in enumerate(splits):

                    col = sc_idx * num_splits + split_idx
                    ax = fig.add_subplot(inner[0, col])
                    group_axes.append(ax)

                    sub_df = block_df[block_df['Split'] == split].copy()
                    sub_df = sub_df.set_index('Generator').reindex(generators)

                    heatmap_array = sub_df['Value'].to_numpy().reshape(-1, 1)

                    vmin = heatmap_array.min()
                    vmax = heatmap_array.max()

                    heatmap_flat = heatmap_array.flatten()
                    max_idx = np.argmax(heatmap_flat)

                    annot_array = []
                    for i, val in enumerate(heatmap_flat):
                        txt = f"{val:.4f}" if not using_norm_values else f"{val:.3f}"
                        if i == max_idx:
                            txt = r"$\bf{" + txt + "}$"
                        annot_array.append(txt)

                    annot_array = np.array(annot_array).reshape(heatmap_array.shape)

                    # Show colorbar on the last column of each fingerprint type
                    show_colorbar = (split_idx == num_splits - 1)

                    if show_colorbar:
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes(
                            "right",
                            size="5%",
                            pad=0.05
                        )
                    else:
                        cax = None

                    sns.heatmap(
                        heatmap_array,
                        annot=annot_array,
                        fmt="",
                        cmap=cmap_custom,
                        ax=ax,
                        cbar=show_colorbar,
                        cbar_ax=cax,
                        annot_kws={
                            "size": 16,
                            "color": "black"
                        },
                        vmin=vmin,
                        vmax=vmax
                    )

                    ax.set_aspect("auto")

                    # X-axis = split labels
                    ax.set_xticks([0.5])
                    ax.set_xticklabels([split], rotation=0, ha="center", fontsize=14)

                    # -----------------------------------------
                    # Y LABELS  (GENS only in first column)
                    # -----------------------------------------
                    if rec_idx == 0 and sc_idx == 0 and split_idx == 0:
                        new_labels = [
                            g.replace('_epsilon', '\n epsilon')
                             .replace('_mut_r', '\n mut_r')
                             .replace('addcarbon', 'AddCarbon')
                             .replace('_250k', '')
                            for g in generators
                        ]

                        ax.set_yticks(np.arange(len(generators)) + 0.5)
                        ax.set_yticklabels(new_labels, rotation=0, fontsize=15)
                        ax.set_ylabel(metric, fontsize=16, fontweight="bold", labelpad=8)

                    else:
                        ax.set_yticks([])
                        ax.set_ylabel("")

            # -----------------------------------------
            # TOP TITLE = RECEPTOR
            # -----------------------------------------
            if met_idx == 0:

                p0 = group_axes[0].get_position()
                p1 = group_axes[-1].get_position()

                x_mid = (p0.x0 + p1.x1) / 2
                y_top = p0.y1 + 0.018

                fig.text(
                    x_mid,
                    y_top,
                    receptor.replace('_', ' '),
                    ha="center",
                    va="bottom",
                    fontsize=16,
                    fontweight="bold"
                )

                # Add fingerprint type labels dynamically
                for fp_idx, fp_type in enumerate(type_fps):
                    start_col = fp_idx * num_splits
                    end_col = start_col + num_splits - 1
                    
                    p0 = group_axes[start_col].get_position()
                    p1 = group_axes[end_col].get_position()

                    x_mid = (p0.x0 + p1.x1) / 2
                    y_top = p0.y1 + 0.004

                    fig.text(
                        x_mid,
                        y_top,
                        fp_type,
                        ha="center",
                        va="bottom",
                        fontsize=14
                    )

    # -----------------------------------------
    # FINAL TOUCHES
    # -----------------------------------------
    if title:
        fig.suptitle(title, fontsize=14, y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_name:
        save_current_figure(save_name, save_folder=save_folder, bbox_inches='tight', include_pdf=True)
    
    plt.show()


    
#================================================
def load_effect_thresholds_csv(csv_path: str, ph4: False) -> dict:
    """
    Reads effect_size_thresholds.csv and returns:
      {("RS","csk"):(t1,t2,t3), ("RS","murcko"):(...), ...}

    Expected columns:
      Metric, Scaffold,
      Trivial Δ (≤25%), Small Δ (25–50%), Moderate Δ (50–75%), Large Δ (>75%)
    """
    df = pd.read_csv(csv_path)

    def norm_scaffold(s: str) -> str:
        s = str(s).strip().lower()
        if "csk" in s:
            return "csk"
        if "murcko" in s:
            return "murcko"
        return s
    

    def last_number(text: str) -> float:
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(text))
        if not nums:
            raise ValueError(f"Cannot parse number from: {text}")
        return float(nums[-1])

    if ph4:
        out = {}
        for _, r in df.iterrows():
            metric = str(r["Metric"]).strip().upper()
            scaffold = norm_scaffold(r["PH4"])
            t1 = last_number(r["Trivial Δ (≤25%)"])
            t2 = last_number(r["Small Δ (25–50%)"])
            t3 = last_number(r["Moderate Δ (50–75%)"])
            out[(metric, scaffold)] = (t1, t2, t3)
    else:
        out = {}
        for _, r in df.iterrows():
            metric = str(r["Metric"]).strip().upper()
            scaffold = norm_scaffold(r["Scaffold"])
            t1 = last_number(r["Trivial Δ (≤25%)"])
            t2 = last_number(r["Small Δ (25–50%)"])
            t3 = last_number(r["Moderate Δ (50–75%)"])
            out[(metric, scaffold)] = (t1, t2, t3)

    return out


def _mix_with_white(hex_color: str, amount: float) -> tuple:
    """
    amount in [0,1]:
      0 -> original color
      1 -> white
    """
    r, g, b = to_rgb(hex_color)
    return (r + (1 - r) * amount,
            g + (1 - g) * amount,
            b + (1 - b) * amount)


def colormap_for_metric_bins(base_hex: str):
    """
    Returns (ListedColormap, BoundaryNorm) for 4 bins:
      0 Trivial  -> very light tint
      1 Small    -> light tint
      2 Moderate -> medium tint
      3 Large    -> base-ish / darker (strongest)
    """
    # můžete upravit, jak moc je to "světlé"
    colors = [
        _mix_with_white(base_hex, 0.9),  # Trivial (almost white)
        _mix_with_white(base_hex, 0.6),  # Small
        _mix_with_white(base_hex, 0.38),  # Moderate
        _mix_with_white(base_hex, 0.10),  # Large (close to base)
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    return cmap, norm


def _bin_delta(delta_abs: float, t1: float, t2: float, t3: float) -> int:
    # 0=Trivial, 1=Small, 2=Moderate, 3=Large
    if delta_abs <= t1:
        return 0
    if delta_abs <= t2:
        return 1
    if delta_abs <= t3:
        return 2
    return 3


def round_excel(x, ndigits):
    q = Decimal(10) ** -ndigits
    return float(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))


def format_value_for_metric(x, metric, using_norm_values):

    metric = metric.upper()

    if metric in ("RS", "SED"):
        val = round_excel(float(x), 2)
        return f"{val:.2f}"

    if metric == "ASER":
        if using_norm_values:
            mant = round_excel(float(x), 2)
        else:
            mant = float(x) * 100.0
            mant = round_excel(mant, 2)
        return f"{mant:.2f}"

    return f"{x:.2f}"


def plot_combined_heatmap_with_single_column_for_each_metric_rotated_binned(
        generators, receptors, scaffolds, splits,
        metrics=['RS', 'SED', 'ASER'],
        title=None, save_name=None, using_norm_values=False,
        data_folder='', save_folder='',
        inter_metric_wspace=0.15,
        intra_metric_wspace=0.05,
        effect_thresholds_csv="effect_size_thresholds_scaffolds.csv",
        annotate_values=True,
    ):
    """
    Same layout as plot_combined_heatmap_with_single_column_for_each_metric_rotated(),
    but cell COLORS are discrete bins based on effect-size intervals:

      Δ = |best_in_column - value|   (best = max; assumes higher is better)
      bin 0..3 = Trivial/Small/Moderate/Large using thresholds from CSV.

    The cell TEXT annotations can remain the original metric values.
    """

    thresholds = load_effect_thresholds_csv(effect_thresholds_csv, False)

    # Discrete colormap for bins

    metric_base_colors = {
    'RS': "#e97b32",
    'SED': "#97C2F0",
    'ASER': "#71ad48"
    }

    # --- Build a DataFrame with all values (same as original) ---
    data = []
    for gen in generators:
        for receptor in receptors:
            for type_scaffold in scaffolds:
                for type_cluster in splits:

                    # REAL (for binning)
                    df_real = (preprocesing_org)(
                        type_cluster,
                        type_scaffold,
                        generators,
                        receptor,
                        data_folder
                    )

                    # NORMALIZED (for display, only if requested)
                    df_norm = None
                    if using_norm_values:
                        df_norm = (preprocesing)(
                            type_cluster,
                            type_scaffold,
                            generators,
                            receptor,
                            data_folder
                        )

                    for met in metrics:
                        value_real = df_real[df_real.name.str.startswith(gen)][met].iloc[0]
                        if using_norm_values:
                            value_norm = df_norm[df_norm.name.str.startswith(gen)][met].iloc[0]
                        else:
                            value_norm = np.nan

                        data.append([gen, receptor, type_scaffold, type_cluster, met, value_real, value_norm])


    df = pd.DataFrame(
        data,
        columns=['Generator', 'Receptor', 'Scaffold', 'Split', 'Metric', 'Value_real', 'Value_norm']
    )

    # -----------------------------------------
    # ROTATED LAYOUT (same as original)
    # -----------------------------------------
    nrows = len(metrics)      # rows = metrics
    ncols = len(receptors)    # cols = receptors

    fig_width = 1.7 * 4 * ncols + 2
    fig_height = 6 * nrows * 1.3
    fig = plt.figure(figsize=(fig_width, fig_height))

    outer_gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols,
        wspace=inter_metric_wspace,
        hspace=0.1
    )

    for met_idx, metric in enumerate(metrics):
        metric_df = df[df['Metric'] == metric].copy()
        base_hex = metric_base_colors.get(metric.upper(), "#999999")
        cmap_bins, norm_bins = colormap_for_metric_bins(base_hex)

        for rec_idx, receptor in enumerate(receptors):

            inner = outer_gs[met_idx, rec_idx].subgridspec(
                nrows=1,
                ncols=4,
                wspace=intra_metric_wspace,
                hspace=0.0
            )

            group_axes = []

            for sc_idx, scaffold_type in enumerate(["csk", "murcko"]):

                block_df = metric_df[
                    (metric_df['Receptor'] == receptor) &
                    (metric_df['Scaffold'] == scaffold_type)
                ]

                for split_idx, split in enumerate(["dis", "sim"]):

                    col = sc_idx * 2 + split_idx
                    ax = fig.add_subplot(inner[0, col])
                    group_axes.append(ax)

                    sub_df = block_df[block_df['Split'] == split].copy()
                    sub_df = sub_df.set_index('Generator').reindex(generators)

                    # --- REAL values for binning ---
                    values_real = sub_df['Value_real'].to_numpy().reshape(-1, 1)

                    # --- DISPLAY values (real or normalized) for annotations ---
                    if using_norm_values:
                        values_show = sub_df['Value_norm'].to_numpy().reshape(-1, 1)
                    else:
                        values_show = values_real

                     # --- compute bins from |best_real - value_real| ---
                    best_real = float(np.nanmax(values_real))
                    deltas = np.abs(values_real - best_real)

                    t1, t2, t3 = thresholds[(metric.upper(), scaffold_type.lower())]
                    bins = np.vectorize(lambda d: _bin_delta(float(d), t1, t2, t3))(deltas).astype(int)

                    annot_array = None
                    if annotate_values:
                        flat_vals = values_show.flatten()
                        max_idx = int(np.nanargmax(flat_vals))
                        worst_idx = int(np.nanargmin(flat_vals))

                        annot_list = []
                        for i, val in enumerate(flat_vals):
                            txt_core = format_value_for_metric(val, metric, using_norm_values)

                            if i == max_idx:
                                txt = r"$\bf{" + txt_core + "}$"
                            else:
                                txt = f"${txt_core}$"
 
                            annot_list.append(txt)

                        annot_array = np.array(annot_list).reshape(values_show.shape)

                    show_colorbar = (sc_idx == 1 and split_idx == 1)
                    if show_colorbar:
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                    else:
                        cax = None

                    hm = sns.heatmap(
                        bins,
                        annot=annot_array,
                        fmt="",
                        cmap=cmap_bins,
                        norm=norm_bins,
                        ax=ax,
                        cbar=show_colorbar,
                        cbar_ax=cax,
                        annot_kws={"size": 16, "color": "black"},
                    )

                    # colorbar labels
                    if show_colorbar:
                        cbar = hm.collections[0].colorbar
                        cbar.set_ticks([0, 1, 2, 3])
                        cbar.set_ticklabels(["Trivial", "Small", "Moderate", "Large"])

                    ax.set_aspect("auto")

                    ax.set_xticks([0.5])
                    ax.set_xticklabels([split], rotation=0, ha="center", fontsize=14)

                    # Y labels only on first block (same rule as original)
                    if rec_idx == 0 and sc_idx == 0 and split_idx == 0:
                        new_labels = [
                            g.replace('_epsilon', '\n epsilon')
                             .replace('_mut_r', '\n mut_r')
                             .replace('addcarbon', 'AddCarbon')
                            for g in generators
                        ]
                        ax.set_yticks(np.arange(len(generators)) + 0.5)
                        ax.set_yticklabels(new_labels, rotation=0, fontsize=15)
                        if metric == 'ASER' and not using_norm_values:
                            metric_text = "ASER · 10⁻²"
                        else:
                            metric_text = metric
                        ax.set_ylabel(metric_text, fontsize=16, fontweight="bold", labelpad=8)
                    else:
                        ax.set_yticks([])
                        ax.set_ylabel("")

            # Titles at top row (same as original)
            if met_idx == 0:
                p0 = group_axes[0].get_position()
                p1 = group_axes[-1].get_position()
                x_mid = (p0.x0 + p1.x1) / 2
                y_top = p0.y1 + 0.018
                fig.text(x_mid, y_top, receptor.replace('_', ' '),
                         ha="center", va="bottom", fontsize=16, fontweight="bold")

                p0 = group_axes[0].get_position()
                p1 = group_axes[1].get_position()
                x_mid = (p0.x0 + p1.x1) / 2
                y_top = p0.y1 + 0.004
                fig.text(x_mid, y_top, "CSK", ha="center", va="bottom", fontsize=14)

                p2 = group_axes[2].get_position()
                p3 = group_axes[3].get_position()
                x_mid = (p2.x0 + p3.x1) / 2
                y_top = p2.y1 + 0.004
                fig.text(x_mid, y_top, "MURCKO", ha="center", va="bottom", fontsize=14)

    if title:
        fig.suptitle(title, fontsize=14, y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    if save_name:
        save_current_figure(save_name, save_folder=save_folder, bbox_inches='tight', include_pdf=True)

    plt.show()



def plot_combined_heatmap_with_single_column_for_each_metric_rotated_binned_pharm(
        generators, receptors, phfps, splits,
        metrics=['RS', 'SED', 'ASER'],
        title=None, save_name=None, using_norm_values=False,
        data_folder='', save_folder='',
        inter_metric_wspace=0.3,
        intra_metric_wspace=0.05,
        user_threshold = 0.7,
        threshold_sim = None,  # Threshold for 'sim' split
        threshold_dis = None,  # Threshold for 'dis' split
        effect_thresholds_csv="effect_size_thresholds_ph4_rdkit.csv",
        annotate_values=True,
    ):
    """
    Same layout as plot_combined_heatmap_with_single_column_for_each_metric_rotated(),
    but cell COLORS are discrete bins based on effect-size intervals:

      Δ = |best_in_column - value|   (best = max; assumes higher is better)
      bin 0..3 = Trivial/Small/Moderate/Large using thresholds from CSV.

    The cell TEXT annotations can remain the original metric values.
    """

    thresholds = load_effect_thresholds_csv(effect_thresholds_csv, True)

    # Discrete colormap for bins

    metric_base_colors = {
    'RS': "#e97b32",
    'SED': "#97C2F0",
    'ASER': "#71ad48"
    }

    # --- Build a DataFrame with all values (same as original) ---
    data = []
    for gen in generators:
        for receptor in receptors:
            for type_phfp in phfps:
                for type_cluster in splits:

                    # Determine threshold based on split type
                    if threshold_sim is not None and 'sim' in type_cluster.lower():
                        current_threshold = threshold_sim
                    elif threshold_dis is not None and 'dis' in type_cluster.lower():
                        current_threshold = threshold_dis
                    else:
                        current_threshold = user_threshold

                    # REAL (for binning)
                    df_real = (preprocesing_org)(
                        type_cluster,
                        type_phfp,
                        generators,
                        receptor,
                        data_folder, True, current_threshold
                    )

                    # NORMALIZED (for display, only if requested)
                    df_norm = None
                    if using_norm_values:
                        df_norm = (preprocesing)(
                            type_cluster,
                            type_phfp,
                            generators,
                            receptor,
                            data_folder, True, current_threshold
                        )

                    for met in metrics:
                        value_real = df_real[df_real.name.str.startswith(gen)][met].iloc[0]
                        if using_norm_values:
                            value_norm = df_norm[df_norm.name.str.startswith(gen)][met].iloc[0]
                        else:
                            value_norm = np.nan

                        data.append([gen, receptor, type_phfp, type_cluster, met, value_real, value_norm])


    df = pd.DataFrame(
        data,
        columns=['Generator', 'Receptor', 'phfp', 'Split', 'Metric', 'Value_real', 'Value_norm']
    )

        # -----------------------------------------
    # DYNAMIC LAYOUT BASED ON type_fps LENGTH
    # -----------------------------------------
    num_fps = len(phfps)
    num_splits = len(splits)
    cols_per_receptor = num_fps * num_splits  # Total columns per receptor
    
    nrows = len(metrics)      # rows = metrics
    ncols = len(receptors)   # cols = receptors

    # Adjust figure width based on number of fingerprint types
    fig_width = 1.7 * cols_per_receptor * ncols + 2
    fig_height = 6 * nrows * 1.3

    fig = plt.figure(figsize=(fig_width, fig_height))

    outer_gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols,
        wspace=inter_metric_wspace,
        hspace=0.1
    )

    for met_idx, metric in enumerate(metrics):
        metric_df = df[df['Metric'] == metric].copy()
        base_hex = metric_base_colors.get(metric.upper(), "#999999")
        cmap_bins, norm_bins = colormap_for_metric_bins(base_hex)

        for rec_idx, receptor in enumerate(receptors):

            inner = outer_gs[met_idx, rec_idx].subgridspec(
                nrows=1,
                ncols=cols_per_receptor,
                wspace=intra_metric_wspace,
                hspace=0.0
            )

            group_axes = []

            for sc_idx, phfp_type in enumerate(phfps):

                block_df = metric_df[
                    (metric_df['Receptor'] == receptor) &
                    (metric_df['phfp'] == phfp_type)
                ]

                for split_idx, split in enumerate(["dis", "sim"]):

                    col = sc_idx * 2 + split_idx
                    ax = fig.add_subplot(inner[0, col])
                    group_axes.append(ax)

                    sub_df = block_df[block_df['Split'] == split].copy()
                    sub_df = sub_df.set_index('Generator').reindex(generators)

                    # --- REAL values for binning ---
                    values_real = sub_df['Value_real'].to_numpy().reshape(-1, 1)

                    # --- DISPLAY values (real or normalized) for annotations ---
                    if using_norm_values:
                        values_show = sub_df['Value_norm'].to_numpy().reshape(-1, 1)
                    else:
                        values_show = values_real

                     # --- compute bins from |best_real - value_real| ---
                    best_real = float(np.nanmax(values_real))
                    deltas = np.abs(values_real - best_real)

                    t1, t2, t3 = thresholds[(metric.upper(), phfp_type.lower())]
                    bins = np.vectorize(lambda d: _bin_delta(float(d), t1, t2, t3))(deltas).astype(int)

                    annot_array = None
                    if annotate_values:
                        flat_vals = values_show.flatten()
                        max_idx = int(np.nanargmax(flat_vals))
                        worst_idx = int(np.nanargmin(flat_vals))

                        annot_list = []
                        for i, val in enumerate(flat_vals):
                            txt_core = format_value_for_metric(val, metric, using_norm_values)

                            if i == max_idx:
                                txt = r"$\bf{" + txt_core + "}$"
                            else:
                                txt = f"${txt_core}$"
 
                            annot_list.append(txt)

                        annot_array = np.array(annot_list).reshape(values_show.shape)

                    show_colorbar = (sc_idx == (num_fps-1) and split_idx == (num_splits-1))
                    if show_colorbar:
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                    else:
                        cax = None

                    hm = sns.heatmap(
                        bins,
                        annot=annot_array,
                        fmt="",
                        cmap=cmap_bins,
                        norm=norm_bins,
                        ax=ax,
                        cbar=show_colorbar,
                        cbar_ax=cax,
                        annot_kws={"size": 16, "color": "black"},
                    )

                    # colorbar labels
                    if show_colorbar:
                        cbar = hm.collections[0].colorbar
                        cbar.set_ticks([0, 1, 2, 3])
                        cbar.set_ticklabels(["Trivial", "Small", "Moderate", "Large"])

                    ax.set_aspect("auto")

                    ax.set_xticks([0.5])
                    ax.set_xticklabels([split], rotation=0, ha="center", fontsize=14)

                    # Y labels only on first block (same rule as original)
                    if rec_idx == 0 and sc_idx == 0 and split_idx == 0:
                        new_labels = [
                            g.replace('_epsilon', '\n epsilon')
                             .replace('_mut_r', '\n mut_r')
                             .replace('addcarbon', 'AddCarbon')
                             .replace('enamine', 'Enamine')
                            for g in generators
                        ]
                        ax.set_yticks(np.arange(len(generators)) + 0.5)
                        ax.set_yticklabels(new_labels, rotation=0, fontsize=15)
                        if metric == 'ASER' and not using_norm_values:
                            metric_text = "ASER · 10⁻²"
                        else:
                            metric_text = metric
                        ax.set_ylabel(metric_text, fontsize=16, fontweight="bold", labelpad=8)
                    else:
                        ax.set_yticks([])
                        ax.set_ylabel("")

            # Titles at top row (same as original)
            if met_idx == 0:
                p0 = group_axes[0].get_position()
                p1 = group_axes[-1].get_position()
                x_mid = (p0.x0 + p1.x1) / 2
                y_top = p0.y1 + 0.018
                fig.text(x_mid, y_top, receptor.replace('_', ' '),
                         ha="center", va="bottom", fontsize=16, fontweight="bold")

                # Add fingerprint type labels dynamically
                for fp_idx, fp_type in enumerate(phfps):
                    start_col = fp_idx * num_splits
                    end_col = start_col + num_splits - 1
                    
                    p0 = group_axes[start_col].get_position()
                    p1 = group_axes[end_col].get_position()

                    x_mid = (p0.x0 + p1.x1) / 2
                    y_top = p0.y1 + 0.004

                    if fp_type == 'rdkit':
                        fp_type_str = 'RDKit'
                    else:
                        fp_type_str = fp_type

                    fig.text(
                        x_mid,
                        y_top,
                        fp_type_str,
                        ha="center",
                        va="bottom",
                        fontsize=14
                    )

    if title:
        fig.suptitle(title, fontsize=14, y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    if save_name:
        save_current_figure(save_name, save_folder=save_folder, bbox_inches='tight', include_pdf=True)

    plt.show()
