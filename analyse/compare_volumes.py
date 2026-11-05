import argparse
import logging
import os
import os.path as op

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from utils.logger import BLUE, GREEN, RESET, configure_logging

logger = logging.getLogger(__name__)
configure_logging(dev_mode=True, module_name=__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare volumes across multiple CSV files."
    )
    parser.add_argument(
        "--csvs",
        required=True,
        nargs='+',
        help="CSV files to compare (space-separated)",
    )
    parser.add_argument(
        "--labels",
        required=True,
        nargs='+',
        help="Labels for each CSV (same order)",
    )
    parser.add_argument(
        "--out_dir",
        default=".",
        help="Output directory for plots.",
    )
    return parser.parse_args()


def plot_distribution_ridgeline_by_label(dfs, dataset_labels, out_dir):
    """Create ridgeline plot for each label across all datasets (stacked vertically)."""
    label_cols = [col for col in dfs[0].columns if col.startswith("label_")]
    label_ids = sorted([int(col.split("_")[1]) for col in label_cols])
    
    colors = ['#FFB6C1', '#87CEEB', '#90EE90', '#FFD700', '#FF6347', '#DDA0DD']
    
    n_labels = len(label_ids)
    fig, axes = plt.subplots(n_labels, 1, figsize=(14, 3*n_labels))
    
    # Handle single subplot case
    if n_labels == 1:
        axes = [axes]
    
    for label_idx, label_id in enumerate(label_ids):
        ax = axes[label_idx]
        
        # Collect volumes for this label across all datasets
        all_volumes = []
        for df in dfs:
            col_name = f"label_{label_id}"
            if col_name in df.columns:
                vols = df[col_name].dropna().values
                all_volumes.extend(vols)
        
        if not all_volumes:
            ax.text(0.5, 0.5, f'No data for Label {label_id}', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(f"Label {label_id}", fontsize=12, fontweight='bold')
            continue
        
        # Define x range
        x_min = min(all_volumes) * 0.95
        x_max = max(all_volumes) * 1.05
        x_range = np.linspace(x_min, x_max, 300)
        
        # Plot each dataset
        for idx, (df, dataset_label) in enumerate(zip(dfs, dataset_labels)):
            col_name = f"label_{label_id}"
            if col_name not in df.columns:
                continue
            
            volumes = df[col_name].dropna().values
            if len(volumes) < 2:
                continue
            
            # Calculate KDE
            kde = stats.gaussian_kde(volumes)
            density = kde(x_range)
            
            # Normalize
            density = density / density.max() * 0.8
            
            # Plot
            color = colors[idx % len(colors)]
            ax.fill_between(x_range, idx, idx + density, alpha=0.7, label=dataset_label, color=color)
            ax.plot(x_range, idx + density, color=color, linewidth=2)
        
        # Formatting
        ax.set_xlabel("Volume", fontsize=11)
        ax.set_ylabel("Dataset", fontsize=11)
        ax.set_title(f"Label {label_id}", fontsize=12, fontweight='bold')
        ax.set_yticks(range(len(dfs)))
        ax.set_yticklabels(dataset_labels, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        
        # Add legend only to first subplot
        if label_idx == 0:
            ax.legend(loc='upper right', fontsize=10)
    
    fig.suptitle("Distribution Comparison: All Labels (Stacked)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = op.join(out_dir, "distribution_ridgeline_by_label.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved: %s", plot_path)


def plot_barplot(df, dataset_label, out_dir):
    """Create bar plot for a dataset."""
    label_cols = [col for col in df.columns if col.startswith("label_")]
    label_ids = sorted([int(col.split("_")[1]) for col in label_cols])
    
    # Calculate mean and std for each label
    means = [df[f"label_{l}"].mean() for l in label_ids]
    stds = [df[f"label_{l}"].std() for l in label_ids]
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(label_ids))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
    
    ax.set_xlabel("Labels", fontsize=12)
    ax.set_ylabel("Mean Volume", fontsize=12)
    ax.set_title(f"Bar Plot: {dataset_label}", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in label_ids])
    ax.grid(axis="y", alpha=0.3)
    
    plot_path = op.join(out_dir, f"barplot_{dataset_label.replace(' ', '_').replace('(', '').replace(')', '').lower()}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info("Saved: %s", plot_path)


def plot_barplot_comparison(dfs, dataset_labels, out_dir):
    """Create comparison bar plot with all datasets."""
    label_cols = [col for col in dfs[0].columns if col.startswith("label_")]
    label_ids = sorted([int(col.split("_")[1]) for col in label_cols])
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(16, 7))
    
    x = np.arange(len(label_ids))
    width = 0.2
    
    for i, (df, label) in enumerate(zip(dfs, dataset_labels)):
        means = [df[f"label_{l}"].mean() for l in label_ids]
        ax.bar(x + i*width - (len(dfs)-1)*width/2, means, width, label=label, alpha=0.8)
    
    ax.set_xlabel("Labels", fontsize=12)
    ax.set_ylabel("Mean Volume", fontsize=12)
    ax.set_title("Comparison: All Datasets", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in label_ids])
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    
    plot_path = op.join(out_dir, "barplot_comparison.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info("Saved: %s", plot_path)


def plot_heatmap_difference(df1, df2, label1, label2, out_dir):
    """Create heatmap showing difference between two datasets (by subject_id only)."""
    label_cols = [col for col in df1.columns if col.startswith("label_")]
    label_ids = sorted([int(col.split("_")[1]) for col in label_cols])
    
    # Keep only first occurrence per subject_id
    df1_clean = df1.drop_duplicates(subset=["subject_id"], keep="first").copy()
    df2_clean = df2.drop_duplicates(subset=["subject_id"], keep="first").copy()
    
    # Check for dropped rows
    if len(df1) > len(df1_clean):
        logger.warning("Dropped %d duplicate subjects from %s, keeping first occurrence", 
                      len(df1) - len(df1_clean), label1)
    if len(df2) > len(df2_clean):
        logger.warning("Dropped %d duplicate subjects from %s, keeping first occurrence", 
                      len(df2) - len(df2_clean), label2)
    
    # Merge on subject_id
    df_merged = df1_clean.merge(
        df2_clean[["subject_id"] + [f"label_{l}" for l in label_ids]],
        on="subject_id",
        suffixes=("_1", "_2")
    )
    
    if len(df_merged) == 0:
        logger.warning("No matching subjects between %s and %s", label1, label2)
        return
    
    # Calculate difference
    diff_data = np.zeros((len(df_merged), len(label_ids)))
    for j, l in enumerate(label_ids):
        diff_data[:, j] = df_merged[f"label_{l}_2"] - df_merged[f"label_{l}_1"]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, len(df_merged)*0.6))
    
    # Use centered colormap (blue = negative, red = positive)
    vmax = np.abs(diff_data).max()
    im = ax.imshow(diff_data, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    
    ax.set_xlabel("Labels", fontsize=12)
    ax.set_ylabel("Subject", fontsize=12)
    ax.set_title(f"Difference: {label2} - {label1}", fontsize=14, fontweight="bold")
    
    # Set x-axis
    ax.set_xticks(range(len(label_ids)))
    ax.set_xticklabels([f"L{l}" for l in label_ids], fontsize=11)
    
    # Set y-axis
    subject_ids = df_merged["subject_id"].values
    ax.set_yticks(range(len(subject_ids)))
    ax.set_yticklabels(subject_ids, fontsize=11)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Difference (Volume)", fontsize=11)
    
    plot_path = op.join(out_dir, f"diff_{label1.replace(' ', '_').replace('(', '').replace(')', '').lower()}_vs_{label2.replace(' ', '_').replace('(', '').replace(')', '').lower()}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved: %s", plot_path)


def main(args: argparse.Namespace):
    os.makedirs(args.out_dir, exist_ok=True)
    
    if len(args.csvs) != len(args.labels):
        raise ValueError("Number of CSVs must match number of labels")
    
    # Read all CSVs
    dfs = []
    for csv_path, label in zip(args.csvs, args.labels):
        logger.info("Reading %s%s%s (%s)", BLUE, csv_path, RESET, label)
        df = pd.read_csv(csv_path)
        dfs.append(df)
        logger.info("  %d subjects, %d columns", len(df), len(df.columns))
    
    # Create ridgeline plot by label (distribution)
    logger.info("Creating distribution ridgeline plot by label...")
    plot_distribution_ridgeline_by_label(dfs, args.labels, args.out_dir)
    
    # Create individual bar plots
    logger.info("Creating individual plots...")
    for df, dataset_label in zip(dfs, args.labels):
        logger.info("Processing: %s", dataset_label)
        plot_barplot(df, dataset_label, args.out_dir)
    
    # Create comparison plot
    logger.info("Creating comparison plot...")
    plot_barplot_comparison(dfs, args.labels, args.out_dir)
    
    # Create difference heatmaps (no ratio heatmaps)
    logger.info("Creating difference heatmaps...")
    for i in range(len(dfs) - 1):
        for j in range(i + 1, len(dfs)):
            plot_heatmap_difference(dfs[i], dfs[j], args.labels[i], args.labels[j], args.out_dir)
    
    logger.info("%sComparison complete!%s", GREEN, RESET)
    logger.info("Plots saved to: %s", args.out_dir)


if __name__ == "__main__":
    main(parse_args())