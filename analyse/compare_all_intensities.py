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
        description="Compare intensities across CSV files."
    )
    parser.add_argument("--csvs", required=True, nargs='+', help="CSV files")
    parser.add_argument("--labels", required=True, nargs='+', help="Labels for each CSV")
    parser.add_argument("--out_dir", default=".", help="Output directory")
    return parser.parse_args()


def plot_barplot(df, dataset_label, out_dir):
    """Bar plot for mean intensity per label."""
    label_cols = [col for col in df.columns if col.startswith("label_") and col.endswith("_mean")]
    label_ids = sorted([int(col.split("_")[1]) for col in label_cols])
    
    if not label_ids:
        return
    
    means = [df[f"label_{l}_mean"].mean() for l in label_ids]
    stds = [df[f"label_{l}_std"].mean() for l in label_ids]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(label_ids))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
    ax.set_xlabel("Labels", fontsize=12)
    ax.set_ylabel("Mean Intensity", fontsize=12)
    ax.set_title(f"Intensity: {dataset_label}", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in label_ids], fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    
    plot_path = op.join(out_dir, f"intensity_barplot_{dataset_label.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info("Saved: %s", plot_path)

def plot_distribution_with_histogram(dfs, out_dir):
    """KDE + Histogram by label with same density scale."""
    label_cols = [col for col in dfs[0].columns if col.startswith("label_") and col.endswith("_mean")]
    label_ids = sorted([int(col.split("_")[1]) for col in label_cols])
    
    if not label_ids:
        return
    
    # First pass: calculate max density across all labels
    max_density = 0
    for label_id in label_ids:
        all_intensities = []
        for df in dfs:
            col_name = f"label_{label_id}_mean"
            if col_name in df.columns:
                intensities = df[col_name].dropna().values
                all_intensities.extend(intensities)
        
        if all_intensities and len(all_intensities) > 1:
            try:
                kde = stats.gaussian_kde(all_intensities)
                x_min, x_max = min(all_intensities) * 0.95, max(all_intensities) * 1.05
                x_range = np.linspace(x_min, x_max, 300)
                density = kde(x_range)
                max_density = max(max_density, density.max())
            except:
                pass
    
    # Second pass: plot with same scale
    n_labels = len(label_ids)
    fig, axes = plt.subplots(n_labels, 1, figsize=(12, 3.5*n_labels))
    
    if n_labels == 1:
        axes = [axes]
    
    for label_idx, label_id in enumerate(label_ids):
        ax = axes[label_idx]
        
        all_intensities = []
        
        for df in dfs:
            col_name = f"label_{label_id}_mean"
            if col_name in df.columns:
                intensities = df[col_name].dropna().values
                all_intensities.extend(intensities)
        
        if not all_intensities:
            ax.text(0.5, 0.5, f'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        x_min, x_max = min(all_intensities) * 0.95, max(all_intensities) * 1.05
        
        # Histogram + KDE plot
        ax.hist(all_intensities, bins=30, density=True, alpha=0.6, color='#FFB6C1', label='Histogram')
        
        try:
            kde = stats.gaussian_kde(all_intensities)
            x_range = np.linspace(x_min, x_max, 300)
            ax.plot(x_range, kde(x_range), 'b-', linewidth=2, label='KDE (Scipy)')
        except:
            pass
        
        ax.set_xlabel("Intensity", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"Label {label_id}", fontsize=12, fontweight='bold')
        ax.set_ylim(0, max_density * 1.1)  # Même échelle pour tous
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
    
    fig.suptitle("Intensity Distribution by Label", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = op.join(out_dir, "intensity_distribution_histogram.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved: %s", plot_path)
    
def main(args: argparse.Namespace):
    os.makedirs(args.out_dir, exist_ok=True)
    
    if len(args.csvs) != len(args.labels):
        raise ValueError("Number of CSVs must match number of labels")
    
    # Read CSVs
    dfs = []
    for csv_path, label in zip(args.csvs, args.labels):
        if not op.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        
        logger.info("Reading %s%s%s", BLUE, csv_path, RESET)
        df = pd.read_csv(csv_path)
        dfs.append(df)
        logger.info("  %d rows, %d columns", len(df), len(df.columns))
    
    # Create plots
    logger.info("Creating plots...")
    for df, dataset_label in zip(dfs, args.labels):
        plot_barplot(df, dataset_label, args.out_dir)
    
    plot_distribution_with_histogram(dfs, args.out_dir)
    
    logger.info("%sComplete!%s", GREEN, RESET)


if __name__ == "__main__":
    main(parse_args())