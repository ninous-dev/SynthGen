import argparse
import logging
import os
import os.path as op
import re

import pandas as pd
import numpy as np
import torchio as tio
from utils.logger import BLUE, GREEN, RESET, configure_logging

logger = logging.getLogger(__name__)
configure_logging(dev_mode=True, module_name=__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute intensity statistics per label from images masked with labels."
    )
    parser.add_argument("--csv", required=True, help="CSV file with subject data")
    parser.add_argument("--image_pattern", required=True, help="Pattern for image columns (e.g., 'sim_')")
    parser.add_argument("--mask_column", required=True, help="Column name for mask paths")
    parser.add_argument("--out_dir", required=True, help="Output directory for CSV files")
    return parser.parse_args()


def compute_intensity_stats(img: tio.ScalarImage, mask: tio.LabelMap) -> dict:
    """Compute intensity statistics for each label."""
    img_data = img.data.squeeze()
    mask_data = mask.data.squeeze()
    
    unique_labels = mask_data.unique()
    stats_dict = {}
    
    for label in unique_labels:
        if label == 0:
            continue
        
        intensities = img_data[mask_data == label].numpy()
        
        if len(intensities) == 0:
            continue
        
        stats_dict[int(label)] = {
            'mean': float(intensities.mean()),
            'std': float(intensities.std()),
            'median': float(np.median(intensities)),
            'min': float(intensities.min()),
            'max': float(intensities.max()),
            'count': int(len(intensities)),
        }
    
    return stats_dict


def extract_contrast_number(col_name: str) -> str:
    """Extract number from column name like sim_contrast_1 -> 1."""
    match = re.search(r'_(\d+)$', col_name)
    return match.group(1) if match else ""


def main(args: argparse.Namespace):
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Read CSV
    df = pd.read_csv(args.csv)
    
    # Check required columns
    if "subject_id" not in df.columns:
        raise ValueError("CSV must have 'subject_id' column")
    if "gen_folder" not in df.columns:
        raise ValueError("CSV must have 'gen_folder' column")
    if args.mask_column not in df.columns:
        raise ValueError(f"CSV must have '{args.mask_column}' column")
    
    # Find all image columns matching pattern
    image_columns = [col for col in df.columns if col.startswith(args.image_pattern)]
    
    if not image_columns:
        raise ValueError(f"No columns found matching pattern '{args.image_pattern}'")
    
    logger.info("Found %s%d%s subjects", BLUE, len(df), RESET)
    logger.info("Image columns: %s%d%s: %s", BLUE, len(image_columns), RESET, image_columns)
    logger.info("Mask column: %s%s%s", BLUE, args.mask_column, RESET)
    
    # Process each subject and image
    all_stats = []
    all_labels = set()
    
    for idx, row in df.iterrows():
        subject_id = row["subject_id"]
        gen_folder = row["gen_folder"]
        mask_path = row[args.mask_column]
        
        logger.info("[%d/%d] Processing %s%s%s", idx + 1, len(df), BLUE, subject_id, RESET)
        
        # Check mask
        if pd.isna(mask_path) or mask_path == "":
            logger.warning("  No mask path for %s", subject_id)
            continue
        
        if not op.exists(mask_path):
            logger.warning("  Mask not found: %s", mask_path)
            continue
        
        try:
            mask = tio.LabelMap(mask_path)
            
            # Process each image column
            for img_col in image_columns:
                img_path = row[img_col]
                contrast_num = extract_contrast_number(img_col)
                
                if pd.isna(img_path) or img_path == "":
                    continue
                
                if not op.exists(img_path):
                    logger.warning("    Image not found: %s", img_path)
                    continue
                
                try:
                    img = tio.ScalarImage(img_path)
                    stats = compute_intensity_stats(img, mask)
                    
                    all_labels.update(stats.keys())
                    all_stats.append({
                        "subject_id": subject_id,
                        "gen_folder": gen_folder,
                        "contrast_number": contrast_num,
                        "stats": stats,
                    })
                except Exception as e:
                    logger.warning("    Error processing %s: %s", img_col, str(e))
                    continue
        
        except Exception as e:
            logger.error("  Error processing %s: %s", subject_id, str(e))
            continue
    
    # Sort labels
    sorted_labels = sorted(list(all_labels))
    logger.info("Found labels: %s", sorted_labels)
    
    # Build output dataframe
    output_rows = []
    stat_names = ['mean', 'std', 'median', 'min', 'max', 'count']
    
    for entry in all_stats:
        row = {
            "subject_id": entry["subject_id"],
            "gen_folder": entry["gen_folder"],
            "contrast_number": entry["contrast_number"],
        }
        
        for label in sorted_labels:
            label_stats = entry["stats"].get(label, {})
            for stat_name in stat_names:
                col_name = f"label_{label}_{stat_name}"
                row[col_name] = label_stats.get(stat_name, None)
        
        output_rows.append(row)
    
    df_out = pd.DataFrame(output_rows)
    
    # Save CSV
    pattern_name = args.image_pattern.rstrip('_')
    csv_path = op.join(args.out_dir, f"intensities_{pattern_name}_{args.mask_column}.csv")
    df_out.to_csv(csv_path, index=False)

    logger.info("%sResults saved%s", GREEN, RESET)
    logger.info("Path: %s", csv_path)
    logger.info("Shape: %d rows x %d labels", len(df_out), len(sorted_labels))


if __name__ == "__main__":
    main(parse_args())