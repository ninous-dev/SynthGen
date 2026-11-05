import argparse
import logging
import os
import os.path as op

import pandas as pd
import torchio as tio
from utils.logger import BLUE, GREEN, RESET, configure_logging

logger = logging.getLogger(__name__)
configure_logging(dev_mode=True, module_name=__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute volume per label from CSV with subject_id and file paths."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="CSV file with columns: subject_id, gen_folder, and file columns",
    )
    parser.add_argument(
        "--column",
        required=True,
        help="Column name to analyze (lab, amap, hrlab, etc).",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for CSV file.",
    )
    parser.add_argument(
        "--in_mm3",
        action="store_true",
        help="Compute volume in mm³ instead of voxels.",
    )
    return parser.parse_args()


def compute_label_volumes(img: tio.LabelMap, in_mm3: bool = False) -> dict:
    """Compute volume per label using torchio spacing."""
    data = img.data.squeeze()
    
    if in_mm3:
        voxel_volume = img.spacing[0] * img.spacing[1] * img.spacing[2]
    else:
        voxel_volume = 1.0
    
    unique_labels = data.unique()
    volumes = {}
    
    for label in unique_labels:
        if label == 0:
            continue
        n_voxels = (data == label).sum().item()
        volume = n_voxels * voxel_volume
        volumes[int(label)] = float(volume)
    
    return volumes


def main(args: argparse.Namespace):
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Read CSV
    df = pd.read_csv(args.csv)
    
    # Check columns
    if "subject_id" not in df.columns:
        raise ValueError("CSV must have 'subject_id' column")
    
    if "gen_folder" not in df.columns:
        raise ValueError("CSV must have 'gen_folder' column")
    
    if args.column not in df.columns:
        raise ValueError(f"CSV must have '{args.column}' column")
    
    logger.info("Found %s%d%s subjects", BLUE, len(df), RESET)
    logger.info("Using column: %s%s%s", BLUE, args.column, RESET)
    
    # Collect all volumes and find all labels
    all_volumes = []
    all_labels = set()
    
    for idx, row in df.iterrows():
        subject_id = row["subject_id"]
        gen_folder = row["gen_folder"]
        path = row[args.column]
        
        logger.info("[%d/%d] Processing %s%s%s", idx + 1, len(df), BLUE, subject_id, RESET)
        
        # Skip if path is empty
        if pd.isna(path) or path == "":
            logger.warning("  No path for %s", subject_id)
            continue
        
        if not op.exists(path):
            logger.warning("  File not found: %s", path)
            continue
        
        try:
            img = tio.LabelMap(path)
            volumes = compute_label_volumes(img, in_mm3=args.in_mm3)
            
            # Collect all labels found
            all_labels.update(volumes.keys())
            
            # Add to results
            all_volumes.append({
                "subject_id": subject_id,
                "gen_folder": gen_folder,
                "volumes": volumes,
            })
        except Exception as e:
            logger.error("  Error processing %s: %s", subject_id, str(e))
            continue
    
    # Sort labels
    sorted_labels = sorted(list(all_labels))
    logger.info("Found labels: %s", sorted_labels)
    
    # Build output dataframe
    output_rows = []
    for entry in all_volumes:
        row = {
            "subject_id": entry["subject_id"],
            "gen_folder": entry["gen_folder"],
        }
        
        # Add volumes for each label
        for label in sorted_labels:
            col_name = f"label_{label}"
            row[col_name] = entry["volumes"].get(label, None)
        
        output_rows.append(row)
    
    df_out = pd.DataFrame(output_rows)
    
    # Save CSV
    unit = "mm3" if args.in_mm3 else "voxels"
    csv_path = op.join(args.out_dir, f"volumes_{args.column}_{unit}.csv")
    df_out.to_csv(csv_path, index=False)

    logger.info("%sResults saved to %s%s (%s)%s", GREEN, RESET, BLUE, csv_path, RESET)
    logger.info("Shape: %d subjects x %d labels", len(df_out), len(sorted_labels))
    print("\n" + df_out.to_string())


if __name__ == "__main__":
    main(parse_args())