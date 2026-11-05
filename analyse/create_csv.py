import argparse
import os
import os.path as op
import glob
import logging
import re

import pandas as pd

from utils.logger import BLUE, GREEN, ORANGE, RED, RESET, configure_logging

logger = logging.getLogger(__name__)
configure_logging(dev_mode=True, module_name=__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create CSV with paths from generate_mida_tmp* folders."
    )
    parser.add_argument(
        "--root_dir",
        required=True,
        help="Root directory containing generate_mida_tmp* folders.",
    )
    parser.add_argument(
        "--out_csv",
        default="subjects_paths.csv",
        help="Output CSV file.",
    )
    return parser.parse_args()


def find_generate_folders(root_dir: str):
    """Find all generate_mida_tmp* folders."""
    pattern = op.join(root_dir, "generate_mida_tmp*")
    folders = sorted(glob.glob(pattern))
    
    if not folders:
        raise FileNotFoundError(f"No generate_mida_tmp* folders found in {root_dir}")
    
    logger.info("Found %s%d%s generate folders", BLUE, len(folders), RESET)
    return folders


def extract_subject_id(fname: str) -> str:
    """Extract subject_id from filename - handles different formats."""
    # Cherche le pattern _mida_Suj_XX avec un numéro à deux chiffres
    match = re.search(r"_mida_Suj_(\d+)", fname)
    if match:
        subject_num = match.group(1)
        return f"mida_Suj_{subject_num}"
    return None


def get_file_type(fname: str) -> str:
    """Determine file type from filename."""
    if fname.startswith("Amap_"):
        return "amap"
    elif fname.startswith("Lab_") and not fname.startswith("LowRes"):
        return "lab"
    elif fname.startswith("HRLab_"):
        return "hrlab"
    elif fname.startswith("NoLes_lab_"):
        return "noleslab"
    elif fname.startswith("LowResNoLes_lab_"):
        return "lowresnoles"
    elif fname.startswith("Sim_") and "Contrast" in fname:
        match = re.search(r"Contrast(\d+)", fname)
        return f"sim_contrast_{match.group(1)}" if match else None
    elif fname.startswith("Ano_") and fname.endswith(".nii.gz"):
        return "ano"
    elif fname.startswith("4DLab_"):
        return "4dlab"
    elif fname.startswith("Original_"):
        return "original"
    elif fname.endswith(".csv"):
        return "csv"
    
    return None


def find_subject_files(generate_folder: str):
    """Find all files for each subject."""
    subjects = {}
    
    # Scan anomaly_maps directory first
    amaps_dir = op.join(generate_folder, "anomaly_maps")
    if op.isdir(amaps_dir):
        logger.info("  Scanning anomaly_maps/")
        for fname in os.listdir(amaps_dir):
            # Ignore lock files
            if fname.endswith(".lock"):
                continue
            
            if not fname.endswith(".nii.gz"):
                continue
                
            subject_id = extract_subject_id(fname)
            file_type = get_file_type(fname)
            
            if subject_id and file_type:
                if subject_id not in subjects:
                    subjects[subject_id] = {}
                subjects[subject_id][file_type] = op.join(amaps_dir, fname)
    
    # Scan root generate folder
    logger.info("  Scanning root folder")
    for fname in os.listdir(generate_folder):
        full_path = op.join(generate_folder, fname)
        
        # Skip directories
        if op.isdir(full_path):
            continue
        
        # Skip lock files
        if fname.endswith(".lock"):
            continue
        
        subject_id = extract_subject_id(fname)
        file_type = get_file_type(fname)
        
        print(f"{RED} Found file {RESET}: {fname}, subject_id: {subject_id}, file_type: {ORANGE}{file_type}{RESET}")
        
        if subject_id and file_type:
            if subject_id not in subjects:
                subjects[subject_id] = {}
            subjects[subject_id][file_type] = full_path
    
    return subjects


def main(args: argparse.Namespace):
    generate_folders = find_generate_folders(args.root_dir)
    
    all_results = []
    
    for gen_folder in generate_folders:
        gen_name = op.basename(gen_folder)
        logger.info("Processing %s%s%s", BLUE, gen_name, RESET)
        
        subjects = find_subject_files(gen_folder)
        logger.info("  Found %s%d%s subjects", BLUE, len(subjects), RESET)
        
        for subject_id, files in subjects.items():
            row = {"subject_id": subject_id, "gen_folder": gen_name}
            row.update(files)
            all_results.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    df = df.fillna("")
    
    # Sort by subject_id and gen_folder
    df = df.sort_values(["subject_id", "gen_folder"]).reset_index(drop=True)
    
    df.to_csv(args.out_csv, index=False)
    
    logger.info("%sResults saved to %s%s%s", GREEN, BLUE, args.out_csv, RESET)
    logger.info("Total entries: %d", len(df))
    logger.info("Columns: %s", ", ".join(df.columns.tolist()))


if __name__ == "__main__":
    main(parse_args())