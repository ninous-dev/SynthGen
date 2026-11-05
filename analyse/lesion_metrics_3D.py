import argparse
import fnmatch
import logging
import os
import os.path as op
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torchio as tio
from scipy import ndimage
from utils.logger import BLUE, GREEN, RESET, YELLOW, RED, configure_logging


logger = logging.getLogger(__name__)
configure_logging(dev_mode=True, module_name=__name__)


@dataclass
class Args:
    generate_dir: str
    pred_dir: Optional[str]
    pred_pattern: str
    out_dir: str
    model_num: str
    modality_name: str
    connectivity: int
    min_lesion_size: int
    compute_dice: bool
    pred_is_binary: bool
    pred_lesion_label: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate lesion segmentation on generated synthetic subjects."
    )
    parser.add_argument(
        "--generate_dir",
        required=True,
        help="Path to the generate_{label_set}_{tmp} output directory.",
    )
    parser.add_argument(
        "--pred_dir",
        default=None,
        help="Directory containing model predictions (optional).",
    )
    parser.add_argument(
        "--pred_pattern",
        default="*{subject_name}*.nii.gz",
        help=(
            "Glob pattern to match prediction files to subjects. "
            "Supports {subject_name}, {label_set}, {subject_id}."
        ),
    )
    parser.add_argument(
        "--out_dir",
        default=".",
        help="Output directory for metrics files.",
    )
    parser.add_argument(
        "--model_num",
        default="model",
        help="Model identifier used in output filenames.",
    )
    parser.add_argument(
        "--modality_name",
        default="T1",
        help="Modality identifier used in output filenames.",
    )
    parser.add_argument(
        "--connectivity",
        default=3,
        type=int,
        choices=[1, 2, 3],
        help="Connectivity for lesion labeling: 1=6-conn, 2=18-conn, 3=26-conn.",
    )
    parser.add_argument(
        "--min_lesion_size",
        default=0,
        type=int,
        help="Minimum lesion size in voxels (smaller lesions are ignored).",
    )
    parser.add_argument(
        "--compute_dice",
        action="store_true",
        help="Also compute the global Dice score.",
    )
    parser.add_argument(
        "--pred_is_binary",
        action="store_true",
        help="Predictions are already binary masks.",
    )
    parser.add_argument(
        "--pred_lesion_label",
        default=None,
        type=int,
        help="If predictions are multi-label, value of the lesion class.",
    )
    return parser.parse_args()


def parse_subject_name(subject_name: str) -> Dict[str, str]:
    """
    Parse a generated subject name like 'MIDA_Suj_01'.

    Returns
    -------
    dict
        {'label_set': 'MIDA', 'subject_id': '01'}
    """
    m = re.match(r"(.+)_Suj_(\d+)", subject_name)
    if m:
        return {"label_set": m.group(1), "subject_id": m.group(2)}
    return {"label_set": subject_name, "subject_id": ""}


def find_generated_subjects(generate_dir: str) -> List[Tuple[str, str]]:
    """
    Find all generated subjects with an anomaly map.

    Returns
    -------
    list of (subject_name, amap_path)
    """
    amaps_dir = op.join(generate_dir, "anomaly_maps")
    if not op.isdir(amaps_dir):
        raise FileNotFoundError(
            f"No 'anomaly_maps' directory found in {generate_dir}. "
            "Did the generator run with anomaly transforms?"
        )

    subjects = []
    for fname in sorted(os.listdir(amaps_dir)):
        if fname.startswith("Amap_") and fname.endswith(".nii.gz"):
            subject_name = fname[len("Amap_") : -len(".nii.gz")]
            subjects.append((subject_name, op.join(amaps_dir, fname)))

    if not subjects:
        raise ValueError(f"No anomaly maps (Amap_*.nii.gz) found in {amaps_dir}")

    logger.info("Found %s%d%s generated subjects with anomaly maps", BLUE, len(subjects), RESET)
    return subjects


def find_prediction(
    subject_name: str, pred_dir: str, pred_pattern: str
) -> Optional[str]:
    """
    Find the prediction file matching a subject.
    """
    if pred_dir is None:
        return None

    info = parse_subject_name(subject_name)
    label_set = info["label_set"]
    subject_id = info["subject_id"]

    # Try with substitutions first, then fall back to raw pattern
    patterns = [
        pred_pattern.format(
            subject_name=subject_name,
            label_set=label_set,
            subject_id=subject_id,
        ),
        pred_pattern,
    ]

    candidates = []
    for pattern in patterns:
        for fname in os.listdir(pred_dir):
            if fnmatch.fnmatch(fname, pattern):
                candidates.append(op.join(pred_dir, fname))
        if candidates:
            break

    if len(candidates) == 0:
        logger.warning(
            "%sNo prediction found for %s with pattern '%s'%s",
            YELLOW,
            subject_name,
            pred_pattern,
            RESET,
        )
        return None

    if len(candidates) > 1:
        logger.warning(
            "%sMultiple predictions for %s, using first: %s%s",
            YELLOW,
            subject_name,
            candidates[0],
            RESET,
        )

    return candidates[0]


def load_prediction_as_binary(
    pred_path: str,
    pred_is_binary: bool,
    pred_lesion_label: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a prediction image and return a binary mask.

    Returns
    -------
    tuple
        (binary_data, affine)
    """
    if pred_is_binary:
        img = tio.ScalarImage(pred_path)
    else:
        img = tio.LabelMap(pred_path).data.squeeze(0).numpy().astype(int)

    if labels is None:
        all_labels = np.unique(np.concatenate([np.unique(gt_arr), np.unique(pr_arr)]))
        labels = [int(x) for x in all_labels]

    cm = confusion_matrix(gt_arr.ravel(), pr_arr.ravel(), labels=labels)
    return cm, labels

# ----
# Main
# ----
def run(
    gt_dir: str,
    pred_dir: str,
    out_path,
    connectivity: int = 3,
) -> None:
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)

    out_file = Path(out_path)

    gt_files = sorted(gt_dir.glob("*.nii.gz"))
    if not gt_files:
        print(f"[empty] {gt_dir}")
        return

    total_tp, total_gt = 0, 0
    dice_global_list = []
    lines = []

    header = f"{'Subject':20} {'GT':>6} {'PRED':>6} {'TP':>6} {'FN':>6} {'Recall':>8} {'DiceGlob':>10} {'PredLabels'}"
    lines.append(header)
    lines.append("-" * len(header))

    for gt_path in gt_files:
        stem = gt_path.stem.replace(".nii", "")
        matches = glob.glob(str(pred_dir / f"*{stem}.nii.gz"))
        if not matches:
            logger.info(f"{YELLOW}[skip]{RESET} no pred found for {stem}")
            continue

        pred_path = Path(matches[0])

        # Full label maps
        gt_arr_full = tio.LabelMap(gt_path).data.squeeze(0).numpy().astype(int)
        pr_arr_full = tio.LabelMap(pred_path).data.squeeze(0).numpy().astype(int)

        # Binary masks for metrics
        gt_mask = gt_arr_full > 0
        pr_mask = pr_arr_full == 17

        # Recall lesion-wise
        recall, tp, fn, nb_pred = lesion_recall_any_overlap(gt_mask, pr_mask, connectivity)
        nb_gt = tp + fn
        total_tp += tp
        total_gt += nb_gt

        # Dice
        d_glob = dice_coefficient(gt_mask, pr_mask)
        dice_global_list.append(d_glob)

        # Dictionary of predicted labels (counts)
        # garder seulement les voxels prédits là où GT > 0
        pred_in_gt = pr_arr_full[gt_mask]

        pred_counts = dict(Counter(pred_in_gt.ravel()))
        # on enlève 0 (fond) si tu veux
        if 0 in pred_counts:
            del pred_counts[0]

        pred_counts = dict(sorted(pred_counts.items()))

        line = f"{stem:20} {nb_gt:6} {nb_pred:6} {tp:6} {fn:6} {recall:8.3f} {d_glob:10.3f} {pred_counts}"
        lines.append(line)
        log = f"{stem}: GT={nb_gt:4} | PRED={nb_pred:4} | TP={tp:4} | FN={fn:4} | Recall={recall:.3f} | DiceGlobal={d_glob:.3f} | {pred_counts}"
        logger.info(log)

    # Summary dataset
    micro_recall = (total_tp / total_gt) if total_gt > 0 else float("nan")
    mean_dice_global = float(np.mean(dice_global_list)) if dice_global_list else float("nan")

    logger.info("\n=== SUMMARY ===")
    logger.info(f"subjects: {len(gt_files)}")
    logger.info(f"total_gt_lesions: {total_gt}")
    logger.info(f"total_tp_lesions: {total_tp}")
    logger.info(f"ratio SEP detected global: {micro_recall:.4f}" if not np.isnan(micro_recall) else "micro_recall: NaN")
    logger.info(f"mean_dice_global: {mean_dice_global:.4f}" if not np.isnan(mean_dice_global) else "mean_dice_global: NaN")
    
    lines.append("\n=== SUMMARY ===")
    lines.append(f"{'subjects':20} {len(gt_files)}")
    lines.append(f"{'total_gt_lesions':20} {total_gt}")
    lines.append(f"{'total_tp_lesions':20} {total_tp}")
    lines.append(f"{'micro_recall':20} {micro_recall:.4f}" if not np.isnan(micro_recall) else "micro_recall: NaN")
    lines.append(f"{'mean_dice_global':20} {mean_dice_global:.4f}" if not np.isnan(mean_dice_global) else "mean_dice_global: NaN")

    # Sauvegarde unique
    out_file.write_text("\n".join(lines))
    logger.info(f"Saved summary to {out_file}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lesion evaluation")
    parser.add_argument("--gt_dir", required=True, help="Dossier GT (.nii.gz)")
    parser.add_argument("--pred_dir", required=True, help="Dossier PRED (.nii.gz), match par motif *{stem}.nii.gz")
    parser.add_argument("--connectivity", type=int, default=3, choices=[1, 2, 3], help="1=6N, 2=18N, 3=26N (default)")
    parser.add_argument("--model_num", required=True) # TODO help
    parser.add_argument("--modality_name", required=True) # TODO help
    parser.add_argument("--out_dir") # TODO help
    
    # TODO never used
    # parser.add_argument('--output_dir', required=True)
    # parser.add_argument('--model', required=True)
    # parser.add_argument('--dataset', required=True)
    # parser.add_argument('--output_mask_dir', required=True)
    # parser.add_argument('--modality', required=True)
    # parser.add_argument('--err_dir', required=True)
    # parser.add_argument('--iou_threshold', type=float, default=0.0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_path=f"metrics_summary_{args.model_num}_{args.modality_name}"

    if args.out_dir:
        out_path = op.join(args.out_dir, out_path)
    run(
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        connectivity=args.connectivity,
        out_path=out_path
    )