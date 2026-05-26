import argparse
import ast
import configparser
import re
from distutils.util import strtobool
import csv
import logging
import os
import os.path as op
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import commentjson
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torchio as tio
from filelock import FileLock, Timeout
import fnmatch

# Local imports
import transforms_class as tf
from transforms_class import (
    OneHotTransform,
)
from transforms_class import RandomMorphologyTransform, RandomLabelsToImageOneConstraint, LabelTransform
from utils.json import ArrayTensorJSONEncoder
from utils.logger import BLUE, GREEN, RESET, YELLOW, RED, configure_logging

# Configure logging
logger = logging.getLogger(__name__)
configure_logging(dev_mode=True, module_name=__name__)
# -----------------------------------------------------------------------------
# Data types
# -----------------------------------------------------------------------------
@dataclass
class Args:
    """
    Command line arguments for synthetic subject generation.
    """

    dataset: int
    nb_gen: int
    outputdir: str
    config_filename: str


@dataclass
class DatasetCfg:
    """
    Configuration for a specific dataset/label set.
    """

    subjdir: List[str]


@dataclass
class Config:
    """
    Configuration for the synthetic subject generation.
    """

    transforms: Dict[str, str]
    dataset: Dict[int, DatasetCfg]


# -----------------------------------------------------------------------------
# Args parser
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for synthetic subject generation.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing:
        - dataset (int): labelset dataset.
        - nb_gen (int): number of generated images per subject
        - outputdir (str): directory where transform JSON files are stored
        - config_filename (str): config file name

    """
    parser = argparse.ArgumentParser(description="Batch generate synthetic subjects.")
    parser.add_argument(
        "-d",
        "--dataset",
        help="Dataset IDs corresponding to ids in the config file, default=1",
        default=1,
        type=int,
        required=False,
    )
    
    parser.add_argument(
        "-a",
        "--anomaly_type",
        help="Anomaly type IDs corresponding to ids in the config file, default=1",
        default=None,
        type=int,
        required=False,
    )

    parser.add_argument(
        "-n",
        "--nb_gen",
        help="number of generated images per subject (default=1)",
        default=1,
        type=int,
        required=False,
    )

    parser.add_argument(
        "-o",
        "--outputdir",
        help="output directory where transform JSON files are stored",
        default=".",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-c",
        "--config_filename",
        help="Config filename, default=configs.ini",
        default="configs.ini",
        type=str,
        required=False,
    )

    parser.add_argument(
        "-s",
        "--subject_idx",
        help="Index (1-based) of the subject to generate (optional)",
        default=None,
        type=int,
        required=False,
    )

    parser.add_argument(
        "-z",
        "--nb_contrast",
        help="number of ouput contrast per subject (default 1)",
        default=1,
        type=int,
        required=False,
    )

    parser.add_argument(
        "--get_label_only",
        help="If set, only generate labels without synthetic images",
        action="store_true",
    )
    return parser.parse_args()

def parse_bool(val, default='False'):

        """
        Parse a value into a boolean using distutils.util.strtobool.

        Parameters
        ----------
        val : Any
            The value to parse as a boolean. Can be a string, int, bool, etc.
        default : str or bool, optional
            The default value to use if val is None, default='False'.

        Returns
        -------
        bool
            The parsed boolean value.
        """
        return bool(strtobool(str(val if val is not None else default)))

# -----------------------------------------------------------------------------
# Configs
# -----------------------------------------------------------------------------

def parse_config_list(raw: str) -> List[str]:
    """
    Parse a configuration string into a list of strings.

    This utility handles various config file formats:
    - Python list-like strings (e.g. '["a", "b"]')
    - Multi-line strings (one item per line)
    - Comma-separated values (single line)

    Parameters
    ----------
    raw : str
        The raw string from the config file to parse.

    Returns
    -------
    List[str]
        The parsed list of strings, with whitespace stripped.

    """
    if raw.strip().startswith("["):
        res_list = list(ast.literal_eval(raw))
    else:
        res_list = [p.strip() for p in raw.replace("\r", "").splitlines() if p.strip()]
        if len(res_list) == 1 and "," in raw:
            res_list = [p.strip() for p in raw.split(",") if p.strip()]
    return res_list


def load_config(
    config_dir: str, label_set_num: int, anomalies_num: int, filename: str = "configs.ini"
) -> Config:
    """
    Load the configuration for a given label set and transforms from a config file.

    This function reads a config file (INI format) and loads:
    - The transforms section (JSON definitions for each transform pipeline)
    - The dataset section for the specified label set (paths, regex, options)

    Parameters
    ----------
    config_dir : str
        Directory containing the config file and referenced transform files.

    label_set_num : int
        The label set/dataset number to load.

    anomalies_num : int
        The number of anomalies to load.

    filename : str, optional
        Name of the config file, default="configs.ini".

    Returns
    -------
    Config
        A Config dataclass with:
        - transforms: dict of transform pipelines (TorchIO Compose objects)
        - dataset: dict of dataset configuration parameters for the label set

    Raises
    ------
    ValueError
        If the specified dataset section does not exist in the config file.

    """
    cfgp = configparser.ConfigParser()
    
    # Use absolute path directly, otherwise prepend config directory
    if op.isabs(filename):
        config_file = filename
    else:
        config_file = op.join(config_dir, filename)

    if not op.isfile(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    logger.info("Loading configuration from %s%s%s", BLUE, config_file, RESET)
    cfgp.read(config_file)

    # Transforms config
    transforms = {}
    if "transforms" in cfgp:
        for key, path in cfgp["transforms"].items():
            transforms[key] = get_transform_from_json(path)

        if anomalies_num is not None and anomalies_num != 0:
            if cfgp.has_section("anomalies"):
                json_file = cfgp["anomalies"][str(anomalies_num)]
                transforms["anomalies"] = get_transform_from_json(json_file)
                logger.info("%s[Anomalies]%s Loaded anomaly %s from %s%s%s", BLUE, RESET, anomalies_num, BLUE, json_file, RESET)
            else:
                raise ValueError(f"Config file must have an [anomalies] section with keys for each anomaly type, but got {cfgp.sections()}")

            if transforms["anomalies"] is None or len(transforms["anomalies"].transforms) == 0:
                raise ValueError(f"Anomalies transform for anomaly type {anomalies_num} is empty or not found in config.")

    # Dataset config
    section = f"label_set.dataset{label_set_num}"
    if not cfgp.has_section(section):
        raise ValueError(
            f"Config for the dataset {label_set_num} must exists (section: {section}), but got {cfgp.sections()}"
        )

    model_config = cfgp[section]
    transforms["preprocessing"] = get_transform_from_json(model_config["preprocessing"])
    
    res_model_config = {
        "labels_csv": model_config["labels_csv"],
        "subject_dir": model_config["subject_dir"],
        "do_erode": model_config["do_erode"],
        "label_set_name": model_config["label_set_name"],
        "synth_col": model_config["synth_col"],
        "add_subregions": parse_bool(model_config.get("add_subregions", 'False')),
        "nb_pool": int(model_config.get("nb_pool", "3")),
        "pre_pool_after_preproc_dilate": int(model_config.get("pre_pool_after_preproc_dilate", "0")),
        "directory_path_add_subregions": model_config.get("directory_path_add_subregions", ""),
        "add_subregions_lab": parse_config_list(model_config.get("add_subregions_lab", "")),
    }
    


    return Config(transforms=transforms, dataset=res_model_config)


def get_label_set_config(label_set_num: int, dataset: Dict) -> Tuple[
    str,  # label_set_name
    List[str],  # subject_files
    Dict[str, int],  # label_name_to_value
    Dict[int, int],  # synth_label_to_tissue
    Dict[int, int],  # synth_label_to_target
]:
    """
    Return the configuration for a given label set.

    Loads label mapping, file lists, and label dictionaries for the specified label set/dataset,
    based on the config/dataset dict.

    Parameters
    ----------
    label_set_num : int
        The number identifying the label set (e.g. 1, 2, 3, 4).
    dataset : Dict
        The dataset configuration dictionary (from config.dataset).

    Returns
    -------
    Tuple[str, List[str], Dict[str, int], Dict[int, int], Dict[int, int], Dict[str, List[str]]]
        label_set_name : str
            Name of the label set/dataset
        subject_files : List[str]
            List of subject file paths
        label_name_to_value : Dict[str, int]
            Mapping from label names to label values
        synth_label_to_tissue : Dict[int, int]
            Mapping from synthetic label to tissue class
        synth_label_to_target : Dict[int, int]
            Mapping from synthetic label to target label

    """
    # ---- Extract dataset parameters ----
    #remove nan, due to empty element, for allen and bigbrain, where we add in the
    #NametargetRegV4 missing labels (skull head vasc dura), so we need to skep them
    # when reading other mapping
    def isnotNaN(num):
        return num == num

    label_set_name = dataset["label_set_name"]
    synth_col = dataset["synth_col"]

    logger.info("Loading label set %s%s%s from %s%s%s", BLUE, label_set_name, RESET, BLUE, dataset["labels_csv"], RESET)
    label_name_column = "Name"
    preprocessing_target_map = None
    df = pd.read_csv(dataset["labels_csv"], comment="#")

    label_name_to_value = {ll["Name"]: int(ll[synth_col]) for _, ll in df.iterrows() if not pd.isna(ll[synth_col])}
    synth_label_to_tissue = {int(ll[synth_col]): ll["synth_tissu"] for _, ll in df.iterrows()if not pd.isna(ll[synth_col])}
    
    #TODO make it specific in the config not in the code
    if label_set_num in [1, 3, 4]: # Case MIDA, Skull and BigBrain datasets
        synth_label_to_target = {int(ll[synth_col]): int(ll[synth_col]) for _, ll in df.iterrows()if not pd.isna(ll[synth_col])}
    elif label_set_num in [2, 5, 6]: # Case Vascular and Allen Brain datasets
        synth_label_to_target = {int(ll[synth_col]): int(ll['synth_target']) for _, ll in df.iterrows() if not pd.isna(ll['synth_target'])}
        preprocessing_target_map_name = {ll['NameTargetRegion']: int(ll['synth_targetRegion']) for _, ll in df.iterrows() if not pd.isna(ll['synth_targetRegion'])} # if not nan in synth_targetRegion
        preprocessing_target_map_name = dict(sorted(preprocessing_target_map_name.items(), key=lambda item: item[1]))
        preprocessing_target_map_to_synth = {int(ll[synth_col]): int(ll['synth_targetRegion']) for _, ll in df.iterrows() if not pd.isna(ll['synth_targetRegion'])}   
        preprocessing_target_map = (preprocessing_target_map_name, preprocessing_target_map_to_synth)
    else:
        raise ValueError(f"Label set {label_set_num} not implemented")

    # Link the names of tissue groups to the names of the regions that compose these tissue groups
    label_name_few_to_label_name_all = (
        df.groupby("Stratified_Tissue_Region")[label_name_column]
        .apply(set)
        .apply(list)
        .dropna()
        .to_dict()
    )

    # --- Get files ---
    logger.info("Getting subject files from %s%s%s", BLUE, dataset["subject_dir"], RESET)
    subject_files = [
        os.path.join(dataset["subject_dir"], file)
        for file in os.listdir(dataset["subject_dir"])
        if file.endswith(".nii.gz")
    ]
    if len(subject_files) == 0:
        raise ValueError(f"No subject files found in {dataset['subject_dir']}")


    # --- Return all args ---
    args_files_maps = (
        label_set_name,
        subject_files,
        label_name_to_value,
        synth_label_to_tissue,
        synth_label_to_target,
        label_name_few_to_label_name_all,
        preprocessing_target_map
    )
    return args_files_maps


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def get_transfo_short_name(info_dict: dict, order_str: str) -> str:
    """Generate a short code from a transform order string."""
    order = info_dict.get(order_str, "")
    parts = order.split("_")
    return "".join(p[:3] for p in parts)


def add_prefix_to_filenames(
    file_names: Union[str, Sequence[str]], prefix: str
) -> List[str]:
    """
    Prepend a prefix to the basename of each file path.
    """
    paths = [file_names] if isinstance(file_names, str) else list(file_names)
    return [op.join(op.dirname(p), f"{prefix}{op.basename(p)}") for p in paths]


def rescale_affine_corrected(
    affine: np.ndarray,
    shape: Sequence[int],
    zooms: Sequence[float],
    new_shape: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    Compute a new affine for resampled image to preserve centering.
    """
    shape = np.array(shape)
    new_shape = np.array(new_shape)

    # Compute the original voxel spacing from the affine
    old_spacing = nib.affines.voxel_sizes(affine)
    # Compute the new scaling matrix for the affine
    scale = affine[:3, :3] * (zooms / old_spacing)

    # Compute the center of the old image in world coordinates
    center_old = nib.affines.apply_affine(affine, (shape - 1) / 2)
    # Compute the center of the new image in world coordinates (using new scale)
    center_new = scale @ ((new_shape - 1) / 2)
    # Compute the translation needed to keep the image centered
    trans = center_old - center_new

    # Return the new affine matrix with updated scale and translation
    return nib.affines.from_matvec(scale, trans)


# -----------------------------------------------------------------------------
# Pool and Resample
# -----------------------------------------------------------------------------


def pool_remap(
    image: Union[tio.LabelMap, tio.ScalarImage],
    pooling_size: int = 2,
    ensure_multiple: Optional[int] = None,
    transform_map: Optional[tio.Transform] = None,
    keep_missing_label: bool = True,
    is_label: bool = True,
) -> Union[Tuple[tio.LabelMap, tio.LabelMap], tio.ScalarImage]:
    """
    Downsample (pool) a label map or scalar image, with optional one-hot remapping for labels.

    For label maps, returns both a pooled binary label map and a multi-channel label map.
    For scalar images, returns the pooled image.

    Parameters
    ----------
    image : tio.LabelMap or tio.ScalarImage
        The input image to downsample (label or scalar).

    pooling_size : int, optional
        The size of the pooling kernel, by default=2.

    ensure_multiple : int, optional
        If set, pad the image to ensure its shape is a multiple of this value, by default=None.

    transform_map : tio.Transform, optional
        An optional transform to apply before pooling, by default=None.

    keep_missing_label : bool, optional
        If True, keep all possible label channels up to the max label, by default=True.

    is_label : bool, optional
        If True, treat the input as a label map; otherwise as a scalar image, by default=True.

    Returns
    -------
    tuple of (tio.LabelMap, tio.LabelMap)
        If is_label is True: (binary_labels, multi_channel_labels)

    tio.ScalarImage
        If is_label is False: the pooled scalar image.

    """
    thot_inv = OneHotTransform(invert_transform=True)
    pool = torch.nn.AvgPool3d(kernel_size=pooling_size, ceil_mode=True)
    tpad = tio.EnsureShapeMultiple(ensure_multiple or pooling_size)

    # Apply optional pre-transform
    image = tpad(transform_map(image)) if transform_map else tpad(image)

    # Get the new affine
    original_shape = np.array(image.shape[1:])
    new_shape = original_shape // pooling_size
    new_voxel_size = nib.affines.voxel_sizes(image.affine) * pooling_size
    new_affine = rescale_affine_corrected(
        image.affine, original_shape, new_voxel_size, new_shape
    )

    if is_label:

        labels = image.data.squeeze(0).int()
        label_values = labels.unique()
        nb_channel = (
            int(label_values.max()) + 1 if keep_missing_label else len(label_values)
        )
        binary_map = torch.zeros([nb_channel, *new_shape])

        # For each label/channel, create a binary mask and pool it
        for channel in range(nb_channel):
            mask = labels == (channel if keep_missing_label else label_values[channel])
            binary_map[channel] = pool(mask.float().unsqueeze(0))[0]

        label_image = tio.LabelMap(tensor=binary_map, affine=new_affine)
        binary = thot_inv(label_image)

        return binary, label_image
    else:
        img = image.data.float().unsqueeze(0)
        down = pool(img)[0]
        return tio.ScalarImage(tensor=down, affine=new_affine)


# -----------------------------------------------------------------------------
# History Recording
# -----------------------------------------------------------------------------


def record_history(
    sample: tio.Subject, idx: Optional[int] = None
) -> Dict[str, Union[List, str]]:
    """
    Extract TorchIO transform history and metrics into a dict.
    """
    infos: Dict[str, Union[List, str]] = {}

    # Get history
    is_batch = not isinstance(sample, tio.Subject)
    history = sample.get("history", []) if is_batch else sample.history
    if not history:
        return {}

    metrics = (
        sample.transforms_metrics if is_batch else sample.get("transforms_metrics", [])
    )

    # Select single entry if batch
    relevant_history = history[idx] if is_batch else history
    relevant_metrics = metrics[idx] if is_batch else metrics

    # Flatten nested metric lists
    if len(relevant_metrics) == 1 and isinstance(relevant_metrics[0], list):
        relevant_metrics = relevant_metrics[0]

    infos["history"] = relevant_history
    infos["metrics"] = relevant_metrics

    # Build a reproducible dict of each transform's parameters
    order = []
    counts: Dict[str, int] = {}
    for entry in relevant_history:
        name = entry["name"] if isinstance(entry, dict) else entry.name
        params = {}
        items = (
            entry.items()
            if isinstance(entry, dict)
            else [(arg, getattr(entry, arg)) for arg in entry.args_names]
        )
        for k, v in items:
            params[k] = str(v) if callable(v) else v
        key = f"T_{name}"
        counts[key] = counts.get(key, 0) + 1
        suffix = f"_{counts[key]}" if counts[key] > 1 else ""
        infos[f"{key}{suffix}"] = commentjson.dumps(params, cls=ArrayTensorJSONEncoder)
        order.append(name)

    infos["order"] = "_".join(order)

    if "anomaly_type" in sample:
        infos["order"] += f'_Ano{sample["anomaly_type"]}'

    # Add image path in csv
    for image_key in sample.keys():
        if isinstance(sample[image_key], tio.data.image.LabelMap):
            # Needed because label CombinedLesionTransform
            # Add a str attribute to torchio subject
            infos[f"input_path_{image_key}"] = str(sample[image_key].path)
    return infos


# -----------------------------------------------------------------------------
# Transform Parsing
# -----------------------------------------------------------------------------
def get_transform_from_json(json_file: str) -> tio.Transform:
    """
    Load TorchIO Compose pipeline from a JSON definition.
    
    Parameters
    ----------
    json_file : str
        Path to the JSON file containing the transform definition.

    Returns
    -------
    tio.Transform
        The parsed TorchIO transform pipeline corresponding to the definition.
    """
    logger.info("Loading transform from %s%s%s", BLUE, json_file, RESET)
    with open(json_file, encoding="utf-8") as f:
        data = commentjson.load(f)
    return parse_transform(data["train_transforms"])


def parse_transform(defn: Union[Dict, List]) -> tio.Transform:
    """
    Recursively parse a transform definition (dict or list) into a TorchIO Compose pipeline.
    
    Parameters
    ----------
    defn : dict or list
        The transform definition, which can be:
        - A list of transform definitions (for Compose)
        - A dict with keys "name", "attributes", and optionally "transforms" for nested transforms.
    
    Returns
    -------
    tio.Transform
        The parsed TorchIO transform pipeline corresponding to the definition.
    """
    # Convert List to Compose
    if isinstance(defn, list):
        return tio.Compose([parse_transform(d) for d in defn])

    # Resolve class
    try:
        cls = getattr(tio.transforms, defn["name"])
    except AttributeError:
        cls = getattr(tf, defn["name"])

    name = defn["name"]
    attrs = defn.get("attributes", {}).copy()

    # Extract transforms
    children = None
    if "transforms" in defn:
        children = defn["transforms"]
    elif "transforms" in attrs:
        children = attrs.pop("transforms")

    # Case : Compose
    if name == "Compose":
        transforms = [parse_transform(t) for t in (children or [])]
        return cls(transforms, **attrs)

    # Case : Oneof
    if name == "OneOf":
        if children is None:
            raise ValueError("OneOf requires 'transforms'")

        # Weighted case
        if isinstance(children, list) and all("prob" in t for t in children):
            transform_dict = {
                parse_transform(t): t["prob"] for t in children
            }
            return cls(transform_dict, **attrs)

        # Uniform case
        transforms = [parse_transform(t) for t in children]
        return cls(transforms, **attrs)

    # Generic transform
    return cls(**attrs)

# -----------------------------------------------------------------------------
# CSV Mapping Utilities
# ----------------------------------------------------------------------------
def build_output_path(
    output_dir: str, output_path: str, prefix: str, suffix: str, short_name: str
) -> str:
    """
    Construct output filename with prefix, suffix, and short_name.
    """
    base = op.splitext(op.basename(output_path))[0].replace(".nii", "")
    name = base if base in short_name else base + short_name

    parts = [p for p in [prefix, name, suffix] if p]
    filename = "_".join(parts) + ".nii.gz"

    return op.join(output_dir, filename)


# -----------------------------------------------------------------------------
# Main Subject Generation
# -----------------------------------------------------------------------------
def add_subregions_labels(
    directory_path: str,
    add_subregions_lab: List[str],
    label_image: tio.LabelMap,
    name_map_current: Dict[str, int],
    preprocessing_target_map: Dict[str, int],
    name_col: str = "Name_continuous",
    label_col: str = "label_continuous",
):
    """
    Add subregion labels to a label image using external NIfTI and CSV files.

    For each subregion in `add_subregions_lab`, this function:
      - Loads the corresponding NIfTI and CSV files from `directory_path`.
      - Resolves target and source label IDs using the provided mapping dictionaries.
      - Applies a mask to restrict label assignment to the relevant region.
      - Remaps the subregion labels according to the CSV and inserts them into the output label image.
      - Checks for missing or overlapping labels and raises errors or warnings as appropriate.
      - Ensures that all target regions are covered, either by subregion assignment or by copying from the original label image.

    Parameters
    ----------
    directory_path : str
        Path to the directory containing subregion NIfTI and CSV files.
    add_subregions_lab : list of str
        List of subregion names to add.
    label_image : tio.LabelMap
        The base label image to which subregions will be added.
    name_map_current : dict
        Mapping from region names to target label IDs.
    preprocessing_target_map : dict
        Mapping from target label IDs to source label IDs.
    name_col : str, optional
        Name of the column in the CSV containing region names, by default="Name_continuous".
    label_col : str, optional
        Name of the column in the CSV containing label values, by default="label_continuous".

    Returns
    -------
    tio.LabelMap
        The label image with subregion labels added and checked for consistency.

    Raises
    ------
    RuntimeError
        If the expected NIfTI or CSV files are missing or ambiguous.
    ValueError
        If a mask is empty, if no value is assigned in the mask, or if overlaps are detected.
    """
    out_put_label_image = torch.zeros_like(label_image.data)
    target_not_in_subregion = list(preprocessing_target_map.keys())

    for lab in add_subregions_lab:
        logger.info(f"Processing subregion {BLUE}{lab}{RESET}")
        # File matching between parcel nii and parcel csv
        nii_matches = fnmatch.filter(os.listdir(directory_path), f"*{lab}*.nii.gz")
        csv_matches = fnmatch.filter(os.listdir(directory_path), f"*{lab}*.csv")

        if len(nii_matches) != 1 or len(csv_matches) != 1:
            raise RuntimeError(
                f"{lab}: expected 1 nii + 1 csv, got {nii_matches}, {csv_matches}"
            )

        # Load the label map and CSV for the subregion
        nii_path = os.path.join(directory_path, nii_matches[0])
        csv_path = os.path.join(directory_path, csv_matches[0])

        label_to_add_image = tio.LabelMap(nii_path)
        df_add = pd.read_csv(csv_path)

        # Resolve target IDs and source IDs
        target_ids = {name_map_current[name] for name in df_add[name_col]}
        source_ids = {preprocessing_target_map[target_key] for target_key in target_ids}
        for target_key in target_ids:
            target_not_in_subregion.remove(target_key)

        mask_add = torch.isin(label_image.data, torch.tensor(list(source_ids), device=label_image.data.device))
        
        # Check if the mask is empty
        if not torch.any(mask_add):
            raise ValueError(f"{lab}: mask is empty")
        
        label_to_add_image.data[~mask_add] = 0  # Set to zero outside the mask

        # Remap the label map to target IDs
        dict_remap = dict(zip(df_add[label_col], target_ids))
        label_map_image = tio.RemapLabels(dict_remap)(label_to_add_image)
        
        if torch.all(label_map_image.data[mask_add] == 0):
            raise ValueError(f"No assigned value in dilation mask for {lab}")

        # Add the subregion labels to the output label image
        out_put_label_image[mask_add] = label_map_image.data[mask_add]

        # Check for missing target IDs in the output label image
        unique_labels_in_image = torch.unique(out_put_label_image)
        missing_ids = [target_id for target_id in target_ids if target_id not in unique_labels_in_image]

        if missing_ids:
            # Name of the missing ids
            names_missing = [name for name, target_id in name_map_current.items() if target_id in missing_ids]
            logger.warning(
                f"{RED}{lab}: missing target IDs {missing_ids} Corresponding names {names_missing}{RESET}"
            ) #TODO must be a raise error, because it means that the subregion is not added at all, which is a problem for the generation
            
    # Add remaining target IDs that are not in subregions, and check for overlaps
    target_id_overlaps = []    
    for target_id in target_not_in_subregion:
        # Target ID in the preprocessed subject
        target_id_few = preprocessing_target_map[target_id]
        mask_target = label_image.data == target_id_few
        
        # Overlap with the added subregions
        overlap = out_put_label_image[mask_target]
        if torch.any(overlap != 0):
            target_id_overlaps.append((target_id, torch.unique(overlap)))
        out_put_label_image[mask_target] = target_id

    # Check for overlaps
    if target_id_overlaps:
        overlap_info = ";\n ".join(
            f"target ID {tid} overlaps with labels {olap.cpu().numpy()}"
            for tid, olap in target_id_overlaps
        )
        raise ValueError(f"Overlaps detected for target IDs not in subregions: {overlap_info}")

    # Check for continuous labels
    if not torch.all(torch.diff(torch.unique(out_put_label_image.data)) == 1):
        logger.warning(f"Non-continuous labels after adding subregions: {torch.unique(out_put_label_image.data)}")

    label_image.set_data(out_put_label_image)
    return label_image

def resolve_transform_anatomies(
    transform: tio.Transform,
    label_name_to_value: Dict[str, int],
    label_name_few_to_label_name_all: Dict[str, List[str]],
    synth_map: Optional[Dict[int, int]] = None
):
    """
    Recursively resolve anatomical names in transforms to label values using the provided name map and label groupings.

    Parameters
    ----------
    transform : tio.Transform
        The transform to resolve.
    label_name_to_value : Dict[str, int]
        Mapping from anatomical names to label values.
    label_name_few_to_label_name_all : Dict[str, List[str]]
        Mapping from few-shot label names to all corresponding label names.
    synth_map : Optional[Dict[int, int]]
        Optional mapping from label values to synthetic label values (used for remapping in synthesis).

    Returns
    -------
    tio.Transform
        The transform with resolved anatomical names.

    Raises
    ------
    ValueError
        If the transform is not a LabelTransform.
    """
    # ---- Case Compose ----
    if hasattr(transform, "transforms"):
        for t in transform.transforms:
            resolve_transform_anatomies(t, label_name_to_value, label_name_few_to_label_name_all, synth_map)
        return transform

    # ---- Case OneOf ----
    if hasattr(transform, "transforms_dict"):
        for t in transform.transforms_dict.keys():
            resolve_transform_anatomies(t, label_name_to_value, label_name_few_to_label_name_all, synth_map)
        return transform

    # ---- Case Transform ----

    if isinstance(transform, LabelTransform):
        if not isinstance(transform, RandomMorphologyTransform):
            # Remapping on the final tissue directly
            transform.resolve_anatomy(label_name_few_to_label_name_all)
        transform.resolve_anatomy(label_name_to_value)

    # ---- Case LabelToImageConstraint ----
    if isinstance(transform, RandomLabelsToImageOneConstraint):
        transform.resolve_anatomy(label_name_few_to_label_name_all)
        for key, values in transform.contrasted_labels.items():
            if isinstance(values, list) and "tumor_edema" in values:
                #Remove value that does not contains "edema" (because we want to contrast only the edema part if both are present)
                transform.contrasted_labels[key] = [v for v in values if "edema" in v]


        transform.resolve_anatomy(label_name_to_value)
        transform.resolve_anatomy(synth_map)

    return transform

def reduce_resolution(
    target_map: Dict[int, int],
    subject_spatial: tio.Subject,
    nb_pool: int
) -> Tuple[tio.Subject, tio.LabelMap]:
    # Remap target, only if needed
    if list(target_map.keys()) == list(target_map.values()):
        target_image = subject_spatial.label
    else:
        target_image = tio.RemapLabels(target_map)(subject_spatial.label)

    # Pool labels and remap
    # ---------------------
    label_bin, label_4d = pool_remap(target_image, pooling_size=nb_pool, ensure_multiple=nb_pool*2)
    subject_down = tio.Subject(label=label_4d, label_bin=label_bin)

    #down sample the label without lesion too
    if "__label_without_lesion" in subject_spatial:
        if list(target_map.keys()) == list(target_map.values()):
            target_image = subject_spatial.__label_without_lesion
        else:
            target_image = tio.RemapLabels(target_map)(subject_spatial.__label_without_lesion)
        label_bin_noles, _ = pool_remap(target_image, pooling_size=nb_pool, ensure_multiple=nb_pool*2)
        subject_down.add_image(label_bin_noles, "__label_without_lesionLowRes")


    # herite from extra attribute (used after in label_2_image
    if "anomalies_labels" in subject_spatial:
        subject_down["anomalies_labels"] = subject_spatial["anomalies_labels"]
    if "suffix" in subject_spatial:
        subject_down["suffix"] = subject_spatial["suffix"]


    return subject_down

def label_2_image(
    transforms: List[tio.Transform],
    subject_down: tio.Subject,
    synth_map: Dict[int, int],
    label_name_to_value: Dict[str, int],
) -> tio.Subject:
    """
    Apply label-to-image synthesis and pooling to generate synthetic MRI and label maps.

    Parameters
    ----------
    transforms : List[tio.Transform]
        TorchIO transforms to generate synthetic intensities from labels.
    subject_down : tio.Subject
        Subject after spatial transforms (input for synthesis and remapping).
    synth_map : Dict[int, int]
        Mapping from input label values to synthetic tissue classes (for intensity synthesis).
    label_name_to_value : Dict[str, int]
        Mapping from anatomical names (e.g. 'WM') to label values (used for contrast).

    Returns
    -------
    Tuple[tio.Subject, tio.Subject, tio.LabelMap, tio.LabelMap]
        subject_img : tio.Subject
            Subject with synthetic image (before pooling).
        subject_down : tio.Subject
            Subject with pooled synthetic image.
        label_bin : tio.LabelMap
            Pooled binary label map.
        label_4d : tio.LabelMap
            Pooled multi-channel label map.

    """

    # Label to image transform
    # ------------------------
    subject_synth = tio.RemapLabels(synth_map)(subject_down)
    subject_img = transforms(subject_synth)

    return subject_img

def setup_name_map_current(label_name_to_value, preprocessing_target_map, label_img):
    """
    Prepare the current name map, handling special case for vascular datasets (remap lGM/rGM to GM).
    Returns (name_map_current, label_img)
    """
    name_map_current = label_name_to_value.copy() if preprocessing_target_map is None else preprocessing_target_map[0].copy()
    if preprocessing_target_map is not None:
        # Special case vascular, remap left and right GM
        if "GM" not in name_map_current:
            tmap = tio.RemapLabels({name_map_current["lGM"]: 1, name_map_current["rGM"]: 1})
            label_img = tmap(label_img)  # Concatenate left and right GM
            name_map_current["GM"] = 1
    name_map_current = dict(sorted(name_map_current.items(), key=lambda item: item[1]))
    return name_map_current, label_img
    
def generate_one_subject(
    sujname: str,
    label_file: str,
    transforms: Dict[str, tio.Transform],
    label_name_to_value: Dict[str, int],
    synth_map: Dict[int, int],
    target_map: Dict[int, int],
    preprocessing_target_map: Dict[int, int],
    output_name: str,
    output_dir: str = ".",
    add_subregions: bool = True,
    directory_path_add_subregions: str = "",
    add_subregions_lab: List[str] = None,
    nb_pool: int = 3,
    pre_pool_after_preproc_dilate: int = 0,
    nb_out_contrast: int = 1,
    get_label_only: bool = False,
    label_name_few_to_label_name_all: Dict[str, str] = None
) -> tio.Subject:
    """
    Generate one synthetic subject (image and labels), apply transforms, and save outputs to disk.

    Parameters
    ----------
    sujname : str
        Name of the subject (used for logging).

    label_file : str
        Path to the input label map (.nii.gz).

    transforms : Dict[str, tio.Transform]
        Dictionary of transforms (keys: 'anomalies', 'spatial', 'label_to_image', 'intensity').

    label_name_to_value : Dict[str, int]
        Mapping from anatomical structure names (e.g. 'GM', 'WM') to label values.

    synth_map : Dict[int, int]
        Mapping from label values to synthetic tissue classes (used for intensity generation).

    target_map : Dict[int, int]
        Mapping from label values to target label values (used for evaluation).

    output_name : str
        Base name for the output files (without path).

    output_dir : str, optional
        Output directory where results are saved, by default=current directory.

    add_subregions : bool, optional
        Whether to add subregion labels, by default=True.

    directory_path_add_subregions : str, optional
        Directory path where subregion label files are located, by default="".

    add_subregions_lab : List[str], optional
        List of subregion labels to add, by default=None.

    nb_pool : int, optional
        Pooling kernel size for downsampling images and labels, by default=3.

    pre_pool_after_preproc_dilate : int, optional
        label_img = tio.LabelMap(label_file) 
        name_map_current, label_img = setup_name_map_current(label_name_to_value, preprocessing_target_map, label_img) 
        
    label_name_few_to_label_name_all : Dict[str, str], optional
        Mapping from short label names to full label names (used for remapping in subregions), by default=None.

    Returns
    -------
    tio.Subject
        The transformed subject with anomalies included, for post-processing or tracking.

    """
    # -------------------------
    # Load and preprocess label
    # Erode and dilate
    # -------------------------
    label_img = tio.LabelMap(label_file)
    name_map_current, label_img = setup_name_map_current(label_name_to_value, preprocessing_target_map, label_img)
    
    subject = tio.Subject(label=label_img)
    subject["suffix"] = ""

    # -----------------------
    # Preprocessing transform
    # -----------------------
    logger.info(
        "%s==>%s Starting %sPreprocessing%s transforms", GREEN, RESET, BLUE, RESET
    )

    preprocessing_transform = resolve_transform_anatomies(
        transforms["preprocessing"], name_map_current, label_name_few_to_label_name_all
    )
    logger.info("%sResolved anatomies for preprocessing transforms%s", GREEN, RESET)
    preprocessed_subject = preprocessing_transform(subject)
    logger.info(
        "%sPreprocessing%s done -> suffix %s%s%s",
        GREEN,
        RESET,
        BLUE,
        preprocessed_subject["suffix"],
        RESET,
    )
    if pre_pool_after_preproc_dilate==2 :
        # Special case for BigBrain starting label at 0.1 mm, and we want to continue with 0.25mm
        label_bin, _ = pool_remap(preprocessed_subject.label, pooling_size=2, ensure_multiple=4)
        treslice = tio.Resample(target=0.25)
        label_bin = treslice(label_bin)
        preprocessed_subject.remove_image('label')
        preprocessed_subject.add_image(label_bin, 'label')
        print(f"resample image to {label_bin}")

    if add_subregions:
        logger.info("%s==>%s Start adding labels with ass and yep", GREEN, RESET)
        label_img = add_subregions_labels(
            directory_path_add_subregions,
            add_subregions_lab,
            preprocessed_subject.label,
            label_name_to_value, 
            preprocessing_target_map[1]
        )
        
        # Use remove_image/add_image to properly replace the label in the Subject.
        # Direct assignment (subject.label = ...) does not update TorchIO's internal state.
        # This ensures both the attribute and the internal dict are consistent, avoiding subtle bugs downstream.
        preprocessed_subject.remove_image('label')
        preprocessed_subject.add_image(label_img, 'label')

    # -------------------
    # Anomalies transform (optional)
    # -------------------
    if "anomalies" in transforms:
        preprocessed_subject["suffix"] += "Ano"
        logger.info("%s==>%s Starting %sAnomalies%s transforms", GREEN, RESET, BLUE, RESET)
        
        transforms_anomalies = resolve_transform_anatomies(
            transforms["anomalies"], label_name_to_value, label_name_few_to_label_name_all
        )
        logger.info("%sResolved anatomies for anomalies transforms%s", GREEN, RESET)

        subject_anomalies = transforms_anomalies(preprocessed_subject)
        
        if "lesion_label_names" in subject_anomalies:
            # Retrieve the lesion label name
            min_lesion_value = min(subject_anomalies["lesion_label_names"].values())
            label_names_with_min_value = [k for k, v in label_name_to_value.items() if v == min_lesion_value]
            
            # Check that there is exactly one label name with the minimum lesion value
            if len(label_names_with_min_value) > 1:
                raise NotImplementedError(f"Multiple lesion labels not supported yet")
            if len(label_names_with_min_value) == 0:
                raise ValueError(f"No label with value {min_lesion_value} found in label_name_to_value, needed to assign lesion labels")
            
            label_name_to_value.update(subject_anomalies["lesion_label_names"])
            label_name_few_to_label_name_all[label_names_with_min_value[0]] = list(subject_anomalies["lesion_label_names"].keys())
            
            # Remove lesion_label_names from subject_anomalies to avoid confusion later
            # and also remove the lesion_label from label_name_to_value to avoid confusion with the original lesion label
            for idx, (_, value) in enumerate(subject_anomalies["lesion_label_names"].items()):
                if idx == 0:
                    first_value = synth_map[value]
                else:
                    synth_map[value] = first_value + idx
            label_name_to_value.pop(label_names_with_min_value[0], None)

        if "anomaly_type" in subject_anomalies:
            logger.info("Anomalies transforms to apply: %s%s%s", BLUE, subject_anomalies["anomaly_type"], RESET)

    else:
        logger.info("%s==>%s No %sAnomalies%s transforms specified", GREEN, RESET, BLUE, RESET)
        subject_anomalies = preprocessed_subject

    # Create output directory and symlink to the original label
    os.makedirs(output_dir, exist_ok=True)
    original_file = build_output_path(
            output_dir, output_name, "Original", subject_anomalies["suffix"], "test"
        )
    if not os.path.isfile(original_file):
        os.symlink(label_file, original_file)

    # -----------------
    # Spatial transform
    # -----------------
    logger.info("%s==>%s Starting %sSpatial%s transforms", GREEN, RESET, BLUE, RESET)

    subject_spatial = transforms["spatial"](subject_anomalies)
    

    # If only label output is requested
    if get_label_only:
        out_label = build_output_path(
            output_dir, output_name, "Lab", subject_spatial["suffix"], "test"
        )
        subject_spatial.label.save(out_label)
        logger.info(
            "%sSaved label only output%s %s%s%s",
            YELLOW,
            YELLOW,
            BLUE,
            out_label,
            RESET,
        )

        return subject_anomalies

    logger.info("%s==>%s Starting %sDownsampling%s transforms", GREEN, RESET, BLUE, RESET)
    subject_down = reduce_resolution(target_map, subject_spatial, nb_pool)

    for num_contrast in range(1, nb_out_contrast + 1):
        # -----------------------
        # Label 2 image transform
        # -----------------------
        logger.info(
            "%s==>%s Starting %sLabel2Image%s transforms contrast %s%s/%s%s", GREEN, RESET, BLUE, RESET, GREEN, num_contrast, nb_out_contrast, RESET
        )

        transforms_labelToImage = resolve_transform_anatomies(
            transforms["label_to_image"], label_name_to_value, label_name_few_to_label_name_all, synth_map
        )
        subject_img, subject_down, label_bin, label_4d = label_2_image(
            transforms_labelToImage,
            target_map,
            subject_spatial,
            synth_map,
            label_name_to_value )

        # --------------------
        # Intensity transforms
        # --------------------
        logger.info("%s==>%s Starting %sIntensity%s transforms", GREEN, RESET, BLUE, RESET)
        if "intensity" in transforms:
            subject_img = transforms["intensity"](subject_img)

        if num_contrast == 1:
            # ------------------------
            # Record transform history
            # ------------------------
            info1 = record_history(subject_img)
            #info2 = record_history(subject_down) (

            short_name = "st" #get_transfo_short_name(info1, "order") + get_transfo_short_name(info2, "order")
            info1.pop("history", None)
            #info2.pop("history", None)

            # ---------------
            # Prepare outputs
            # ---------------

            out_label = build_output_path(
                output_dir, output_name, "Lab", subject_img["suffix"], short_name
            )
            out_lab4d = build_output_path(
                output_dir, output_name, "4DLab", subject_img["suffix"], short_name
            )
            out_image = build_output_path(
                output_dir, output_name, "Sim", subject_img["suffix"], short_name
            )
            out_labHighRes = build_output_path(
                output_dir, output_name, "HRLab", subject_img["suffix"], short_name
            )

            subject_down.label_bin.save(out_label)
            if "__label_without_lesionLowRes" in subject_down:
                out_label_nolesLR = build_output_path(
                    output_dir, output_name, "LowResNoLes_lab", subject_img["suffix"], short_name
                )
                subject_down.__label_without_lesionLowRes.save(out_label_nolesLR)
                
            subject_img.label.save(out_lab4d)
            
            subject_spatial["label"].save(out_labHighRes)
            logger.info("%s==> Save%s label_bin %s%s%s", GREEN, RESET, BLUE, out_label, RESET)
            logger.info("%s==> Save%s label_4d  %s%s%s", GREEN, RESET, BLUE, out_lab4d, RESET)
            logger.info("%s==> Save%s label     %s%s%s", GREEN, RESET, BLUE, out_labHighRes, RESET)
            if add_subregions:
                output_csv_label = build_output_path(
                    output_dir, output_name, "Names", subject_img["suffix"], short_name)
                output_csv_label = output_csv_label.replace(".nii.gz", ".csv")
                dff = pd.DataFrame(label_name_to_value.items(), columns=['Name', 'Label'])
                dff.to_csv(output_csv_label, index=False)

            if "lesion_label_names" in subject_anomalies:
                ano_path = build_output_path(
                    output_dir, output_name, "Ano", subject_img["suffix"], short_name
                )
                subject_anomalies["label"].save(ano_path)
                logger.info("%s==> Save%s anomalies label %s%s%s", YELLOW, RESET, BLUE, ano_path, RESET)
                add_anomalies_map(sujname, subject_anomalies, output_dir, label_file)

            # Case where we want to save the label without lesion
            if "__label_without_lesion" in subject_spatial:
                out_label_noles = build_output_path(
                    output_dir, output_name, "NoLes_lab", subject_img["suffix"], short_name
                )
                # TODO check it
                subject_spatial["__label_without_lesion"].save(out_label_noles)
                logger.info("%s==> Saved%s label without lesion %s%s%s", GREEN, RESET, BLUE, out_label_noles, RESET)

            # -----------------
            # Write history CSV
            # -----------------
            csv_path = out_image.replace(".nii.gz", ".csv").replace("Sim", "CSV")

            with open(csv_path, "w", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(info1.items())
                #writer.writerows(info2.items())

        if nb_out_contrast > 1:
            # Replace "_ContrastN.nii.gz" or ".nii.gz" with the new contrast suffix
            out_image = re.sub(r'_Contrast\d+\.nii\.gz|\.nii\.gz', f'_Contrast{num_contrast}.nii.gz', out_image)
            
        subject_img.synth_mri.save(out_image)
        logger.info("%s==> Save%s synth_mri %s%s%s", GREEN, RESET, BLUE, out_image, RESET)


    return subject_anomalies


# --- Utils for ano map common file between process---
def _is_empty_or_missing(path: str) -> bool:
    return (not os.path.exists(path)) or os.path.getsize(path) == 0


def _load_labelmap_with_retries(
    path: str, retries: int = 20, delay: float = 0.25
) -> tio.LabelMap:
    """
    Attempt to load a label map from disk, with retries in case of read failure or missing file.

    Parameters
    ----------
    path : str
        Path to the .nii.gz file to load as a label map.
    retries : int, optional
        Maximum number of retry attempts (default is 20).
    delay : float, optional
        Delay in seconds between retries (default is 0.25).

    Returns
    -------
    tio.LabelMap
        The loaded label map.

    Raises
    ------
    Exception
        Re-raises the last exception encountered during loading, if all retries fail.

    RuntimeError
        If no exception occurred but the file could not be loaded after all retries.

    """
    last_err = None
    for _ in range(retries):
        try:
            if _is_empty_or_missing(path):
                # Still empty or missing -> wait and retry
                time.sleep(delay)
                continue
            return tio.LabelMap(path)

        except Exception as e:  # catch all exceptions during loading
            last_err = e
            time.sleep(delay)

    if last_err is not None:
        raise last_err

    raise RuntimeError(f"Failed to load label map at {path} after retries")


def _atomic_save_labelmap(tensor: torch.Tensor, affine, final_path: str) -> None:
    """
    Save a labelmap atomically by writing to a temporary file and replacing the target.

    Parameters
    ----------
    tensor : torch.Tensor
        Labelmap tensor to save. Can be 3D or 4D; a channel dimension is added if missing.

    affine : array-like
        Affine transformation matrix associated with the image.

    final_path : str
        Destination path for the final .nii.gz file. A temporary file is created
        and atomically renamed to this path.

    Notes
    -----
    Uses a temporary file with process ID and UUID to avoid collisions.
    Ensures atomic replacement on the same filesystem.

    """
    tmp_path = f"{final_path}.tmp-{os.getpid()}-{uuid.uuid4().hex}.nii.gz"

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    tio.LabelMap(tensor=tensor, affine=affine).save(tmp_path)
    print(YELLOW, "XXX saved temporary file", RESET, tmp_path)

    # Replace atomic in the file system
    os.replace(tmp_path, final_path)


# --- Parallel safe block ---
def add_mask_safely(subject_anomalies: dict, anomalies_path: str, mask: torch.Tensor):
    """
    Safely add an anomaly mask to the anomaly map on disk, with file locking.

    Parameters
    ----------
    subject_anomalies : dict
        Dictionary containing the subject's data. Must include
        ``subject_anomalies['label']`` as a ``tio.LabelMap``.

    anomalies_path : str
        Path to the final anomaly map (.nii.gz).

    mask : torch.Tensor
        Anomaly mask to add. Shape must match the label map
        (3D or 4D with leading channel dimension).
        Converted internally to ``torch.int16``.

    Raises
    ------
    ValueError
        If the mask shape does not match the base anomaly map.

    RuntimeError
        If the file lock cannot be acquired within the timeout.

    """
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    mask = mask.to(dtype=torch.int16)

    # Lock file (only one process at a time on this path)
    lock = FileLock(anomalies_path + ".lock")
    try:
        # Increase timeout if necessary when many jobs are writing to the same file
        with lock.acquire(timeout=600):
            # Current effect
            # zero if the file does not exist (or is empty), otherwise load with retries
            if _is_empty_or_missing(anomalies_path):
                base = torch.zeros_like(
                    subject_anomalies["label"].data, dtype=torch.int16
                )
                affine = subject_anomalies["label"].affine

            else:
                lm = _load_labelmap_with_retries(anomalies_path)
                base = lm.data.to(dtype=torch.int16)
                affine = lm.affine

            # Safety: aligning the shapes
            if base.shape != mask.shape:
                raise ValueError(
                    f"Shape mismatch: base {tuple(base.shape)} vs mask {tuple(mask.shape)}"
                )

            # Atomic save
            _atomic_save_labelmap(base + mask, affine, anomalies_path)

    except Timeout as e:
        raise RuntimeError(f"Lock timeout on {anomalies_path}") from e


def add_anomalies_map(
    subject_name: str, subject_anomalies: tio.Subject, output_dir, label_file
):
    """
    Add an anomaly mask to the anomaly map for a given subject, and save label file if needed.

    Parameters
    ----------
    subject_name : str
        Unique identifier for the subject (used in filenames).

    subject_anomalies : tio.Subject
        Subject containing data related to anomaly for the subject.
        Must include:
        - 'label' : a `tio.LabelMap`
        - 'lesion_label_names' : dict mapping lesion label names to their values
        - 'anomaly_type' : str, used in the output filename

    output_dir : str
        Output directory path where anomaly maps and label maps are saved.

    label_file : str
        Path to the original label map (.nii.gz) to be saved if not already present.

    Notes
    -----
    This function:
    - Builds anomaly and label map paths.
    - Computes the anomaly mask using ``torch.isin``.
    - Saves the label file if not already saved.
    - Calls `add_mask_safely` to add the anomaly mask to the existing map.

    """
    # ---- paths ----
    amaps_dir = op.join(output_dir, "anomaly_maps")
    os.makedirs(amaps_dir, exist_ok=True)

    anomalies_path = op.join(amaps_dir, f"Amap_{subject_name}.nii.gz")
    subject_path = op.join(amaps_dir, f"Lab_{subject_name}.nii.gz")

    # ---- mask of anomalies ----
    mask = torch.isin(
        subject_anomalies["label"].data,
        torch.tensor(list(subject_anomalies["lesion_label_names"].values())),
    )

    # ---- save subjects and ano map ----
    if not op.exists(subject_path):
        tio.LabelMap(label_file).save(subject_path)
        logger.info("Save subject %s%s%s", YELLOW, subject_path, RESET)

    add_mask_safely(subject_anomalies, anomalies_path, mask)

    logger.info("Subject anomalies map updated %s%s%s", BLUE, anomalies_path, RESET)

def log_subject_example(subject_idx, subject_number, total_files, example_num, total_examples):
    """
    Log the subject/example progress message in a consistent format.
    """
    if subject_idx is None:
        logger.info(
            "Subject %s%s/%s%s example : %s%s/%s%s",
            GREEN,
            subject_number,
            total_files,
            RESET,
            GREEN,
            example_num,
            total_examples,
            RESET,
        )
    else:
        logger.info(
            "Subject %s%s%s example : %s%s/%s%s",
            GREEN,
            subject_number,
            RESET,
            GREEN,
            example_num,
            total_examples,
            RESET,
        )

def main(args: argparse.Namespace):
    """
    Run the main entry point for the script.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.

    """
    # Get configuration
    config = load_config(args.outputdir, args.dataset, args.anomaly_type, args.config_filename)
    nb_example_per_subject = args.nb_gen

    (
        label_set_name,
        files,
        label_name_to_value,
        synth_label_to_tissue,
        synth_label_to_target,
        label_name_few_to_label_name_all,
        preprocessing_target_map
    ) = get_label_set_config(args.dataset, config.dataset)

    # Create temporary output_dir name
    tmp_name = op.basename(tempfile.TemporaryDirectory().name)
    output_dir = op.join(args.outputdir, f"generate_{label_set_name}_{tmp_name}")

    if args.subject_idx is not None:
        files = [files[args.subject_idx - 1]]

    if len(files) == 0:
        raise ValueError("No valid subject found.")
    logger.info("Running on %s%d%s subject files", BLUE, len(files), RESET)
    
    # Loop over subjects and generate examples
    for ngen in range(nb_example_per_subject):
        for subject_number, label_file in enumerate(files, start=args.subject_idx or 1):
            if subject_number > 1 and len(files) > 1:
                # Reload config for each subject
                # to avoid issues with transforms containing random elements that are not re-instantiated at each generation loop
                config = load_config(args.outputdir, args.dataset, args.anomaly_type, args.config_filename)

            sujname = f"{label_set_name}_Suj_{subject_number:02}"
            fout = f"gen{ngen + 1:03}_{sujname}.nii.gz"

            logger.info("%s=== Start generation ===%s", GREEN, RESET)
            log_subject_example(args.subject_idx, subject_number, len(files), ngen + 1, nb_example_per_subject)

            logger.info("Label file: %s%s%s", BLUE, label_file, RESET)
            logger.info("Computing base name %s%s%s", BLUE, fout, RESET)

            generate_one_subject(
                sujname,
                label_file,
                config.transforms,
                label_name_to_value,
                synth_label_to_tissue,
                synth_label_to_target,
                preprocessing_target_map,
                fout,
                output_dir=output_dir,
                add_subregions=config.dataset["add_subregions"],
                directory_path_add_subregions=config.dataset[
                    "directory_path_add_subregions"
                ],
                add_subregions_lab=config.dataset["add_subregions_lab"],
                nb_pool=config.dataset["nb_pool"],
                pre_pool_after_preproc_dilate=config.dataset["pre_pool_after_preproc_dilate"],
                nb_out_contrast=args.nb_contrast,
                get_label_only=args.get_label_only,
                label_name_few_to_label_name_all=label_name_few_to_label_name_all
            )


if __name__ == "__main__":
    main(parse_args())
