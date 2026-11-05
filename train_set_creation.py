import ast
from glob import glob
import json
import logging
import os
import re
import shutil
from typing import Dict, List, Union
import subprocess # nosec B404

import pandas as pd
from tqdm import tqdm
from utils.create_jobs import create_jobs
from utils.file import get_directories, get_files, get_parent_path
import os.path as op
import torchio as tio
from itertools import product
import argparse

from utils.logger import BLUE, GREEN, RED, RESET
logger = logging.getLogger(__name__)

# Example of run:
# python train_set_creation.py -g -d -p -plan -j --config config_nnunet_dataset_train_job_creation/config_dataset_train_job_nnunet_csf_deform_cergm.ini

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the nnUNet pipeline.
    
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
        
    Raises
    ------
    argparse.ArgumentTypeError
        If the dataset number is not provided or is invalid.
    """
    parser = argparse.ArgumentParser(
        description="nnUNet pipeline: dataset creation, planning, training and job setup.",
        epilog=f"{RED}On the cluster{RESET} this is the command to allocate the maximum number of cpus : {BLUE}srun -p gpu-cenir --ntasks=1 --cpus-per-task=56 --mem=0 --time=90-00:00:00 --pty --account=cenir bash{RESET}"
    )

    parser.add_argument("--generate-folder", "-g", action="store_true", help="Generate raw data folders from source directories")
    parser.add_argument("--nnunet-dataset", "-d", action="store_true", help="Create the nnUNet dataset from preprocessed images and labels")
    parser.add_argument("--add-previous-dataset", "-p", action="store_true", help="Add dataset 716 (without tumor) to the current dataset")
    parser.add_argument("--plan-and-preprocess", "-plan", action="store_true", help="Run preprocessing and experiment planning")
    parser.add_argument("--create-jobs", "-j", action="store_true", help="Create SLURM jobs for training")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the .ini configuration file"
    )

    args = parser.parse_args()
    
    return args


def load_label_csv(
        csv_path: str,
        col_name: str,
        col_synth: str,
        col_synth_tissu: str,
        col_target: str,
        col_target_few: str,
        col_name_few: str
    ) -> tuple[Dict, Dict, Dict, Dict, Dict]:
    """
    Load a CSV file containing label mappings and return dictionaries for label remapping.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing label mappings.

    col_name : str
        Column name for the label names.

    col_synth : str
        Column name for the synthetic label values.

    col_synth_tissu : str
        Column name for the synthetic tissue values.

    col_target : str
        Column name for the target label values.

    col_target_few : str
        Column name for the target few label values.

    col_name_few : str
        Column name for the few label names.
        
    Returns
    -------
    tuple[Dict, Dict, Dict, Dict, Dict]
        
    """
    # Read the csv
    df = pd.read_csv(csv_path, comment='#')

    # Label remap name to synth
    dict_lab = {row[col_name]: row[col_synth] for _, row in df.iterrows()}

    # Synth remap to synth tissu
    dict_map_synth = {row[col_synth]: row[col_synth_tissu] for _, row in df.iterrows()}

    # Synth remap to target
    dict_map_target = {row[col_synth]: row[col_target] for _, row in df.iterrows()}

    # Target remap to target few
    dict_map_target_few = {row[col_target]: row[col_target_few] for _, row in df.iterrows()}

    # Name few to target few
    dict_lab_few = {row[col_name_few]: row[col_target_few] for _, row in df.iterrows()}

    return dict_lab, dict_map_synth, dict_map_target, dict_map_target_few, dict_lab_few

def get_label_set_map(set_num:int, add_tumor_labs=False):
    """
    Map label sets for post-synthetic remapping in nnUNet region task.
    Unifies all label set configs into a single flexible function.

    Parameters
    ----------
    set_num : int
        1 = MIDA, 2 = VASC, 3 = SKULL

    add_tumor_labs : bool
        Whether to extend mapping with synthetic tumor labels (Anomalies)

    Returns
    -------
    dict
        Mapping and file list used for region-based remapping.
    """
    names = {1: "mida", 2: "vasc", 3: "skul"}

    if set_num == 1:
        # --- MIDA ---
        sujdir = ['/network/iss/opendata/data/template/MIDA_v1.0/MIDA_v1_voxels/mida_all/']
        label_csv = '/network/iss/opendata/data/template/remap/my_synth/mida_labels.csv'

        dict_lab, dict_map_synth, dict_map_target, dict_map_target_RegFew, dict_lab_RegFew = load_label_csv(
            label_csv,
            col_name='Name',
            col_synth='value',
            col_synth_tissu='synth_tissue_29',
            col_target='value',
            col_target_few='synth_targetRegionFew',
            col_name_few='NameTargetRegionFew',
        )

        fins = get_files(sujdir, '^csf_veine_r025s05')

    elif set_num == 2:
        # --- VASC ---
        dirvas = '/network/iss/cenir/analyse/irm/users/romain.valabregue/segment_RedNucleus/vascular_pc3D/preproc/'
        sujdir = get_directories(dirvas, ['(AR$|SO$)', 'synth_v3'])

        # CSV 1: label only
        label_csv_lab = '/network/iss/opendata/data/template/remap/my_synth/Svas_synth_v3.csv'
        df_lab = pd.read_csv(label_csv_lab)
        dict_lab = {row['Name']: row['synth'] for _, row in df_lab.iterrows()}
        dict_lab['Hippo'] = dict_lab['Hyp']  # special fix for Hippo label

        # CSV 2: full mapping
        label_csv_synth_target = '/network/iss/opendata/data/template/remap/my_synth/brain_and_skull_and_Ass_vascular_v3_label_DS709.csv'
        _, dict_map_synth, dict_map_target, dict_map_target_RegFew, dict_lab_RegFew = load_label_csv(
            label_csv_synth_target,
            col_name='Name',
            col_synth='synth',
            col_synth_tissu='synth_tissu',
            col_target='synth_target',
            col_target_few='synth_targetRegionFew',
            col_name_few='NameTargetRegionFew',
        )
        fins = get_files(sujdir, '^r025_s05.*gz')

    elif set_num == 3:
        # --- SKULL ---
        sujdir = get_directories(
            '/network/iss/cenir/analyse/irm/users/romain.valabregue/segment_RedNucleus/Skull/',
            ['.*', 'slicer2']
        )
        label_csv = '/network/iss/opendata/data/template/remap/my_synth/brain_and_skull_and_head_Ultra_v2_label_DS708_notumor.csv'

        dict_lab, dict_map_synth, dict_map_target, dict_map_target_RegFew, dict_lab_RegFew = load_label_csv(
            label_csv,
            col_name='Name',
            col_synth='synth',
            col_synth_tissu='synth_tissu',
            col_target='synth',
            col_target_few='synth_region_few',
            col_name_few='Name_region_few',
        )

        dict_lab['vascular_brain'] = dict_lab.get('Vas_brain', dict_lab.get('vascular_brain'))

        fins = get_files(sujdir, '^inVe.*down')
    else:
        raise ValueError(f"Unknown label_set_num: {set_num}")

    # --- Postprocessing: sorting name maps ---
    dict_lab = dict(sorted(dict_lab.items(), key=lambda item: item[1]))
    dict_lab_RegFew = dict(sorted(dict_lab_RegFew.items(), key=lambda item: item[1]))


    # --- Optional: add tumor label mappings ---
    if add_tumor_labs:
        # Special case because Anomalies have been added before the remap target so their value start with last dict_map_target
        max_remap_key = max(dict_map_target.keys())
        max_remap_val = max(dict_map_target_RegFew.values())
        extended_tmap = {max_remap_key + i: max_remap_val + 1 for i in range(1, 4)}
        dict_map_target_RegFew.update(extended_tmap)
        dict_lab_RegFew['Anomalies'] = max_remap_val + 1

    return {
        "name": names[set_num],
        "files": fins,
        "name_map": dict_lab,
        "synth_map": dict_map_synth,
        "target_map": dict_map_target,
        "target_map_RegionFew": dict_map_target_RegFew,
        "name_map_RegionFew": dict_lab_RegFew
    }

def resolve_conflict(dst: str) -> str:
    """
    If the destination file exists, adds _1, _2... before the extension.
    Handles compound extensions like .nii.gz properly.
    """
    if not op.exists(dst):
        return dst

    dirname = op.dirname(dst)
    filename = op.basename(dst)

    # Split base and extension (handle .nii.gz or similar)
    match = re.match(r"^(.*?)(\.[^.]+(?:\.[^.]+)?)$", filename)
    if match:
        base, ext = match.groups()
    else:
        base, ext = filename, ''

    i = 1
    new_dst = op.join(dirname, f"{base}_{i}{ext}")
    while op.exists(new_dst):
        i += 1
        new_dst = op.join(dirname, f"{base}_{i}{ext}")

    logger.info(f"[{dirname}] Conflict: '{filename}' → '{op.basename(new_dst)}'")
    return new_dst

def copy_move_link_files(src_files:  List[str], dst_files: Union[str, List[str]], file_type: str = 'link') -> None:
    """
    Move or link files from `src_files` to `dst_files`.

    Parameters
    ----------
    src_files : str or list of str
        Source file(s).
    dst : str or list of str
        Destination file(s) or a destination directory.
    file_type : str
        'link' to create symbolic links, 'move' to move files, 'copy' to copy files
    """
    if isinstance(dst_files, str) and op.isdir(dst_files):
        dst_files = [op.join(dst_files, op.basename(file)) for file in src_files]
    elif isinstance(dst_files, str):
        raise ValueError("dst must be a folder if it is a string else it is a list")

    if len(src_files) != len(dst_files):
        raise ValueError("`src_files` and `dst` must have the same length after processing.")

    for src, dst in zip(src_files, dst_files):
        final_dst = resolve_conflict(dst)

        if file_type == 'link':
            os.symlink(src, final_dst)
        elif file_type == 'move':
            shutil.move(src, final_dst)
        elif file_type == 'copy':
            shutil.copy2(src, final_dst)
        else:
            raise ValueError(f"Unknown type: {file_type}. Use 'link', 'copy' or 'move'.")

def regroupe_generated_data(
    raw_data_dir: str,
    dirdata: str,
    dir_regex: str,
    prefix: str = '',
    create_outdirs: bool = True,
    output_prefix: str = "synth",
):
    """
    Groups all generated files (binary and 4D) into common output folders,
    renaming them to avoid filename conflicts.

    Parameters
    ----------
    raw_data_dir : str
        Final output directory where regrouped files will be placed.
    dirdata : str or list of str
        One or more directories containing generate_* folders.
    dir_regex : str
        Pattern to match source subdirectories.
    prefix : str
        Prefix used to name the output folders.
    create_outdirs : bool
        Whether to create output folders if they don't exist.
    output_prefix : str
        Prefix for the output files (default="synth").
    """
    # Find all subdirectories matching the pattern
    subdirs = get_directories(dirdata, f'{dir_regex}_{prefix}')
    logger.info(f"Found {BLUE}{len(subdirs)}{RESET} directories matching '{BLUE}{dir_regex}_{prefix}{RESET}' in {BLUE}{dirdata}{RESET}")
    
    if not subdirs:
        raise RuntimeError(f"No directories found matching '{dir_regex}_{prefix}' in {dirdata}. Check your input path or regex.")

    # Out folder
    dout_bin = op.join(raw_data_dir, f'{prefix}_{output_prefix}_bin')
    dout_4d = op.join(raw_data_dir, f'{prefix}_{output_prefix}_4D')
    
    logger.info(f"Output folders: {BLUE}{dout_bin}{RESET}, {BLUE}{dout_4d}{RESET}")

    if create_outdirs:
        os.makedirs(dout_bin, exist_ok=True)
        os.makedirs(dout_4d, exist_ok=True)
    else:
        if not (op.isdir(dout_bin) and op.isdir(dout_4d)):
            raise RuntimeError("Output folders do not exist and create_outdirs=False.")

    # Loop over folders
    for i, dirgen in enumerate(tqdm(subdirs)):
        # --- Binary files processing ---
        files_bin = get_files(dirgen, '^[SL]')
        fnames_bin = get_parent_path(files_bin)[1]
        renamed_bin = [f'{dout_bin}/{name[:7]}{i:03}{name[10:]}' for name in fnames_bin]
        copy_move_link_files(src_files=files_bin, dst_files=renamed_bin, file_type='link')

        # --- 4D files generation ---
        files_4d = get_files(dirgen, '^[4d]')
        fnames_4d = get_parent_path(files_4d)[1]
        renamed_4d = [f'{dout_4d}/{name[:9]}{i:03}{name[12:]}' for name in fnames_4d]
        copy_move_link_files(src_files=files_4d, dst_files=renamed_4d, file_type='link')

def get_next_image_filename(images_path: str, base_name: str) -> str:
    """
    Finds the next available filename of the form '{base_name}_{XXXX}_0000.nii.gz'
    in the given directory by detecting the highest existing XXXX value.

    Parameters
    ----------
    images_path : str
        Directory to search in.
    base_name : str
        Prefix of the filename.

    Returns
    -------
    str
        Filename (not full path) of the next available file.
    """
    pattern = re.compile(rf"^{re.escape(base_name)}_(\d{{4}})_0000\.nii\.gz$")
    max_idx = -1

    for fname in os.listdir(images_path):
        match = pattern.match(fname)
        if match:
            max_idx = max(max_idx, int(match.group(1)))

    next_index = max_idx + 1
    return f"{base_name}_{next_index:04}", next_index

def create_nnunet_dataset_from_nii(
        img_files: List[str],
        lab_files: List[str],
        label_dict: Dict[str, int],
        dataset_directory: str,
        base_name: str = 'RRR',
        lab_one_hot: bool = False,
        tmap_lab: Dict = None,
        region_mask: bool = False,
        tmap_fil_up: Dict = None
    ):
    """
    Create a nnUNet dataset from provided image and label files.
    
    Parameters
    ----------
    img_files : List[str]
        List of paths to image files.

    lab_files : List[str]
        List of paths to label files.

    label_dict : Dict[str, int]
        Dictionary mapping label names to integer values.

    dataset_directory : str
        Directory where the dataset will be created.

    base_name : str, optional
        Base name for the output files (default is 'RRR').

    lab_one_hot : bool, optional
        Whether to convert labels to one-hot encoding (default is False).

    tmap_lab : Dict, optional
        Dictionary for label remapping (default is None).

    region_mask : bool, optional
        Whether to create a region mask from labels (default is False).

    tmap_fil_up : Dict, optional
        Dictionary for additional label remapping (default is None).
        
    """
    images_path = op.join(dataset_directory, 'imagesTr')
    labels_path = op.join(dataset_directory, 'labelsTr')

    for directory in [dataset_directory, images_path, labels_path]:
        if not op.isdir(directory):
            os.mkdir(directory)

    if len(lab_files) != len(img_files):
        raise ValueError(f"Different count of labels and images, got {len(lab_files)} and {len(img_files)}")

    for image_file, label_file in tqdm(zip(img_files, lab_files), total=len(lab_files)):
        output_filename, last_idx = get_next_image_filename(images_path, base_name)
        image_output_path = op.join(images_path, output_filename + "_0000.nii.gz")  # channels number
        if not op.isfile(image_output_path):
            os.symlink(image_file, image_output_path)

        output_filename_label = output_filename + '.nii.gz'
        label_output_path = op.join(labels_path, output_filename_label)
        if lab_one_hot:
            # Must transform 4d label to 3D os.symlink(label_file, labels_path + output_filename_label)
            if not op.isfile(label_output_path):
                label_image = tio.LabelMap(label_file)
                label_image = thoti(label_image) # TODO move here or is it onehottransform ?
                label_image.save(label_output_path)

        elif tmap_lab is not None:
            if tmap_fil_up is not None:
                # added for tumor label which add 1 to 3 extra label (being not continuous with healthy labels)
                max_remap_key = max(tmap_lab.keys())
                max_remap_val = max(tmap_lab.values())

                label_values = tio.LabelMap(label_file).data.unique().to(int).numpy()

                extra_labels = label_values[label_values > max_remap_key]
                extended_tmap = {label: max_remap_val + 1 for label in extra_labels}
                
                logger.info(f"Extended tmap {extended_tmap}")

                new_tmap_lab = tmap_lab.copy()
                new_tmap_lab.update(extended_tmap)

                label_image = tio.RemapLabels(new_tmap_lab)(label_image)
                label_dict[tmap_fil_up] = max_remap_val + 1
            else:
                label_image = tmap_lab(tio.LabelMap(label_file))

            label_image.save(label_output_path)

            if region_mask:
                label_image['data'][label_image.data > 0] = 1
                mask_output_path = op.join(images_path, output_filename + "_0001.nii.gz")
                label_image.save(mask_output_path)
        else:
            os.symlink(label_file, label_output_path)

    label_dict = {k: v for k, v in sorted(label_dict.items(), key=lambda item: item[1])}  # sort

    if region_mask:
        json_dic = {
            "channel_names": {"0": "T1", "1": "mask"},
            'labels': label_dict,
            "numTraining": last_idx + 1,
            "file_ending": ".nii.gz"
        }
    else:
        json_dic = {
            "channel_names": {"0": "T1"},
            'labels': label_dict,
            "numTraining": last_idx + 1,
            "file_ending": ".nii.gz"
        }
    with open(op.join(dataset_directory, 'dataset.json'), 'w') as file:
        json.dump(json_dic, file, indent=4, sort_keys=False)


def nnunet_train_job(
        datasetNum: int,
        nnunet_directory: str,
        nb_folds: int,
        nb_cpus: int,
        jobdir_name: str,
        job_name: str,
        nb_preproc_cpus:str = 12,
        plan_model: List[str] = None,
        plan_type: List[str] = None,
        init_model_dir: str =None,
        server: str = 'ICM'
    ) -> None:
    """
    Create and submit nnUNet training jobs for specified dataset.
    
    Parameters
    ----------
    datasetNum : int
        nnUNet dataset number.
    nnunet_directory : str
        Directory where nnUNet datasets are stored.
    nb_folds : int
        Number of folds for cross-validation.
    nb_cpus : int
        Number of CPUs to allocate for each job.
    jobdir_name : str
        Name of the directory where jobs will be stored.
    job_name : str
        Name of the job to be created.
    nb_preproc_cpus : int, optional
        Number of CPUs for preprocessing (default is 12).
    plan_model : List[str], optional
        List of plan models to use (default is ['3d_fullres']).
    plan_type : List[str], optional
        List of plan types to use (default is ['-p nnUNetPlans']).
    init_model_dir : str, optional
        Directory containing initial model checkpoints (default is None).
    server : str, optional
        Server type for job submission (default is 'ICM').
    
    Raises
    ------
    ValueError
        If the provided `plan_model` or `plan_type` is invalid.
    RuntimeError
        If the nnUNet directory does not exist or is not a directory.
    
    Notes
    -----
    This function prepares and submits nnUNet training jobs for a specified dataset.
    It constructs the necessary command-line arguments and job parameters, then creates the jobs using the `create_jobs` function.
    The function assumes that the nnUNet environment is set up correctly and that the necessary directories exist.
    It also handles the case where an initial model directory is provided, allowing for training from a pre-existing checkpoint.
    If `plan_model` or `plan_type` are not provided, they default to `['3d_fullres']` and `['-p nnUNetPlans']`, respectively.
    """
    if plan_model is None:
        plan_model = ['3d_fullres']
        
    if plan_type is None:
        plan_type = [ '-p nnUNetPlans']

    cmd_ini = f'''
source /network/iss/cenir/software/irm/conda-forge/etc/profile.d/conda.sh
conda activate ptorch23

SOFT_PATH=/network/iss/cenir/software/irm/
export PYTHONPATH=${{SOFT_PATH}}toolbox_python/torchio/src:$PYTHONPATH

export nnUNet_raw="{nnunet_directory}"
export nnUNet_preprocessed="{op.join(nnunet_directory, "Preproc")}"
export nnUNet_results="{op.join(nnunet_directory, "Results")}"

export nnUNet_n_proc_DA={nb_preproc_cpus}
nnUNetv2_train {datasetNum}'''

    jobs=[] # 40 for V100  24 for A100 ?
    for fold, (pm, pt) in product(range(nb_folds), zip(plan_model, plan_type)):
        if init_model_dir is not None:
            dir_mod = get_directories(init_model_dir, f'fold_{fold}')
            final_checkpoint_file = get_files(dir_mod, 'checkpoint_final.pth')[0]
            jobs.append(f'{cmd_ini} {pm} {fold} {pt} -pretrained_weights {final_checkpoint_file}')
        else:
            jobs.append(f'{cmd_ini} {pm} {fold} {pt} ')

    job_directory = op.join(nnunet_directory, "job")
    job_params = dict()
    job_params['raw_data_directory'] = op.join(job_directory, jobdir_name)
    job_params['jobs'] = jobs
    job_params['job_name'] = job_name
    job_params['cpus_per_task'] = nb_cpus
    job_params['job_pack'] = 1


    if server == 'ICM':
        job_params['cluster_queue'] = '-p gpu-ampere,gpu-cenir --gres=gpu:1 --mem 200G --account=cenir'
        job_params['walltime'] = '200:00:00'
    else:
        job_params['cluster_queue'] = '-C v100-32g --qos=qos_gpu-t4 --account=ezy@v100 --gres=gpu:1'
        job_params['walltime'] = '20:00:00'
        job_params['walltime'] = '100:00:00'


    create_jobs(job_params)



def load_and_validate_config(config_path: str, plan_and_preprocess: bool, previous_dataset: bool) -> Dict[str, Union[str, List[str]]]:
    """
    Load and validate the configuration file.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    plan_and_preprocess : bool
        Whether planning/preprocessing is expected.
    previous_dataset : bool
        Whether a previous dataset must be considered.

    Returns
    -------
    Dict[str, Union[str, List[str]]]
        Parsed configuration values.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If the configuration is invalid or required keys are missing.
    KeyError
        If a required key is not found.
    """
    logger.info(f"Config_path {config_path}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = {}
    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # Skip empty lines and comments
            if "=" not in line:
                raise ValueError(f"Invalid line in config: {line}")
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Convert list-style strings to actual lists
            if value.startswith("[") and value.endswith("]"):
                value = ast.literal_eval(value)
                if not isinstance(value, list):
                    raise ValueError(f"Invalid list for key '{key}': {value}")
            elif value == "None":
                value = None

            config[key] = value

    # === Validate required keys ===
    required_config = [
        "dir_datas",
        "output_prefix",
        "dir_regex",
        "suffix",
        "server",
        "raw_data_dir"
    ]

    required_paths = [
        "nnunet_directory",
    ]

    required_ints = [
        "nb_cpus",
        "nb_preproc_cpus",
        "dataset_num",
        "nb_folds"
    ]

    if plan_and_preprocess:
        for key in ["nnunet_plan", "nnunet_config"]:
            if key not in config:
                raise KeyError(f"Missing required plan_and_preprocess config key: '{key}'")

    if previous_dataset:
        required_paths.append("pretrained_dir")
        for key in ["previous_dataset"]:
            if key not in config:
                raise KeyError(f"Missing required previous_dataset config key: '{key}'")
            required_paths.append(key)

    for k in required_paths + required_ints + required_config:
        if k not in config:
            raise KeyError(f"Missing required config key: '{k}'")

    # === Validate directories ===
    for path_key in required_paths:
        path_val = config[path_key]
        if not os.path.isdir(path_val):
            raise FileNotFoundError(f"Path does not exist: {path_key} = {path_val}")

    for key in required_ints:
        try:
            config[key] = int(config[key])
        except ValueError as e:
            raise ValueError(f"Invalid integer value for key '{key}': {config[key]}") from e

    for i, path in enumerate(config["dir_datas"]):
        if not glob(path + "*"):
            raise FileNotFoundError(f"dir_datas[{i}] does not match any existing path: {path}")

    return config

#TODO dead code to remove 
    #export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

                # new_tmap_lab = {k: v for k, v in tmap_lab.items()}
                # new_tmap_lab.update(extended_tmap)
    # dict_lab = {k: v for k, v in sorted(dict_lab.items(), key=lambda item: item[1])}  #  remove
    # dict_lab_RegFew = {k: v for k, v in sorted(dict_lab_RegFew.items(), key=lambda item: item[1])} #  remove
        # remove ?
        #job_params['cluster_queue'] = '-C a100 --account=ezy@a100  --qos=qos_gpu_a100-t3 --gres=gpu:1'
        #job_params['cluster_queue'] = '-p gpu-ampere,gpu-cenir --gres=gpu:1 --mem 200G'
    #job_params['mem'] = 32000
        #fins = get_files(sujdir, '^csf_veine_r025s05_mida') # alternative option

# dir_datas = ['/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/Vascular4_tumor']
# dataset_name = 'Dataset717_MixLowDill_Ano'
# nnunet_directory = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/'
    # output_prefix="synth"
        # nnunet_directory = "/network/iss/cenir/analyse/irm/users/ines.khemir/nnunet"
    # raw_data_dir = "/network/iss/cenir/analyse/irm/users/ines.khemir/nnunet/raw_data"
        # previous_dataset = "/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/Dataset716_MixLowDill/"
    # pretrained_dir = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/Results/Dataset716_MixLowDill/nnUNetTrainerNoDA__nnUNetResEncUNetXLPlans__3d_fullres/'
    # avant ... pretrained_dir = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/Results/Dataset715_MixSuj6/nnUNetTrainerNoDA__nnUNetResEncUNetXLPlans__3d_fullres'
    # avant ... pretrained_dir = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/Results/Dataset712_Vasc2suj_v3_Few/nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres/'
        # attention avec cette option pas de --c du coup il ecrase tout si le job se relance !!!
    # sur amper 14 cpu mais 24 sur gpu-cenir
    # lancer que le premier job (array=1) et attendre le debut du training ... mias peut etre plus utile a partir 2.6.0

def _resolve_exe(exe: str) -> str:
    path = shutil.which(exe)
    if not path:
        raise FileNotFoundError(f"Cannot find executable: {exe}")
    return os.path.realpath(path)

def _safe_run(cmd: list[str]) -> None:
    cmd = [str(ele) for ele in cmd]
    subprocess.run(cmd, check=True) # nosec B603

def process_nnunet_dataset_creation(set_map_args, output_prefix, config, idx, combinations, dataset_directory):
    """
    Process the creation of a nnUNet dataset from images and labels.

    Parameters
    ----------
    set_map_args : dict
        Arguments containing label mappings and set name.
    output_prefix : str
        Prefix for output files.
    config : dict
        Configuration dictionary.
    idx : int
        Current combination index.
    combinations : list
        List of all combinations.
    dataset_directory : str
        Directory where the dataset will be created.
    """
    logger.info(f"{GREEN}Generate folder{RESET}")

    dict_map_target = set_map_args['target_map_RegionFew']
    label_dict = set_map_args['name_map_RegionFew']
    subset_dir = f'{set_map_args["name"]}_{output_prefix}_bin'
    bin_directory = op.join(config["raw_data_dir"], subset_dir)

    img_files = get_files(bin_directory, 'Sim.*gz')
    lab_files = get_files(bin_directory, '^Lab.*gz')

    logger.info(f"Processing dataset creation {GREEN}[{idx}/{len(combinations)}]{RESET} ==> {BLUE}{subset_dir}{RESET}")
    logger.info(f"{BLUE}{len(img_files)}{RESET} inputs {BLUE}{len(lab_files)}{RESET} labels")

    create_nnunet_dataset_from_nii(
        img_files,
        lab_files,
        label_dict,
        dataset_directory,
        base_name='RRR',
        tmap_lab=tio.RemapLabels(dict_map_target),
    )
    logger.info(f"{GREEN}Done{RESET}")

if __name__ == '__main__':
    args = parse_args()
    config = load_and_validate_config(
        args.config,
        args.plan_and_preprocess,
        args.add_previous_dataset
    )
    
    dataset_num = config["dataset_num"]
    suffix = config["suffix"]
    dataset_directory = op.join(config["nnunet_directory"], f'Dataset{dataset_num}_{suffix}')

    combinations = list(product(range(1, 4), zip(config["dir_datas"], config["output_prefix"])))
    for idx, (set_map_nb, (dir_data, output_prefix)) in enumerate(combinations, 1):
        set_map_args = get_label_set_map(set_map_nb, add_tumor_labs=True)
        subset_dir = f"{dir_data}_{set_map_args['name']}"

        logger.info(f"Processing{GREEN}[{idx}/{len(combinations)}]{RESET} ==> {BLUE}{subset_dir}{RESET}")

        # Generate folder for each subset
        if args.generate_folder:
            logger.info(f"{GREEN}Generate folder link to raw datadir{RESET}")

            regroupe_generated_data(
                raw_data_dir=config["raw_data_dir"],
                dirdata=subset_dir,
                dir_regex=config["dir_regex"],
                prefix=set_map_args['name'],
                output_prefix=output_prefix
            )
        # Create Dataset
        if args.nnunet_dataset:
            logger.info(f"{GREEN}Create nnUNet dataset{RESET}")
            process_nnunet_dataset_creation(
                set_map_args,
                output_prefix,
                config,
                idx,
                combinations,
                dataset_directory
            )

    if args.add_previous_dataset:
        label_dict = set_map_args['name_map_RegionFew']
        previous_dataset = config["previous_dataset"]

        # Add the previous dataset
        img_files = get_files(op.join(previous_dataset, 'imagesTr'),'.*gz')
        lab_files = get_files(op.join(previous_dataset, 'labelsTr'),'.*gz')

        logger.info(f"{GREEN}Processing dataset creation ==> {RESET}{BLUE}{previous_dataset}{RESET}")
        logger.info(f"{BLUE}{len(img_files)}{RESET} inputs {BLUE}{len(lab_files)}{RESET} labels")

        nb_suj = create_nnunet_dataset_from_nii(
            img_files,
            lab_files,
            label_dict,
            dataset_directory,
            base_name='RRR',
        )

        logger.info(f"{GREEN}Done{RESET}")

    nnunet_plan = "nnUNetPlans"
    if args.plan_and_preprocess:
        nnunet_plan = config["nnunet_plan"]
        nnunet_config = config["nnunet_config"]
        nb_cpus_plan_and_preprocess = int(os.environ.get("SLURM_CPUS_ON_NODE", 1))

        logger.info(f"""{GREEN}Plan and preprocess ==> {RESET} with :
    {BLUE}{nb_cpus_plan_and_preprocess}{RESET} cpus
    {BLUE}{nnunet_config}{RESET} nnunet config
    {BLUE}{nnunet_plan}{RESET} nnunet plan
    """)

        # Preprocessing
        plan_preproc_exe = _resolve_exe("nnUNetv2_plan_and_preprocess")
        _safe_run([
            plan_preproc_exe, "-d", dataset_num, "-c", nnunet_config, "-np", nb_cpus_plan_and_preprocess
        ])

        # Planification with custom planner
        plan_exp_exe = _resolve_exe("nnUNetv2_plan_experiment")
        _safe_run([
            plan_exp_exe, "-d", dataset_num, "-pl", nnunet_plan
        ])

    if args.create_jobs:
        jobdir_name = f'DS{dataset_num}_{suffix}'
        nnunet_train_job(
            datasetNum=dataset_num,
            nnunet_directory=config["nnunet_directory"],
            jobdir_name=jobdir_name,
            nb_folds=config["nb_folds"],
            nb_cpus=config["nb_cpus"],
            nb_preproc_cpus= config["nb_preproc_cpus"],
            job_name=f'DS{dataset_num}_train',
            init_model_dir=config["pretrained_dir"],
            plan_model=['3d_fullres'],
            plan_type=[f'-p {nnunet_plan}'],
            server=config["server"]
        )

