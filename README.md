
# Synthetic Brain MRI Generator

Synthetic brain MRI generator with anatomical labels and controlled anomalies

## Table of Contents
- [Overview](#overview)
- [Latest Article](#latest-article)
- [Data](#data)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Main Commands](#command-line-arguments)
- [Additional Documentation](#additional-documentation)
- [Configuration: configs.ini](#configuration-configsini)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

This project generates synthetic brain MRI images from anatomical label maps, with advanced anomaly management (lesions, tumors, etc.) and configurable pipelines via TorchIO.

Main features:

- Spatial transformations (rotations, elastic deformations, etc.)
- Label preprocessing and subregion mapping
- Anomaly generation (lesions, tumors, etc.)
- Intensity-based image synthesis
- Multi-contrast MRI generation
- Parallel processing with file locking
- Full transformation history tracking

## Latest Article

For details and recent results, see the latest article:

**SIAM: Head and Brain MRI Segmentation from Few High-Quality Templates via Synthetic Training** — *Romain Valabregue, Ines Khemir, Eric Badinet, François Rousseau, Guillaume Auzias, Reuben Dorent* ([arXiv:2605.02737](https://arxiv.org/abs/2605.02737))

## Data

The label templates used by this project are available on Zenodo

### Download and extract

```bash
wget -O siam_label_template_v1.tar.gz "https://zenodo.org/records/20428684/files/siam_label_template.tar.gz?download=1"
mkdir -p data
tar -xzf siam_label_template_v1.tar.gz -C data
```
## Installation

### 1. Clone the repository
```bash
git clone https://github.com/ninous-dev/SynthGen.git
cd SynthGen
```

### 2. Create and activate a conda environment
```bash
conda create -n synth_gen python=3.10
conda activate synth_gen
```

### 3. Install PyTorch
Follow instructions at [pytorch.org](https://pytorch.org/get-started/locally/) for your CUDA version, e.g.:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Install dependencies for the test suite
```bash
pip install pytest pytest-cov
```

### 6. Install git hooks
The pre-commit hook automatically runs the full test suite **with coverage** before each commit.
Run this **once** after cloning:
```bash
bash install_hooks.sh
```
> To bypass it exceptionally: `git commit --no-verify`

## Repository Structure

| Folder/File | Description |
|-----------------|-------------|
| `generate.py` | Main generation script |
| `transforms_class/` | Custom TorchIO transforms |
| `utils/` | Utility functions |
| `analyse/` | Post-generation analysis scripts |
| `check_datasets/` | Dataset validation and checking |
| `configs_anomalies_examples/` | Example anomaly transform JSON files (MS, Tumor, Ventricles) |
| `jobs_run_cluster/`, `generate.sh`, `batch_generate.sh` | Batch/SLURM execution scripts |
| `requirements.txt` | Python dependencies |
| `README_cluster_ICM` | ICM makefile launch process |

## Command Line Arguments
Argument	Short	Default	Description of the [GENERATION](generate.py)

| Argument           | Short | Default      | Description                          |
|--------------------|-------|-------------|--------------------------------------|
| --dataset          | -d    | 1           | Dataset ID from config               |
| --nb_gen           | -n    | 1           | Images per subject                   |
| --outputdir        | -o    | .           | Output directory                     |
| --config_filename  | -c    | configs.ini | Config file name                     |
| --subject_idx      | -s    | None        | Process specific subject (1-based index) |
| --nb_contrast      | -z    | 1           | Output contrasts per subject         |
| --anomaly_type     | -a    | None        | Anomaly type ID                      |
| --get_label_only   | -     | False       | Generate labels only (no images)     |

## Configuration: `configs.ini`

The `configs.ini` file defines all paths and parameters for the generation pipeline. It is structured in three types of sections:

### `[transforms]`

Paths to the JSON transform files applied to all datasets:

```ini
[transforms]
spatial     = /path/to/transform_spatial.json
label_to_image = /path/to/transform_lab2img.json
intensity   = /path/to/transform_intensity.json
```

### `[anomalies]`

Optional. Maps anomaly IDs (used with `-a`) to their transform JSON files:

```ini
[anomalies]
1 = /path/to/transform_deform_all_AnoTumor.json
2 = /path/to/transform_deform_all_AnoMS.json
```

Example files are available in [`configs_anomalies_examples/`](configs_anomalies_examples/).

### `[label_set.datasetN]`

One section per dataset (indexed from 1). Used with `-d N`.

| Key | Required | Description |
|-----|----------|-------------|
| `label_set_name` | yes | Short name for the dataset |
| `subject_dir` | yes | Directory containing the label files |
| `labels_csv` | yes | CSV file mapping label names to values |
| `preprocessing` | yes | Path to the preprocessing transform JSON |
| `synth_col` | yes | Column name in CSV for synthesis mapping |
| `do_erode` | yes | Whether to apply erosion (`True`/`False`) |
| `add_subregions` | no | Add subregion labels (`True`/`False`) |
| `directory_path_add_subregions` | no | Directory with subregion label files |
| `add_subregions_lab` | no | List of subregion label patterns (regex supported) |
| `nb_pool` | no | Pooling factor (default: 3) |
| `pre_pool_after_preproc_dilate` | no | Set to `2` for BigBrain-type datasets |

Example:

```ini
[label_set.dataset1]
label_set_name = mida
subject_dir    = /path/to/mida
labels_csv     = /path/to/mida_labels.csv
preprocessing  = /path/to/transform_preprocessing_mida.json
do_erode       = True
synth_col      = synth
file_regex     = ^csf_veine_r025s05
```

## TestSuite

The test suite covers all transforms in `transforms_class/` and utilities in `transforms_class/utils/`.

### Run tests

```bash
python -m pytest
```

This automatically runs with coverage (configured in `pytest.ini`):
- **Terminal report**: lines not covered are shown inline
- **HTML report**: generated in `coverage_html/index.html`

### Pre-commit hook

The hook runs the full suite before every `git commit` and blocks the commit if any test fails.  
Install once with:
```bash
bash install_hooks.sh
```
> To bypass exceptionally: `git commit --no-verify`

## Contributing

Contributions are welcome! For suggestions, bug reports, or feature requests, please open an issue or pull request.

## License

This project is licensed under [LICENSE](LICENSE).

## Contact

Ines Khemir <ikhemir@hotmail.fr>
Romain Valabregue <romain.valabregue@upmc.fr>
