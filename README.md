TODO refaire presque enierement
# Synthetic MRI Generator: Code & Pipeline Overview

This repository provides a modular pipeline for generating synthetic MRI images and labels from anatomical label maps, with support for random morphology edits, spatial and intensity transforms, anomaly simulation, and flexible dataset configuration.

---

## Code Structure & Main Logic

### Main script: `generate.py`
- **Entry point** for the synthetic data generation pipeline.
- Loads configuration files (INI and JSON), parses dataset and transform parameters.
- Iterates over subjects and generates synthetic images/labels using a sequence of transforms.
- Records all steps, parameters, and random seeds in a CSV log for reproducibility.

### Key modules:
- `generate.py`: Orchestrates the full pipeline, argument parsing, config loading, and output management.
- `transforms_class/`: Custom TorchIO-compatible transforms for domain-specific deformations (e.g., MS, tumor, CSF, etc).
- `utils/`: Utility functions for file I/O, logging, and job management.
- `analyse/`: Scripts for post-generation analysis and metrics.
- `check_datasets/`: Scripts for dataset validation and label checking.

### Pipeline Steps (per subject):
1. **Load label map** (`.nii.gz`).
2. **Preprocessing** (optional): e.g., erosion, subregion addition, label remapping.
3. **Morphological edits**: Random dilations or boundary tweaks (CSF, GM, WM).
4. **Spatial transforms**: Affine, elastic, or custom deformations (configurable via JSON).
5. **Anomaly simulation**: Optional, via anomaly-specific transform configs (e.g., MS, tumor, ventricles).
6. **Label remapping**: To target and synthetic label sets (for ground truth and image synthesis).
7. **Label-to-image synthesis**: Generates intensity images from label maps (configurable via JSON).
8. **Intensity transforms**: Bias field, noise, motion, rescaling, etc.
9. **Pooling/downsampling**: Ensures output shapes are compatible and realistic.
10. **Save outputs**: NIfTI images (`Sim*.nii.gz`, `Lab*.nii.gz`, `4DLab*.nii.gz`), CSV logs, and anomaly maps.

All steps are modular and controlled by configuration files.

---

## Configuration System

The pipeline is controlled by a main INI file and several JSON files:

- **INI config**: Specifies datasets, label CSVs, subject directories, and which transform JSONs to use.
- **Transform JSONs**: Define the sequence and parameters for spatial, label-to-image, and intensity transforms.
- **Anomaly JSONs**: Define anomaly-specific transforms (e.g., MS, tumor, ventricles).

See `siam_config/v4.2/configs/` for real examples.

---

## Example Directory Layout

```bash
synthetic-mri-generator/
├── generate.py
├── requirements.txt
├── transforms_class/
├── utils/
├── analyse/
├── check_datasets/
├── ...
├── configs/
│   ├── configs.ini
│   ├── transform_spatial.json
│   ├── transform_lab2img.json
│   ├── transform_intensity.json
│   └── anomalies/
│       ├── transform_anomalies_deform_csf_ms.json
│       ├── transform_anomalies_deform_ventricules_head.json
│       └── transform_anomalies_tumor.json
```

---

## How to Use the Pipeline

### 1. Prepare configuration files
Each `[label_set.datasetX]` section defines a dataset. Example config files are shown for reference; you must create or adapt your own configs locally.
- Each transform JSON defines a pipeline step (spatial, label-to-image, intensity, anomaly).

### 2. Run the generator

```bash
python generate.py \
  --dataset 1 \
  --nb_gen 2 \
  --outputdir /path/to/configs \
  --config_filename configs.ini \
  --subject_idx 1
```

#### Main arguments
- `--dataset, -d`: Dataset section number (matches `[label_set.datasetX]` in INI).
- `--nb_gen, -n`: Number of synthetic examples per subject.
- `--outputdir, -o`: Directory containing configs and where outputs will be written.
- `--config_filename, -c`: Config file name (default=`configs.ini`).
- `--subject_idx, -s`: (Optional) Only generate for a specific subject index.

### 3. Outputs
- Synthetic images, labels, and logs are saved in the output directory, organized by dataset and anomaly.
- Anomaly maps are accumulated and saved per subject/anomaly type.

---

## Configuration File Details

### INI Example
```ini
[transforms]
spatial = transform_spatial.json
label_to_image = transform_lab2img.json
intensity = transform_intensity.json

[anomalies]
1 = anomalies/transform_anomalies_deform_csf_ms.json
2 = anomalies/transform_anomalies_deform_ventricules_head.json
3 = anomalies/transform_anomalies_tumor.json

[label_set.dataset1]
label_set_name = mida
subject_dir = /path/to/mida
labels_csv = /path/to/mida_labels.csv
preprocessing = /path/to/transform_preprocessing_mida.json
do_erode = True
file_regex = ^csf_veine_r025s05
synth_col = synth
```

### Label CSV Example
Each dataset expects a specific CSV structure. For example, MIDA:

```csv
Name,Name_original,synth,synth_tissue_29,synth_targetRegionFew,NameTargetRegionFew,Stratified_Tissue_Region
Background,Background,50,0,0,background,background
Dura,Dura,1,13,13,Dura,Dura
GM,Brain Gray Matter,10,1,1,GM,GM
WM,Brain White Matter,12,2,2,WM,WM
...etc
```

---

## Transform Pipeline Examples

### Spatial Transform JSON
```json
{
  "train_transforms": [
    { "name": "RandomAffine", "attributes": { "p": 0.95, "degrees": 20, "scales": 0.3, "translation": 10 } },
    { "name": "RandomElasticDeformation", "attributes": { "p": 0.75, "num_control_points": 18, "max_displacement": 6 } }
  ]
}
```

### Label-to-Image Transform JSON
```json
{
  "train_transforms": [
    { "name": "RandomLabelsToImageOneConstraint", "attributes": { "label_key": "label", "image_key": "synth_mri", "default_mean": [0, 1], "default_std": [0.005, 0.05], "minimal_contrasted": 0.1 } }
  ]
}
```

### Intensity Transform JSON
```json
{
  "train_transforms": [
    { "name": "RandomBiasField", "attributes": { "p": 1, "coefficients": 0.5, "order": 3 } },
    { "name": "RandomMotionFromTimeCourse", "attributes": { "p": 0.5, ... } },
    { "name": "RandomNoise", "attributes": { "mean": 0, "std": [0.01, 0.1] } },
    { "name": "RescaleIntensity", "attributes": { "out_min_max": [0, 1], "percentiles": [1, 99] } }
  ]
}
```

### Anomaly Transform JSON
See the `anomalies/` folder for detailed configs for each anomaly type (MS, tumor, ventricles, etc).

---

## Advanced Features

- **Anomaly simulation**: Add MS, tumor, or ventricle deformations using anomaly configs.
- **Flexible label mapping**: Each dataset can define its own label CSV and mapping logic.
- **Custom transforms**: Extend with your own transforms in `transforms_class/`.
- **Job batching**: Use the provided shell scripts to generate and submit SLURM jobs for large-scale data generation.

---

## Troubleshooting & Tips

- Check logs and CSV outputs for full provenance of each generated subject.
- Ensure your label CSV matches the expected columns for your dataset.
- For reproducibility, set random seeds at the start of your script.
- For performance, run on fast storage and batch jobs using the provided scripts.

---

## Contact

Dev: Ines Khemir <ikhemir@hotmail.fr>, Romain Valabregue <gitlab@mg.gitlab.com>