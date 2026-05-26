# Synthetic MRI Job Workflow

This document describes the workflow and scripts for generating synthetic MRI datasets and managing SLURM jobs.

## 1. Directory Preparation

Before generating jobs, create the necessary directory structure:

```bash
bash create_anomaly_dirs.sh
```

## 2. Batch Job Generation

Generate all job scripts and SLURM launchers for all anomaly/dataset combinations:

```bash
bash batch_generate.sh
```

This will populate the `generate_anomalies/` directory with job scripts and launchers.

## 3. Submitting Jobs

To submit all jobs for a specific anomaly or dataset, use the generated launcher scripts:

- For a specific anomaly:
	```bash
	bash generate_anomalies/jobs_gen_109/jobs_by_anomaly/launch_<DATASET>_<ANOMALY>.sh
	```
- For a specific dataset:
	```bash
	bash generate_anomalies/jobs_gen_109/jobs_by_dataset/launch_<DATASET>.sh
	```

## 4. Cleaning and Relaunching

To clean previously generated files and relaunch jobs:

```bash
bash clean_generated_file.sh
bash batch_generate.sh
```

## Notes
- Edit `generate.sh` to adjust job parameters, anomaly types, or dataset lists.
- Output, logs, and job scripts are organized by dataset and anomaly for clarity.
- Check the generated SLURM scripts before launching on the cluster.
