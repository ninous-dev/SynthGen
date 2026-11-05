# Running on the ICM Cluster (SLURM)

This guide explains how to use the provided scripts to launch synthetic MRI generation in batch mode on the ICM SLURM cluster.

## Configuration: `env.sh`

Before submitting any job, edit `jobs_run_cluster/env.sh` to match your environment:

| Variable | Description |
|----------|-------------|
| `conda activate <env>` | Name of your conda environment |
| `SOFT_PATH` | Path to the software directory containing TorchIO |
| `ROOT_PROG` | Root directory of this project |
| `GENERATION_DIR` | Output directory where generated data will be saved |
| `CONFIG_SRC` | Path to the `configs.ini` config file |
| `ANO_LIST` | Space-separated list of anomaly IDs to process (e.g. `"1 2"`) |
| `ANO_NAMES` | Corresponding anomaly names (same order as `ANO_LIST`) |
| `NB_JOBS` | Number of SLURM array jobs to submit |
| `NB_CONTRASTS` | Number of MRI contrasts to generate |
| `DEFAULT_DSNUM` | Default dataset index |

---

## Prerequisites
- Access to the ICM cluster with SLURM
- Compatible Python modules/environment (see `requirements.txt`)
- Scripts available in the `jobs_run_cluster/` folder

## Script Structure
- `makefile`: makes it easy to submit jobs for different datasets
- `generation_ano.sh`: bash script called by SLURM to launch the generation
- `env.sh`: environment script (modules, variables, etc.)

## Using the Makefile

Go to the `jobs_run_cluster` folder:

```bash
cd jobs_run_cluster
```

### Submit jobs for a dataset

- For MIDA:
  ```bash
  make sub_mida
  ```
- For Vascular:
  ```bash
  make sub_vascular
  ```
- For Vascular2:
  ```bash
  make sub_vascular2
  ```
- For Skull:
  ```bash
  make sub_skull
  ```
- For BigBrain:
  ```bash
  make sub_bigBrain
  ```
- For Allen:
  ```bash
  make sub_allen
  ```
- To launch all datasets:
  ```bash
  make sub_all
  ```
- To launch all with specific config and 1000 subjects:
  ```bash
  make sub_all_ano FULL=1
  ```

Each command prepares the log directories and submits SLURM jobs via the `generation_ano.sh` script with the appropriate parameters (dataset, memory, anomalies, etc.).

### Customization

You can modify in the `makefile`:
- Number of CPUs per task (`cpu_per_task`)
- Allocated memory (`mem`)
- Number of jobs in the array (`jobarray`)
- Datasets and anomalies to process (`DSname`, `DSnum`, `ANO_LIST`, etc.)

### Example of generated SLURM submission

The makefile generates commands like:

```bash
sbatch -p compute --array=1-50 --cpus-per-task=10 -t 1-00:00:00 --mem 70000 \
  --job-name=mida_csf_ms \
  -o /.../logs/mida_csf_ms-%A_%a.log \
  -e /.../logs/mida_csf_ms-%A_%a.err \
  generation_ano.sh -d 1 -z 3 -a 1
```

## Logs and Results
- Execution logs are in `.../logs/`
- Results are in the folder defined by `job_maindir` in the makefile

## Tips
- Check that the paths in the makefile match your working directories
- Adjust resources (CPU/memory) according to the job size
- Check the logs for monitoring and debugging

---

For any questions, contact the project maintainers.
