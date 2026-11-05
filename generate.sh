#!/usr/bin/env bash
# Usage: ./generate.sh <ANOMALY> <DATASET> <DATASET_NUM> <NUM_JOBS> <NB_CONTRAST>

set -euo pipefail

# -------------------- Args --------------------
if [[ "$#" -ne 5 ]]; then
    echo "Usage: $0 <ANOMALY> <DATASET> <DATASET_NUM> <NUM_JOBS> <NB_CONTRAST>"
    exit 1
fi

ANOMALY="$1"
DATASET="$2"
DATASET_NUM="$3"
NUM_JOBS="$4"
NB_CONTRAST="$5"

# (none=0, deform_csf=1, deform_ventricules_head=2, tumor=3)
case "$ANOMALY" in
    "none") ANOMALY_NUM=0 ;;
    "deform_csf_ms") ANOMALY_NUM=1 ;;
    "deform_ventricules_head") ANOMALY_NUM=2 ;;
    "tumor") ANOMALY_NUM=3 ;;
    *) echo "Unknown anomaly type: $ANOMALY" ; exit 1 ;;
esac

# -------------------- Config --------------------
PARTITION="bioinfo,compute"
CPUS=2
MEM="100G"
if [[ "$DATASET" == "allen" || "$DATASET" == "bigBrain" || "$DATASET" == "skull" ]]; then
    MEM="200G"
fi

TIME="1-00:00:00"
BASE_DIR="generate_anomalies"
REAL_PATH="$(readlink -f .)"
BASE_DIR="${REAL_PATH}/${BASE_DIR}"

JOB_DIR="${BASE_DIR}/jobs_gen_109"
JOB_SCRIPT_DIR="${JOB_DIR}/jobs_scripts"
LOG_DIR="${BASE_DIR}/logs_gen109/logs_${DATASET}/logs_${DATASET}_${ANOMALY}"
OUT_DIR="${BASE_DIR}/generation_${DATASET}/generation_${DATASET}_${ANOMALY}"

DATASET_LAUNCH_DIR="${JOB_DIR}/jobs_by_dataset"
ANOMALIES_LAUNCH_DIR="${JOB_DIR}/jobs_by_anomaly"

LAUNCHER="launch_${DATASET}_${ANOMALY}.sh"

mkdir -p "$JOB_DIR" "$JOB_SCRIPT_DIR" "$LOG_DIR" "$OUT_DIR"
mkdir -p "$DATASET_LAUNCH_DIR" "$ANOMALIES_LAUNCH_DIR"

# -------------------- JOB SCRIPT PATH --------------------
JOB_SCRIPT="${JOB_SCRIPT_DIR}/generation_${DATASET}_${ANOMALY}.sh"

# -------------------- SLURM ARRAY SCRIPT --------------------

cat > "$JOB_SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH -p ${PARTITION}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --array=1-${NUM_JOBS}
#SBATCH --job-name=gen_${DATASET}_${ANOMALY}
#SBATCH -o ${LOG_DIR}/log_${DATASET}_${ANOMALY}_%A_%a.out
#SBATCH -e ${LOG_DIR}/err_${DATASET}_${ANOMALY}_%A_%a.err

source /network/iss/cenir/software/irm/conda-forge/etc/profile.d/conda.sh
conda activate ptorch23

SOFT_PATH=/network/iss/cenir/software/irm/
export PYTHONPATH=\${SOFT_PATH}toolbox_python/torchio/src:\$PYTHONPATH

set -euo pipefail

python ${REAL_PATH}/generate.py \\
    -o ${OUT_DIR} \\
    -d ${DATASET_NUM} \\
    -c configs.ini \\
    -a ${ANOMALY_NUM} \\
    -z ${NB_CONTRAST}

EOF

chmod +x "$JOB_SCRIPT"

# -------------------- LAUNCHER PER ANOMALY --------------------
ANOMALIES_LAUNCH="${ANOMALIES_LAUNCH_DIR}/${LAUNCHER}"

if [[ ! -f "$ANOMALIES_LAUNCH" ]]; then
    cat > "$ANOMALIES_LAUNCH" <<EOF
#!/usr/bin/env bash
echo "Launching dataset/anomaly: ${DATASET}/${ANOMALY}"
EOF
    chmod +x "$ANOMALIES_LAUNCH"
fi

grep -qxF "sbatch ${JOB_SCRIPT}" "$ANOMALIES_LAUNCH" || echo "sbatch ${JOB_SCRIPT}" >> "$ANOMALIES_LAUNCH"

# -------------------- LAUNCHER PER DATASET --------------------
DATASET_LAUNCHER="${DATASET_LAUNCH_DIR}/launch_${DATASET}.sh"

if [[ ! -f "$DATASET_LAUNCHER" ]]; then
    cat > "$DATASET_LAUNCHER" <<EOF
#!/usr/bin/env bash
echo "Launching dataset: ${DATASET}"
EOF
    chmod +x "$DATASET_LAUNCHER"
fi

grep -qxF "sbatch ${JOB_SCRIPT}" "$DATASET_LAUNCHER" || echo "sbatch ${JOB_SCRIPT}" >> "$DATASET_LAUNCHER"

# -------------------- Done --------------------
echo "Created/updated:"
echo "  - Array job         : $JOB_SCRIPT"
echo "  - Launcher anomaly  : $ANOMALIES_LAUNCH"
echo "  - Launcher dataset  : $DATASET_LAUNCHER"