#!/bin/bash
# Batch script to generate jobs for all anomaly/dataset combinations
# Usage: ./batch_generate.sh

ANOMALIES=(deform_csf_ms deform_ventricules_head tumor)
DATASETS=(mida vascular1 skull bigBrain allen vascular2)
DATASET_NUMS=(1 2 3 4 5 6)
NUM_JOBS=30
NB_CONTRAST=4

for i in ${!DATASETS[@]}; do
    DATASET=${DATASETS[$i]}
    DATASET_NUM=${DATASET_NUMS[$i]}
    for ANOMALY in "${ANOMALIES[@]}"; do
        # Skip deform_ventricules_head for bigBrain and allen datasets
        if [[ "$DATASET" == "bigBrain" && "$ANOMALY" == "deform_ventricules_head" ]]; then
            continue
        elif [[ "$DATASET" == "allen" && "$ANOMALY" == "deform_ventricules_head" ]]; then
            continue
        fi
        ./generate.sh "$ANOMALY" "$DATASET" "$DATASET_NUM" "$NUM_JOBS" "$NB_CONTRAST"
    done
done
echo "All job scripts generated for all anomaly/dataset combinations."
