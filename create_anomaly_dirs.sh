#!/bin/bash
# Script to create directories for config files in generate_anomalies/ for each anomaly type
# Usage: ./create_anomaly_dirs.sh

ANOMALIES=(deform_csf_ms deform_ventricules_head tumor)
DATASETS=(mida vascular1 skull bigBrain allen vascular2)

for DATASET in "${DATASETS[@]}"; do
    DIR="generate_anomalies/generation_${DATASET}"
    mkdir -p "$DIR"
    echo "Created directory: $DIR"

    for ANOMALY in "${ANOMALIES[@]}"; do
        SUBDIR="${DIR}/generation_${DATASET}_${ANOMALY}"
        mkdir -p "$SUBDIR"
        echo "Created subdirectory: $SUBDIR"

        ln -sf /network/iss/opendata/data/template/siam_label/v4.2/configs/configs.ini "$SUBDIR/configs.ini"

    done
done

echo "All anomaly/dataset config directories created."
