#!/bin/bash
# Script to create symbolic links for transform_*.json and configs.ini
# Usage: ./link_configs.sh

ANOMALIES=(deform_csf_ms deform_ventricules_head tumor)
DATASETS=(mida vascular1 skull bigBrain allen, vascular2)

for ANOMALY in "${ANOMALIES[@]}"; do
    SRC_DIR="generate_anomalies/generation_${ANOMALY}"
    for DATASET in "${DATASETS[@]}"; do
        DEST_DIR="${SRC_DIR}/generation_${ANOMALY}_${DATASET}"
        mkdir -p "$DEST_DIR"
        # Link all transform_*.json files
        for FILE in "$SRC_DIR"/transform_*.json; do
            [ -e "$FILE" ] || continue
            ln -sf "$(realpath --relative-to="$DEST_DIR" "$FILE")" "$DEST_DIR/"
        done
        # Link configs.ini if it exists
        if [ -e "$SRC_DIR/configs.ini" ]; then
            ln -sf "$(realpath --relative-to="$DEST_DIR" "$SRC_DIR/configs.ini")" "$DEST_DIR/"
        fi
    done
done

echo "Symbolic links created for all anomaly/dataset directories."
