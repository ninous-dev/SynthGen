#!/bin/bash
source /network/iss/cenir/software/irm/conda-forge/etc/profile.d/conda.sh
conda activate ptorch23

SOFT_PATH=/network/iss/cenir/software/irm/
export PYTHONPATH=${SOFT_PATH}toolbox_python/torchio/src:$PYTHONPATH
export ROOT_PROG=/network/iss/cenir/analyse/irm/users/ines.khemir/synth_ct_test/Synthetic-mri-generator
export PYTHONPATH=${ROOT_PROG}:$PYTHONPATH

ROOT_DIR="/network/iss/cenir/analyse/irm/users/ines.khemir/gen_testv5.0/anomalies/mida/Ano_deform_all_AnoMS"
ANALYSE_DIR="$ROOT_DIR/analyse"

mkdir -p "$ANALYSE_DIR"
echo "Analysis directory created"

python "${ROOT_PROG}/analyse/create_csv.py" --root_dir "$ROOT_DIR" --out_csv "$ANALYSE_DIR/subjects.csv"
echo "CSV created"

# Colonnes in voxels
for col in lab; do
    python "${ROOT_PROG}/analyse/compute_volumes.py" \
        --csv "$ANALYSE_DIR/subjects.csv" \
        --column "$col" --out_dir "$ANALYSE_DIR/volumes_voxels"
    echo "Volumes $col calculated"
done

# Colonnes in mm3
for col in amap hrlab original lowresnoles ano noleslab 4dlab; do
    python "${ROOT_PROG}/analyse/compute_volumes.py" \
        --csv "$ANALYSE_DIR/subjects.csv" \
        --column "$col" --in_mm3 --out_dir "$ANALYSE_DIR/volumes_mm3"
    echo "Volumes $col calculated"
done

python "${ROOT_PROG}/analyse/compare_volumes.py" \
    --csvs "$ANALYSE_DIR/volumes_lab_voxels.csv" "$ANALYSE_DIR/volumes_lab_voxels.csv" \
    --labels "Lab (voxels)" "Lab 2(voxels)" \
    --out_dir "$ANALYSE_DIR/volumes_comparison"
echo "Comparison plots created"

python "${ROOT_PROG}/analyse/compute_intensities.py" \
    --csv "$ANALYSE_DIR/subjects.csv" \
    --image_pattern "sim_" \
    --mask_column lab \
    --out_dir "$ANALYSE_DIR"
echo "Intensities computed"

# Compare intensities
python "${ROOT_PROG}/analyse/compare_all_intensities.py" \
    --csvs "$ANALYSE_DIR/intensities_sim_lab.csv" \
    --labels "sim (Lab)" \
    --out_dir "$ANALYSE_DIR/intensities"
echo "Intensity plots created"


echo ""
echo "========================================"
echo "Analysis complete! Files in: $ANALYSE_DIR"
ls -lh "$ANALYSE_DIR"/volumes_*.csv
echo "========================================"