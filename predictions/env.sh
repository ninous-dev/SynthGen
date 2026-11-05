# source /network/iss/cenir/software/irm/conda-forge/etc/profile.d/conda.sh
# conda activate ptorch23

# SOFT_PATH=/network/iss/cenir/software/irm/
# export PYTHONPATH=${SOFT_PATH}toolbox_python/torchio/src:$PYTHONPATH

# export SIAM_MODEL_DIR=/network/iss/cenir/analyse/irm/users/ines.khemir/predictions/models
# # export SIAM_MODEL_DIR="/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet"
# export nnUNet_raw="/network/iss/cenir/analyse/irm/users/ines.khemir/nnunet/"
# # export nnUNet_raw="/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet"
# export nnUNet_preprocessed=$nnUNet_raw"Preproc/"
# export nnUNet_results=$nnUNet_raw"Results/"

source /network/iss/cenir/software/irm/conda-forge/etc/profile.d/conda.sh
conda activate ptorch23

SOFT_PATH=/network/iss/cenir/software/irm/
export PYTHONPATH=${SOFT_PATH}toolbox_python/torchio/src:$PYTHONPATH

export nnUNet_raw="/network/iss/cenir/analyse/irm/users/ines.khemir/nnunet/"
export nnUNet_preprocessed=$nnUNet_raw"Preproc/"
export nnUNet_results=$nnUNet_raw"Results/"

export SIAM_MODEL_DIR="/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/Results"

echo "nnUNet_raw: $nnUNet_raw"
# siam-pred -i /network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/Brast_LGG/Flair_LGG -o /network/iss/cenir/analyse/irm/users/ines.khemir/predictions/Brats_LGG_Flair -m 50 -device cpu
export DATA_DIR="/network/iss/cenir/analyse/irm/users/ines.khemir/predictions/Brats_LGG_Flair_70"
siam-pred -i $DATA_DIR/Flair_LGG -o $DATA_DIR/predictions -m 70 -device cpu
