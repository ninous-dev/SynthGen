source /network/iss/cenir/software/irm/conda-forge/etc/profile.d/conda.sh
conda activate ptorch23

SOFT_PATH=/network/iss/cenir/software/irm/
export PYTHONPATH=${SOFT_PATH}toolbox_python/torchio/src:$PYTHONPATH
export ROOT_PROG=/network/iss/cenir/analyse/irm/users/ines.khemir/synth_v5.0/Synthetic-mri-generator
# export ROOT_PROG=/network/iss/cenir/analyse/irm/users/ines.khemir/synth_ct_test/Synthetic-mri-generator

# V4.2
# export GENERATION_DIR=/network/iss/cenir/analyse/irm/users/ines.khemir/gen_test42/anomalies
# export CONFIG_SRC=/network/iss/opendata/data/template/siam_label/v4.2/configs/configs.ini

# # V5.0
# export GENERATION_DIR=/network/iss/cenir/analyse/irm/users/ines.khemir/gen_testv5.0/anomalies
# export CONFIG_SRC=/network/iss/opendata/data/template/siam_label/v5.0_test/configs/configs.ini

# # ct_test
# export GENERATION_DIR=/network/iss/cenir/analyse/irm/users/ines.khemir/gen_testtopbrain/anomalies
# export CONFIG_SRC=/network/iss/opendata/data/template/siam_label/ct_test_config/configs/configs.ini

# V6.0
# export GENERATION_DIR=/network/iss/cenir/analyse/irm/users/ines.khemir/gen_testv6.0/deform_csf_without_affine
# export CONFIG_SRC=/network/iss/opendata/data/template/siam_label/v6.0_test/configs/configs.ini
export GENERATION_DIR_TEST=/network/iss/cenir/analyse/irm/users/ines.khemir/gen_testv6.0/test_all_ano
export CONFIG_SRC_TEST=/network/iss/opendata/data/template/siam_label/v6.0_test/configs/configs.ini

export GENERATION_DIR=/network/iss/cenir/analyse/irm/users/ines.khemir/gen_testv6.0/anomalies_final
export CONFIG_SRC=/network/iss/opendata/data/template/siam_label/v6.0/configs/configs.ini

export ANO_LIST="1 2 3 4 5 6"
export ANO_NAMES="deform_all_AnoTumor deform_all_AnoMS deform_all_AnoMSnohead deform_all_AnoTumornohead deform_all_Ano deform_all_AnoNoHead"
export NB_JOBS=3
export NB_CONTRASTS=4
export DEFAULT_DSNUM=2
export PARTITION=bioinfo
export CPU_PER_TASK=24
export DEFAULT_MEM=100G
export DEFAULT_TIMELIM=1-00:00:00

export DSNAME=(mida vascular skull bigBrain allen vascular2)
