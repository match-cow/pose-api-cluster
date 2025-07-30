#!/bin/bash

REQ_PATH=$1
FILENAME=$2

module purge
module load GCC/11.2.0
module load Eigen/3.4.0
module load Miniconda3/24.7.1-0
module load CUDA/11.8.0

source /sw/apps/software/noarch/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
conda activate /bigwork/nhkwmeng/foundationpose

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

cd /bigwork/nhkwmeng/foundationpose/FoundationPose

PYTHONPATH=. python run_demo.py \
    --test_scene_dir "$REQ_PATH" \
    --mesh_file "$REQ_PATH/mesh/$FILENAME.ply" \
    --debug_dir "/bigwork/nhkwmeng/foundationpose/FoundationPose/debug"

