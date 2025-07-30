#!/bin/bash

module load GCC/11.2.0
module load Eigen/3.4.0
module load Miniconda3/24.7.1-0
module load CUDA/11.8.0

# Set CUDA environment
export CUDA_HOME=/sw/apps/software/arch/Core/CUDA/11.8.0
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Activate Conda environment
source /sw/apps/software/noarch/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
conda activate /bigwork/nhkwmeng/foundationpose

# Set DIR used in pose_api_server.py
export DIR=/bigwork/nhkwmeng/foundationpose

# Kill anything on port 5000
PID=$(ss -tulnp | grep ':5000' | awk '{print $NF}' | cut -d',' -f2 | cut -d'=' -f2)
if [ -n "$PID" ]; then
    echo "Port 5000 in use by PID $PID. Killing it..."
    kill -9 $PID
fi

# Start the server
cd /bigwork/nhkwmeng/foundationpose/FoundationPose
PYTHONPATH=. python ../pose_api_server.py > ../pose_api.log 2>&1 &
