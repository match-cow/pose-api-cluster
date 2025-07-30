# 6D Pose Estimation API (Cluster Backend)

This repository provides a Flask-based REST API for 6D object pose estimation on SLURM-managed GPU clusters.  
Each inference request is dispatched as a separate compute job using `srun`.

The system is designed to be **backend-agnostic**: it currently uses [FoundationPose](https://github.com/NVlabs/FoundationPose) as the default model, but can be extended to support other pose estimation pipelines. The input is base64-encoded RGB-D, mask, and mesh; the output is a 4×4 SE(3) pose matrix.

---

## Features

- REST API endpoint: `/foundationpose`
- Accepts base64-encoded RGB-D input, mask, and PLY mesh
- Launches per-request compute jobs via SLURM (`srun`)
- Uses FoundationPose to return 6D pose as 4×4 matrix
- Modular backend architecture for future models

---

## Folder Structure

```
foundationpose/
├── FoundationPose/              # FoundationPose codebase (run_demo.py, weights, etc.)
├── pose_api_server.py           # Flask server
├── run_request_once.sh          # Launches per-request SLURM job
├── launch_cluster_server.sh     # Starts the API server with correct environment
├── start_compute_node.sh        # Optional helper to open interactive compute shell
└── README.md
```

---

## Environment Setup (on Compute Node)

This project was developed and tested on the [LUIS HPC Cluster](https://docs.cluster.uni-hannover.de/doku.php?id=start), which uses SLURM for resource management and supports modules, Miniconda, and GPU nodes.

```bash
# Load modules
module purge
module load GCC/11.2.0
module load Eigen/3.4.0
module load CUDA/11.8.0
module load Miniconda3/24.7.1-0

# Set up Conda
source /sw/apps/software/noarch/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
conda create -y -p /bigwork/youruser/foundationpose python=3.9
conda activate /bigwork/youruser/foundationpose

# Install dependencies
cd /bigwork/youruser/foundationpose/FoundationPose
pip install -r requirements.txt

# Replace problematic packages
pip uninstall -y numpy scipy
pip install numpy==1.24.4 scipy==1.10.1

# GPU libraries
pip install --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git
pip install --no-cache-dir kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

# Set CUDA paths
export CUDA_HOME=/sw/apps/software/arch/Core/CUDA/11.8.0
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

(Optional) Compile BundleSDF extension:

```bash
cd bundlesdf/mycuda
python setup.py install
```

---

## Running the API Server

To run the Flask server with CUDA and graphics support, launch it **from a compute node**.

### Step 1: Allocate a GPU node

Run this from the login node:

```bash
./start_compute_node.sh
```

Which does:

```bash
srun --partition=gpu --gres=gpu:1 --pty --time=01:00:00 --mem=16G bash --login
```

This opens a GPU-enabled interactive shell.

### Step 2: Start the Flask server

Inside the compute node:

```bash
cd /path/to/foundationpose
./launch_cluster_server.sh
```

This script:

- Loads modules (CUDA, Miniconda, etc.)
- Activates the environment
- Exports `CUDA_HOME` and `LD_LIBRARY_PATH`
- Exports `DIR` used internally
- Starts `pose_api_server.py` on port 5000
- Logs output to `pose_api.log`

You may need to kill port 5000 manually if restarting:

```bash
lsof -i :5000
kill <PID>
```

---

## API Usage

### Endpoint

```
POST /foundationpose
Content-Type: application/json
```

### Request JSON

```json
{
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "images": [
    {
      "filename": "scene1",
      "rgb": "<base64 PNG>",
      "depth": "<base64 PNG>"
    }
  ],
  "mask": "<base64 PNG>",
  "mesh": "<base64 PLY>"
}
```

### Response JSON

```json
{
  "status": "Pose estimation complete",
  "transformation_matrix": [[...], [...], [...], [...]]
}
```

### Error Responses

| Code | Meaning                                |
|------|----------------------------------------|
| 400  | Missing or invalid fields              |
| 401  | JSON parse error                       |
| 402  | Base64 decode error                    |
| 500  | Inference or matrix validation failure |

---

## Runtime Behavior

The system uses a two-tier architecture:

1. **API Server (Flask)** runs on a GPU compute node (started manually via srun):
   - Receives and validates incoming POST requests
   - Writes input data to `saved_requests/<uuid>/`
   - Launches a separate job using `srun` to run inference

2. **SLURM Job** is dispatched per request:
   - Runs on a compute node with GPU
   - Executes `run_request_once.sh`, which launches `run_demo.py`
   - Writes 4×4 pose matrix to `debug/ob_in_cam/`
   - Returns result to Flask for response

---

## Output Structure (Per Request)

```
saved_requests/<uuid>/
├── cam_K.txt
├── rgb/       scene1.png
├── depth/     scene1.png
├── masks/     scene1.png
└── mesh/      scene1.ply

debug/ob_in_cam/scene1.txt       # Output 4×4 matrix
```

---

## Matrix Validation

The returned transformation matrix is checked before sending:

- The 3×3 rotation block must be orthonormal
- Determinant of the rotation must be ~1.0
- Invalid matrices trigger a 500 error response

---

## Architecture Overview

```
[Client Request]
       ↓
[Flask API Server (on compute node)]
       ↓
   srun launches new job
       ↓
[Second compute node runs run_demo.py]
       ↓
[Pose matrix saved → Flask returns JSON]
```

---

## Notes

- Each request is isolated; inference runs on its own compute node.
- Only one request is handled at a time (no parallel Flask workers).
- Assumes large-memory GPUs (e.g. A100 80GB).
- Flask is for demonstration. For production, use a WSGI server (e.g. gunicorn).
- Port conflicts might have to be managed manually (port 5000).
