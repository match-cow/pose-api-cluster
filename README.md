# 6D Pose Estimation API — Cluster Backend (SLURM)

This backend provides a **Flask-based REST API** for 6-DoF object pose estimation using RGB-D input and 3D meshes. It is designed to run on **SLURM-managed GPU clusters**.

Each API request is processed by launching a new SLURM job via `srun`, using [FoundationPose](https://github.com/NVlabs/FoundationPose) as the default model. The backend is modular — you can swap in other 6D pose models with minimal changes.

> This is an extension of the [original single-GPU backend](https://github.com/match-now/pose-api), adapted for cluster environments.

---

## 1. Overview

- **Input**: RGB image(s), depth map(s), binary mask, camera intrinsics, and a `.ply` mesh — all base64-encoded  
- **Output**: 4×4 SE(3) object-to-camera pose matrix (per frame)  
- **Interface**: REST API (JSON over HTTP)  
- **Execution**: Each request spawns a separate SLURM job using `srun`  
- **Requirements**: GPU-enabled compute node for both server and inference jobs  

This project was developed and tested on the [LUIS HPC Cluster](https://docs.cluster.uni-hannover.de/doku.php?id=start), but should work on any SLURM-based cluster with GPU access and support for `conda`, `CUDA`, and Python.

---

## 2. Repository Structure

```
pose-api/
├── FoundationPose/               # FoundationPose backend (run_demo.py, weights/, etc.)
│   ├── run_demo.py
│   ├── weights/
│   ├── debug/
│   └── ...
├── pose_api_server.py            # Flask server logic
├── run_request_once.sh           # SLURM job script to handle each request
├── launch_cluster_server.sh      # Starts Flask API with proper env setup
├── start_compute_node.sh         # Helper to launch GPU-enabled interactive shell
├── pose_api.log                  # Flask server log file
└── README.md
```

> A modified copy of [FoundationPose](https://github.com/NVlabs/FoundationPose) is included under `FoundationPose/`, along with pretrained weights. No additional downloads are required.

---

## 3. Setup Instructions

### 3.1 Clone the Repository Locally

On your own machine:

```bash
git clone https://github.com/match-now/pose-api.git
cd pose-api
```

---

### 3.2 Upload the Repo to Your Cluster

If you're not already working directly on the cluster, use `scp` or `rsync` to upload:

```bash
# Example using scp
scp -r pose-api youruser@your.cluster.edu:/bigwork/youruser/

# Or using rsync
rsync -av pose-api/ youruser@your.cluster.edu:/bigwork/youruser/pose-api/
```

> If your cluster allows it, you can also clone the repo directly on the login node.

---

### 3.3 Request a GPU Compute Node

> You **must run the server on a compute node with a GPU**, or it will crash when loading CUDA-based models.

From the login node:

```bash
cd /bigwork/youruser/pose-api
bash start_compute_node.sh
```

This runs:

```bash
srun --partition=gpu --gres=gpu:1 --pty --time=01:00:00 --mem=16G bash --login
```

You now have a terminal on a GPU node.

---

### 3.4 Set Up the Python Environment (First Time Only)

In the GPU shell:

```bash
module purge
module load GCC/11.2.0
module load Eigen/3.4.0
module load CUDA/11.8.0
module load Miniconda3/24.7.1-0

# Set up conda
source /sw/apps/software/noarch/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
conda create -y -p /bigwork/youruser/foundationpose python=3.9
conda activate /bigwork/youruser/foundationpose
```

Then install dependencies:

```bash
cd FoundationPose
pip install -r requirements.txt

# Replace problematic packages
pip uninstall -y numpy scipy
pip install numpy==1.24.4 scipy==1.10.1

# GPU libraries
pip install --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git
pip install --no-cache-dir kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html
```

Optional (if using BundleSDF):

```bash
cd bundlesdf/mycuda
python setup.py install
```

---

## 4. Running the Server

### Step 1 — Start the Server

Still inside the GPU node shell:

```bash
cd /bigwork/youruser/pose-api
bash launch_cluster_server.sh
```

This:
- Loads modules
- Activates the conda environment
- Sets environment variables (e.g., `DIR`)
- Kills any process already on port 5000
- Starts the Flask server in the background

Logs are saved to `pose_api.log`. You can monitor them with:

```bash
tail -f pose_api.log
```

---

### Step 2 — Confirm It’s Running

```bash
curl http://localhost:5000
# Should return: "it is running!"
```

If something else is already on port 5000, run:

```bash
lsof -i :5000
kill <PID>
```

---

## 5. API Usage

### 5.1 Endpoint

```
POST /foundationpose
Content-Type: application/json
```

---

### 5.2 Request Format

```json
{
  "camera_matrix": [
    ["fx", 0, "cx"],
    [0, "fy", "cy"],
    [0, 0, 1]
  ],
  "images": [
    {
      "filename": "scene1",
      "rgb": "<base64 encoded PNG>",
      "depth": "<base64 encoded PNG>"
    }
  ],
  "mask": "<base64 encoded PNG>",
  "mesh": "<base64 encoded PLY>"
}
```

- Images must match in size  
- `.ply` mesh only  
- Only one object per request (mask + mesh is shared across frames)

---

### 5.3 Example Request (cURL)

```bash
curl -X POST http://localhost:5000/foundationpose \
     -H "Content-Type: application/json" \
     -d @request.json | jq
```

> Tip: You can reuse `saved_requests/<uuid>/` from previous jobs to build new requests by re-encoding files to base64.

---

## 6. Output

### 6.1 JSON Response

```json
{
  "status": "Pose estimation complete",
  "transformation_matrix": [
    ["r11", "r12", "r13", "tx"],
    ["r21", "r22", "r23", "ty"],
    ["r31", "r32", "r33", "tz"],
    [0, 0, 0, 1]
  ]
}
```

The matrix is SE(3), row-major, and maps object → camera.

Pose validity is checked before returning:
- Rotation must be orthogonal (RᵀR ≈ I)
- Determinant of rotation must be ≈ 1

---

### 6.2 Files Saved Per Request

```
FoundationPose/saved_requests/<uuid>/
├── cam_K.txt
├── rgb/scene1.png
├── depth/scene1.png
├── masks/scene1.png
└── mesh/scene1.ply

FoundationPose/debug/ob_in_cam/scene1.txt
```

---

## 7. Runtime Behavior

### Architecture

```
[Client sends JSON request]
        ↓
[Flask server on GPU node]
        ↓
[srun launches run_request_once.sh]
        ↓
[run_demo.py runs inference job]
        ↓
[4×4 matrix saved and returned]
```

After inference, GPU memory is cleared:

```python
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
gc.collect()
```

---

## 8. Error Codes

| Code | Meaning                             |
|------|-------------------------------------|
| 200  | Success                             |
| 400  | Invalid input fields or shapes      |
| 401  | Empty or malformed JSON             |
| 402  | Nested or base64 decoding error     |
| 500  | Pose failed or invalid matrix       |

---

## 9. Limitations

- Only `.ply` mesh supported  
- One object per request (shared mask + mesh)  
- Requires GPU (no CPU fallback)  
- Flask is not production-ready — use `gunicorn` for deployment  
- Port conflicts must be resolved manually  
- One request handled at a time (no queueing/multiprocessing)

---

## 10. Extending to Other Models

You can integrate another model by:

- Replacing `run_pose_estimation()` in `pose_api_server.py`
- Maintaining the same input/output interface:
  - Input: base64 JSON
  - Output: 4×4 SE(3) matrix

Suggested replacements: GDR-Net, CosyPose, your own tracker, etc.

---

## 11. Attribution

This backend uses [FoundationPose](https://github.com/NVlabs/FoundationPose):

```bibtex
@article{wen2023foundationpose,
  title     = {FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects},
  author    = {Bowen Wen and Wei Yang and Jan Kautz and Stan Birchfield},
  journal   = {arXiv preprint arXiv:2312.08344},
  year      = {2023},
  url       = {https://arxiv.org/abs/2312.08344}
}
```

Please cite their work if you use this system in research or development.
