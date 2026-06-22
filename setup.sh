#!/bin/bash
set -e

ENV_NAME=sketch3d
PYTHON_VERSION=3.10

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "==> Creating conda environment: $ENV_NAME (python=$PYTHON_VERSION)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "ERROR: Conda environment '$ENV_NAME' already exists."
    echo "Remove it first with:"
    echo "  conda deactivate"
    echo "  conda env remove -n $ENV_NAME"
    exit 1
fi

conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

echo ""
echo "==> Activating $ENV_NAME"
conda activate "$ENV_NAME"

echo ""
echo "==> [1/6] Upgrading pip tools"
python -m pip install --upgrade pip setuptools wheel

echo ""
echo "==> [2/6] Installing NumPy compatibility version"
pip install "numpy<2"

echo ""
echo "==> [3/6] Installing PyTorch 2.7.0 with CUDA 12.8"
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

echo ""
echo "==> Checking PyTorch installation"
python - <<'PY'
import torch
import numpy as np

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("numpy:", np.__version__)

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY

echo ""
echo "==> [4/6] Installing CUDA 12.8 toolkit into conda env"
conda install -c "nvidia/label/cuda-12.8.0" cuda-toolkit cuda-nvcc -y

echo ""
echo "==> Checking CUDA toolkit"
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export CUDAToolkit_ROOT="$CONDA_PREFIX"
export CUDA_TOOLKIT_ROOT_DIR="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$CONDA_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

which nvcc
nvcc --version

echo ""
echo "==> [5/6] Installing build tools"
conda install -c conda-forge cmake ninja -y
pip install scikit-build-core pybind11

echo ""
echo "==> [6/6] Installing torchmcubes with CUDA 12.8"

export CMAKE_PREFIX_PATH="$(python - <<'PY'
import torch
print(torch.utils.cmake_prefix_path)
PY
)"

export CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc -DCUDAToolkit_ROOT=$CONDA_PREFIX -DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX -DCUDA_INCLUDE_DIRS=$CONDA_PREFIX/targets/x86_64-linux/include"

pip install --no-cache-dir --no-build-isolation git+https://github.com/tatsy/torchmcubes.git

echo ""
echo "==> Installing remaining dependencies"
pip install -r requirement.txt

echo ""
echo "==> Final check"
python - <<'PY'
import torch
import numpy as np

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("numpy:", np.__version__)

try:
    import torchmcubes
    print("torchmcubes: OK")
except Exception as e:
    print("torchmcubes: FAILED")
    print(e)
    raise

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY

echo ""
echo "Setup complete."
echo ""
echo "Run the demo with:"
echo "  conda activate $ENV_NAME"
echo "  python demo.py samples/sample_01.png --device cuda"
