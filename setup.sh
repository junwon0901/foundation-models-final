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
echo "==> [1/5] Upgrading pip tools"
python -m pip install --upgrade pip setuptools wheel

echo ""
echo "==> [2/5] Installing PyTorch 2.2.2 with CUDA 12.1"
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "==> Checking PyTorch installation"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo ""
echo "==> [3/5] Installing build tools"
pip install cmake ninja pybind11

echo ""
echo "==> [4/5] Installing torchmcubes"

if [ ! -f /usr/local/cuda/include/cuda.h ]; then
    echo "WARNING: /usr/local/cuda/include/cuda.h not found."
    echo "torchmcubes CUDA build may fail."
fi

export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

pip install --no-cache-dir git+https://github.com/tatsy/torchmcubes.git

echo ""
echo "==> [5/5] Installing remaining dependencies"
pip install -r requirement.txt

echo ""
echo "==> Final check"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

try:
    import torchmcubes
    print("torchmcubes: OK")
except Exception as e:
    print("torchmcubes: FAILED")
    print(e)
    raise
PY

echo ""
echo "Setup complete."
echo ""
echo "Run the demo with:"
echo "  conda activate $ENV_NAME"
echo "  python demo.py samples/sample_01.png --device cuda"
