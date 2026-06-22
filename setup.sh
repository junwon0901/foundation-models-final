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
echo "==> [1/3] Upgrading pip tools"
python -m pip install --upgrade pip setuptools wheel

echo ""
echo "==> [2/3] Installing PyTorch"
pip install torch torchvision

echo ""
echo "==> [3/3] Installing dependencies"
pip install -r requirement.txt

echo ""
echo "==> Final check"
python - <<'PY'
import torch
import numpy as np
import torchmcubes

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("numpy:", np.__version__)
print("torchmcubes: OK")
PY

echo ""
echo "Setup complete."
echo ""
echo "Run the demo with:"
echo "  conda activate $ENV_NAME"
echo "  python demo.py samples/sample_01.png --device cuda"
