#!/bin/bash
set -e

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "==> Creating conda environment: sketch3d (python=3.10)"
conda create -n sketch3d python=3.10 -y

echo "==> Activating sketch3d"
conda activate sketch3d

echo ""
echo "==> [1/3] Installing PyTorch"
pip install torch torchvision

echo ""
echo "==> [2/3] Installing torchmcubes"
if pip install git+https://github.com/tatsy/torchmcubes.git 2>/dev/null; then
    echo "torchmcubes installed with CUDA support."
else
    echo "CUDA build failed (likely header permissions). Falling back to CPU build."
    CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=NOTFOUND" pip install git+https://github.com/tatsy/torchmcubes.git
fi

echo ""
echo "==> [3/3] Installing remaining dependencies"
pip install -r requirement.txt

echo ""
echo "Setup complete."
echo "Run the demo with:"
echo "  conda activate sketch3d"
echo "  python demo.py samples/sample_01.png --device cuda"
