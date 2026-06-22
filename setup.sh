#!/bin/bash
set -e

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "==> Creating conda environment: sketch3d (python=3.10)"
conda create -n sketch3d python=3.10 -y

echo "==> Activating sketch3d"
conda activate sketch3d

echo ""
echo "==> [1/4] Installing PyTorch"
pip install torch torchvision

echo ""
echo "==> [2/4] Installing CUDA toolkit into conda env"
conda install -c "nvidia/label/cuda-13.1.0" cuda-toolkit cuda-nvcc -y 2>/dev/null || \
conda install -c nvidia cuda-toolkit cuda-nvcc -y

echo ""
echo "==> [3/4] Installing torchmcubes"
CMAKE_ARGS="-DCUDA_INCLUDE_DIRS=$CONDA_PREFIX/targets/x86_64-linux/include" \
  PATH=$CONDA_PREFIX/bin:$PATH \
  pip install git+https://github.com/tatsy/torchmcubes.git

echo ""
echo "==> [4/4] Installing remaining dependencies"
pip install -r requirement.txt

echo ""
echo "Setup complete."
echo "Run the demo with:"
echo "  conda activate sketch3d"
echo "  python demo.py samples/sample_01.png --device cuda"
