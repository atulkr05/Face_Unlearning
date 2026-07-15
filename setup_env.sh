#!/bin/bash
# Face Identity Unlearning — Environment Setup
# Run from the challenge root: bash face_unlearning/setup_env.sh
set -e

VENV="/DATA2/Atul/2027/challenge/face_env"
PIP="$VENV/bin/pip"

echo "=== Installing core ML stack ==="
$PIP install --upgrade pip wheel setuptools

# PyTorch (CUDA 12.1 for H200)
$PIP install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing diffusion & transformer libraries ==="
$PIP install \
    diffusers==0.27.2 \
    transformers==4.40.1 \
    accelerate==0.30.1 \
    safetensors==0.4.3 \
    peft==0.10.0 \
    huggingface_hub==0.22.2

echo "=== Installing face analysis stack ==="
$PIP install \
    insightface==0.7.3 \
    onnxruntime-gpu==1.18.0 \
    opencv-python-headless==4.9.0.80

echo "=== Installing image & numerical libraries ==="
$PIP install \
    Pillow==10.3.0 \
    numpy==1.26.4 \
    scipy==1.13.0 \
    scikit-learn==1.4.2 \
    matplotlib==3.9.0

echo "=== Installing evaluation libraries ==="
$PIP install \
    clean-fid==0.1.35 \
    lpips==0.1.4 \
    pytorch-fid==0.3.0 \
    tqdm==4.66.4 \
    pandas==2.2.2 \
    einops==0.7.0

echo "=== Installing dev utilities ==="
$PIP install pytest ipython rich

echo ""
echo "=== All packages installed ==="
$VENV/bin/python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"
