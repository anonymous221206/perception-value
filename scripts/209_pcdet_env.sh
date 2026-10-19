#!/usr/bin/env bash
# Follow-up E7, on the training machine: a private environment for OpenPCDet's released nuScenes LiDAR detectors.
# Nothing of the machine or of other environments is changed: conda prefix under deep_ext, its own CUDA compiler and
# C++ compiler (the system gcc is newer than nvcc supports), PyTorch and spconv from their wheels, OpenPCDet at a
# fixed commit with its CUDA ops built for the A5000 (sm_86). Log: $D/logs/pcdet_env.log
set -eu
B=${B:?set B to the work root on the GPU machine}
D=$B/deep_ext
E=$D/envs/pcdet
SRC=$D/third_party/OpenPCDet
COMMIT=233f849829b6ac19afb8af8837a0246890908755   # master, 2025-10-08 (the repository has no release tag after v0.5.2)
CONDA=$B/miniforge3/bin/conda
export TMPDIR=$B/tmp PIP_CACHE_DIR=$D/pip_cache
mkdir -p $D/envs $D/third_party $D/logs $TMPDIR
[ -x $E/bin/python ] || $CONDA create -y -p $E -c conda-forge python=3.10 gxx_linux-64=11 gcc_linux-64=11 ninja
$CONDA install -y -p $E -c nvidia/label/cuda-12.1.1 cuda-nvcc cuda-cudart-dev cuda-cccl cuda-libraries-dev
P=$E/bin/python
$P -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
$P -m pip install "setuptools<70"          # torch 2.1's extension builder imports pkg_resources
$P -m pip install spconv-cu120==2.3.6 numpy==1.26.4 numba==0.59.1 llvmlite==0.42.0 easydict pyyaml scikit-image \
  tqdm SharedArray tensorboardX nuscenes-devkit==1.1.11 gdown==5.2.0 transformers==4.45.2 onnx==1.16.2 \
  av2==0.3.6 kornia==0.6.12          # OpenPCDet imports its Argoverse 2 dataset (av2) unconditionally; kornia 0.6 for torch 2.1
$P -m pip install --no-deps torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html   # the dynamic pillar encoder of CenterPoint-PointPillar
[ -d $SRC/.git ] || git clone https://github.com/open-mmlab/OpenPCDet.git $SRC
git -C $SRC checkout -q $COMMIT
cd $SRC
export CUDA_HOME=$E TORCH_CUDA_ARCH_LIST=8.6 CC=$E/bin/x86_64-conda-linux-gnu-gcc CXX=$E/bin/x86_64-conda-linux-gnu-g++
$P setup.py develop
$P -c "import torch, spconv, pcdet; from pcdet.ops.iou3d_nms import iou3d_nms_cuda; print('ok', torch.__version__, torch.version.cuda, pcdet.__version__)"
git -C $SRC rev-parse HEAD
