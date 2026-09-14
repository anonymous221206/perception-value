#!/bin/bash
# Build the CPU-only Python 3.9 environment for the nuPlan simulation stack (PDM-Closed, IDM).
#
# nuplan-devkit needs Python >= 3.9, while the main environment is 3.8 (tied to NVIDIA's Jetson torch build).
# Both external planners are rule-based, so the simulation stack installs without GPU torch.  Native
# dependencies come from conda-forge (aarch64 builds exist); pure-Python packages from pip.  Exact versions used:
# environment/requirements-nuplan.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
ENV="${1:-nuplan}"
MAMBA="${MAMBA:-mamba}"

"$MAMBA" create -y -n "$ENV" -c conda-forge python=3.9 \
  numpy=1.23 scipy pandas pyarrow "shapely>=2.0" geopandas fiona rasterio pyogrio rtree \
  py-opencv casadi scikit-learn matplotlib pillow tqdm joblib cachetools psutil requests \
  urllib3 sympy typer ujson aiofiles nest-asyncio sqlalchemy=1.4.27 bokeh=2.4.3 \
  "hydra-core>=1.1" pyquaternion tornado jupyter

P="$("$MAMBA" run -n "$ENV" python -c 'import sys; print(sys.executable)')"
"$P" -m pip install --no-input retry positional-encodings control guppy3 pyinstrument \
  aioboto3 boto3 s3fs moto mock docker grpcio grpcio-tools || true
# the pinned checkouts (environment/setup_third_party.sh), without letting pip re-resolve their dependency graph
"$P" -m pip install --no-deps -e third_party/nuplan_devkit
"$P" -m pip install --no-deps -e third_party/tuplan_garage
"$P" -c "import nuplan, tuplan_garage; print('nuPlan environment ready')"
echo "export RAP_NUPLAN_PREFIX=$(dirname "$(dirname "$P")")"
