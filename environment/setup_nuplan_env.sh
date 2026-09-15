#!/bin/bash
# Build the CPU-only Python 3.9 environment for the nuPlan simulation stack (PDM-Closed, IDM).
#
# nuplan-devkit needs Python >= 3.9, while the main environment is 3.8 (tied to NVIDIA's Jetson torch build).
# Both external planners are rule-based and run on CPU. The devkit still imports, at module level:
#   * pytest, reached by the scenario builder and both planners (every nuPlan stage);
#   * torch, pytorch-lightning, timm, tensorboard and ray, reached by run_simulation.py (stage E1).
# Native dependencies come from conda-forge (aarch64 builds exist); pure-Python packages from pip, pinned to the
# versions used (environment/requirements-nuplan.txt). Usage: bash environment/setup_nuplan_env.sh [env name]
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
  aioboto3 boto3 s3fs moto mock docker grpcio grpcio-tools
"$P" -m pip install --no-input pytest==8.4.2 torch==2.3.1 torchvision==0.18.1 pytorch-lightning==2.6.0 \
  timm==1.0.29 tensorboard==2.21.0 ray==2.51.2 urllib3==1.26.20
# the pinned checkouts (environment/setup_third_party.sh), without letting pip re-resolve their dependency graph
"$P" -m pip install --no-deps -e third_party/nuplan_devkit
"$P" -m pip install --no-deps -e third_party/tuplan_garage

PREFIX="$(dirname "$(dirname "$P")")"
RAP_NUPLAN_PREFIX="$PREFIX" environment/bin/py-nuplan - <<'PY'
import importlib
for m in ("nuplan.planning.simulation.planner.idm_planner",
          "tuplan_garage.planning.simulation.planner.pdm_planner.pdm_closed_planner",
          "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder",
          "nuplan.planning.script.run_simulation"):
    importlib.import_module(m)
print("nuPlan environment ready: planners, scenario builder and run_simulation import")
PY
echo "export RAP_NUPLAN_PREFIX=$PREFIX"
