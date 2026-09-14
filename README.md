# Allocating perception compute by decision value: code release

This repository contains the code, intermediate caches and final results for an evaluation of when extra
perception compute changes a downstream decision.

The perception pair is YOLOv8s at 320 and 640 input resolution. Its effect on decisions is measured on three
tracks:

| track | data | downstream decision makers |
|---|---|---|
| nuScenes | nuScenes | braking controller; PKL's published planner |
| KITTI | KITTI tracking | braking controller; Planner B |
| nuPlan | nuPlan mini | published PDM-Closed and IDM planners, first under a transported detection profile, then under real detections |

Allocation signals are scored under frame quotas and under measured latency and energy budgets.

## Repository layout

```
README.md, LICENSE, reproduce.py   entry points
configs/        frozen splits, benchmark scenario list, paths.env.example
data/           dataset instructions and helpers; data/cache/ = shipped intermediate caches
docs/           result reports and generated tables
environment/    environment pins, setup scripts, device check, Python wrappers (bin/)
results/final/  every final table, check file and figure-data export
results/raw/    shipped intermediate runs that later stages read
scripts/        the pipeline, numbered in order (index: scripts/README.md); scripts/legacy/ = early exploration
src/rap/        library code
tests/          unit tests and negative controls
third_party/    PROVENANCE.md; prior-work code is cloned here at pinned commits
```

## 1. Reproduce every table from the shipped caches (any machine, CPU only)

`data/cache/` and `results/raw/` ship the detections, decision tables, planner outputs and nuPlan branch results
that the paper's tables are computed from. The statistics on top of them are deterministic, so on any machine:

```bash
conda create -n rap python=3.8 && conda activate rap
pip install -r environment/requirements-cached.txt
cp configs/paths.env.example configs/paths.env    # set RAP_EDGE_PREFIX to this environment's prefix
source configs/paths.env
python reproduce.py --tier cached --verify
```

`--verify` regenerates each table and compares it with the shipped file. Numbers are compared to 1e-9 relative
and text exactly. `python -m pytest tests` runs the unit tests; tests that need KITTI files are skipped when the
dataset is absent.

| result | file(s) | stage |
|---|---|---|
| benchmark table (frame quotas, all tracks, all baselines) | `results/final/benchmark_table.csv`, `benchmark_cells.csv`, `docs/benchmark_tables.md` | C2–C3 |
| per-mode operating-point calibration | `results/final/calibration_*.csv/json`, `docs/calibration_tables.md` | C4–C5 |
| detection-list and pixel routers | `results/final/benchmark_table_routers.csv`, `docs/routers_tables.md` | C6–C7 |
| measured latency and energy budgets | `results/final/benchmark_budget_routers.csv` | full tier (H6) |
| nuPlan with real perception | `results/final/nuplan_real_perception_*.csv/json`, overlays, `docs/nuplan_real_perception_tables.md` | C9–C11 |
| nuPlan allocation on real-perception decision values | `results/final/benchmark_table_nuplan_real.csv`, `benchmark_budget_nuplan_real.csv` | C12 |
| figure data (BEV objects, gallery, budget curves) | `results/final/fig_*` | full tier (L1) |

Reports that interpret these tables are in `docs/` (`iclr_*.md`).

## 2. Full reproduction from raw data

1. Datasets. Download nuScenes (v1.0-trainval metadata, blob 01, map expansion), KITTI tracking and nuPlan
   (v1.1 mini split, maps) and arrange them as described in `data/README.md`. Then run
   `python data/verify_datasets.py`.
2. Weights: `bash data/download_models.sh`.
3. Prior-work code at pinned commits: `bash environment/setup_third_party.sh`.
4. Environments.
   * Main: Python 3.8 with NVIDIA's Jetson PyTorch and JetPack's TensorRT (`environment/requirements-edge.txt`).
   * nuPlan simulation: `bash environment/setup_nuplan_env.sh`.
5. `python reproduce.py --tier full --list` prints every stage in order. Run them all with
   `python reproduce.py --tier full`, or resume at a stage with `--from <id>`.

Stage markers: `[D]` needs datasets; `[H]` is hardware-dependent. The full pipeline takes several days on the
reference device. The nuPlan planner runs alone take about 8 hours.

## Hardware: read before running hardware-dependent stages

All latency, energy and TensorRT detection results were produced on this reference platform:

| component | version |
|---|---|
| device | NVIDIA Jetson AGX Xavier (MAXN) |
| JetPack / L4T | 5.1.6 / R35.6.5 |
| CUDA | 11.4 |
| cuDNN | 8.6 |
| TensorRT | 8.5.2.2 |
| PyTorch | 2.1.0a0+nv23.6 |

`python environment/check_device.py` compares your machine with it. On any other machine:

* engine build, profiling, detection and budget stages print a warning;
* stages that would overwrite shipped caches stop unless `RAP_ALLOW_OTHER_DEVICE=1` is set;
* latency, energy and allocator overheads will not match the paper;
* FP16 TensorRT detections can differ slightly, and so can every table computed from re-run detections.

Use the cached tier to reproduce the paper's numbers exactly.

## Third-party code and data

* **Third-party code.** nuplan-devkit, tuPlan Garage (PDM-Closed), PKL (planning-centric-metrics) and TIP are
  cloned at the commits in `third_party/PROVENANCE.md` and used unmodified, except for the deviations listed
  there.
* **Detector weights.** The public Ultralytics YOLOv8s and RT-DETR-l releases.
* **Dataset-derived files.** Caches, result tables, overlays and gallery images remain under their datasets'
  licenses: nuScenes and nuPlan CC BY-NC-SA 4.0, KITTI CC BY-NC-SA 3.0.

## License

The code is released under the MIT license (see `LICENSE`).
