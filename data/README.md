# Data

This release has three layers of data:

| layer | where | shipped? | needed for |
|---|---|---|---|
| raw datasets | `$RAP_DATASETS` (default `datasets/`) | **no**: each requires registration and acceptance of its terms | `reproduce.py --tier full` |
| detector weights | `$RAP_MODELS` (default `models/`) | no: `data/download_models.sh` fetches the public Ultralytics weights | `--tier full` |
| intermediate caches | `data/cache/`, `results/raw/` | **yes** | `--tier cached` (every table, on any machine) |

`python data/verify_datasets.py` checks the layout below.

## Raw datasets (download them yourself)

All three datasets require an account and acceptance of their terms of use, so no script can download them
for you. After downloading, arrange them as follows under `$RAP_DATASETS`:

```
datasets/
├── nuscenes/trainval/          nuScenes v1.0-trainval (https://www.nuscenes.org/nuscenes#download)
│   ├── v1.0-trainval/          metadata (all JSON tables)
│   ├── samples/CAM_FRONT/      from blob "trainval01" (the 85 scenes, 3,376 CAM_FRONT keyframes used here)
│   └── maps/expansion/         map expansion pack v1.3 (needed by PKL/TIP rasters)
├── kitti_tracking/training/    KITTI multi-object tracking (https://www.cvlibs.net/datasets/kitti/eval_tracking.php)
│   ├── image_02/  label_02/  calib/  oxts/
└── nuplan/                     nuPlan (https://www.nuscenes.org/nuplan#download)
    ├── nuplan-v1.1/splits/mini/        v1.1 "Mini Split" databases (64 logs; the benchmark uses 34)
    ├── nuplan-maps-v1.0/               maps
    └── sensor_blobs_cam_f0/            fetched by scripts/112 (below)
```

## nuPlan front-camera images (partial fetch)

The nuPlan mini camera data is distributed as nine archives of 45–54 GB, each bundling all cameras of several
logs. `scripts/112_nuplan_fetch_cam_f0.py` avoids downloading them. It reads each archive's ZIP central
directory by HTTP Range and fetches only the 12,921 CAM_F0 images inside the benchmark's scenario windows
(2.77 GB). Every member is checked by CRC32 and JPEG decode.

The archive links are only shown after logging in to the official download page, so they are not included.

1. Save the "Mini Sensors Metadata" file as `configs/nuplan_mini_sensor.txt`.
2. Put the links of "Camera 0" … "Camera 8" in `configs/nuplan_camera_urls.txt`, one `<index><TAB><url>` per line.
3. Run the fetch, in three stages:
   ```
   environment/bin/py-edge scripts/112_nuplan_fetch_cam_f0.py --stage dirs
   environment/bin/py-edge scripts/112_nuplan_fetch_cam_f0.py --stage plan
   environment/bin/py-edge scripts/112_nuplan_fetch_cam_f0.py --stage fetch
   ```

The fetch is capped at 400 MB of directory data and 3.5 GB of image data. It stops, rather than falling back to
a whole-archive download, if a server refuses Range requests.

## Follow-up experiments

The follow-ups (`docs/followup_index.md`) read further data under `$RAP_DATASETS` or on the GPU machine; each script's
header names its inputs. NAVSIM (E2, E11) needs OpenScene v1.1's `navsim_logs/{test,mini}`, the CAM_F0
`sensor_blobs` of both splits and the nuPlan maps, set up as NAVSIM's install guide describes (`OPENSCENE_DATA_ROOT`,
`NUPLAN_MAPS_ROOT`). E11 also needs navtest's CAM_L0 and CAM_R0 images: `scripts/219_openscene_cams_fetch.sh` fetches
them from OpenScene's public test camera archives (about 128 GB downloaded, 32 GB kept). E3 needs no dataset: it
runs Bench2Drive's 220 routes in CARLA 0.9.15 (`third_party/PROVENANCE.md`).

## Licenses of dataset-derived files

The shipped caches, result tables, overlays and gallery images are derived from the datasets. They are
provided for non-commercial research under each dataset's own license:

| dataset | license |
|---|---|
| nuScenes | CC BY-NC-SA 4.0 |
| nuPlan | CC BY-NC-SA 4.0 |
| KITTI | CC BY-NC-SA 3.0 |
| OpenScene / NAVSIM (derived from nuPlan) | CC BY-NC-SA 4.0 |
| Argoverse 1.1, Argoverse 2 | CC BY-NC-SA 4.0 |

The MIT license in `LICENSE` covers the code only.
