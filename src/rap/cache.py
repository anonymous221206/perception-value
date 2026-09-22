"""Ragged per-frame detection cache: flat arrays plus offsets, one file per (seq, mode).

The cached geometry (`geo_*`) is the monocular lift in the **camera** frame, as the detection scripts wrote it.  When
`rap.frames` selects another frame, `DetCache` re-lifts the cached boxes on read with `rap.mono.predicted_geometry`
and the unit's camera -> ego transform from `data/cache/cam_to_ego.json` (Task 23).  The boxes, confidences and classes
never change; the stored camera-frame arrays stay available through `geo_camera`.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np

from . import frames

COARSE_ID = {"vehicle": 0, "person": 1, "cyclist": 2}
ID_COARSE = np.array(["vehicle", "person", "cyclist"])

DET_ARRAYS = ("xyxy", "conf", "entropy", "margin", "binent")
GEO_ARRAYS = ("z", "z_ground", "z_height", "lat_min", "lat_max", "ttc", "box_h")


def save(path: Path, frames: list[int], dets: list[dict], geos: list[dict] | None,
         frame_scalars: dict[str, list]) -> None:
    counts = np.array([len(d["conf"]) for d in dets], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    out = {"frames": np.asarray(frames, np.int32), "offsets": offsets}
    for k in DET_ARRAYS:
        stacked = [np.asarray(d[k]) for d in dets]
        out[k] = (np.concatenate(stacked, axis=0) if any(len(s) for s in stacked)
                  else np.zeros((0, 4) if k == "xyxy" else 0, np.float32)).astype(np.float32)
    out["coarse_id"] = np.concatenate(
        [np.array([COARSE_ID[c] for c in d["coarse"]], np.int8) for d in dets]
    ) if len(dets) else np.zeros(0, np.int8)
    if geos is not None:
        for k in GEO_ARRAYS:
            out[f"geo_{k}"] = np.concatenate([np.asarray(g[k], np.float32) for g in geos]) \
                if len(geos) else np.zeros(0, np.float32)
    for k, v in frame_scalars.items():
        out[f"fs_{k}"] = np.asarray(v, np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)


KITTI_CACHES = {"det", "det512", "rtdetr_kitti", "rtdetr_kitti_mid", "modesel"}
NUSC_CACHES = {"nusc_det_tv"}
PERM = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])     # camera (x right, y down, z fwd) -> ego


@functools.lru_cache(maxsize=1)
def _table() -> dict:
    from .paths import CACHE
    f = Path(CACHE) / "cam_to_ego.json"
    if not f.exists():
        raise SystemExit(f"missing {f}: run scripts/143_cam_to_ego_table.py")
    return json.loads(f.read_text())


class Camera:
    """One unit's intrinsics, lift camera height, frame interval and camera -> ego transform in each frame."""

    def __init__(self, dataset: str, unit: str):
        from . import kitti
        from .kitti import Calib
        e = _table()[dataset].get(unit)
        if e is None:
            raise KeyError(f"{dataset} unit {unit!r} is not in cam_to_ego.json")
        self.dataset, self.unit = dataset, unit
        if "P2" in e:                                            # KITTI: load_calib's P2, tabulated by 143
            P2 = np.array(e["P2"], dtype=float)
        elif dataset == "KITTI":
            P2 = kitti.load_calib(unit).P2                       # a table written before P2 was tabulated
        else:
            P2 = np.zeros((3, 4))
            P2[:3, :3] = np.array(e["K"], dtype=float)
        # the lift reads fx, fy, cx and cy from P2 and nothing else of the calibration
        self.calib = Calib(P2=P2, R_rect=np.eye(4), Tr_velo_cam=np.eye(4), Tr_imu_velo=np.eye(4))
        self.cam_h, self.dt = float(e["cam_h"]), float(e["dt"])
        self.R, self.t = np.array(e["R"], dtype=float), np.array(e["t"], dtype=float)
        self.t22 = np.array(e["t_task22"], dtype=float)

    def transform(self, frame: str):
        if frame == "ego":
            return self.R, self.t
        if frame == "identity":
            return PERM, np.zeros(3)
        if frame == "task22":
            return PERM, np.array([self.t22[0], self.t22[1], 0.0])
        raise ValueError(f"no transform for frame {frame!r}")


def camera_for(path: Path) -> Camera:
    """The camera of a cache file `<cache>/<mode>/<unit>.npz`, by its cache directory."""
    path = Path(path)
    root = path.parent.parent.name
    dataset = "KITTI" if root in KITTI_CACHES else "nuScenes" if root in NUSC_CACHES else None
    if dataset is None:
        raise ValueError(f"{path}: no camera -> ego transform for cache {root!r}")
    return Camera(dataset, path.stem)


@functools.lru_cache(maxsize=256)
def _relifted(path: str, frame: str, stamp: tuple) -> dict:
    """Every frame of one cache file re-lifted in `frame`, with ttc checked against the stored value (gate G3)."""
    from . import mono
    c = DetCache(Path(path), frame="camera")
    cam = camera_for(Path(path))
    R, t = cam.transform(frame)
    out = {k: [] for k in GEO_ARRAYS}
    prev = None
    for i in range(len(c)):
        d = c.det(i)
        g = mono.predicted_geometry(d, prev, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
        ttc = np.asarray(g["ttc"], np.float32)
        stored = c.z["geo_ttc"][c._slice(i)]
        if ttc.shape != stored.shape or not np.array_equal(ttc, stored):
            raise RuntimeError(f"gate G3: ttc differs from the stored value under the {frame} frame, "
                               f"{path} frame {i}; ttc must be invariant to a rigid transform")
        for k in GEO_ARRAYS:
            out[k].append(np.asarray(g[k], np.float32))
        prev = d
    return {k: (np.concatenate(v) if v else np.zeros(0, np.float32)) for k, v in out.items()}


class DetCache:
    """Per-frame views over one cached sequence.

    The arrays are materialised on construction rather than read through the lazy
    `NpzFile`: indexing a lazy npz re-inflates the whole array on every access, which
    made per-frame reads dominate table building by a factor of ~15.

    `frame` (default: `rap.frames.current()`) selects the geometry `geo(i)` returns; see the module docstring.
    """

    def __init__(self, path: Path, frame: str | None = None):
        with np.load(path, allow_pickle=False) as z:
            self.z = {k: z[k] for k in z.files}
        self.frames = self.z["frames"]
        self.offsets = self.z["offsets"]
        self.index = {int(f): i for i, f in enumerate(self.frames)}
        self.has_geo = "geo_z" in self.z
        self.path = Path(path)
        self.frame = frame or frames.current()
        self.geo_cam = {k: self.z[f"geo_{k}"] for k in GEO_ARRAYS} if self.has_geo else {}
        if self.has_geo and self.frame != "camera":
            st = self.path.stat()
            relifted = _relifted(str(self.path.resolve()), self.frame, (st.st_size, st.st_mtime_ns))
            for k, v in relifted.items():
                self.z[f"geo_{k}"] = v

    def __len__(self) -> int:
        return len(self.frames)

    def _slice(self, i: int) -> slice:
        return slice(int(self.offsets[i]), int(self.offsets[i + 1]))

    def det(self, i: int) -> dict:
        s = self._slice(i)
        d = {k: self.z[k][s] for k in DET_ARRAYS}
        d["coarse"] = ID_COARSE[self.z["coarse_id"][s]]
        for k in ("n_cand", "n_cand_raw", "n_post"):
            d[k] = float(self.z[f"fs_{k}"][i])
        return d

    def geo(self, i: int) -> dict:
        s = self._slice(i)
        return {k: self.z[f"geo_{k}"][s] for k in GEO_ARRAYS}

    def geo_camera(self, i: int) -> dict:
        """The stored camera-frame geometry, whatever frame `geo` returns."""
        s = self._slice(i)
        return {k: self.geo_cam[k][s] for k in GEO_ARRAYS}

    def scalars(self, i: int) -> dict:
        return {k[3:]: float(v[i]) for k, v in self.z.items() if k.startswith("fs_")}
