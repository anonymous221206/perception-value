"""Argoverse 2 sensor reader for the follow-up track E8b (addendum E8 in docs/followup_protocol.md, E8b rules).

`rap.argo.AV1Log`'s interface on 214's copy: the tables as CSV (annotations, poses, calibration, the sweep-to-image
pairs) and the paired ring_front_center images. The image is cropped below row 1830 (the ego vehicle's hood); the
intrinsic matrix is unchanged. Whether the image is undistorted first is gate E8-0's choice (`undistort`).
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from .geometry import GEOM_FIELDS, _fill_range_rate

AV2_IMG_W, AV2_IMG_H = 1550, 1830          # after the hood crop (released 1550x2048)
HOOD_ROW = 1830
AV2_DT = 0.1
SPEED_WINDOW_S = 0.1
AV2_CLASSES = {"REGULAR_VEHICLE": "vehicle", "LARGE_VEHICLE": "vehicle", "BOX_TRUCK": "vehicle", "TRUCK": "vehicle",
               "TRUCK_CAB": "vehicle", "BUS": "vehicle", "ARTICULATED_BUS": "vehicle", "SCHOOL_BUS": "vehicle",
               "VEHICULAR_TRAILER": "vehicle", "RAILED_VEHICLE": "vehicle", "PEDESTRIAN": "person",
               "OFFICIAL_SIGNALER": "person", "BICYCLE": "cyclist", "BICYCLIST": "cyclist", "MOTORCYCLE": "cyclist",
               "MOTORCYCLIST": "cyclist", "WHEELED_RIDER": "cyclist"}


def _rot(r):
    return Rotation.from_quat([r.qx, r.qy, r.qz, r.qw]).as_matrix()


class AV2Log:
    def __init__(self, tables: str | Path, images: str | Path, split: str, log: str):
        self.log, self.split = log, split
        self.tdir = Path(tables) / split / log
        self.idir = Path(images) / split / log / "sensors" / "cameras" / "ring_front_center"
        it = pd.read_csv(self.tdir / "intrinsics.csv").set_index("sensor_name").loc["ring_front_center"]
        self.K = np.array([[it.fx_px, 0.0, it.cx_px], [0.0, it.fy_px, it.cy_px], [0.0, 0.0, 1.0]])
        self.dist = np.array([it.k1, it.k2, 0.0, 0.0, it.k3])
        ex = pd.read_csv(self.tdir / "extrinsics.csv").set_index("sensor_name").loc["ring_front_center"]
        self.R_cam, self.t_cam = _rot(ex), np.array([ex.tx_m, ex.ty_m, ex.tz_m], float)
        fr = pd.read_csv(self.tdir / "frames.csv")
        self.frames = list(zip(fr.sweep_ns.astype(int), fr.image_ns.astype(int)))
        self.dropped = 0
        p = pd.read_csv(self.tdir / "poses.csv")
        self._pose = {int(r.timestamp_ns): (_rot(r), np.array([r.tx_m, r.ty_m, r.tz_m], float))
                      for r in p.itertuples()}
        self.pose_ts = np.array(sorted(self._pose))
        a = pd.read_csv(self.tdir / "annotations.csv")
        a = a[a.category.isin(AV2_CLASSES) & (a.num_interior_pts >= 1)]
        self._ann = {int(t): g for t, g in a.groupby("timestamp_ns")}
        self._ann_all = pd.read_csv(self.tdir / "annotations.csv")

    def image_path(self, i: int) -> Path:
        return self.idir / f"{self.frames[i][1]}.jpg"

    def pose(self, ts: int):
        return self._pose[int(ts)]

    def speeds(self) -> np.ndarray:
        """As `rap.argo.AV1Log.speeds`: forward ego speed from the pose about 0.1 s before the image time."""
        out = np.zeros(len(self.frames))
        for i, (_, tc) in enumerate(self.frames):
            Rc, pc = self.pose(tc)
            earlier = self.pose_ts[self.pose_ts <= tc - 0.5 * SPEED_WINDOW_S * 1e9]
            if len(earlier):
                t0 = int(earlier[np.argmin(np.abs(earlier - (tc - SPEED_WINDOW_S * 1e9)))])
                _, p0 = self.pose(t0)
                d, dt = pc - p0, (tc - t0) / 1e9
            else:
                t1 = self.frames[min(i + 1, len(self.frames) - 1)][1]
                _, p1 = self.pose(t1)
                d, dt = p1 - pc, (t1 - tc) / 1e9
            out[i] = float((Rc.T @ d)[0] / dt) if dt > 1e-6 else 0.0
        return out

    def ground_offset_samples(self) -> np.ndarray:
        a = self._ann_all
        a = a[(a.category == "REGULAR_VEHICLE") & (a.tx_m > 3) & (a.tx_m < 30) & (a.ty_m.abs() < 6)]
        return (a.tz_m - a.height_m / 2).to_numpy(float)

    def geometry(self) -> np.ndarray:
        """`rap.argo.AV1Log.geometry` on Argoverse 2 cuboids (centre, quaternion, length / width / height), clipped
        to the cropped image."""
        rows = []
        for fi, (ts, tc) in enumerate(self.frames):
            if ts not in self._ann:
                continue
            Rs, ps = self.pose(ts)
            Rc, pc = self.pose(tc)
            for r in self._ann[ts].itertuples():
                c = np.array([r.tx_m, r.ty_m, r.tz_m], float)
                Rb = _rot(r)
                l, w, h = float(r.length_m), float(r.width_m), float(r.height_m)
                xy = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
                foot = np.stack([Rb @ np.array([x, y, -h / 2]) + c for x, y in xy])
                top = foot + Rb @ np.array([0.0, 0.0, h])
                allc = (Rc.T @ ((Rs @ np.vstack([foot, top]).T).T + ps - pc).T).T
                foot_ego = allc[:4]
                cam = (self.R_cam.T @ (allc - self.t_cam).T).T
                front = cam[:, 2] > 0.5
                if front.sum() < 4:
                    continue
                uv = (self.K @ cam[front].T).T
                uv = uv[:, :2] / uv[:, 2:3]
                x1, y1 = uv.min(0)
                x2, y2 = uv.max(0)
                x1, x2 = np.clip([x1, x2], 0, AV2_IMG_W - 1)
                y1, y2 = np.clip([y1, y2], 0, AV2_IMG_H - 1)
                if (x2 - x1) < 4 or (y2 - y1) < 4:
                    continue
                rows.append((self.log, fi, r.track_uuid, r.category, AV2_CLASSES[r.category], x1, y1, x2, y2,
                             foot_ego[:, 0].min(), foot_ego[:, 0].mean(), foot_ego[:, 1].mean(),
                             foot_ego[:, 1].min(), foot_ego[:, 1].max()))
        if not rows:
            return np.zeros(0, dtype=GEOM_FIELDS)
        tid = {t: i for i, t in enumerate(sorted({r[2] for r in rows}))}
        out = np.zeros(len(rows), dtype=GEOM_FIELDS)
        for i, r in enumerate(rows):
            out[i]["seq"], out[i]["frame"], out[i]["track_id"] = r[0][:16], r[1], tid[r[2]]
            out[i]["type"], out[i]["coarse"] = r[3][:16], r[4]
            for j, k in enumerate(("x1", "y1", "x2", "y2", "long_near", "long_cen", "lat_cen", "lat_min", "lat_max")):
                out[i][k] = r[5 + j]
            out[i]["ttc"] = np.inf
        _fill_range_rate(out, halfwidth=2, dt=AV2_DT)
        return out


def load_image(log: AV2Log, i: int, maps=None):
    """The detector's input: optionally undistorted (OpenCV remap tables from `undistort_maps`), then cropped."""
    import cv2
    img = cv2.imread(str(log.image_path(i)))
    if maps is not None:
        img = cv2.remap(img, maps[0], maps[1], cv2.INTER_LINEAR)
    return img[:HOOD_ROW]


def undistort_maps(log: AV2Log):
    import cv2
    return cv2.initUndistortRectifyMap(log.K, log.dist, None, log.K, (AV2_IMG_W, 2048), cv2.CV_32FC1)


def logs(tables: str | Path):
    """(split, log) of the copied logs: validation first, then the fitting logs, each sorted."""
    t = Path(tables)
    return [(sp, p.name) for sp in ("val", "train") for p in sorted((t / sp).iterdir()) if p.is_dir()]


def make_adapter(tables, images, split_logs):
    from .argo import unit_name
    by_unit = {unit_name(l): (sp, l) for sp, l in split_logs}
    cache: dict = {}

    def get(u):
        if u not in cache:
            cache[u] = AV2Log(tables, images, *by_unit[u])
        return cache[u]

    class AV2Adapter:
        name = "argoverse2"

        @staticmethod
        @functools.lru_cache(maxsize=None)
        def geometry(seq):
            return get(seq).geometry()

        @staticmethod
        @functools.lru_cache(maxsize=None)
        def speeds(seq):
            return get(seq).speeds()

    return AV2Adapter
