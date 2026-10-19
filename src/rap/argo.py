"""Argoverse readers for the follow-up track E8 (addendum E8 in docs/followup_protocol.md).

The interface is `rap.nusc`'s: per log, the decision frames in temporal order, an image per frame, the ground-truth
geometry array of `rap.geometry.GEOM_FIELDS` in the ego frame (x forward, y left, z up), the ego speed, and the camera
entry the detection cache re-lifts with (`rap.cache.Camera`).

Argoverse 1.1 (tracking, ring_front_center): the labels are per LiDAR sweep (10 Hz); a decision frame is a labelled
sweep, with the ring_front_center image nearest in time (at most 50 ms away). The released images carry radial lens
distortion; detection reads them undistorted with the published intrinsics (`undistort_maps`), so the pinhole model
of the lift and of the ground-truth projection holds.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import GEOM_FIELDS, _fill_range_rate

AV1_IMG_W, AV1_IMG_H = 1920, 1200
AV1_DT = 0.1                    # labelled sweeps, 10 Hz
MAX_PAIR_S = 0.05               # largest sweep-to-image gap of a decision frame
MIN_POINTS = 1                  # Section 3's visibility rule: at least one LiDAR point inside the box (217's counts)
SPEED_WINDOW_S = 0.1            # ego speed: backward difference over about this long
# the detector's coarse class of each Argoverse 1 label class evaluated; the others (ON_ROAD_OBSTACLE, OTHER_MOVER,
# ANIMAL, STROLLER, WHEELCHAIR) have no counterpart among the detector's traffic classes and are not reference objects
AV1_CLASSES = {"VEHICLE": "vehicle", "LARGE_VEHICLE": "vehicle", "BUS": "vehicle", "SCHOOL_BUS": "vehicle",
               "TRAILER": "vehicle", "EMERGENCY_VEHICLE": "vehicle", "PEDESTRIAN": "person",
               "BICYCLE": "cyclist", "BICYCLIST": "cyclist", "MOTORCYCLE": "cyclist", "MOTORCYCLIST": "cyclist",
               "MOPED": "cyclist"}


def _ts(name: str) -> int:
    return int(name.rsplit("_", 1)[1].split(".")[0])


def _quat_wxyz(q) -> np.ndarray:
    return Rotation.from_quat(np.roll(np.asarray(q, float), -1)).as_matrix()


class AV1Log:
    """One Argoverse 1.1 tracking log."""

    def __init__(self, root: str | Path, log: str):
        self.log = log
        self.dir = Path(root) / log
        cal = json.loads((self.dir / "vehicle_calibration_info.json").read_text())
        cam = next(c["value"] for c in cal["camera_data_"] if c["key"] == "image_raw_ring_front_center")
        self.K = np.array([[cam["focal_length_x_px_"], cam["skew_"], cam["focal_center_x_px_"]],
                           [0.0, cam["focal_length_y_px_"], cam["focal_center_y_px_"]], [0.0, 0.0, 1.0]])
        k1, k2, k3 = cam["distortion_coefficients_"]
        self.dist = np.array([k1, k2, 0.0, 0.0, k3])                       # OpenCV order: radial k1, k2, (p1, p2), k3
        self.R_cam = _quat_wxyz(cam["vehicle_SE3_camera_"]["rotation"]["coefficients"])   # camera -> ego
        self.t_cam = np.array(cam["vehicle_SE3_camera_"]["translation"], float)
        self.city = json.loads((self.dir / "city_info.json").read_text())["city_name"]
        sweeps = sorted(_ts(p.name) for p in (self.dir / "per_sweep_annotations_amodal").glob("*.json"))
        cams = np.array(sorted(_ts(p.name) for p in (self.dir / "ring_front_center").glob("ring_front_center_*.jpg")))
        self.frames, self.dropped = [], 0
        for s in sweeps:
            j = int(np.argmin(np.abs(cams - s)))
            if abs(int(cams[j]) - s) <= MAX_PAIR_S * 1e9:
                self.frames.append((s, int(cams[j])))
            else:
                self.dropped += 1
        self.pose_ts = np.array(sorted(_ts(p.name) for p in (self.dir / "poses").glob("*.json")))
        self.interpolated: set = set()            # image times without a released pose (counted, see `pose`)
        self._pts = None                          # LiDAR points per box (217), read by `geometry`

    def points(self) -> dict:
        """{(sweep timestamp, track uuid): LiDAR points inside the box}, counted by 217 from the released sweeps."""
        if self._pts is None:
            from .paths import CACHE
            f = Path(CACHE) / "xl" / "e8a_lidar_counts" / f"{self.log}.csv"
            if not f.exists():
                raise SystemExit(f"missing {f}: the reference needs 217's point counts (E8 addendum, after gate E8-0)")
            import csv
            self._pts = {(int(r["timestamp"]), r["track_label_uuid"]): int(r["num_pts"]) for r in csv.DictReader(open(f))}
        return self._pts

    def image_path(self, i: int) -> Path:
        return self.dir / "ring_front_center" / f"ring_front_center_{self.frames[i][1]}.jpg"

    @functools.lru_cache(maxsize=None)
    def _released(self, ts: int):
        p = json.loads((self.dir / "poses" / f"city_SE3_egovehicle_{ts}.json").read_text())
        return _quat_wxyz(p["rotation"]), np.array(p["translation"], float)

    @functools.lru_cache(maxsize=None)
    def pose(self, ts: int):
        """ego -> city at `ts`: the released pose, or, where a log releases none at that time, the interpolation of
        the two released poses around it (translation linear, rotation spherical)."""
        ts = int(ts)
        j = int(np.searchsorted(self.pose_ts, ts))
        if j < len(self.pose_ts) and self.pose_ts[j] == ts:
            return self._released(ts)
        self.interpolated.add(ts)
        if j == 0 or j == len(self.pose_ts):
            return self._released(int(self.pose_ts[min(j, len(self.pose_ts) - 1)]))
        t0, t1 = int(self.pose_ts[j - 1]), int(self.pose_ts[j])
        (R0, p0), (R1, p1) = self._released(t0), self._released(t1)
        w = (ts - t0) / (t1 - t0)
        from scipy.spatial.transform import Slerp
        R = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([R0, R1])))([w]).as_matrix()[0]
        return R, (1 - w) * p0 + w * p1

    def labels(self, i: int) -> list:
        return json.loads((self.dir / "per_sweep_annotations_amodal" /
                           f"tracked_object_labels_{self.frames[i][0]}.json").read_text())

    def speeds(self) -> np.ndarray:
        """Forward ego speed at each frame's image time, from the pose about 0.1 s earlier (any sensor's pose); the
        first frames of a log, which have none, use the next frame instead."""
        out = np.zeros(len(self.frames))
        for i, (_, tc) in enumerate(self.frames):
            Rc, pc = self.pose(tc)
            earlier = self.pose_ts[self.pose_ts <= tc - 0.5 * SPEED_WINDOW_S * 1e9]
            if len(earlier):
                t0 = int(earlier[np.argmin(np.abs(earlier - (tc - SPEED_WINDOW_S * 1e9)))])
                R0, p0 = self.pose(t0)
                d, dt = pc - p0, (tc - t0) / 1e9
            else:
                t1 = self.frames[min(i + 1, len(self.frames) - 1)][1]
                R1, p1 = self.pose(t1)
                d, dt = p1 - pc, (t1 - tc) / 1e9
            out[i] = float((Rc.T @ d)[0] / dt) if dt > 1e-6 else 0.0
        return out

    def ground_offset_samples(self) -> np.ndarray:
        """Box bottoms (ego z) of vehicles 3-30 m ahead within 6 m laterally, for the camera-height rule."""
        z = []
        for i in range(len(self.frames)):
            for a in self.labels(i):
                c = a["center"]
                if a["label_class"] == "VEHICLE" and 3.0 < c["x"] < 30.0 and abs(c["y"]) < 6.0:
                    z.append(c["z"] - a["height"] / 2.0)
        return np.array(z)

    def geometry(self) -> np.ndarray:
        """The structured array `rap.geometry.sequence_geometry` returns, as `rap.nusc` builds it: boxes of the
        evaluated classes with at least one LiDAR point, projected (pinhole, undistorted image) with at least four corners in front, clipped to the
        image, at least 4 px wide and high; footprint in the ego frame at the image time."""
        rows = []
        Kc = self.K
        for fi, (ts, tc) in enumerate(self.frames):
            Rs, ps = self.pose(ts)
            Rc, pc = self.pose(tc)
            pts = self.points()
            for a in self.labels(fi):
                coarse = AV1_CLASSES.get(a["label_class"])
                if coarse is None or pts.get((ts, a["track_label_uuid"]), 0) < MIN_POINTS:
                    continue
                c = np.array([a["center"][k] for k in "xyz"], float)
                q = a["rotation"]
                Rb = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
                l, w, h = float(a["length"]), float(a["width"]), float(a["height"])
                xy = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
                foot = np.stack([Rb @ np.array([x, y, -h / 2]) + c for x, y in xy])
                top = foot + Rb @ np.array([0.0, 0.0, h])
                allc = np.vstack([foot, top])                                   # ego at the sweep time
                allc = (Rc.T @ ((Rs @ allc.T).T + ps - pc).T).T                 # -> ego at the image time
                foot_ego = allc[:4]
                cam = (self.R_cam.T @ (allc - self.t_cam).T).T
                front = cam[:, 2] > 0.5
                if front.sum() < 4:
                    continue
                uv = (Kc @ cam[front].T).T
                uv = uv[:, :2] / uv[:, 2:3]
                x1, y1 = uv.min(0)
                x2, y2 = uv.max(0)
                x1, x2 = np.clip([x1, x2], 0, AV1_IMG_W - 1)
                y1, y2 = np.clip([y1, y2], 0, AV1_IMG_H - 1)
                if (x2 - x1) < 4 or (y2 - y1) < 4:
                    continue
                rows.append((self.log, fi, a["track_label_uuid"], a["label_class"], coarse, x1, y1, x2, y2,
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
        _fill_range_rate(out, halfwidth=2, dt=AV1_DT)      # 10 Hz, as KITTI
        return out


def undistort_maps(log: AV1Log):
    """OpenCV remap tables taking the released image to the pinhole image with the same intrinsics."""
    import cv2
    return cv2.initUndistortRectifyMap(log.K, log.dist, None, log.K, (AV1_IMG_W, AV1_IMG_H), cv2.CV_32FC1)


def unit_name(log: str) -> str:
    """Logs are named by the first 16 characters of their id (the geometry array's `seq` field holds 16)."""
    return log[:16]


def make_adapter(root: str | Path, logs):
    """`rap.decision.KittiAdapter`'s interface keyed by unit name."""
    by_unit = {unit_name(l): l for l in logs}
    cache: dict = {}

    def get(u):
        if u not in cache:
            cache[u] = AV1Log(root, by_unit[u])
        return cache[u]

    class AV1Adapter:
        name = "argoverse1"

        @staticmethod
        @functools.lru_cache(maxsize=None)
        def geometry(seq):
            return get(seq).geometry()

        @staticmethod
        @functools.lru_cache(maxsize=None)
        def speeds(seq):
            return get(seq).speeds()

    return AV1Adapter
