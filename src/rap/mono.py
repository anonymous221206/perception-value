"""Criticality estimated from CHEAP detections alone — the deployable counterpart
to `geometry.py`.

Nothing here may touch ground truth, the FULL prediction, or a future frame. It
uses only: the CHEAP 2D boxes of the current frame, the CHEAP boxes of the
*previous* frame, and the camera calibration (known at build time on a real
vehicle).

Two independent range cues are used because each fails differently:
  * ground-plane cue: z = f_y * h_cam / (v_bottom - c_y), exact for an object
    standing on a flat road, degenerate near the horizon;
  * height-prior cue: z = f_y * H_class / h_box, robust near the horizon but
    wrong for atypically sized objects.
"""
from __future__ import annotations

import numpy as np

from .kitti import Calib, FRAME_DT

# Typical physical heights [m] used as the class-conditional size prior.
HEIGHT_PRIOR = {"vehicle": 1.55, "person": 1.72, "cyclist": 1.72}
CAMERA_HEIGHT = 1.65  # KITTI cam2 height above the road [m]


def range_ground(y2: np.ndarray, calib: Calib, min_px: float = 2.0,
                 cam_h: float = CAMERA_HEIGHT) -> np.ndarray:
    """Distance from where the box bottom meets the road plane."""
    dv = np.maximum(y2 - calib.cy, min_px)
    return calib.fy * cam_h / dv


def range_height(y1: np.ndarray, y2: np.ndarray, coarse: np.ndarray,
                 calib: Calib, min_px: float = 2.0) -> np.ndarray:
    """Distance from the apparent box height and a class height prior."""
    h = np.maximum(y2 - y1, min_px)
    prior = np.array([HEIGHT_PRIOR.get(c, 1.6) for c in coarse], dtype=np.float64)
    return calib.fy * prior / h


def fuse_range(z_ground: np.ndarray, z_height: np.ndarray, y2: np.ndarray,
               calib: Calib) -> np.ndarray:
    """Trust the ground cue when the box bottom is well below the horizon."""
    dv = y2 - calib.cy
    w_ground = np.clip((dv - 5.0) / 25.0, 0.0, 1.0)
    return w_ground * z_ground + (1 - w_ground) * z_height


def lateral_offset(x1: np.ndarray, x2: np.ndarray, z: np.ndarray, calib: Calib):
    """Lateral extent of the object in the ego frame (+left), from the box edges.

    The camera x axis points right, the ego y axis points left, so the sign flips.
    A constant lateral mounting offset of cam2 is absorbed into the principal point,
    which is what a deployed system would calibrate anyway.
    """
    xl = (x1 - calib.cx) * z / calib.fx
    xr = (x2 - calib.cx) * z / calib.fx
    lat_max, lat_min = -xl, -xr           # flip right-handedness
    return lat_min, lat_max


def ttc_from_scale(curr_h: np.ndarray, prev_h: np.ndarray, dt: float = FRAME_DT,
                   max_ttc: float = 1e3) -> np.ndarray:
    """Scale-change TTC: a box growing by a factor s over dt hits in dt/(s-1).

    Purely monocular and calibration-free; this is the classic looming cue.
    """
    ttc = np.full(len(curr_h), np.inf)
    ok = np.isfinite(prev_h) & (prev_h > 1) & (curr_h > 1)
    growth = np.zeros_like(ttc)
    growth[ok] = curr_h[ok] / prev_h[ok]
    expanding = ok & (growth > 1.0 + 1e-3)
    ttc[expanding] = dt / (growth[expanding] - 1.0)
    return np.minimum(ttc, max_ttc)


def associate_prev(curr_xyxy: np.ndarray, prev_xyxy: np.ndarray,
                   iou_thr: float = 0.3) -> np.ndarray:
    """Greedy IoU association of current boxes to the previous frame's boxes.

    Returns, per current box, the previous box height or NaN. Frame-to-frame motion
    at 10 Hz is small enough that plain IoU is adequate; this is the cheapest
    association a real gate could afford.
    """
    out = np.full(len(curr_xyxy), np.nan)
    if len(prev_xyxy) == 0 or len(curr_xyxy) == 0:
        return out
    iou = box_iou(curr_xyxy, prev_xyxy)
    taken = set()
    for ci in np.argsort(-iou.max(axis=1)):
        order = np.argsort(-iou[ci])
        for pj in order:
            if iou[ci, pj] < iou_thr:
                break
            if pj not in taken:
                taken.add(int(pj))
                out[ci] = prev_xyxy[pj, 3] - prev_xyxy[pj, 1]
                break
    return out


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n,4) x (m,4) -> (n,m) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


# The camera-frame lift measures range along the camera optical axis and lateral extent about the camera axis, and
# the planners read those as ego-frame quantities.  The camera is not at the ego origin (KITTI's cam2 is ~1.1 m
# forward and ~0.26 m right of the IMU origin; nuScenes's CAM_FRONT is 1.70 m forward) and is not level.  Task 22
# Part C shifted the result by the translation (`cam_offset`, kept for reproducibility); Task 23 made the ego frame the
# shipped convention (`cam_to_ego`, see `_lift_ego`), chosen through `rap.frames`.
GEOM_SHIFT_Z = ("z", "z_ground", "z_height")
GEOM_SHIFT_LAT = ("lat_min", "lat_max")


def apply_cam_offset(geo: dict, cam_offset) -> dict:
    """The camera-to-ego translation applied to an already-lifted geometry.

    Exact: the fused range is a convex combination whose weight depends on `y2 - c_y` only, so shifting after the
    fusion equals shifting each cue before it, and the lateral extent is computed from the camera-frame depth, so
    the translation reaches it only through `t_y`.  `140_lift_offset_outcomes.py` checks this against the lift.
    """
    tx, ty = float(cam_offset[0]), float(cam_offset[1])
    out = dict(geo)
    for k in GEOM_SHIFT_Z:
        if k in out:
            out[k] = out[k] + tx
    for k in GEOM_SHIFT_LAT:
        if k in out:
            out[k] = out[k] + ty
    return out


def _ego_x(R, t, X, Y, Z):
    """Ego-frame x of camera-frame points, written out so the identity transform is exact."""
    return R[0, 0] * X + R[0, 1] * Y + R[0, 2] * Z + t[0]


def _ego_y(R, t, X, Y, Z):
    return R[1, 0] * X + R[1, 1] * Y + R[1, 2] * Z + t[1]


def _lift_ego(x1, y1, x2, y2, coarse, calib: Calib, cam_h: float, R, t, min_px: float = 2.0) -> dict:
    """The lift in the ego frame (Task 23; the exact definition is in the pre-registration record).

    `R`, `t` take camera coordinates (x right, y down, z forward) to ego coordinates (x forward, y left, z up).
    The camera -> ego rotation enters where the camera's attitude acts on the estimate, and only there:

      * D, the pixel distance of the box bottom below the **ego** horizon, replaces `y2 - c_y` both in the ground cue
        and in the fusion weight, so the two share one horizon.  With the road normal n = R^T (0, 0, -1) expressed in
        camera coordinates, D = f_y * (n . ray) for the bottom-centre ray, computed in pixel units.
      * the height cue (a ratio of pixel heights) is unchanged, and the cues are fused as camera depths along the one
        bottom-centre ray, exactly as shipped;
      * the fused point and the two bottom corners at that depth are then expressed in the ego frame.

    Pitch therefore enters once, through D; it is never applied a second time to the fused point.  With R the pure
    axis permutation and t = 0 every step performs the shipped operations plus exact multiplications by 0 and +-1,
    so the result equals the camera-frame lift (gate G1); with t = (t_x, t_y, .) it equals Task 22's shift (G2).
    """
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    n0, n1, n2 = -R[2, 0], -R[2, 1], -R[2, 2]
    uc = 0.5 * (x1 + x2)
    dv = y2 - calib.cy
    D = n0 * ((calib.fy / calib.fx) * (uc - calib.cx)) + n1 * dv + n2 * calib.fy
    zg = calib.fy * cam_h / np.maximum(D, min_px)                    # the ground cue on the corrected horizon
    zh = range_height(y1, y2, coarse, calib)                          # unchanged
    w = np.clip((D - 5.0) / 25.0, 0.0, 1.0)                           # the same horizon as the ground cue
    zc = np.clip(w * zg + (1 - w) * zh, 0.5, 200.0)                   # fused camera depth, as shipped
    yb = (y2 - calib.cy) * zc / calib.fy
    xl = (x1 - calib.cx) * zc / calib.fx
    xr = (x2 - calib.cx) * zc / calib.fx
    xc = (uc - calib.cx) * zc / calib.fx
    y_left, y_right = _ego_y(R, t, xl, yb, zc), _ego_y(R, t, xr, yb, zc)
    zgc, zhc = np.clip(zg, 0.5, 200.0), np.clip(zh, 0.5, 200.0)
    return {"z": _ego_x(R, t, xc, yb, zc),
            "z_ground": _ego_x(R, t, (uc - calib.cx) * zgc / calib.fx, (y2 - calib.cy) * zgc / calib.fy, zgc),
            "z_height": _ego_x(R, t, (uc - calib.cx) * zhc / calib.fx, (y2 - calib.cy) * zhc / calib.fy, zhc),
            "lat_min": np.minimum(y_left, y_right), "lat_max": np.maximum(y_left, y_right)}


def predicted_geometry(det: dict, prev_det: dict | None, calib: Calib,
                       cam_h: float = CAMERA_HEIGHT, dt: float = FRAME_DT, cam_offset=None,
                       cam_to_ego=None) -> dict:
    """Per-detection estimated geometry: range, lateral extent, TTC.

    With neither option this is the camera-frame lift as first shipped.  `cam_to_ego=(R, t)` is the ego-frame lift of
    Task 23 (see `_lift_ego`); `cam_offset=(t_x, t_y)` is Task 22's translation-only shift of the camera-frame result.
    The two cannot be combined.
    """
    if cam_offset is not None and cam_to_ego is not None:
        raise ValueError("cam_offset and cam_to_ego cannot be combined: the translation would be applied twice")
    xyxy = det["xyxy"].astype(np.float64)
    n = len(xyxy)
    if n == 0:
        z = np.zeros(0)
        return {"z": z, "z_ground": z, "z_height": z, "lat_min": z, "lat_max": z,
                "ttc": z, "box_h": z}
    x1, y1, x2, y2 = xyxy.T
    prev_h = (associate_prev(xyxy, prev_det["xyxy"].astype(np.float64))
              if prev_det is not None else np.full(n, np.nan))
    ttc = ttc_from_scale(y2 - y1, prev_h, dt)
    if cam_to_ego is not None:
        g = _lift_ego(x1, y1, x2, y2, det["coarse"], calib, cam_h, *cam_to_ego)
        # ttc is a ratio of box heights in the image and cannot depend on a rigid transform; it is taken from the one
        # computation above, never re-derived, and the frame-aware cache asserts it against the stored value (G3)
        g.update({"ttc": ttc, "box_h": y2 - y1})
        return g
    zg = range_ground(y2, calib, cam_h=cam_h)
    zh = range_height(y1, y2, det["coarse"], calib)
    z = np.clip(fuse_range(zg, zh, y2, calib), 0.5, 200.0)
    lat_min, lat_max = lateral_offset(x1, x2, z, calib)
    g = {"z": z, "z_ground": np.clip(zg, 0.5, 200.0), "z_height": np.clip(zh, 0.5, 200.0),
         "lat_min": lat_min, "lat_max": lat_max, "ttc": ttc, "box_h": y2 - y1}
    return g if cam_offset is None else apply_cam_offset(g, cam_offset)
