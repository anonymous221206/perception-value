"""Follow-up E3: PDM-Lite (Bench2Drive's carla_garage expert) with perception in the loop, for the benchmark's modes.

Rule (as the benchmark's nuPlan and NAVSIM tracks): every vehicle / walker whose 3D box projects into the front camera
(1600x900, FOV 70, Bench2Drive's CAM_FRONT pose) is kept only if a detection of the active mode matches it (Hungarian,
class-compatible, IoU >= 0.3); unmatched in-camera actors are hidden from the planner; unmatched detections become
static obstacles lifted on the ground plane (class-median sizes, heading along the viewing ray); actors outside the
camera are untouched. Map, route, traffic lights and scenario information stay privileged.

Environment:
  PERC_BRANCH   privileged | cheap (192x320) | full (384x640) | u960 (544x960)      COCO YOLOv8s, threshold 0.25
  DEEP_WEIGHTS  YOLOv8s weights (PyTorch FP16 on the simulation GPU; the benchmark's letterbox and decoding)
  SHADOW=1      also run the full mode on every tick and log whether the perceived object set would differ
  FORK_TICK, FORK_TICKS (10), FORK_BRANCH (full | cheap), WINDOW_TICKS (160)
                fork mode (addendum E3 in docs/followup_protocol.md): cheap except ticks [FORK_TICK, FORK_TICK + FORK_TICKS), which use
                FORK_BRANCH; after FORK_TICK + WINDOW_TICKS the planner gets the privileged scene (no detector cost; the
                rest of the route is not scored)
  SAVE_PATH     per-tick logs, one JSON line per tick, in ticks_<tag>.jsonl
Per tick: ego speed, position, the braking controller's requirement a* from the true actors (benchmark corridor and
parameters), the perceived-set signatures of the active mode (and of the full mode in shadow), and an ego-actor box
overlap flag.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import carla
import numpy as np
from scipy.optimize import linear_sum_assignment

from autopilot import AutoPilot

sys.path.insert(0, os.environ.get("DEEP_SRC", str(Path(__file__).resolve().parents[1] / "src")))
from rap import planner as Pl                                                    # noqa: E402
from rap.detect import Mode, TwoFidelityDetector                                 # noqa: E402

W, H, FOV = 1600, 900, 70.0
CAM_POSE = dict(x=0.80, y=0.0, z=1.60, roll=0.0, pitch=0.0, yaw=0.0)
IOU_THR, THR = 0.3, 0.25
PRIORS = {"vehicle": (4.5, 1.9, 1.6), "person": (0.6, 0.6, 1.75), "cyclist": (1.8, 0.7, 1.7)}
TYPE = {"vehicle": "vehicle.pseudo.phantom", "person": "walker.pedestrian.phantom", "cyclist": "vehicle.pseudo.phantom"}
MODES = {"cheap": Mode("b2d_c320", 192, 320), "full": Mode("b2d_f640", 384, 640), "u960": Mode("b2d_u960", 544, 960)}
PP = Pl.PlannerParams()


def get_entry_point():
    return "DeepPilot"


def actor_class(a):
    if a.type_id.startswith("walker"):
        return "person"
    nw = int(a.attributes.get("number_of_wheels", 4)) if hasattr(a, "attributes") else 4
    return "cyclist" if nw == 2 else "vehicle"


COMPAT = {"vehicle": {"vehicle"}, "person": {"person"}, "cyclist": {"cyclist", "person", "vehicle"}}


def iou(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])                    # noqa: E731
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-9)


class PseudoActor:
    """A static phantom obstacle with the actor API PDM-Lite uses."""
    _next = 10_000_000

    def __init__(self, loc, yaw, size, cls):
        PseudoActor._next += 1
        self.id = PseudoActor._next
        self.type_id = TYPE[cls]
        self.attributes = {"role_name": "phantom", "number_of_wheels": "4"}
        L, Wd, Ht = size
        self._tf = carla.Transform(loc, carla.Rotation(yaw=yaw))
        self.bounding_box = carla.BoundingBox(carla.Location(0, 0, Ht / 2), carla.Vector3D(L / 2, Wd / 2, Ht / 2))
        self.is_alive = True

    def get_location(self):
        return self._tf.location

    def get_transform(self):
        return self._tf

    def get_velocity(self):
        return carla.Vector3D(0, 0, 0)

    def get_acceleration(self):
        return carla.Vector3D(0, 0, 0)

    def get_angular_velocity(self):
        return carla.Vector3D(0, 0, 0)

    def get_control(self):
        return carla.WalkerControl() if self.type_id.startswith("walker") else carla.VehicleControl()


class PerceivedActors(list):
    """ActorList stand-in: iterable, filter(pattern) with CARLA's fnmatch semantics on type_id."""

    def filter(self, pattern):
        import fnmatch
        return PerceivedActors([a for a in self if fnmatch.fnmatch(a.type_id, pattern)])

    def find(self, aid):
        return next((a for a in self if a.id == aid), None)


class WorldProxy:
    """carla.World with get_actors() / get_actor() answering from the perceived scene; everything else delegated."""

    def __init__(self, world, actors):
        self._w, self._actors = world, actors

    def get_actors(self, *a):
        return self._actors

    def get_actor(self, aid):
        return self._actors.find(aid) or self._w.get_actor(aid)

    def __getattr__(self, k):
        return getattr(self._w, k)


class DeepPilot(AutoPilot):
    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(path_to_conf_file, route_index, traffic_manager)
        self.branch = os.environ.get("PERC_BRANCH", "privileged")
        self.shadow = os.environ.get("SHADOW", "0") == "1"
        self.fork_tick = int(os.environ.get("FORK_TICK", "-1"))
        self.fork_ticks = int(os.environ.get("FORK_TICKS", "10"))
        self.fork_branch = os.environ.get("FORK_BRANCH", "full")
        self.window = int(os.environ.get("WINDOW_TICKS", "160"))
        self.tag = os.environ.get("DEEP_TAG", self.branch)
        self.tick = 0
        self.log = []
        self.K = np.array([[W / (2 * math.tan(math.radians(FOV) / 2)), 0, W / 2],
                           [0, W / (2 * math.tan(math.radians(FOV) / 2)), H / 2], [0, 0, 1]])
        if self.branch != "privileged" or self.fork_tick >= 0:
            self.det = TwoFidelityDetector(os.environ["DEEP_WEIGHTS"], device="cuda:0", backend="torch")

    def sensors(self):
        s = list(super().sensors())
        s.append({"type": "sensor.camera.rgb", **CAM_POSE, "width": W, "height": H, "fov": FOV, "id": "rgb_front_perc"})
        return s

    def _cam_world(self):
        tf = self._vehicle.get_transform()
        return np.array(tf.get_matrix()) @ np.array(carla.Transform(carla.Location(CAM_POSE["x"], CAM_POSE["y"], CAM_POSE["z"])).get_matrix())

    def _project(self, actor, Twc_inv):
        corners = actor.bounding_box.get_world_vertices(actor.get_transform())
        P = np.array([[c.x, c.y, c.z, 1.0] for c in corners]) @ Twc_inv.T
        x, y, z = P[:, 1], -P[:, 2], P[:, 0]
        if (z > 0.1).sum() < 4:
            return None
        z = np.maximum(z, 0.1)
        u, v = self.K[0, 0] * x / z + self.K[0, 2], self.K[1, 1] * y / z + self.K[1, 2]
        x1, y1, x2, y2 = max(u.min(), 0), max(v.min(), 0), min(u.max(), W - 1), min(v.max(), H - 1)
        return None if x2 - x1 < 2 or y2 - y1 < 2 else (x1, y1, x2, y2)

    def _ground(self, u, v, Twc):
        d = np.array([1.0, (u - self.K[0, 2]) / self.K[0, 0], -(v - self.K[1, 2]) / self.K[1, 1]])
        dw = Twc[:3, :3] @ d
        o = Twc[:3, 3]
        gz = self._vehicle.get_transform().location.z
        if dw[2] >= -1e-6:
            return None
        s = (gz - o[2]) / dw[2]
        p = o + s * dw
        return p, math.degrees(math.atan2(dw[1], dw[0]))

    def _detect(self, img, mode):
        d = self.det.detect(img, MODES[mode])[0]
        k = d["conf"] >= THR
        return {"xyxy": d["xyxy"][k].astype(float), "coarse": np.asarray(d["coarse"])[k]}

    def _perceived(self, img, mode):
        """The planner's actor list under `mode`, and a signature of what changed against the true scene."""
        actors = self._world.get_actors()
        others = [a for a in list(actors.filter("*vehicle*")) + list(actors.filter("*walker*")) if a.id != self._vehicle.id]
        Twc = self._cam_world()
        Twc_inv = np.linalg.inv(Twc)
        proj = {a.id: self._project(a, Twc_inv) for a in others}
        inc = [a for a in others if proj[a.id] is not None
               and a.get_location().distance(self._vehicle.get_location()) < self.config.detection_radius]
        det = self._detect(img, mode)
        dx, dc = det["xyxy"], np.asarray(det["coarse"]).astype(str)
        matched, used = set(), set()
        if inc and len(dx):
            M = iou([proj[a.id] for a in inc], dx)
            for i, a in enumerate(inc):
                M[i, [j for j in range(len(dx)) if dc[j] not in COMPAT[actor_class(a)]]] = 0
            r, c = linear_sum_assignment(-M)
            for i, j in zip(r, c):
                if M[i, j] >= IOU_THR:
                    matched.add(inc[i].id); used.add(j)
        hidden = {a.id for a in inc} - matched
        out = PerceivedActors([a for a in actors if a.id not in hidden])
        phantoms = []
        for j in range(len(dx)):
            if j in used:
                continue
            b = dx[j]
            g = self._ground(0.5 * (b[0] + b[2]), b[3], Twc)
            if g is None:
                continue
            p, yaw = g
            rng = math.hypot(p[0] - Twc[0, 3], p[1] - Twc[1, 3])
            if rng > 80.0 or rng < 0.5:
                continue
            cls = dc[j] if dc[j] in PRIORS else "vehicle"
            L, Wd, Ht = PRIORS[cls]
            cx, cy = p[0] + math.cos(math.radians(yaw)) * L / 2, p[1] + math.sin(math.radians(yaw)) * L / 2
            out.append(PseudoActor(carla.Location(cx, cy, p[2]), yaw, PRIORS[cls], cls))
            phantoms.append((round(cx, 1), round(cy, 1), cls))
        sig = hashlib.md5(json.dumps([sorted(hidden), sorted(phantoms)]).encode()).hexdigest()[:12]
        return out, {"in_cam": len(inc), "hidden": len(hidden), "phantoms": len(phantoms), "sig": sig}

    def _a_star(self):
        """The braking controller's requirement from the true actors (ego frame: x forward, y left)."""
        ego = self._vehicle.get_transform()
        yaw = math.radians(ego.rotation.yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        v = float(self._get_forward_speed())
        ev = self._vehicle.get_velocity()
        dist, lo, hi, ttc = [], [], [], []
        for a in list(self._world.get_actors().filter("*vehicle*")) + list(self._world.get_actors().filter("*walker*")):
            if a.id == self._vehicle.id:
                continue
            pts = []
            for p in a.bounding_box.get_world_vertices(a.get_transform()):
                dx, dy = p.x - ego.location.x, p.y - ego.location.y
                pts.append((c * dx + s * dy, -(-s * dx + c * dy)))      # CARLA's y points right; ego y left
            pts = np.array(pts)
            near = float(pts[:, 0].min()) - self._vehicle.bounding_box.extent.x
            if near <= 0 or near > PP.max_range:
                continue
            av = a.get_velocity()
            closing = (ev.x - av.x) * c + (ev.y - av.y) * s
            dist.append(near); lo.append(float(pts[:, 1].min())); hi.append(float(pts[:, 1].max()))
            ttc.append(near / closing if closing > 0.1 else np.inf)
        a, _ = Pl.required_decel(np.array(dist), np.array(lo), np.array(hi), np.array(ttc), v, PP)
        return a, v

    def _overlap(self):
        """Ego box overlaps any actor box (2D, oriented) -- the collision flag of the window outcome."""
        eb = self._vehicle.bounding_box.get_world_vertices(self._vehicle.get_transform())
        e = np.array([[p.x, p.y] for p in eb])
        ec = e.mean(0)
        for a in list(self._world.get_actors().filter("*vehicle*")) + list(self._world.get_actors().filter("*walker*")):
            if a.id == self._vehicle.id or a.get_location().distance(self._vehicle.get_location()) > 10:
                continue
            o = np.array([[p.x, p.y] for p in a.bounding_box.get_world_vertices(a.get_transform())])
            if _sat(e, o):
                return True
        return False

    def _mode_now(self):
        if self.fork_tick < 0:
            return self.branch
        if self.tick >= self.fork_tick + self.window:
            return "privileged"
        if self.fork_tick <= self.tick < self.fork_tick + self.fork_ticks:
            return self.fork_branch
        return "cheap"

    def _get_control(self, input_data, plant):
        mode = self._mode_now()
        a_star, v = self._a_star()
        loc = self._vehicle.get_location()
        rec = {"tick": self.tick, "mode": mode, "v": v, "x": loc.x, "y": loc.y, "a_star": a_star,
               "overlap": self._overlap()}
        if mode == "privileged" or "rgb_front_perc" not in input_data:
            res = super()._get_control(input_data, plant)
        else:
            img = np.ascontiguousarray(input_data["rgb_front_perc"][1][:, :, :3])
            perceived, info = self._perceived(img, mode)
            rec.update({f"{k}": v_ for k, v_ in info.items()})
            if self.shadow and mode != "full":
                _, sh = self._perceived(img, "full")
                rec.update({f"shadow_{k}": v_ for k, v_ in sh.items()})
            real = self._world
            self._world = WorldProxy(real, perceived)
            try:
                res = super()._get_control(input_data, plant)
            finally:
                self._world = real
        # AutoPilot._get_control returns (control, driving_data), or a bare control in one early branch
        ctl = res[0] if isinstance(res, tuple) else res
        rec.update({"throttle": float(ctl.throttle), "brake": float(ctl.brake), "steer": float(ctl.steer)})
        self.log.append(rec)
        self.tick += 1
        return res

    def destroy(self, results=None):
        if os.environ.get("SAVE_PATH"):
            p = Path(os.environ["SAVE_PATH"]) / f"ticks_{self.tag}.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a") as fh:
                fh.write(json.dumps({"routes": Path(os.environ.get("ROUTES", "r")).stem, "branch": self.branch,
                                     "fork_tick": self.fork_tick, "fork_branch": self.fork_branch,
                                     "route_index": getattr(self, "route_index", None), "ticks": self.log}) + "\n")
        try:
            super().destroy(results)
        except TypeError:
            super().destroy()


def _sat(a, b):
    """Separating-axis test for two convex quadrilaterals given by their corners (any order of a box's 8 vertices)."""
    from scipy.spatial import ConvexHull
    pa, pb = a[ConvexHull(a).vertices], b[ConvexHull(b).vertices]
    for poly in (pa, pb):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-e[1], e[0]])
            ra, rb = pa @ n, pb @ n
            if ra.max() < rb.min() or rb.max() < ra.min():
                return False
    return True
