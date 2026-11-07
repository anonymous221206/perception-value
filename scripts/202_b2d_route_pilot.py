"""Follow-up E3 (c): 195's pilot (PDM-Lite with perception in the loop) with per-frame routing between the cheap and
full modes (addendum E3c in docs/followup_protocol.md). 195 itself is unchanged.

A decision frame is 10 ticks (0.5 s): frame k = ticks [10k, 10k + 10). At a frame's first tick the cheap mode runs and
the router decides from its output; an escalated frame uses the full mode on all its 10 ticks, otherwise cheap.

Environment (besides 195's; PERC_BRANCH=cheap):
  ROUTER=dev     cheap throughout with the full mode in shadow (SHADOW=1). Every tick also logs the braking
                 controller's requirement under each mode's perceived scene (a_req_cheap, a_req_full), and each frame's
                 first tick the gate features of the cheap output: the development data of the gate (203 --stage fit)
  ROUTER=gate    escalate a frame when the gate's prediction >= tau (ROUTER_MODEL: pickle with model, cols, tau)
  ROUTER=random  escalate a frame with probability ROUTER_BUDGET (numpy generator seeded with ROUTER_SEED)
Gate features, E2's gate columns without the driving command: counts at the operating threshold (0.25) and at the
detector floor (0.10), confidence sum (kept) and max (all), summed binary entropy (kept, all), largest and summed box
area (kept, fraction of the image), lowest box bottom (kept), plus the ground-plane distance of the nearest kept
detection and the ego speed.
Logged per routed frame (first tick): route_score, escalate, gate (the features).
"""
from __future__ import annotations

import importlib.util
import math
import os
import pickle
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("b2d_pilot_195", Path(__file__).with_name("195_b2d_pilot.py"))
P195 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P195)

FRAME, FLOOR = 10, 0.10
GATE_COLS = ["n_det", "n_cand", "conf_sum", "conf_max", "ent_sum", "unc_sum", "area_max", "area_sum", "bottom_max",
             "near_dist", "ego_speed"]


def get_entry_point():
    return "RoutePilot"


class RoutePilot(P195.DeepPilot):
    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(path_to_conf_file, route_index, traffic_manager)
        self.router = os.environ.get("ROUTER", "dev")
        self.budget = float(os.environ.get("ROUTER_BUDGET", "0.2"))
        self.rng = np.random.default_rng(int(os.environ.get("ROUTER_SEED", "0")))
        self.gate = None
        if self.router == "gate":
            with open(os.environ["ROUTER_MODEL"], "rb") as fh:
                self.gate = pickle.load(fh)
        self.frame_mode = "cheap"
        self._raw, self._pa = {}, {}

    # one detector pass per (tick, mode): the router's cheap pass is reused by the planner's perceived scene
    def _detect(self, img, mode):
        key = (self.tick, mode)
        if key not in self._raw:
            self._raw = {k: v for k, v in self._raw.items() if k[0] == self.tick}
            self._raw[key] = self.det.detect(img, P195.MODES[mode])[0]
        d = self._raw[key]
        k = d["conf"] >= P195.THR
        return {"xyxy": d["xyxy"][k].astype(float), "coarse": np.asarray(d["coarse"])[k]}

    def _perceived(self, img, mode):
        out, info = super()._perceived(img, mode)
        self._pa[mode] = out
        return out, info

    def _mode_now(self):
        return self.frame_mode if self.router in ("gate", "random") else super()._mode_now()

    def _a_req(self, actors):
        """The braking controller's requirement under a perceived scene (195's _a_star on that actor list)."""
        real = self._world
        self._world = P195.WorldProxy(real, actors)
        try:
            return float(self._a_star()[0])
        finally:
            self._world = real

    def _gate_features(self, d, v):
        conf = np.asarray(d["conf"], float)
        xyxy = np.asarray(d["xyxy"], float).reshape(-1, 4)
        ent = np.asarray(d["binent"], float)
        fl = conf >= FLOOR
        conf, xyxy, ent = conf[fl], xyxy[fl], ent[fl]
        k = conf >= P195.THR
        area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]) / (P195.W * P195.H) if len(xyxy) else np.zeros(0)
        bottom = xyxy[:, 3] / P195.H if len(xyxy) else np.zeros(0)
        near = 200.0
        if k.any():
            Twc = self._cam_world()
            for b in xyxy[k]:
                g = self._ground(0.5 * (b[0] + b[2]), b[3], Twc)
                if g is not None:
                    near = min(near, math.hypot(g[0][0] - Twc[0, 3], g[0][1] - Twc[1, 3]))
        return {"n_det": int(k.sum()), "n_cand": int(len(conf)), "conf_sum": float(conf[k].sum()),
                "conf_max": float(conf.max()) if len(conf) else 0.0, "ent_sum": float(ent[k].sum()),
                "unc_sum": float(ent.sum()), "area_max": float(area[k].max()) if k.any() else 0.0,
                "area_sum": float(area[k].sum()), "bottom_max": float(bottom[k].max()) if k.any() else 0.0,
                "near_dist": float(near), "ego_speed": float(v)}

    def _decide(self, img):
        self._detect(img, "cheap")
        f = self._gate_features(self._raw[(self.tick, "cheap")], self._get_forward_speed())
        if self.router == "gate":
            s = float(self.gate["model"].predict(np.array([[f[c] for c in self.gate["cols"]]], float))[0])
            esc = s >= self.gate["tau"]
        else:
            s = float(self.rng.random())
            esc = s < self.budget
        self.frame_mode = "full" if esc else "cheap"
        return {"route_score": s, "escalate": bool(esc), "gate": f}

    def _get_control(self, input_data, plant):
        dec, t = None, self.tick
        if self.router in ("gate", "random") and t % FRAME == 0:
            if "rgb_front_perc" in input_data:
                dec = self._decide(np.ascontiguousarray(input_data["rgb_front_perc"][1][:, :, :3]))
            else:
                self.frame_mode = "cheap"
        self._pa = {}
        res = super()._get_control(input_data, plant)
        rec = self.log[-1]
        if dec is not None:
            rec.update(dec)
        if self.router == "dev" and "cheap" in self._pa and "full" in self._pa:
            rec["a_req_cheap"] = self._a_req(self._pa["cheap"])
            rec["a_req_full"] = self._a_req(self._pa["full"])
            if t % FRAME == 0 and (t, "cheap") in self._raw:
                rec["gate"] = self._gate_features(self._raw[(t, "cheap")], rec["v"])
        return res
