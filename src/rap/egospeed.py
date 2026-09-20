"""The ego speed an allocator is allowed to read.

Ego speed is a signal in several stages (`trivial_ego_speed`, and four of the trivial heuristics of the core
matrix). A signal computed at frame *i* may use only what is known at frame *i*.

  * KITTI: `KittiAdapter.speeds` reads the OXTS forward velocity of the frame itself
    (`kitti.load_oxts(seq)[:, 8]`), an instantaneous measurement -- already causal.
  * nuPlan: `91_nuplan_cheap_features.py` reads `ego.dynamic_car_state.rear_axle_velocity_2d.x` at the current
    iteration -- already causal.
  * nuScenes: `NuScenesDB.ego_speeds` differentiates the ego poses with a *centred* difference and so reads the
    pose half a second ahead. `NuScenesDB.ego_speeds_causal` is the backward difference that replaces it in
    signals; `138_causal_ego_speed.py` tabulates it once, per frame, into `data/cache/nusc_v_ego_causal.csv`.

The decision cost keeps the centred value: there `v_ego` is the vehicle's own state, CHEAP and FULL see the same
number, and V = J(CHEAP) - J(FULL) does not move (Task 22 Part B pre-registration).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .paths import CACHE

CAUSAL_CSV = Path(CACHE) / "nusc_v_ego_causal.csv"
COL = "v_ego_causal"
MODE = "causal"           # set by a stage's `--ego_speed`; see `add_argument`
_TABLE = None


def table() -> pd.DataFrame:
    global _TABLE
    if _TABLE is None:
        if not CAUSAL_CSV.exists():
            raise SystemExit(f"missing {CAUSAL_CSV}: run scripts/138_causal_ego_speed.py first")
        t = pd.read_csv(CAUSAL_CSV)
        _TABLE = t.astype({"seq": str, "frame": int})
    return _TABLE


def attach(d: pd.DataFrame, dataset: str, col: str = COL) -> pd.DataFrame:
    """Add the causal ego speed of every row. On tracks that are already causal it is the shipped `v_ego`."""
    if col in d.columns:
        return d
    if dataset != "nuScenes" or MODE == "centred":
        d[col] = d["v_ego"].to_numpy(float)
        return d
    k = d[["seq", "frame"]].astype({"seq": str, "frame": int}).reset_index(drop=True)
    m = k.merge(table()[["seq", "frame", COL, "first_frame"]], on=["seq", "frame"], how="left",
                validate="one_to_one")
    assert m[COL].notna().all(), f"the causal ego-speed table misses {int(m[COL].isna().sum())} frames"
    v = m[COL].to_numpy(float)
    if MODE == "causal_drop_first":
        # the sensitivity the pre-registration promised: a scene's first frame carries no measured speed, so
        # instead of the defined 0 it is put out of reach of any quota
        v = np.where(m.first_frame.to_numpy(int) == 1, -np.inf, v)
    d[col] = v
    return d


def add_argument(ap):
    """`--ego_speed {causal,centred,causal_drop_first}`.

    `centred` reproduces the pre-fix signal and is the sanity gate; `causal_drop_first` is the sensitivity to the
    rule for a scene's first frame, which it puts out of reach instead of giving it a speed of 0.
    """
    ap.add_argument("--ego_speed", choices=["causal", "centred", "causal_drop_first"], default="causal",
                    help="which ego speed the signals read; 'centred' reproduces the pre-Task-22B values")


def configure(args):
    global MODE
    MODE = args.ego_speed
    print(f"  ego-speed signal: {MODE}", flush=True)
