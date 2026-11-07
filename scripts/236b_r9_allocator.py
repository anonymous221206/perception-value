#!/usr/bin/env python
"""Follow-up round 3, R9: the allocator 151_profile_allocator.py times (R9_ALLOC = the pickle 236 --stage crit writes).

R3's extractor on R3's 200 fixed frames (the export `score_input` receives carries no cached geometry, so `inp` only sets
the pace), then the model as plain numpy: tree walks for the gradient-boosted models, the averaged nets for the MLP.
"""
from __future__ import annotations

import importlib.util, os, pickle, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _t231():
    s = importlib.util.spec_from_file_location("t231", ROOT / "scripts" / "231_r3_gate_cost.py")
    m = importlib.util.module_from_spec(s)
    sys.modules["t231"] = m
    s.loader.exec_module(m)
    return m


def predictor(spec):
    if spec["kind"] == "gbm":
        trees, base = spec["trees"], spec["base"]

        def f(x):
            s = base
            for t in trees:
                n = t[0]
                while not n[0]:
                    n = t[n[3]] if x[n[1]] <= n[2] else t[n[4]]
                s += n[5]
            return s
        return f
    mu, sd, nets = spec["mu"], spec["sd"], spec["nets"]

    def f(x):
        z = np.clip((x - mu) / sd, -10.0, 10.0).astype(np.float32)
        s = 0.0
        for w in nets:
            h = np.maximum(w[0] @ z + w[1], 0.0)
            h = np.maximum(w[2] @ h + w[3], 0.0)
            s += float((w[4] @ h + w[5])[0])
        return s / len(nets)
    return f


if "--check" in sys.argv:
    spec = pickle.loads(Path(sys.argv[sys.argv.index("--check") + 1]).read_bytes())
    assert callable(predictor(spec))
    print("allocator spec ok", spec["kind"], spec["subset"], len(spec["cols"]))
    sys.exit(0)

_spec = pickle.loads(Path(os.environ["R9_ALLOC"]).read_bytes())
_t = _t231()
_items = _t.profile_frames()
_ext = _t.extractor(_spec["subset"], _spec["cols"])
_pred = predictor(_spec)
_i = {"k": 0}


def score_input(inp):
    it = _items[_i["k"] % len(_items)]
    _i["k"] += 1
    return float(_pred(_ext(it)[0]))
