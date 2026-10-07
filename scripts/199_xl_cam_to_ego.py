#!/usr/bin/env python
"""Follow-up E1/E5: the camera -> ego table (143's nuScenes rule) for every trainval scene with detections.

143 tabulates CAM_FRONT's calibrated_sensor of each scene's first sample (R, t), that sample's intrinsics, the lift's
camera height and the frame interval, for the frozen 85 scenes. This writes the same entries for all 850 scenes into a
separate file, data/cache/cam_to_ego_xl.json, which `rap.cache` merges under the frozen table: a scene already in
cam_to_ego.json must get exactly its frozen entry (checked here), so the frozen track reads the same values.
"""
from __future__ import annotations

import importlib.util, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap.nusc import FRAME_DT as NUSC_DT, NuScenesDB                            # noqa: E402
from rap.paths import CACHE, NUSCENES_XL                                        # noqa: E402

s = importlib.util.spec_from_file_location("t143", ROOT / "scripts" / "143_cam_to_ego_table.py")
t143 = importlib.util.module_from_spec(s)
s.loader.exec_module(t143)


def main():
    root = NUSCENES_XL
    db = NuScenesDB(root, "v1.0-trainval")
    by = {db.scene_name(sc): sc for sc in db.scenes}
    out = {}
    for n in sorted(p.stem for p in (Path(CACHE) / "nusc_det_xl" / "ns_cheap_320").glob("*.npz")):
        tok = db.samples(by[n])[0]
        R, t = db._cam_to_ego(tok)
        c = db.calib(tok)
        out[n] = {"R": t143.lst(R), "t": t143.lst(t), "t_task22": t143.lst(t), "K": t143.lst(c.P2[:3, :3]),
                  "cam_h": t143.NUSC_CAM_H, "dt": NUSC_DT}
    frozen = json.loads((Path(CACHE) / "cam_to_ego.json").read_text())["nuScenes"]
    bad = [n for n in frozen if out.get(n) != frozen[n]]
    if bad:
        raise SystemExit(f"entries differ from the frozen table for {len(bad)} scenes, e.g. {bad[:3]}")
    (Path(CACHE) / "cam_to_ego_xl.json").write_text(json.dumps({"nuScenes": out}, indent=1))
    print(f"wrote cam_to_ego_xl.json: {len(out)} scenes; the {len(frozen)} frozen entries are identical")


if __name__ == "__main__":
    main()
