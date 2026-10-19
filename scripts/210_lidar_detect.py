#!/usr/bin/env python
"""Follow-up E7, on the training machine (209's environment): OpenPCDet's released nuScenes CenterPoint checkpoints on
the keyframes E7 reads. addendum E7 in docs/followup_protocol.md.

  root     OpenPCDet's data root (links to the dataset and to 205's sweeps) and its 10-sweep infos; the infos of E7's
           scenes (the 150 official-validation scenes and E1's 228 new test scenes) -> nuscenes_infos_10sweeps_e7.pkl
  detect   --model {pp,voxel}: boxes per keyframe, in the ego frame -> <out>/<model>/<scene>.npz, det_annos.pkl
  eval     gate E7-0: OpenPCDet's nuScenes evaluation on the 150 validation scenes -> <out>/<model>/gate_e7_0.json

The configurations are used as released. Two settings make inference repeatable, and neither changes the model:
OpenPCDet draws the sweep order and shuffles the points at test time, so numpy is seeded with the sample's index
before each sample is read.
"""
from __future__ import annotations

import argparse, csv, json, os, pickle, sys, time
from pathlib import Path

import numpy as np

# OpenPCDet 233f849 still uses numpy's removed aliases (np.int, np.float, np.bool), e.g. in the 2D backbone of the
# PointPillars and TransFusion models; they were the builtins, so restoring them changes no computation.
for _a, _t in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _a):
        setattr(np, _a, _t)

B = Path(os.environ["B"])                                # the work root on the GPU machine
D = B / "deep_ext"
PCDET = D / "third_party" / "OpenPCDet"
ROOT = D / "nusc_lidar" / "pcdet_root"                  # OpenPCDet's DATA_PATH; VERSION v1.0-trainval below it
OUT = D / "nusc_lidar" / "e7_det"
NS = B / "datasets" / "nuscenes"
MODELS = {"pp": ("cbgs_dyn_pp_centerpoint", "1UvGm6mROMyJzeSRu7OD1leU_YWoAZG7v", 50.03),
          "voxel": ("cbgs_voxel0075_voxelnext", "1IV7e7G9X-61KXSjMGtQo579pzDNbhwvf", 60.53),     # amendment 1
          "transfusion": ("transfusion_lidar", "1cuZ2qdDnxSwTCsiXWwbqCGF-uoazTXbz", 64.58)}       # secondary
INFOS = "nuscenes_infos_10sweeps_e7.pkl"


def stage_root(args):
    v = ROOT / "v1.0-trainval"
    v.mkdir(parents=True, exist_ok=True)
    for name, src in (("samples", NS / "samples"), ("maps", NS / "maps"), ("v1.0-trainval", NS / "v1.0-trainval"),
                      ("sweeps", D / "nusc_lidar" / "sweeps")):
        if not (v / name).exists():
            (v / name).symlink_to(src)
    sys.path.insert(0, str(PCDET))
    from pcdet.datasets.nuscenes.nuscenes_dataset import create_nuscenes_info
    if not (v / "nuscenes_infos_10sweeps_val.pkl").exists():
        create_nuscenes_info(version="v1.0-trainval", data_path=ROOT, save_path=ROOT, max_sweeps=10)
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(v), verbose=False)
    split = list(csv.DictReader(open(args.scene_split)))
    want = {r["scene"] for r in split if r["official"] == "val" or (r["group"] != "frozen85" and r["deep_split"] == "test")}
    infos = []
    for f in ("nuscenes_infos_10sweeps_train.pkl", "nuscenes_infos_10sweeps_val.pkl"):
        for i in pickle.load(open(v / f, "rb")):
            sc = nusc.get("scene", nusc.get("sample", i["token"])["scene_token"])["name"]
            if sc in want:
                i["scene_name"] = sc
                infos.append(i)
    missing = [s["lidar_path"] for i in infos for s in i["sweeps"] if not (v / s["lidar_path"]).exists()]
    pickle.dump(infos, open(v / INFOS, "wb"))
    print(f"{len(infos)} keyframes of {len({i['scene_name'] for i in infos})} scenes; missing sweep files: {len(missing)}")
    if missing:
        raise SystemExit("sweeps missing: " + ", ".join(missing[:3]))


def checkpoint(model):
    name, gid, _ = MODELS[model]
    f = D / "weights" / "pcdet" / f"{name}.pth"
    if not f.exists():
        import gdown
        f.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=gid, output=str(f), quiet=False)
    return f


def build(model, workers):
    sys.path.insert(0, str(PCDET))
    os.chdir(PCDET / "tools")                         # the configurations name their base configuration relative to it
    from easydict import EasyDict
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
    from pcdet.models import build_network
    from pcdet.utils import common_utils
    get = NuScenesDataset.__getitem__

    def seeded(self, index):                          # repeatable sweep order and point shuffle
        np.random.seed(index)
        return get(self, index)
    NuScenesDataset.__getitem__ = seeded
    cfg_from_yaml_file(f"cfgs/nuscenes_models/{MODELS[model][0]}.yaml", cfg)
    cfg.DATA_CONFIG.DATA_PATH = str(ROOT)
    cfg.DATA_CONFIG.INFO_PATH = EasyDict({"train": [], "test": [INFOS]})
    logger = common_utils.create_logger()
    ds, loader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1,
                                     dist=False, workers=workers, logger=logger, training=False)
    net = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=ds)
    net.load_params_from_file(filename=str(checkpoint(model)), logger=logger, to_cpu=False)
    return cfg, ds, loader, net.cuda().eval()


def stage_detect(args):
    import torch
    from pcdet.models import load_data_to_gpu
    cfg, ds, loader, net = build(args.model, args.workers)
    out = OUT / args.model
    out.mkdir(parents=True, exist_ok=True)
    by_token = {i["token"]: i for i in ds.infos}
    det_annos, per_scene, times = [], {}, []
    t0 = time.time()
    with torch.no_grad():
        for k, batch in enumerate(loader):
            load_data_to_gpu(batch)
            torch.cuda.synchronize(); t1 = time.time()
            pred, _ = net(batch)
            torch.cuda.synchronize(); times.append(time.time() - t1)
            annos = ds.generate_prediction_dicts(batch, pred, cfg.CLASS_NAMES)
            det_annos += annos
            a = annos[0]
            info = by_token[a["metadata"]["token"]]
            T = np.linalg.inv(info["ref_from_car"])               # LiDAR -> ego at the keyframe
            b = np.asarray(a["boxes_lidar"], float).reshape(-1, 9)
            xyz = (T[:3, :3] @ b[:, :3].T).T + T[:3, 3]
            yaw = b[:, 6] + np.arctan2(T[1, 0], T[0, 0])
            vel = (T[:2, :2] @ b[:, 7:9].T).T
            per_scene.setdefault(info["scene_name"], []).append(
                (a["metadata"]["token"], np.hstack([xyz, b[:, 3:6], yaw[:, None], vel]), np.asarray(a["score"], float),
                 np.asarray(a["name"])))
            if k % 500 == 0:
                print(f"  {k} / {len(ds)} [{time.time() - t0:.0f}s]", flush=True)
    for sc, rows in per_scene.items():
        n = np.array([len(r[2]) for r in rows])
        np.savez_compressed(out / f"{sc}.npz", tokens=np.array([r[0] for r in rows]),
                            offsets=np.concatenate([[0], np.cumsum(n)]),
                            boxes=np.concatenate([r[1] for r in rows]) if n.sum() else np.zeros((0, 9)),
                            score=np.concatenate([r[2] for r in rows]) if n.sum() else np.zeros(0),
                            name=np.concatenate([r[3] for r in rows]) if n.sum() else np.zeros(0, "U1"))
    pickle.dump(det_annos, open(out / "det_annos.pkl", "wb"))
    t = np.array(times[20:]) if len(times) > 220 else np.array(times)
    json.dump({"model": MODELS[args.model][0], "keyframes": len(det_annos), "scenes": len(per_scene),
               "latency_ms_median_200": float(np.median(t[:200]) * 1e3), "seconds": time.time() - t0,
               "checkpoint_md5": os.popen(f"md5sum {checkpoint(args.model)}").read().split()[0]},
              open(out / "summary.json", "w"), indent=1)
    print("wrote", out)


def stage_eval(args):
    cfg, ds, loader, net = build(args.model, 0)
    out = OUT / args.model
    det = pickle.load(open(out / "det_annos.pkl", "rb"))
    val = {i["token"] for i in pickle.load(open(ROOT / "v1.0-trainval" / "nuscenes_infos_10sweeps_val.pkl", "rb"))}
    det = [a for a in det if a["metadata"]["token"] in val]
    s, r = ds.evaluation(det, cfg.CLASS_NAMES, output_path=str(out / "eval_val150"))
    m = json.load(open(out / "eval_val150" / "metrics_summary.json"))
    g = {"model": MODELS[args.model][0], "keyframes": len(det), "mAP": 100 * m["mean_ap"], "NDS": 100 * m["nd_score"],
         "published_mAP": MODELS[args.model][2]}
    g["pass"] = bool(abs(g["mAP"] - g["published_mAP"]) <= 1.0)
    json.dump(g, open(out / "gate_e7_0.json", "w"), indent=1)
    print(s, "\ngate E7-0:", g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["root", "detect", "eval"], required=True)
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scene_split", default=str(D / "code" / "scene_split.csv"))
    args = ap.parse_args()
    {"root": stage_root, "detect": stage_detect, "eval": stage_eval}[args.stage](args)


if __name__ == "__main__":
    main()
