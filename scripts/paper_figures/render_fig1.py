"""Render Figure 1: two gallery frames with the braking controller's corridor drawn on the road.

Same crop and overlays as render_gallery.py (decisive object in red/green, missed reference object dashed yellow,
other detections thin white), plus the controller's driving corridor: the band |y| < 1.2 m of the ego frame on the
ground plane, 4-80 m ahead (the controller's range), projected into the image with the CAM_FRONT calibration of each sample
(fig1_calibration.json, from the nuScenes calibrated_sensor table).

Usage: python render_fig1.py <release>/results/final/fig_gallery ../figs/fig1
"""
import json
import sys

import numpy as np
from PIL import Image, ImageDraw

import render_gallery as G

FRAMES = [("figure1_scene-0055_03", "neg1"), ("figure1_scene-0065_24", "pos4")]
HALF_W, Z0, Z1 = 1.2, 4.0, 80.0
FILL, EDGE = (80, 200, 255, 60), (80, 200, 255, 220)


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def project(pts_ego, cal):
    R, t, K = quat_to_R(cal["rotation"]), np.array(cal["translation"]), np.array(cal["K"])
    p = (pts_ego - t) @ R            # ego -> camera: R^T (p - t), row form
    uv = (p @ K.T)
    return uv[:, :2] / uv[:, 2:3]


def corridor_polygon(cal, ox, oy):
    z = np.linspace(Z0, Z1, 60)
    left = np.stack([z, np.full_like(z, HALF_W), np.zeros_like(z)], 1)
    right = np.stack([z, np.full_like(z, -HALF_W), np.zeros_like(z)], 1)
    uvl, uvr = project(left, cal), project(right, cal)
    to = lambda uv: [((u - ox) * G.SC, (v - oy) * G.SC) for u, v in uv]
    return to(uvl), to(uvr)


def overlay_corridor(img, cal, ox, oy):
    l, r = corridor_polygon(cal, ox, oy)
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.polygon(l + r[::-1], fill=FILL)
    d.line(l, fill=EDGE, width=4)
    d.line(r, fill=EDGE, width=4)
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def main(src, out):
    cals = json.load(open("fig1_calibration.json"))
    for stem, name in FRAMES:
        d = json.load(open(f"{src}/{stem}.json"))
        cal = cals[name]
        assert cal["filename"] in d["image_path"], (cal["filename"], d["image_path"])
        img = Image.open(f"{src}/{d['image_copy']}").convert("RGB")
        considered = [b for b in d["full_boxes"] if b["considered_by_controller"]]
        dec = max(considered, key=lambda b: b["a_req_obj"])
        gt = {g["gt_id"]: g for g in d["gt_boxes"]}
        ref = gt.get(dec.get("matched_gt_id")) if dec.get("matched_gt_id") is not None else None
        boxes = [dec["xyxy"]] + ([ref["xyxy_projected"]] if ref else [])
        cx = (min(b[0] for b in boxes) + max(b[2] for b in boxes)) / 2
        cy = (min(b[1] for b in boxes) + max(b[3] for b in boxes)) / 2
        ox = int(min(max(cx - G.CW / 2, 0), img.width - G.CW))
        oy = int(min(max(cy - G.CH / 2, 0), img.height - G.CH))
        crop = img.crop((ox, oy, ox + G.CW, oy + G.CH)).resize((G.OW, G.OH), Image.LANCZOS)
        crop = overlay_corridor(crop, cal, ox, oy)

        cheap = crop.copy()
        draw = ImageDraw.Draw(cheap)
        for b in d["cheap_boxes"]:
            draw.rectangle(G.to_crop(b["xyxy"], ox, oy), outline=G.WHITE, width=2)
        if ref is not None:
            G.dashed(draw, G.to_crop(ref["xyxy_projected"], ox, oy), G.YELLOW, 6)
        cheap.save(f"{out}/{name}_cheap.jpg", quality=92)

        full = crop.copy()
        draw = ImageDraw.Draw(full)
        for b in d["full_boxes"]:
            if b is not dec:
                draw.rectangle(G.to_crop(b["xyxy"], ox, oy), outline=G.WHITE, width=2)
        draw.rectangle(G.to_crop(dec["xyxy"], ox, oy), outline=G.RED if ref is None else G.GREEN, width=6)
        full.save(f"{out}/{name}_full.jpg", quality=92)
        print(name, "decisive", dec["class"], "lat", round(dec["x_lateral_m"], 2), "fwd", round(dec["z_forward_m"], 1),
              "in_corridor", dec["in_corridor"], "| reference lat", None if ref is None else round(ref["x_lateral_m"], 2),
              "in_corridor", None if ref is None else ref["in_corridor"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
