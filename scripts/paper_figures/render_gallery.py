"""Render the qualitative gallery (Figures gallery-main and gallery) from the release's fig_gallery export.

Input: results/final/fig_gallery/ of the release (per-frame JSON with every cheap, full and reference box, the
controller's corridor flags and required accelerations, and a copy of the 1600x900 CAM_FRONT image).
The decisive object is the full-mode detection the controller considers with the largest required acceleration
(it equals the frame's a_req.full). It is an added detection (red) when it matches no reference object, and a
recovered object (green on full, dashed yellow on cheap at the reference box) when its reference object is missed
by the cheap mode. Each pair is cropped to the same 640x360 window around the decisive object and upscaled to
1120x630. Every other detection is a thin white box.

Usage: python render_gallery.py <release>/results/final/fig_gallery ../figs/gallery_ego
"""
import json
import sys

from PIL import Image, ImageDraw

FRAMES = [("negative_1_scene-0055_03", "neg1"), ("negative_2_scene-0098_22", "neg2"),
          ("negative_3_scene-0101_38", "neg3"), ("negative_4_scene-0102_32", "neg4"),
          ("positive_3_scene-0053_25", "pos3"), ("positive_4_scene-0065_24", "pos4"),
          ("positive_5_scene-0048_08", "pos5"), ("positive_6_scene-0054_30", "pos6")]
CW, CH, OW, OH = 640, 360, 1120, 630
SC = OW / CW
WHITE, RED, GREEN, YELLOW = (255, 255, 255), (210, 32, 38), (34, 160, 52), (245, 200, 0)


def to_crop(xyxy, ox, oy):
    x0, y0, x1, y1 = xyxy
    return [(x0 - ox) * SC, (y0 - oy) * SC, (x1 - ox) * SC, (y1 - oy) * SC]


def dashed(draw, xyxy, colour, width, dash=18, gap=10):
    x0, y0, x1, y1 = xyxy
    for (a, b), (c, d) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        length = ((c - a) ** 2 + (d - b) ** 2) ** 0.5 or 1
        t = 0
        while t < length:
            e = min(t + dash, length)
            draw.line([(a + (c - a) * t / length, b + (d - b) * t / length),
                       (a + (c - a) * e / length, b + (d - b) * e / length)], fill=colour, width=width)
            t += dash + gap


def main(src, out):
    for stem, name in FRAMES:
        d = json.load(open(f"{src}/{stem}.json"))
        img = Image.open(f"{src}/{d['image_copy']}").convert("RGB")
        considered = [b for b in d["full_boxes"] if b["considered_by_controller"]]
        dec = max(considered, key=lambda b: b["a_req_obj"])
        gt = {g["gt_id"]: g for g in d["gt_boxes"]}
        ref = gt.get(dec.get("matched_gt_id")) if dec.get("matched_gt_id") is not None else None
        assert ref is None or not ref["matched_by_cheap"], stem
        boxes = [dec["xyxy"]] + ([ref["xyxy_projected"]] if ref else [])
        cx = (min(b[0] for b in boxes) + max(b[2] for b in boxes)) / 2
        cy = (min(b[1] for b in boxes) + max(b[3] for b in boxes)) / 2
        ox = int(min(max(cx - CW / 2, 0), img.width - CW))
        oy = int(min(max(cy - CH / 2, 0), img.height - CH))
        crop = img.crop((ox, oy, ox + CW, oy + CH)).resize((OW, OH), Image.LANCZOS)

        cheap = crop.copy()
        draw = ImageDraw.Draw(cheap)
        for b in d["cheap_boxes"]:
            draw.rectangle(to_crop(b["xyxy"], ox, oy), outline=WHITE, width=2)
        if ref is not None:
            dashed(draw, to_crop(ref["xyxy_projected"], ox, oy), YELLOW, 6)
        cheap.save(f"{out}/{name}_cheap.jpg", quality=92)

        full = crop.copy()
        draw = ImageDraw.Draw(full)
        for b in d["full_boxes"]:
            if b is not dec:
                draw.rectangle(to_crop(b["xyxy"], ox, oy), outline=WHITE, width=2)
        draw.rectangle(to_crop(dec["xyxy"], ox, oy), outline=RED if ref is None else GREEN, width=6)
        full.save(f"{out}/{name}_full.jpg", quality=92)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
