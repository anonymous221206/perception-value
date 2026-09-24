"""Write fig_transform.tex: nDG(label) - nDG(raw V) for the two R1 regression routers at a 20% budget, ten core cells.

Input: results/final/target_transform_heatmap.csv of the release (Task 30, post hoc). A star marks a paired 95%
interval that excludes zero.
Usage (from the repository root): python scripts/paper_figures/make_fig_transform.py results/final/target_transform_heatmap.csv
"""
import sys

import pandas as pd

CELLS = [("KITTI", "mono", "brake"), ("KITTI", "mono", "traj"), ("KITTI", "oracle", "brake"), ("KITTI", "oracle", "traj"),
         ("nuScenes", "mono", "brake"), ("nuScenes", "mono", "plan_ade"), ("nuScenes", "mono", "plan_fde"),
         ("nuScenes", "oracle", "brake"), ("nuScenes", "oracle", "plan_ade"), ("nuScenes", "oracle", "plan_fde")]
CELL_LABEL = {"brake": r"$q_{\mathrm{brake}}$", "traj": r"$q_{\mathrm{traj}}$", "plan_ade": "ADE", "plan_fde": "FDE"}
ROWS = [("R1_mlp_reg", "rankV", r"MLP, rank$(V)$"), ("R1_mlp_reg", "secdfV", r"MLP, signed CDF$(V)$"),
        ("R1_mlp_reg", "Q", "MLP, ORIC reward"), ("R1_mlp_reg", "G", "MLP, AP swap"),
        ("R1_gbm_reg", "rankV", r"GBM, rank$(V)$"), ("R1_gbm_reg", "secdfV", r"GBM, signed CDF$(V)$"),
        ("R1_gbm_reg", "Q", "GBM, ORIC reward"), ("R1_gbm_reg", "G", "GBM, AP swap")]
W, H, SAT = 1.02, 0.40, 0.45

h = pd.read_csv(sys.argv[1])
h = h[h.quota == 0.2]
out = [r"\documentclass[10pt,border=1pt]{standalone}", r"\usepackage{times}", r"\usepackage{amsmath}",
       r"\usepackage{xcolor}", r"\usepackage{tikz}", r"\definecolor{cNeg}{HTML}{C0392B}",
       r"\definecolor{cPos}{HTML}{1F6FB4}", r"\begin{document}", r"\begin{tikzpicture}[font=\scriptsize]"]
for j, (tr, geo, sysm) in enumerate(CELLS):
    x = (j + 0.5) * W
    out.append(rf"\node[align=center, anchor=south, text width={W:.2f}cm] at ({x:.3f},0.03) {{{'nuSc.' if tr == 'nuScenes' else tr}\\{geo}\\{CELL_LABEL[sysm]}}};")
for i, (arch, lab, name) in enumerate(ROWS):
    y = -(i + 0.5) * H - (0.12 if i >= 4 else 0)
    out.append(rf"\node[anchor=east] at (-0.08,{y:.3f}) {{{name}}};")
    for j, (tr, geo, sysm) in enumerate(CELLS):
        r = h[(h.architecture == arch) & (h.label == lab) & (h.track == tr) & (h.geometry == geo) & (h.system == sysm)]
        assert len(r) == 1, (arch, lab, tr, geo, sysm, len(r))
        r = r.iloc[0]
        v = float(r.minus_V)
        star = r.minus_V_lo > 0 or r.minus_V_hi < 0
        mix = int(round(100 * min(abs(v) / SAT, 1.0)))
        col = f"{'cPos' if v > 0 else 'cNeg'}!{mix}!white"
        txt = ("0.00" if abs(v) < 0.005 else f"{v:+.2f}".replace("-", "$-$")) + ("*" if star else "")
        tc = "white" if mix > 55 else "black"
        x0, y0 = j * W, y - H / 2
        out.append(rf"\fill[{col}] ({x0:.3f},{y0:.3f}) rectangle ({x0 + W:.3f},{y0 + H:.3f});")
        body = r"\textbf{" + txt + "}" if star else txt
        out.append(rf"\node[text={tc}] at ({x0 + W / 2:.3f},{y:.3f}) {{{body}}};")
out.append(rf"\draw[gray!55] (0,0) rectangle ({10 * W:.3f},{-8 * H - 0.12:.3f});")
out.append(rf"\draw[gray!55] (0,{-4 * H - 0.06:.3f}) -- ({10 * W:.3f},{-4 * H - 0.06:.3f});")
out += [r"\end{tikzpicture}", r"\end{document}"]
open("fig_transform.tex", "w").write("\n".join(out) + "\n")
print("wrote fig_transform.tex")
