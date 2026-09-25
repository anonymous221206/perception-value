"""Write the fig_bev_*.dat files from a release's results/final/fig_bev_objects.csv.gz.

usage: python make_fig_bev_data.py <release>/results/final
Objects the braking controller considers (in its corridor, within 80 m) that differ between the modes, on nuScenes
frames with non-zero decision value under oracle geometry; z = forward, x = lateral (m), ego frame.
"""
import sys

import pandas as pd

T = {"fpadd": "fp_added_by_full", "fprem": "fp_removed_by_full", "misslost": "miss_lost_by_full",
     "missrec": "miss_recovered_by_full"}
o = pd.read_csv(sys.argv[1] + "/fig_bev_objects.csv.gz")
x = o[(o.geometry == "oracle") & (o.system == "brake") & (o.considered_by_controller == True)]
for sg, nm in ((-1, "harmed"), (1, "helped")):
    for k, t in T.items():
        y = x[(x.sign_V == sg) & (x.type == t)]
        with open(f"fig_bev_{nm}_{k}.dat", "w") as f:
            f.write("z x\n")
            for z, lat in zip(y.z_forward_m, y.x_lateral_m):
                f.write(f"{z:.3f} {lat:.3f}\n")
        print(nm, k, len(y))
frames = x.groupby("sign_V")[["scene", "frame"]].apply(lambda g: len(g.drop_duplicates()))
print("frames by sign:", frames.to_dict())
