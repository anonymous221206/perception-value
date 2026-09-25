"""Write fig_gap.tex: where decision value is lost (three panels), from the release's Task 29 outputs.

(a) benefit captured (right) and harm selected (left) by each allocator at a 20% selection budget, mean over the ten
    core cells, as shares of the all-cheap loss, with the oracle on top;
(b) nDG at a 20% budget when the allocator is free (selection budget) and when its measured latency is charged
    (measured budget), mean over the ten core cells; "inf." marks an allocator that cannot run within the budget;
(c) decision value lost by choosing the allocator on validation units by the missed-object perception gain instead of
    by decision value, target-free pool, 20% budget, per cell with its paired 95% interval; the marker shows which
    signal the decision objective chose (the perception objective chose uncertainty in every cell but PDM-Closed).
Usage: python make_fig_gap.py <release>/results/final
"""
import sys

import pandas as pd

R = sys.argv[1]
b = pd.read_csv(f"{R}/cached_analyses_benefit_harm.csv")
sel = pd.concat([pd.read_csv(f"{R}/benchmark_table.csv"), pd.read_csv(f"{R}/benchmark_table_routers.csv")])
meas = pd.read_csv(f"{R}/benchmark_budget_routers.csv")
s = pd.read_csv(f"{R}/cached_analyses_selection.csv")

# (a)
y = b[(b.track != "nuPlan") & (b.quota == 0.2)]
assert y.groupby("signal").size().eq(10).all()
m = y.groupby("signal")[["benefit_share", "harm_share", "gain_share", "oracle_gain_share"]].mean()
NAME = {"trivial_ego_speed": "ego speed", "gate_gbm": "GBM gate", "gate_ridge": "ridge gate",
        "R1_mlp_reg": "router MLP, $V$", "R1_mlp_clf": "router MLP, $V{>}0$", "R1_gbm_reg": "router GBM, $V$",
        "R1_gbm_clf": "router GBM, $V{>}0$", "random": "random", "criticality_cheap": "cheap criticality",
        "uncertainty": "uncertainty", "R2_cnn_clf": "pixel router"}
order = m.sort_values("gain_share", ascending=False).index.tolist()
rows_a = [("oracle", float(m.oracle_gain_share.iloc[0]), 0.0, float(m.oracle_gain_share.iloc[0]))]
rows_a += [(NAME[k], m.loc[k, "benefit_share"], m.loc[k, "harm_share"], m.loc[k, "gain_share"]) for k in order]

# (b)
sel = sel[(sel.track != "nuPlan") & (sel.quota == 0.2) & ((sel.split == "test") | sel.split.isna())]
meas = meas[(meas.track != "nuPlan") & (meas.unit == "ms") & (meas.budget_level == 0.2)]
spec = [("random", "random", "random"), ("uncertainty", "uncertainty", "uncertainty"),
        ("cheap criticality", "criticality_cheap", "criticality_cheap"), ("ridge gate", "gate_ridge", "gate_ridge"),
        ("GBM gate", "gate_gbm", "gate_gbm_batched"), ("router MLP, $V$", "R1_mlp_reg", "R1_mlp_reg"),
        ("router MLP, $V{>}0$", "R1_mlp_clf", "R1_mlp_clf"), ("router GBM, $V$", "R1_gbm_reg", "R1_gbm_reg"),
        ("router GBM, $V{>}0$", "R1_gbm_clf", "R1_gbm_clf"), ("pixel router", "R2_cnn_clf", "R2_cnn_clf")]
bars = []
for lab, s_sig, m_sig in spec:
    a_ = sel[sel.signal == s_sig]
    m_ = meas[meas.signal == m_sig]
    assert len(a_) == 10 and len(m_) == 10, (s_sig, len(a_), len(m_))
    free = float(a_.eta.mean())
    feas = bool(m_.feasible.all())
    assert feas or not m_.feasible.any(), (m_sig, "mixed feasibility")
    charged = float(m_.eta.mean()) if feas else None
    bars.append((lab, free, charged))
rand = [b_ for b_ in bars if b_[0] == "random"][0][1]
bars = [b_ for b_ in bars if b_[0] != "random"]
# (c)
v = s[(s.section == "validation_selection") & (s.pool == "P1") & (s.quota == 0.2) & (s.included == True)
      & (s.objective == "E_perc_dE")]
pooled = s[(s.section == "validation_selection_pooled") & (s.pool == "P1") & (s.quota == 0.2)
           & (s.objective == "E_perc_dE")].iloc[0]
assert len(v) == 11
assert (v[v.track != "nuPlan"].selected_perc == "uncertainty").all()
SYS = {"brake": "brake", "traj": "traj.", "plan_ade": "ADE", "plan_fde": "FDE", "pdm_closed": "PDM-Closed"}
MARK = {"trivial_ego_speed": ("diamond*", "cDE"), "criticality_cheap": ("triangle*", "cTIP"),
        "random": ("square*", "cRand"), "uncertainty": ("o", "cOracle")}
cells = []
for _, r in v.iterrows():
    name = f"{'nuSc.' if r.track == 'nuScenes' else r.track} {r.geometry} {SYS[r.system]}" if r.track != "nuPlan" \
        else "PDM-Closed scalar"
    same = r.selected_dec == r.selected_perc
    cells.append((name, -r.diff_share, -r.diff_share_hi, -r.diff_share_lo, r.selected_dec, same))
rank = {"KITTI": 0, "nuSc.": 1, "PDM-Closed": 2}
cells.sort(key=lambda c: (rank[c[0].split()[0]], c[0]))

o = [r"\documentclass[10pt,border=1pt]{standalone}", r"\usepackage{times}", r"\usepackage{amsmath}",
     r"\usepackage{xcolor}", r"\usepackage{tikz}", r"\usepackage{pgfplots}", r"\pgfplotsset{compat=1.18}",
     r"\definecolor{cOracle}{HTML}{222222}", r"\definecolor{cRand}{HTML}{8A8A8A}", r"\definecolor{cPKL}{HTML}{1F6FB4}",
     r"\definecolor{cRisk}{HTML}{C0392B}", r"\definecolor{cDE}{HTML}{2E8B57}", r"\definecolor{cTIP}{HTML}{E8762C}",
     r"\begin{document}", r"\begin{tikzpicture}[font=\scriptsize]",
     r"\pgfplotsset{every axis/.append style={axis line style={gray!60}, tick style={gray!60},"
     r" tick label style={font=\scriptsize}, label style={font=\scriptsize}, title style={font=\scriptsize,"
     r" yshift=-1mm}, scale only axis, height=4.1cm, y dir=reverse, grid=major, grid style={gray!12}}}"]

# panel (a): diverging bars
n = len(rows_a)
yl = ",".join("{" + (r"\textbf{oracle}" if r[0] == "oracle" else r[0]) + "}" for r in rows_a)
o += [rf"\begin{{axis}}[name=pa, width=2.45cm, xmin=-0.03, xmax=0.20, ymin=-0.6, ymax={n - 0.4},"
      r" scaled ticks=false, xtick={0,0.1,0.2}, xticklabels={0,0.1,0.2},"
      rf" ytick={{{','.join(str(i) for i in range(n))}}}, yticklabels={{{yl}}},"
      r" xlabel={$\leftarrow$ incurred harm $\mid$ captured benefit $\rightarrow$}, title={(a) benefit and harm, 20\% budget}]"]
o.append(rf"\draw[gray!70] (axis cs:0,-0.6) -- (axis cs:0,{n - 0.4});")
for i, (lab, ben, harm, gain) in enumerate(rows_a):
    col = "cOracle!70" if lab == "oracle" else ("cRand!80" if lab == "random" else "cDE!75")
    o.append(rf"\fill[{col}] (axis cs:0,{i - 0.32}) rectangle (axis cs:{ben:.5f},{i + 0.32});")
    if harm > 0:
        o.append(rf"\fill[cRisk!85] (axis cs:{-harm:.5f},{i - 0.32}) rectangle (axis cs:0,{i + 0.32});")
    o.append(rf"\draw[black, line width=0.9pt] (axis cs:{gain:.5f},{i - 0.4}) -- (axis cs:{gain:.5f},{i + 0.4});")
o.append(r"\end{axis}")

# panel (b): nDG free vs charged
n = len(bars)
yl = ",".join("{" + lab + "}" for lab, _, _ in bars)
o += [rf"\begin{{axis}}[name=pb, at={{(pa.east)}}, anchor=west, xshift=21mm, width=2.2cm,"
      rf" xmin=-0.02, xmax=0.25, scaled ticks=false, xtick={{0,0.1,0.2}}, xticklabels={{0,0.1,0.2}},"
      rf" ytick={{{','.join(str(i) for i in range(n))}}}, yticklabels={{{yl}}}, ymin=-0.6, ymax={n - 0.4},"
      r" xlabel={mean nDG, 20\% budget}, title={(b) charging the allocator's cost}, clip=false]"]
o.append(rf"\draw[cRand, densely dotted, line width=0.8pt] (axis cs:{rand:.4f},-0.6) -- (axis cs:{rand:.4f},{n - 0.4});")
for i, (lab, free, charged) in enumerate(bars):
    if lab == "random":
        continue
    if charged is not None:
        o.append(rf"\draw[cPKL!80, line width=0.9pt] (axis cs:{free:.4f},{i}) -- (axis cs:{charged:.4f},{i});")
        o.append(rf"\fill[cPKL] (axis cs:{charged:.4f},{i}) circle (1.7pt);")
    else:
        if free > 0.04:  # label to the left, inside the axis, so it cannot run into panel (c)
            o.append(rf"\node[cRisk, font=\scriptsize, anchor=east, inner sep=1pt] at (axis cs:{free - 0.012:.4f},{i}) {{inf.}};")
        else:
            o.append(rf"\node[cRisk, font=\scriptsize, anchor=west, inner sep=1pt] at (axis cs:{max(free, 0) + 0.012:.4f},{i}) {{inf.}};")
    o.append(rf"\draw[cPKL, line width=0.8pt, fill=white] (axis cs:{free:.4f},{i}) circle (1.7pt);")
o.append(r"\end{axis}")
o.append(r"\node[anchor=north, font=\scriptsize, align=center] at ($(pb.south)+(-4mm,-8.5mm)$) {"
         r"\tikz{\draw[cPKL, line width=0.8pt, fill=white] (0,0) circle (1.7pt);}~allocator free\;"
         r"\tikz{\fill[cPKL] (0,0) circle (1.7pt);}~cost charged\\"
         r"\tikz{\draw[cRand, densely dotted, line width=0.8pt] (0,0) -- (0.4,0);}~random\;"
         r"{\color{cRisk}inf.}~cannot run};")

# panel (c): value lost by choosing on perception gain
n = len(cells)
yl = ",".join("{" + c[0] + "}" for c in cells) + ",{\\textbf{pooled}}"
o += [rf"\begin{{axis}}[name=pc, at={{(pb.east)}}, anchor=west, xshift=22mm, width=2.05cm,"
      r" xmin=-0.28, xmax=0.62, scaled ticks=false, xtick={-0.2,0,0.2,0.4,0.6},"
      r" xticklabels={$-0.2$,0,0.2,0.4,0.6},"
      rf" ytick={{{','.join(str(i) for i in range(n + 1))}}}, yticklabels={{{yl}}}, ymin=-0.6, ymax={n + 0.6},"
      r" xlabel={decision value lost}, title={(c) choosing by perception gain}]"]
o.append(rf"\draw[gray!70] (axis cs:0,-0.6) -- (axis cs:0,{n + 0.6});")
for i, (_, loss, lo, hi, dec, same) in enumerate(cells):
    ci = "cRisk" if lo > 0 else "gray!70"
    o.append(rf"\draw[{ci}, line width=0.8pt] (axis cs:{lo:.4f},{i}) -- (axis cs:{hi:.4f},{i});")
    mk, col = MARK["uncertainty" if same else dec]
    fill = ", fill=white" if same else ""
    o.append(rf"\addplot[only marks, mark={mk}, mark size=1.7pt, {col}{fill}] coordinates {{({loss:.4f},{i})}};")
o.append(rf"\draw[cRisk, line width=1.1pt] (axis cs:{-pooled.diff_share_hi:.4f},{n}) -- "
         rf"(axis cs:{-pooled.diff_share_lo:.4f},{n});")
o.append(rf"\addplot[only marks, mark=*, mark size=2.0pt, cRisk] coordinates {{({-pooled.diff_share:.4f},{n})}};")
o.append(r"\end{axis}")
# legend for (c): what the decision objective chose
o.append(r"\node[anchor=north, font=\scriptsize, align=center] at ($(pc.south)+(-9mm,-8.5mm)$) {"
         r"decision objective's choice:\\"
         r"\tikz{\node[diamond, fill=cDE, inner sep=1.3pt]{};}~ego speed\;"
         r"\tikz{\node[regular polygon, regular polygon sides=3, fill=cTIP, inner sep=0.9pt]{};}~criticality\;"
         r"\tikz{\node[rectangle, fill=cRand, inner sep=1.5pt]{};}~random\;"
         r"\tikz{\draw[cOracle] (0,0) circle (1.6pt);}~same};")
o += [r"\end{tikzpicture}", r"\end{document}"]
tex = "\n".join(o) + "\n"
tex = tex.replace(r"\usepackage{pgfplots}", "\\usepackage{pgfplots}\n\\usetikzlibrary{calc,shapes.geometric}")
open("fig_gap.tex", "w").write(tex)
print("wrote fig_gap.tex; oracle", round(rows_a[0][1], 4), "pooled loss", round(-pooled.diff_share, 4))
