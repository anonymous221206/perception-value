# Paper figures

Two scripts that draw figures of the paper from this release's artifacts. Run from the repository root.

* `render_gallery.py` renders the qualitative gallery, cheap and full crops of eight frames (16 JPEGs, 1120 x 630),
  from the gallery export: `mkdir -p gallery && python scripts/paper_figures/render_gallery.py results/final/fig_gallery gallery`
* `make_fig_transform.py` writes `fig_transform.tex` (TikZ; compile with pdflatex), the target-transform heatmap at
  20 %, from Task 30's table: `python scripts/paper_figures/make_fig_transform.py results/final/target_transform_heatmap.csv`
