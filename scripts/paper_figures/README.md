# Paper figures

Six scripts that draw figures of the paper from this release's artifacts. They read `results/final/` and write into
the current directory, or into the directory given. Python with pandas, NumPy and Pillow.

* `render_gallery.py` renders the qualitative gallery: cheap and full crops of eight frames (16 JPEGs, 1120 x 630),
  from the gallery export.
  `mkdir -p gallery && python scripts/paper_figures/render_gallery.py results/final/fig_gallery gallery`
* `render_fig1.py` renders Figure 1: the two frames exported by name as `results/final/fig_gallery/figure1_*`, with
  the braking controller's corridor projected with each sample's camera calibration (`fig1_calibration.json`, from
  the nuScenes calibrated-sensor table). It reads the calibration from its own directory, so run it there:
  `mkdir -p fig1 && (cd scripts/paper_figures && python render_fig1.py ../../results/final/fig_gallery ../../fig1)`
* `make_fig_gap.py` writes `fig_gap.tex` (TikZ): where decision value is lost, from the Task 29 outputs.
  `python scripts/paper_figures/make_fig_gap.py results/final`
* `make_fig_bev_data.py` writes the `fig_bev_*.dat` files for the bird's-eye-view figure from
  `results/final/fig_bev_objects.csv.gz`.
  `python scripts/paper_figures/make_fig_bev_data.py results/final`
* `make_fig_transform.py` writes `fig_transform.tex` (TikZ), the target-transform heatmap at 20 %, from Task 30's
  table.
  `python scripts/paper_figures/make_fig_transform.py results/final/target_transform_heatmap.csv`

Compile the `.tex` outputs with pdflatex.
