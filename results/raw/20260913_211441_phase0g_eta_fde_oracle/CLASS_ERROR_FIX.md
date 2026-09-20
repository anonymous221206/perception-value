# The nuScenes files in this directory carry the corrected coarse class labels

Until the class-error fix, the reference class of a nuScenes object was `TYPE_TO_COARSE.get(type, "vehicle")`, a map
written for KITTI's type names. nuScenes geometry stores `type` as the category prefix, so every reference object was
labelled `vehicle` and every correctly detected pedestrian, bicycle or motorcycle counted as a class error. It
reaches one primitive, `cls`, and through it E3 and the three E5 variants, on nuScenes only.

This release ships the **corrected** files, because they are the ones the reported numbers come from. They were
produced by the same unchanged code on the corrected tables; nothing else about this run changed, and no KITTI or
nuPlan file in this directory is affected.

* what moved, old value beside new: `results/final/class_error_fix.csv`
* the report, with the checks: `docs/iclr_class_error_fix.md`
* the code: `rap.geometry.coarse_classes`, `rap.nusc.coarse_class`, `scripts/136_class_error_fix.py`
  (stage N5) and `scripts/137_class_error_figures.py` (stage C25)
