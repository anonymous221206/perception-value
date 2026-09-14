"""Skip the tests that read KITTI calibration files when the KITTI dataset is not present.

Everything else runs without any dataset.
"""
import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap.paths import KITTI_CALIB                                               # noqa: E402

NEEDS_KITTI = ("load_calib", "load_labels", "load_oxts", "sequence_geometry")


def pytest_collection_modifyitems(config, items):
    if KITTI_CALIB.exists():
        return
    skip = pytest.mark.skip(reason=f"KITTI tracking not found at {KITTI_CALIB.parent.parent} (set RAP_DATASETS or RAP_KITTI)")
    for item in items:
        fn = getattr(item, "function", None)
        try:
            src = inspect.getsource(fn) if fn else ""
        except (OSError, TypeError):
            src = ""
        try:
            msrc = inspect.getsource(item.module)
        except (OSError, TypeError):
            msrc = ""
        # decision tables are built from KITTI labels and geometry inside module-level helpers
        if any(k in src for k in NEEDS_KITTI) or "decision.build(" in msrc:
            item.add_marker(skip)
