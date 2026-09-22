"""The frame the monocular lift reports its geometry in — one switch, one default (Task 23).

`rap.mono.predicted_geometry` measures range along the camera optical axis and lateral extent about the camera axis;
the braking controller, Planner B and the 3D submissions read those as ego-frame quantities. Task 23 moved the shipped
convention to the ego frame (the camera -> ego rigid transform applied inside the lift, see `rap.mono`). The camera
frame stays reproducible behind this switch.

  camera    the lift as first shipped: camera-frame geometry read as ego-frame
  ego       the camera -> ego transform applied inside the lift            (the default)
  identity  gate G1 only: the ego-frame code path with the identity transform, which must reproduce `camera`
  task22    gate G2 only: the ego-frame code path with no rotation and Task 22 Part C's translation

Every stage whose output depends on the lift reads its frame from here, writes its runs under its usual tag with a
suffix for any frame but `camera`, and finds its inputs through `rap.runs`, so each frame's runs stay separate.
"""
from __future__ import annotations

import os
import sys

FRAMES = ("camera", "ego", "identity", "task22")
DEFAULT = "ego"


def _from_argv() -> str | None:
    """`--frame` on the command line, read at import: stages that resolve their input runs at module level do so
    before arguments are parsed, and must already see the frame they will run in."""
    for i, a in enumerate(sys.argv):
        if a == "--frame" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--frame="):
            return a.split("=", 1)[1]
    return None


_current = _from_argv() or os.environ.get("RAP_FRAME", DEFAULT)
if _current not in FRAMES:
    raise ValueError(f"RAP_FRAME={_current!r}; expected one of {FRAMES}")

# caches written by the planner chain from lifted geometry: each frame gets its own sibling directory
FRAME_CACHES = ("nusc_submissions", "nusc_submissions_calib", "planner_d")

# A named variant of this frame's runs, one further tag suffix (Task 23: C25 re-expressed on the ego-frame tables runs
# its "before" stages as `<tag>_ego_prefix`).  Runs only: caches are not affected, and `rap.runs` falls back to the
# frame's own run for any tag the variant has not written.
VARIANT = os.environ.get("RAP_RUN_VARIANT", "")


def current() -> str:
    return _current


def set_frame(frame: str) -> None:
    global _current
    if frame not in FRAMES:
        raise ValueError(f"unknown frame {frame!r}; expected one of {FRAMES}")
    _current = frame


def suffix(frame: str | None = None) -> str:
    frame = frame or _current
    return "" if frame == "camera" else f"_{frame}"


def tag(base: str, frame: str | None = None, variant: bool = True) -> str:
    """The run tag of a lift-dependent stage in this frame: `core_matrix`, `core_matrix_ego`, ..., plus the variant."""
    return base + suffix(frame) + (f"_{VARIANT}" if VARIANT and variant else "")


def cache_name(name: str, frame: str | None = None) -> str:
    """A planner-chain cache directory in this frame: `planner_d`, `planner_d_ego`, ..."""
    return name + suffix(frame) if name in FRAME_CACHES else name


def add_argument(ap) -> None:
    ap.add_argument("--frame", choices=list(FRAMES), default=None,
                    help=f"frame of the monocular lift (default {DEFAULT}, or $RAP_FRAME); "
                         "identity and task22 exist for gates G1 and G2")


def configure(args) -> str:
    """Apply `--frame` for this process, export it to child processes, and say which frame is in use.

    Some stages resolve their input runs at import time, before arguments are parsed, so `--frame` is read from the
    command line when this module is imported.  Should the frame still differ here (a caller that set it some other
    way), the process re-executes itself with `RAP_FRAME` set, so that `--frame` and `RAP_FRAME` mean the same thing.
    """
    want = getattr(args, "frame", None)
    if want and want != _current:
        if want not in FRAMES:
            raise ValueError(f"unknown frame {want!r}; expected one of {FRAMES}")
        os.environ["RAP_FRAME"] = want
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(sys.executable, [sys.executable] + sys.argv)
    os.environ["RAP_FRAME"] = _current
    print(f"  lift frame: {_current}" + (f", run variant: {VARIANT}" if VARIANT else ""), flush=True)
    return _current
