"""Which run directory holds the shipped tables a stage should read.

Run directories are named `<date>_<time>_<tag>`.  Several stages used to find theirs with a glob such as
`results/raw/*_core_matrix`, which broke silently when Task 22 Part A wrote corrected nuScenes tables under the longer
tags `core_matrix_classfix` and `core_matrix_postreview_classfix`, and which the anonymous release resolved
differently again, because it ships those corrections in place.  Every stage now asks here.

The answer depends on the frame of the monocular lift (`rap.frames`, Task 23).  A stage whose output depends on the
lift writes its run under `rap.frames.tag(<tag>)` — the usual tag in the camera frame, `<tag>_ego` in the ego frame,
and `_identity` / `_task22` for the gate runs — and its readers resolve the same way, so the frames' runs never mix.
In the camera frame the rule is the one Task 22 set: the newest corrected run if there is one, else the newest
original.
"""
from __future__ import annotations

from pathlib import Path

from . import frames
from .paths import RAW

CORE_TAGS = {"plain": "core_matrix", "postreview": "core_matrix_postreview"}
# the camera-frame Planner B tables the benchmark reads (the `static_obstacles` preset); the ego frame's are tagged
PLANB_CAMERA = "20260912_111225_planner_b_static_fixed"

# every run tag whose content depends on the lift, directly or through a table or score built from it
LIFT_TAGS = {
    "core_matrix", "core_matrix_postreview", "planner_b", "calibration_outcomes", "calibration_plan",
    "calibration_plan_boxes", "submissions", "phase0g_eta_fde_oracle", "phase0g_eta_fde_mono", "plannerC_vs_truth",
    "planner_c", "pkl", "tip", "routers_r1", "router_r2", "benchmark_table", "benchmark_budget",
    "benchmark_budget_1thread", "benchmark_budget_routers", "budget_gate_scores", "streaming_v1_scores",
    "streaming_controllers", "calibration_cells", "target_swap", "target_swap_audit", "causal_threshold",
    "statistics_hardening", "consumer_transfer", "objective_swap", "skip_accounting", "realism_outcomes",
    "realism_controls", "lift_offset_outcomes", "lift_offset_sensitivity", "deployable_gate", "mechanism_table",
    "figure_exports", "planner_d_data", "energy_module_budgets", "budget_feasibility",
    "phase0f_eta", "planner_b_static_fixed", "calibration_direct", "calibration_thresholds", "multifidelity_levels",
    "loss_sensitivity_A", "loss_sensitivity_B", "loss_sensitivity_C",
}


def runs(tag: str) -> list[Path]:
    """Every run with exactly this tag, oldest first. The tag must match in full, not as a prefix."""
    return sorted(p for p in Path(RAW).glob(f"*_{tag}")
                  if p.is_dir() and p.name.split("_", 2)[-1] == tag)


def _exact(tag: str) -> Path:
    found = runs(tag)
    if not found:
        raise SystemExit(f"no results/raw/*_{tag} run found")
    return found[-1]


def resolve_tag(tag: str, frame: str | None = None) -> str:
    """The tag a lift-dependent stage's run carries in this frame."""
    return frames.tag(tag, frame) if tag in LIFT_TAGS else tag


def _with_variant(tag: str, frame: str | None) -> Path:
    """The newest run of a lift-dependent tag in this frame; under a run variant, the variant's run if it wrote one."""
    if frames.VARIANT:
        found = runs(frames.tag(tag, frame))
        if found:
            return found[-1]
    return _exact(frames.tag(tag, frame, variant=False))


def frame_runs(tag: str, frame: str | None = None) -> list[Path]:
    """Every run of `tag` in this frame, oldest first; under a run variant, the variant's runs if it wrote any, as
    `latest` resolves.  (`runs(resolve_tag(tag))` has no fallback, so under a variant it silently finds nothing.)"""
    if tag not in LIFT_TAGS:
        return runs(tag)
    if frames.VARIANT:
        found = runs(frames.tag(tag, frame))
        if found:
            return found
    return runs(frames.tag(tag, frame, variant=False))


def latest(tag: str, frame: str | None = None) -> Path:
    """The newest run of `tag` in the current (or given) frame."""
    return _with_variant(tag, frame) if tag in LIFT_TAGS else _exact(tag)


def core_matrix(kind: str = "postreview", frame: str | None = None) -> Path:
    """The run holding the per-frame decision tables in this frame.

    Camera frame: the corrected copy when there is one (Task 22).  Other frames: the one run 52 wrote in that frame,
    which is both the plain and the post-review table set.
    """
    frame = frame or frames.current()
    tag = CORE_TAGS[kind]
    if frame != "camera":
        return _with_variant("core_matrix", frame)
    return (runs(f"{tag}_classfix") or runs(tag) or [_exact(tag)])[-1]


def planner_b(frame: str | None = None) -> Path:
    """The run holding the Planner B per-frame tables in this frame."""
    frame = frame or frames.current()
    return Path(RAW) / PLANB_CAMERA if frame == "camera" else _with_variant("planner_b", frame)
