"""Which run directory holds the shipped tables a stage should read.

Run directories are named `<date>_<time>_<tag>`, and several stages used to find theirs with a glob such as
`results/raw/*_core_matrix`. That broke silently when Task 22 Part A wrote **corrected** copies of the nuScenes
decision tables to runs tagged `core_matrix_classfix` and `core_matrix_postreview_classfix`: the glob does not
match the longer tag, so those stages kept reading the pre-fix tables. Worse, the anonymous release ships the
corrected tables in place, inside the original run directories, so the same glob resolved differently in the two
trees.

Every stage now asks here instead. The rule is one line: take the newest corrected run if one exists, otherwise
the newest original run. Both trees then read the corrected tables.
"""
from __future__ import annotations

from pathlib import Path

from .paths import RAW

CORE_TAGS = {"plain": "core_matrix", "postreview": "core_matrix_postreview"}


def runs(tag: str) -> list[Path]:
    """Every run with exactly this tag, oldest first. The tag must match in full, not as a prefix."""
    return sorted(p for p in Path(RAW).glob(f"*_{tag}")
                  if p.is_dir() and p.name.split("_", 2)[-1] == tag)


def latest(tag: str) -> Path:
    found = runs(tag)
    if not found:
        raise SystemExit(f"no results/raw/*_{tag} run found")
    return found[-1]


def core_matrix(kind: str = "postreview") -> Path:
    """The run holding the per-frame decision tables: the corrected copy when there is one."""
    tag = CORE_TAGS[kind]
    return (runs(f"{tag}_classfix") or runs(tag) or [latest(tag)])[-1]
