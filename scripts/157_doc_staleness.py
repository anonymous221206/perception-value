#!/usr/bin/env python
"""Task 28: which docs/*.md quote numbers that results/final/ no longer holds.

For every tracked document: the commit that last touched it, the results/final files it names (all of results/final
if it names none), and every decimal number it quotes. Each number is looked up in those files twice -- as they were
at the document's own commit, and as they are now -- at the precision the document prints it (also as a percentage
with that many decimals). A number is

  current   found in the files now
  stale     found in the files at the document's commit and nowhere in them now: the value it quoted has moved
  other     found in neither (a number derived in the text, a constant, a number from outside results/final)

A document with at least one stale number is listed as stale, with its examples. Counts quoted as integers ("12 of
78") are not matched (small integers are everywhere in the tables); `files_changed` -- the checked files whose content
differs now from the document's commit -- flags those documents for reading. Nothing under results/final is written;
the audit goes to the run directory only.

In this release, `--task23` defaults to 37c8ce5, the commit that regenerated every result in the ego frame. The
audit reported in docs/iclr_submission_sync.md was run on the development history, which is not part of this release,
so its per-document commit dates, and with them its verdicts, are not reproduced exactly from this history.
"""
from __future__ import annotations

import argparse, io, json, re, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import runmeta                                                         # noqa: E402

NUM = re.compile(r"(?<![\w.])[-+−]?\d+\.(\d{2,6})(?![\d.]*\w)")
FILE = re.compile(r"([A-Za-z0-9_]+\.(?:csv|csv\.gz|json))")
TASK23 = "37c8ce5"            # the commit that regenerated every official result in the ego frame (--task23)


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def values_of(blob: bytes, name: str) -> np.ndarray:
    try:
        if name.endswith(".json"):
            out = []

            def walk(x):
                if isinstance(x, dict):
                    [walk(v) for v in x.values()]
                elif isinstance(x, list):
                    [walk(v) for v in x]
                elif isinstance(x, (int, float)) and not isinstance(x, bool):
                    out.append(float(x))
            walk(json.loads(blob))
            return np.array(out, float)
        d = pd.read_csv(io.BytesIO(blob), compression="gzip" if name.endswith(".gz") else None, low_memory=False)
        v = d.select_dtypes("number").to_numpy(float).ravel()
        return v[np.isfinite(v)]
    except Exception:                                  # noqa: BLE001  an unreadable blob holds no numbers
        return np.zeros(0)


def keys(v: np.ndarray, digits: int) -> set:
    v = np.unique(v)
    return {f"{x:.{digits}f}" for x in v} | {f"{x * 100:.{digits}f}" for x in v}


def at(rev: str, files) -> dict:
    out = {}
    for f in files:
        try:
            blob = subprocess.run(["git", "show", f"{rev}:results/final/{f}"], cwd=ROOT, capture_output=True,
                                  check=True).stdout
        except subprocess.CalledProcessError:
            continue
        out[f] = values_of(blob, f)
    return out


def now(files) -> dict:
    return {f: values_of((ROOT / "results" / "final" / f).read_bytes(), f) for f in files
            if (ROOT / "results" / "final" / f).exists()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task23", default=TASK23, help="the commit that regenerated every result in the ego frame")
    args = ap.parse_args()
    run = runmeta.new_run("doc_staleness", vars(args))
    tracked = set(git("ls-files", "docs").split())
    all_final = [p.name for p in sorted((ROOT / "results" / "final").iterdir())
                 if p.suffix in (".csv", ".json") or p.name.endswith(".csv.gz")]
    rows, untracked = [], sorted(str(p.relative_to(ROOT)) for p in (ROOT / "docs").glob("*.md")
                                 if str(p.relative_to(ROOT)) not in tracked)
    t23 = git("log", "-1", "--format=%cI", args.task23).strip()
    for doc in sorted(p for p in tracked if p.endswith(".md")):
        text = (ROOT / doc).read_text()
        rev = git("log", "-1", "--format=%h", "--", doc).strip()
        when = git("log", "-1", "--format=%cI", "--", doc).strip()
        try:
            then = git("ls-tree", "--name-only", f"{rev}:results/final").split()
        except subprocess.CalledProcessError:            # the document predates results/final
            then = []
        named = sorted({m for m in FILE.findall(text)} & set(all_final + then))
        files = named or all_final
        old, new = at(rev, files), now(files)
        changed = [f for f in files if f in old and subprocess.run(
            ["git", "diff", "--quiet", rev, "--", f"results/final/{f}"], cwd=ROOT).returncode != 0]
        nums = [(m.group(0).replace("−", "-").lstrip("+"), len(m.group(1))) for m in NUM.finditer(text)]
        cache_old, cache_new, cur, stale, other = {}, {}, 0, [], 0
        for tok, dig in nums:
            if dig not in cache_old:
                cache_old[dig] = set().union(*[keys(v, dig) for v in old.values()]) if old else set()
                cache_new[dig] = set().union(*[keys(v, dig) for v in new.values()]) if new else set()
            if tok in cache_new[dig]:
                cur += 1
            elif tok in cache_old[dig]:
                stale.append(tok)
            else:
                other += 1
        rows.append({"doc": doc, "last_commit": rev, "last_commit_date": when[:10],
                     "before_task23_results": when < t23, "files_named": len(named),
                     "files_checked": ";".join(named) if named else "all of results/final",
                     "files_changed": len(changed), "files_changed_named": ";".join(changed) if named else "",
                     "numbers": len(nums), "current": cur, "stale": len(stale), "other": other,
                     "stale_examples": " ".join(list(dict.fromkeys(stale))[:8])})
        print(f"  {doc:52s} {when[:10]}  numbers {len(nums):4d}  current {cur:4d}  stale {len(stale):4d}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(run / "doc_staleness.csv", index=False)
    (run / "untracked.json").write_text(json.dumps(untracked, indent=1))
    print(f"  {int((df.stale > 0).sum())} of {len(df)} tracked documents quote at least one stale number; "
          f"untracked (not audited): {untracked}")
    print("wrote", run)


if __name__ == "__main__":
    main()
