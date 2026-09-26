#!/usr/bin/env python
"""Compare two results/final directories file by file, e.g. a regenerated tree against the shipped one.

For every CSV and JSON file in either directory:
- status: unchanged (byte-identical), changed, new or removed;
- for a changed CSV with the same rows: per column, the number of changed values and the largest absolute change
  (rows written in another order are aligned first);
- every text or boolean value that changed (verdicts, readings, flags such as `beats_*`), with the row's keys;
- for JSON: every leaf that changed.

Writes inventory_files.csv, inventory_columns.csv, reversed_readings.csv and json_changes.csv into --out.
Task 32's inventory (the previous release against this one) is in results/raw/*_shared_history_before/inventory/.
"""
from __future__ import annotations

import argparse, filecmp, json, re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROV = re.compile(r"(_run$|^source_run$|_run_|^run$)")                 # provenance: which run a row came from
FLAG = re.compile(r"(^beats_|survives|collapses|_fires$|^fires|pass|verdict|reading|^interval$|^significant|"
                  r"above_random|below_random|^ok$|_ok$|^win|_win$|^sign$|^sign_|_sign$|reverses|reached|excludes)", re.I)


def leaves(x, pre=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{pre}.{k}" if pre else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{pre}[{i}]")
    else:
        yield pre, x


def read(p):
    kw = dict(keep_default_na=False, na_values=[""], float_precision="round_trip", low_memory=False)
    return pd.read_csv(p, compression="infer", **kw)


def compare_csv(name, a, b, colrows, flips):
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return f"shape {a.shape} -> {b.shape}" + ("" if list(a.columns) == list(b.columns) else ", columns differ")
    num = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])
           and not pd.api.types.is_bool_dtype(a[c])]
    text = [c for c in a.columns if c not in num and not PROV.search(c)]
    # rows written in another order (e.g. appended per dataset): align on the columns whose values are the same
    # multiset in both, when they identify rows uniquely
    same_set = [c for c in a.columns if not FLAG.search(c) and not PROV.search(c)
                and sorted(a[c].astype(str)) == sorted(b[c].astype(str))]
    key = [c for c in same_set if c not in num]
    for c in [c for c in same_set if c in num]:            # numeric columns only as far as needed to be unique
        if key and not a[key].astype(str).duplicated().any():
            break
        key.append(c)
    if key and not all(a[c].astype(str).equals(b[c].astype(str)) for c in key):
        ka, kb = a[key].astype(str), b[key].astype(str)
        if not ka.duplicated().any() and not kb.duplicated().any():
            a = a.assign(_k=ka.agg("|".join, axis=1)).sort_values("_k").drop(columns="_k").reset_index(drop=True)
            b = b.assign(_k=kb.agg("|".join, axis=1)).sort_values("_k").drop(columns="_k").reset_index(drop=True)
    # identifying columns: text columns identical in both; a text column that changed is a reading, not a key
    ids = [c for c in text if a[c].astype(str).equals(b[c].astype(str)) and not FLAG.search(c)]
    if not ids:
        return "no unchanged text column to identify rows"
    n_changed = 0
    for c in num:
        x, y = a[c].to_numpy(float), b[c].to_numpy(float)
        diff = ~((x == y) | (np.isnan(x) & np.isnan(y)))
        if diff.any():
            n_changed += int(diff.sum())
            dd = np.abs(x - y)[diff]
            colrows.append({"file": name, "column": c, "n_changed": int(diff.sum()), "n_rows": len(x),
                            "max_abs_change": float(np.nanmax(dd)) if np.isfinite(dd).any() else np.nan})
    for c in text:
        d = (a[c].astype(str) != b[c].astype(str)).to_numpy()
        if not d.any():
            continue
        n_changed += int(d.sum())
        colrows.append({"file": name, "column": c, "n_changed": int(d.sum()), "n_rows": len(a), "max_abs_change": np.nan})
        for i in np.flatnonzero(d):
            flips.append({"file": name, "column": c, "old": a[c].iloc[i], "new": b[c].iloc[i],
                          "row": "; ".join(f"{k}={a[k].iloc[i]}" for k in ids[:8])})
    return f"{n_changed} values changed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", default=str(ROOT / "results" / "final"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    old, new, out = Path(args.old), Path(args.new), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pat = ("*.csv", "*.csv.gz", "*.json")
    names = sorted({p.relative_to(old).as_posix() for g in pat for p in old.glob(g)}
                   | {p.relative_to(new).as_posix() for g in pat for p in new.glob(g)})
    files, colrows, flips, jrows = [], [], [], []
    for n in names:
        a, b = old / n, new / n
        if not a.exists() or not b.exists():
            files.append({"file": n, "status": "new" if not a.exists() else "removed", "detail": ""})
            continue
        if filecmp.cmp(a, b, shallow=False):
            files.append({"file": n, "status": "unchanged", "detail": ""})
            continue
        if n.endswith(".json"):
            ja, jb = dict(leaves(json.loads(a.read_text()))), dict(leaves(json.loads(b.read_text())))
            k = 0
            for key in sorted(set(ja) | set(jb)):
                va, vb = ja.get(key, "<absent>"), jb.get(key, "<absent>")
                if va != vb and not (isinstance(va, float) and isinstance(vb, float) and np.isnan(va) and np.isnan(vb)):
                    k += 1
                    jrows.append({"file": n, "key": key, "old": va, "new": vb,
                                  "flip": isinstance(va, (bool, str)) or isinstance(vb, (bool, str))})
            files.append({"file": n, "status": "changed", "detail": f"{k} leaves changed"})
            continue
        try:
            detail = compare_csv(n, read(a), read(b), colrows, flips)
        except Exception as e:                                   # noqa: BLE001
            detail = f"not compared: {e!r}"
        files.append({"file": n, "status": "changed", "detail": detail})
    pd.DataFrame(files).to_csv(out / "inventory_files.csv", index=False)
    pd.DataFrame(colrows, columns=["file", "column", "n_changed", "n_rows", "max_abs_change"]).to_csv(
        out / "inventory_columns.csv", index=False)
    pd.DataFrame(flips, columns=["file", "column", "old", "new", "row"]).to_csv(out / "reversed_readings.csv", index=False)
    pd.DataFrame(jrows, columns=["file", "key", "old", "new", "flip"]).to_csv(out / "json_changes.csv", index=False)
    f = pd.DataFrame(files)
    print(f.status.value_counts().to_string())
    print(f"{len(flips)} flipped flag values in CSVs; {sum(r['flip'] for r in jrows)} flipped JSON leaves; {out}")


if __name__ == "__main__":
    main()
