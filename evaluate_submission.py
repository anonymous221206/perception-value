#!/usr/bin/env python3
"""Score an allocator's submission on the benchmark's frozen test split.

    python evaluate_submission.py SUBMISSION.csv [--out SCORED.csv] [--cost_profile PROFILE.json]
                                  [--plan benchmark_table] [--cells KITTI|mono|brake|J ...] [--nboot 1000]

The schema, the rules and how to cite a result are in docs/SUBMITTING.md. The submission is validated first (exactly
the test inputs of every cell it enters, finite scores, the identifiers of its track); any error names the offending
rows and nothing is scored. Scoring is the benchmark's own (`rap.scoring`, moved verbatim from
scripts/92_benchmark_table.py): nDG at the 10, 20, 30 and 50% quotas in exact expectation over random tie-breaks, and
the paired unit bootstrap against random, on the resamples of the table named by --plan. With --cost_profile the
measured latency and energy budgets are scored too, with the allocator's own cost charged on every input and the
benchmark's feasibility rule applied.
"""
from __future__ import annotations

import argparse, hashlib, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from rap import submission as S                                                 # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission", help="the submission CSV (docs/SUBMITTING.md)")
    ap.add_argument("--out", default=None, help="where to write the scored rows (default: <submission>_scored.csv)")
    ap.add_argument("--cost_profile", default=None,
                    help="JSON with the allocator's per-input cost: {\"ms\": ..., \"mJ\": ...}, each a number or "
                         "{track: number}; adds the measured-budget rows")
    ap.add_argument("--plan", default="benchmark_table",
                    choices=["benchmark_table", "routers_r1", "router_r2:nuScenes", "router_r2:KITTI"],
                    help="whose bootstrap resamples to share (default: the benchmark table's)")
    ap.add_argument("--cells", nargs="*", default=None, help="score only these cells (track|geometry|system|target)")
    ap.add_argument("--nboot", type=int, default=S.NBOOT)
    args = ap.parse_args(argv)

    path = Path(args.submission)
    sub = S.load(path)
    profile = S.load_profile(args.cost_profile) if args.cost_profile else None
    try:
        form, scored = S.score(sub, plan=args.plan, nboot=args.nboot, profile=profile, cells_only=args.cells)
    except S.SubmissionError as e:
        print(f"invalid submission {path}:\n{e}", file=sys.stderr)
        return 2
    scored.insert(0, "submission_sha256", hashlib.sha256(path.read_bytes()).hexdigest()[:16])
    out = Path(args.out) if args.out else path.with_name(path.stem + "_scored.csv")
    scored.to_csv(out, index=False)
    sel = scored[scored.track_kind == "selection budget"]
    print(f"{path.name}: form {form}, {sel[list(S.CELL)].drop_duplicates().shape[0]} cells, plan {args.plan}")
    for key, g in sel.groupby(list(S.CELL), sort=False):
        cells = " ".join(f"{int(round(r.quota * 100))}%: {r.ndg:+.3f} [{r.minus_random_lo:+.3f}, {r.minus_random_hi:+.3f}]"
                         f"{'*' if r.interval_excludes_zero else ''}" for r in g.itertuples())
        print(f"  {'|'.join(key):38s} {cells}")
    print("  nDG, and in brackets the paired 95% interval of nDG minus random; * marks an interval excluding zero")
    if profile is not None:
        mb = scored[scored.track_kind == "measured budget"]
        print(f"  measured budgets: {int(mb.feasible.sum())} of {len(mb)} rows feasible")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
