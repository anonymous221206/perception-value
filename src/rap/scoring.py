"""The benchmark's scorer: exact expected gain over random tie-breaks, and the paired unit bootstrap.

Moved verbatim from `scripts/92_benchmark_table.py` (Task 25), so that a submission can be scored by the very code
that scores the benchmark's own signals; 92 imports it back, and every `t92.*` name other stages use still resolves.
"""
from __future__ import annotations

import numpy as np

QUOTAS = (0.10, 0.20, 0.30, 0.50)
EPS = 1e-9


def topk_expect(s, weights, ks):
    """Expected sums of each weight vector over the top-k frames by s, ties broken uniformly.

    Frames strictly above the cut value count fully; the tie group containing the k-th frame
    contributes its total times the share of it that fits.  Also returns the share of each
    selection decided by that tie rather than by the score.
    """
    s = np.round(np.asarray(s, float), 9)
    o = np.argsort(-s, kind="stable")
    ss = s[o]
    new = np.r_[True, ss[1:] != ss[:-1]]
    gid = np.cumsum(new) - 1
    gs = np.flatnonzero(new)
    ge = np.r_[gs[1:], len(ss)]
    ks = np.asarray(ks, int)
    g = gid[np.clip(ks, 1, len(ss)) - 1]
    a, b = gs[g], ge[g]
    frac = (ks - a) / np.maximum(b - a, 1)
    out = []
    for w in weights:
        c = np.r_[0.0, np.cumsum(np.asarray(w, float)[o])]
        out.append(c[a] + (c[b] - c[a]) * frac)
    tie = (ks - a) / np.maximum(ks, 1)                    # the definition in budget.tie_fraction
    return out, tie


def quota_k(n):
    return np.array([max(int(round(q * n)), 1) for q in QUOTAS])


def evaluate(v, units, scores, nboot, rng):
    """Point estimates and paired bootstrap for every signal on one frame set."""
    v = np.asarray(v, float)
    nz = (np.abs(v) > EPS).astype(float)
    n = len(v)
    ks = quota_k(n)
    prize0 = topk_expect(v, [v], ks)[0][0]
    point = {}
    for name, s in scores.items():
        if s is None:                                      # random: every frame tied
            gain = ks / n * v.sum()
            resp, tie = np.full(len(ks), nz.mean()), np.full(len(ks), np.nan)
        else:
            (gain, rsum), tie = topk_expect(s, [v, nz], ks)
            resp = rsum / ks
        point[name] = dict(gain=gain, tie=tie, resp=resp,
                           eta=np.where(prize0 > EPS, gain / np.where(prize0 > EPS, prize0, 1), np.nan))

    uniq = np.unique(units)
    idx = [np.flatnonzero(units == u) for u in uniq]
    draws = {name: np.full((nboot, len(ks)), np.nan) for name in scores}
    dropped = np.zeros(len(ks), int)
    for b in range(nboot):
        t = np.concatenate([idx[i] for i in rng.integers(0, len(uniq), len(uniq))])
        vt = v[t]
        kt = quota_k(len(t))
        pz = topk_expect(vt, [vt], kt)[0][0]
        ok = pz > np.maximum(EPS, 0.25 * prize0)
        dropped += ~ok
        den = np.where(ok, pz, 1.0)
        for name, s in scores.items():
            g = kt / len(t) * vt.sum() if s is None else topk_expect(s[t], [vt], kt)[0][0]
            draws[name][b] = np.where(ok, g / den, np.nan)
    return ks, prize0, point, draws, dropped


def ci(x):
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) >= 50 else (np.nan, np.nan)
