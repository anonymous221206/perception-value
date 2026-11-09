"""R14 learners (docs/followup_protocol.md), shared by 241_r14_exact.py and 242_r15_logdisjoint.py.

Plain numpy and torch; no file of the repository is read here, so a job can run on any machine holding the arrays.

  zscore      z on the fit frames, clipped at +-10 (R9 default 2)
  J           the exact budgeted objective: forced top-k, exact ties broken by frame order
  cem         exact_lin: cross-entropy method over unit-norm theta, best-so-far over every evaluated candidate
  ridge       ridge_V: Ridge(alpha) on z, alpha given (the benchmark gate's RidgeCV alpha on the same fit frames)
  mlp         mlp_V (squared error on scaled V) and exact_mlp (Plackett-Luce REINFORCE, Gumbel-top-k samples)
"""
from __future__ import annotations

import numpy as np

EPS_V = 1e-9                                   # replaced by the caller with rap.scoring.EPS
CLIP = 10.0
POP, ELITE, ITERS, SMOOTH, SIG_FLOOR = 256, 0.1, 400, 0.7, 0.01
SEEDS = (0, 1, 2, 3, 4)
EPOCHS, EVERY, BATCH, WD, M_SAMPLES = 200, 10, 2048, 1e-4, 16


def k_round(n, q):
    return max(int(round(q * n)), 1)


def zscore(X_fit, *others):
    mu, sd = X_fit.mean(0), X_fit.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    f = lambda X: np.clip((X - mu) / sd, -CLIP, CLIP)
    return (f(X_fit),) + tuple(f(X) for X in others) + ((mu, sd),)


# ------------------------------------------------------------------------------------------------ the objective

def topk_mask(s, k):
    """Forced top-k of one score vector; exact ties at the cut go to the earliest frames.

    Returns (mask, ties): ties = frames tied at the cut when the tie straddles it (0 if it does not)."""
    s = np.asarray(s, float)
    o = np.argsort(-s, kind="stable")
    m = np.zeros(len(s), bool)
    m[o[:k]] = True
    tie = s == s[o[k - 1]]
    return m, int(tie.sum()) if (tie & ~m).any() else 0


def J(s, v, k):
    m, _ = topk_mask(s, k)
    return float(v[m].sum())


def J_rows(S, v, k):
    """J of every row of S (c x n), exact ties to the earliest frames (the search's evaluation, float32 scores)."""
    cut = -np.partition(-S, k - 1, axis=1)[:, k - 1:k]              # k-th largest per row
    ge = S >= cut
    out = ge @ v
    over = ge.sum(1) > k                                             # a tie straddles the cut: earliest frames first
    for i in np.flatnonzero(over):
        above = S[i] > cut[i]
        tie = np.flatnonzero(S[i] == cut[i])[: k - int(above.sum())]
        out[i] = v[above].sum() + v[tie].sum()
    return out


# ------------------------------------------------------------------------------------------------ exact_lin

def cem(Z, v, k, mu0, sig0, seed, iters=ITERS, pop=POP, trace=None):
    """CEM over theta (unit norm). Returns (theta, J) of the best candidate evaluated, mu included each iteration."""
    rng = np.random.default_rng(seed)
    d = Z.shape[1]
    mu = np.asarray(mu0, float).copy()
    sig = np.full(d, float(sig0))
    ne = max(int(round(ELITE * pop)), 1)
    best_t, best_j = None, -np.inf
    Z32T = np.ascontiguousarray(Z.T.astype(np.float32))
    for it in range(iters):
        T = mu + sig * rng.standard_normal((pop, d))
        nrm = np.linalg.norm(T, axis=1, keepdims=True)
        T = T / np.where(nrm > 0, nrm, 1.0)
        cand = T
        mn = np.linalg.norm(mu)
        if mn > 0:
            cand = np.vstack([T, mu / mn])
        Jc = J_rows(cand.astype(np.float32) @ Z32T, v, k)
        i = int(np.argmax(Jc))
        if Jc[i] > best_j:
            best_j, best_t = float(Jc[i]), cand[i].copy()
        Jp = Jc[:pop]
        el = T[np.argsort(-Jp, kind="stable")[:ne]]
        mu = SMOOTH * el.mean(0) + (1 - SMOOTH) * mu
        nm = np.linalg.norm(mu)
        mu = mu / nm if nm > 0 else mu
        sig = np.maximum(SMOOTH * el.std(0) + (1 - SMOOTH) * sig, SIG_FLOOR)
        if trace is not None:
            trace.append((it, float(Jc[pop]) if mn > 0 else np.nan, float(Jp.max()), float(np.sort(Jp)[-ne:].mean()), best_j,
                          float(sig.mean())))
    # the final theta is re-scored in float64 so that its J is the one the deployed scorer gets
    return best_t, J(Z @ best_t, v, k)


def ridge_theta(Z, v, alpha):
    """Ridge on z with an intercept (the benchmark's form); the intercept does not change a ranking."""
    from sklearn.linear_model import Ridge
    m = Ridge(alpha=alpha).fit(Z, v)
    return m.coef_.astype(float), float(m.intercept_)


def lin_settings(theta_ridge):
    t = theta_ridge / np.linalg.norm(theta_ridge)
    return [{"init": "ridge", "sigma0": 0.3, "mu0": t}, {"init": "ridge", "sigma0": 1.0, "mu0": t},
            {"init": "zero", "sigma0": 0.3, "mu0": np.zeros_like(t)}, {"init": "zero", "sigma0": 1.0, "mu0": np.zeros_like(t)}]


def fit_exact_lin(Z, v, k, mu0, sig0, iters=ITERS, pop=POP, keep_trace=False):
    """Five seeds; the seed with the highest training J (optimization, not selection)."""
    runs = []
    for seed in SEEDS:
        tr = [] if keep_trace else None
        t, j = cem(Z, v, k, mu0, sig0, seed, iters, pop, tr)
        runs.append((j, seed, t, tr))
    j, seed, t, tr = max(runs, key=lambda r: (r[0], -r[1]))
    return {"theta": t, "J": j, "seed": seed, "J_seeds": [r[0] for r in runs], "trace": {r[1]: r[3] for r in runs} if keep_trace else None}


# ------------------------------------------------------------------------------------------------ MLPs

def batches(seqs, rng, size=BATCH):
    """Whole scenes in shuffled order, cut when a batch reaches `size` frames; a short remainder joins the last batch (R9)."""
    names, inv = np.unique(seqs, return_inverse=True)
    idx = [np.flatnonzero(inv == i) for i in range(len(names))]
    out, cur, n = [], [], 0
    for i in rng.permutation(len(names)):
        cur.append(idx[i])
        n += len(idx[i])
        if n >= size:
            out.append(np.concatenate(cur))
            cur, n = [], 0
    if cur:
        if out:
            out[-1] = np.concatenate([out[-1]] + cur)
        else:
            out.append(np.concatenate(cur))
    return out


def net_of(d, torch):
    return torch.nn.Sequential(torch.nn.Linear(d, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))


def pl_logprob(s, order, torch):
    """Ordered Plackett-Luce log-probability of the first k items `order` (M x k) under logits s (n)."""
    M, k = order.shape
    n = s.shape[0]
    sel = s[order]                                                   # M x k
    mask = torch.ones(M, n, dtype=torch.bool)
    mask.scatter_(1, order, False)
    rest = torch.where(mask, s.expand(M, n), torch.full_like(s.expand(M, n), -float("inf")))
    lse_rest = torch.logsumexp(rest, dim=1, keepdim=True)            # M x 1 (n > k)
    tail = torch.flip(torch.logcumsumexp(torch.flip(sel, [1]), dim=1), [1])   # log sum over sel[j:]
    denom = torch.logaddexp(tail, lse_rest.expand(M, k))
    return (sel - denom).sum(1)


def train_mlp(X, v, seqs, evals, kind, lr, seeds=SEEDS, epochs=EPOCHS, alpha=None, init=None, scale=None, every=EVERY,
              keep_weights=False, m_samples=M_SAMPLES):
    """mlp_V (kind 'V') or exact_mlp (kind 'exact', budget alpha) on z-scored X.

    evals: {name: X_eval (already z-scored)}; X is z-scored. Returns {epoch: {name: seed-averaged scores}}, and, with
    keep_weights, {epoch: [per-seed state lists]}.  init: {epoch-independent list per seed of parameter arrays} or None.
    """
    import torch
    torch.set_num_threads(1)
    tr = torch.as_tensor(X, dtype=torch.float32)
    ev = {k: torch.as_tensor(x, dtype=torch.float32) for k, x in evals.items()}
    vt = torch.as_tensor(v / scale, dtype=torch.float32)
    sums, W = {}, {}
    for si, seed in enumerate(seeds):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        gen = torch.Generator().manual_seed(seed)
        net = net_of(X.shape[1], torch)
        if init is not None:
            with torch.no_grad():
                for p, a in zip(net.parameters(), init[si]):
                    p.copy_(torch.as_tensor(a))
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=WD)
        for epoch in range(1, epochs + 1):
            for b in batches(seqs, rng):
                b = torch.as_tensor(b)
                s = net(tr[b]).squeeze(1)
                if kind == "V":
                    loss = ((s - vt[b]) ** 2).mean()
                else:
                    kb = k_round(len(b), alpha)
                    u = torch.rand(m_samples, len(b), generator=gen).clamp_(1e-12, 1 - 1e-12)
                    g = s.detach().unsqueeze(0) - torch.log(-torch.log(u))
                    order = torch.topk(g, kb, dim=1, sorted=True).indices    # Gumbel-top-k: an exact PL sample
                    R = vt[b][order].sum(1)
                    base = (R.sum() - R) / (m_samples - 1)
                    loss = -((R - base).detach() * pl_logprob(s, order, torch)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            if epoch % every == 0:
                with torch.no_grad():
                    d = sums.setdefault(epoch, {})
                    for k, x in ev.items():
                        d[k] = d.get(k, 0.0) + net(x).squeeze(1).numpy().astype(np.float64)
                if keep_weights:
                    W.setdefault(epoch, []).append([p.detach().numpy().copy() for p in net.parameters()])
    out = {e: {k: x / len(seeds) for k, x in d.items()} for e, d in sums.items()}
    return (out, W) if keep_weights else out
