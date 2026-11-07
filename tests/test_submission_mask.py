"""The decision-mask forms of the submission path (round 3, R10)."""
import numpy as np
import pandas as pd
import pytest

from rap import submission as S

pytestmark = pytest.mark.skipif(not S.VALUES.exists(), reason="no benchmark values in this checkout")


def _frame(cell, z):
    t, ids = S._cell_arrays(cell)
    return t, pd.DataFrame({**dict(zip(S.CELL, map(str, cell))), ids[0]: t[ids[0]].astype(str), ids[1]: t[ids[1]].astype(str), "z": z(t)})


def _cell():
    return [c for c in S.cells() if c[0] == "KITTI"][0]


def test_mask_equals_top_k_gain():
    cell = _cell()
    t, sub = _frame(cell, lambda t: np.zeros(len(t), int))
    k = max(int(round(0.2 * len(t))), 1)
    pick = np.argsort(-t.V.to_numpy(float), kind="stable")[:k]
    z = np.zeros(len(t), int)
    z[pick] = 1
    sub["z"] = z.astype(str)
    form, r = S.score(sub, nboot=20, causal=True)
    row = r[r.quota == 0.2].iloc[0]
    assert form == "mask" and row.mask_within_budget and row.mask_escalated == k
    assert abs(row.gain - t.V.to_numpy(float)[pick].sum()) < 1e-9
    assert abs(row.ndg - 1.0) < 1e-9
    over = r[r.quota == 0.1].iloc[0]
    assert not over.mask_within_budget and over.undefined_reason.startswith("over budget") and np.isnan(over.ndg)


def test_mask_rejects_non_binary_and_mixed():
    cell = _cell()
    _, sub = _frame(cell, lambda t: np.full(len(t), 2))
    with pytest.raises(S.SubmissionError):
        S.validate(sub)
    _, sub = _frame(cell, lambda t: np.zeros(len(t), int))
    sub["score"] = "0"
    with pytest.raises(S.SubmissionError):
        S.validate(sub)


def test_cap_violations():
    z = np.array([1, 1, 1, 0])
    assert S.cap_violations(z, np.array(["a"] * 4), np.arange(4), 0.2) == (2, 1)
