"""Matched-null statistics: each event is compared with its own ordinary days (same ticker, nearby sessions)."""
import math

import numpy as np


def matched_diffs(E: np.ndarray, N: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per matched set: event value minus the mean of its available null values. Returns (diffs, keep mask)."""
    E, N = np.asarray(E, float), np.asarray(N, float).reshape(len(E), -1)
    keep = ~np.isnan(E) & (~np.isnan(N)).any(axis=1)
    with np.errstate(invalid="ignore"):
        d = E - np.nanmean(np.where(keep[:, None], N, 0.0), axis=1)
    return d[keep], keep


def permutation_test(E: np.ndarray, N: np.ndarray, n_perm: int, rng: np.random.Generator) -> tuple[float, float, int, float]:
    """Two-sided matched permutation test of mean(event - mean(its nulls)).

    Under the null hypothesis the event day is just another day for its ticker, so within each matched set
    the "event" label is exchangeable among the set's available values. Each draw relabels one value per set
    at random. Returns (observed statistic, exact permutation p, number of sets used, normal-approximation p).

    The exact p can never go below 1 / (n_perm + 1); the normal-approximation p (observed / std of the
    permutation draws) can, which matters when thousands of tests are corrected together."""
    E, N = np.asarray(E, float), np.asarray(N, float).reshape(len(E), -1)
    d, keep = matched_diffs(E, N)
    n = int(keep.sum())
    if n == 0:
        return np.nan, np.nan, 0, np.nan
    V = np.column_stack([E[keep], N[keep]])                 # n x (1 + k), NaN where a null is missing
    avail = ~np.isnan(V)
    cnt = avail.sum(axis=1)                                  # >= 2 for every kept set
    total = np.where(avail, V, 0.0).sum(axis=1)
    order = np.argsort(~avail, axis=1, kind="stable")        # available column indices first, per row
    pick = np.floor(rng.random((n_perm, n)) * cnt).astype(int)
    cols = np.take_along_axis(np.broadcast_to(order, (n_perm, *order.shape)), pick[..., None], axis=2)[..., 0]
    chosen = V[np.arange(n), cols]                           # n_perm x n
    perm_stats = (chosen - (total - chosen) / (cnt - 1)).mean(axis=1)
    obs = float(d.mean())
    p = (1 + np.sum(np.abs(perm_stats) >= abs(obs) - 1e-15)) / (n_perm + 1)
    sd = float(perm_stats.std())
    p_z = math.erfc(abs(obs) / sd / math.sqrt(2)) if sd > 0 else (0.0 if obs != 0 else 1.0)
    return obs, float(p), n, p_z


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH q-values (step-up), NaN p-values stay NaN and do not count towards m."""
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return q
    ps = p[ok]
    order = np.argsort(ps)
    ranked = ps[order] * m / np.arange(1, m + 1)
    qs = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    out = np.empty(m)
    out[order] = qs
    q[ok] = out
    return q


def trimmed_mean(x: np.ndarray, frac: float) -> float:
    x = np.sort(np.asarray(x, float))
    k = int(np.floor(len(x) * frac))
    x = x[k:len(x) - k] if len(x) - 2 * k > 0 else x
    return float(x.mean()) if len(x) else np.nan


def leave_one_out_sign(d: np.ndarray, labels: np.ndarray) -> tuple[bool, float]:
    """Does the mean keep its sign when any one label (ticker, or quarter) is dropped? Also the weakest mean."""
    d, labels = np.asarray(d, float), np.asarray(labels)
    sign = np.sign(d.mean())
    means = [d[labels != lab].mean() for lab in np.unique(labels) if (labels != lab).any()]
    if not means:
        return False, np.nan
    weakest = min(means, key=lambda m: m * sign)
    return bool(all(np.sign(m) == sign for m in means)), float(weakest)
