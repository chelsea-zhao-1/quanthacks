"""Every statistical test in docs/test_plan.md for the old-news hypothesis, one function per test.

Outcome per event (H1, P): Y_h minus the mean Y_h of its matched ordinary days (src/playground/stats.py,
`matched_diffs`). Y_h = log(RV_h / IV_0) comes from outcome_<label>.csv.

- `h1`        primary: old minus surprise, late people events, earnings excluded, usable; h = 10, 1-month
              bucket. One-sided permutation of the old/surprise labels (prediction: negative).
- `h1b`       old events' Y_h minus the mean Y_h of every ordinary day whose gap move is at least the event's.
              One-sided matched-set permutation (prediction: negative).
- `placebo`   H1 on the placebo filings (prediction: no difference, so its own p-value is two-sided).
- `horizon_profile`  H1, H1b and P at every fixed horizon, bootstrap 95% intervals and BH q-values across
              horizons within each test.
- `sensitivity`      H1 with one setting changed at a time: weight set, cutoff, bucket, strike, category.

Every function returns a tidy table and appends its rows to the test ledger (data/oldnews/ledger.csv).
H2 and the trade record are computed by src/oldnews/trade.py. Results are reported exactly as they come out.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from oldnews import classify as cl
from playground import ledger, stats

SEED = 20261003
N_PERM = 10_000
N_BOOT = 10_000
ALPHA = 0.05
MIN_N = 30                  # fewer qualifying events than this: the result is labelled descriptive
TRIM = 0.05                 # trimmed-mean fraction for the descriptive robustness column
CHUNK = 1_000               # permutations or resamples held in memory at once
HORIZONS = ("1", "2", "3", "5", "10", "21", "42", "63", "expiry")
BUCKETS = ("1m", "2m", "3-6m")
OTMS = (3, 5, 10)
DEPARTURE_TAGS = frozenset({"ceo_departure", "cfo_departure", "executive_officer_departure", "director_departure"})
EXTRA_TAGS = frozenset({"deal_termination", "business_update", "share_repurchase_program"})
CATEGORIES = ("all8", "departures", "plus_extra")
PRIMARY = {"weights": cl.PRIMARY_WEIGHTS, "cutoff": cl.PRIMARY_CUTOFF, "bucket": "1m", "horizon": "10", "otm": 3,
           "category": "all8", "text": "full"}
PREDICTION = {"H1": "negative", "H1b": "negative", "P": "no difference"}
DATA_DIR = cl.DATA_DIR


# ---- inputs ------------------------------------------------------------------------------------------------
def git_head() -> str:
    """Current commit (read-only)."""
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=cl.ROOT)
    return r.stdout.strip() or "unknown"


@dataclass
class Log:
    """Where and under which run id the tests are recorded."""
    path: Path = DATA_DIR / "ledger.csv"
    run_id: str = field(default_factory=ledger.new_run_id)
    head: str = field(default_factory=git_head)


@dataclass
class Inputs:
    classified: pd.DataFrame        # classify.classify output, one row per filing
    nulls: pd.DataFrame             # event_row_id, row_id, round
    gap: pd.DataFrame               # classify.gap_table, indexed by row_id
    outcome: pd.DataFrame           # normalised outcome table


def norm_horizon(h) -> str:
    s = str(h).strip().lower()
    if s in {"exp", "expiry"}:
        return "expiry"
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def _norm_outcome(o: pd.DataFrame) -> pd.DataFrame:
    missing = {"row_id", "bucket", "horizon", "otm", "y", "usable"} - set(o.columns)
    if missing:
        raise ValueError(f"outcome table is missing columns {sorted(missing)}")
    o = o.copy()
    o["horizon"] = o["horizon"].map(norm_horizon)
    otm = pd.to_numeric(o["otm"], errors="coerce").to_numpy(float)
    o["otm"] = np.round(np.where(otm < 1, otm * 100, otm)).astype(int)        # 0.03 and 3 both mean 3%
    o["bucket"] = o["bucket"].astype(str)
    o["usable"] = cl.as_bool(o["usable"])
    o["y"] = pd.to_numeric(o["y"], errors="coerce")
    if o.duplicated(["row_id", "bucket", "horizon", "otm"]).any():
        raise ValueError("outcome table has duplicate row_id x bucket x horizon x otm rows")
    return o


def prepare(events: pd.DataFrame, nulls: pd.DataFrame, gap: pd.DataFrame, outcome: pd.DataFrame,
            classified: pd.DataFrame | None = None, ref: pd.DataFrame | None = None) -> Inputs:
    """Validate and normalise the four input tables. Pass the labelled table (`classified`) or the frozen
    constants (`ref`, from classify.load_frozen); there is no default, so nothing picks a constants file silently."""
    c = (classified if classified is not None else cl.classify(events, gap, ref)).copy()
    for col in ("late", "earnings_excluded", "scored"):
        c[col] = cl.as_bool(c[col])
    c["t_0"] = pd.to_datetime(c["t_0"], errors="coerce")
    c["tags"] = c["tags"].fillna("").astype(str)
    n = nulls[["event_row_id", "row_id", "round"]].copy()
    if n.duplicated(["event_row_id", "round"]).any():
        raise ValueError("nulls table has duplicate event_row_id x round rows")
    return Inputs(c, n, cl.gap_table(gap), _norm_outcome(outcome))


def category_mask(c: pd.DataFrame, group: str, category: str) -> pd.Series:
    """Filings in a test's set before the late / earnings / usability filters."""
    tags = c["tags"].str.split("|").map(set)
    if group == "placebo":
        if category != "all":
            raise ValueError("the placebo set has a single category, 'all'")
        return c["group"] == "placebo"
    people = c["group"] == "people"
    if category == "all8":
        return people
    if category == "departures":                       # at least one departure tag
        return people & tags.map(lambda t: bool(t & DEPARTURE_TAGS))
    if category == "plus_extra":                       # the eight tags, plus late filings with any extra tag
        return people | tags.map(lambda t: bool(t & EXTRA_TAGS))
    raise ValueError(f"unknown category {category!r}")


def universe(c: pd.DataFrame, group: str, category: str) -> pd.Series:
    """Late filings of the set with no nearby earnings filing and a usable score."""
    return category_mask(c, group, category) & c["late"] & ~c["earnings_excluded"] & c["scored"]


def cell(inp: Inputs, bucket: str, horizon: str, otm: int, col: str = "y") -> pd.Series:
    """Usable, finite values of one outcome column for one bucket x horizon x strike, indexed by row_id."""
    o = inp.outcome
    m = (o["bucket"] == bucket) & (o["horizon"] == norm_horizon(horizon)) & (o["otm"] == int(otm)) & o["usable"]
    s = pd.to_numeric(o.loc[m].set_index("row_id")[col], errors="coerce")
    return s[np.isfinite(s)]


def matched_nulls(inp: Inputs, event_ids: pd.Series, y: pd.Series) -> np.ndarray:
    """n_events x n_rounds matrix of the matched ordinary days' values (NaN where missing or unusable)."""
    nl = inp.nulls[inp.nulls["event_row_id"].isin(event_ids)]
    piv = nl.assign(v=nl["row_id"].map(y)).set_index(["event_row_id", "round"])["v"].unstack()
    N = piv.reindex(pd.Index(event_ids)).to_numpy(float)
    return N if N.ndim == 2 and N.shape[1] else np.full((len(event_ids), 1), np.nan)


def _quarters(t: pd.Series) -> np.ndarray:
    return pd.to_datetime(t).dt.to_period("Q").astype(str).to_numpy()


# ---- statistics --------------------------------------------------------------------------------------------
def _label_perm(d: np.ndarray, old: np.ndarray, n_perm: int, rng: np.random.Generator) -> tuple[float, np.ndarray]:
    """Observed mean(old) - mean(rest) and its distribution when the labels are shuffled."""
    n, k = len(d), int(old.sum())
    total = d.sum()
    so = d[old].sum()
    obs = so / k - (total - so) / (n - k)
    out = np.empty(n_perm)
    for s in range(0, n_perm, CHUNK):
        b = min(CHUNK, n_perm - s)
        idx = rng.permuted(np.tile(np.arange(n), (b, 1)), axis=1)[:, :k]
        ps = d[idx].sum(axis=1)
        out[s:s + b] = ps / k - (total - ps) / (n - k)
    return float(obs), out


def _set_perm(E: np.ndarray, N: np.ndarray, n_perm: int, rng: np.random.Generator) -> tuple[float, np.ndarray]:
    """Observed mean(event - mean of its comparison set) and its distribution when, within each set, the
    'event' label is given to a random member (as in playground.stats.permutation_test, kept one-sided).
    Every set must hold the event and at least one comparison value."""
    V = np.column_stack([E, N])
    avail = ~np.isnan(V)
    cnt = avail.sum(axis=1)
    total = np.where(avail, V, 0.0).sum(axis=1)
    order = np.argsort(~avail, axis=1, kind="stable")
    rows = np.arange(len(E))[None, :]
    obs = float((E - (total - E) / (cnt - 1)).mean())
    out = np.empty(n_perm)
    for s in range(0, n_perm, CHUNK):
        b = min(CHUNK, n_perm - s)
        pick = np.floor(rng.random((b, len(E))) * cnt).astype(int)
        chosen = V[rows, order[rows, pick]]
        out[s:s + b] = (chosen - (total - chosen) / (cnt - 1)).mean(axis=1)
    return obs, out


def _p_values(obs: float, perm: np.ndarray, alternative: str) -> tuple[float, float]:
    """(one-sided p in the predicted direction, two-sided p); ties count as extreme."""
    eps = 1e-12
    tail = perm <= obs + eps if alternative == "less" else perm >= obs - eps
    p1 = (1 + int(tail.sum())) / (len(perm) + 1)
    p2 = (1 + int((np.abs(perm) >= abs(obs) - eps).sum())) / (len(perm) + 1)
    return p1, p2


def _boot_means(x: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    out = np.empty(n_boot)
    for s in range(0, n_boot, CHUNK):
        b = min(CHUNK, n_boot - s)
        out[s:s + b] = x[rng.integers(0, len(x), (b, len(x)))].mean(axis=1)
    return out


def _loo_diff(d: np.ndarray, old: np.ndarray, labels: np.ndarray) -> tuple[bool, float]:
    """Does mean(old) - mean(surprise) keep its sign when any one ticker (or quarter) is dropped?"""
    sign = np.sign(d[old].mean() - d[~old].mean())
    vals = []
    for lab in np.unique(labels):
        m = labels != lab
        if (m & old).any() and (m & ~old).any():
            vals.append(d[m & old].mean() - d[m & ~old].mean())
    if not vals:
        return False, np.nan
    return bool(all(np.sign(v) == sign for v in vals)), float(min(vals, key=lambda v: v * sign))


def _empty_row(note: str) -> dict:
    return {"n": 0, "n_old": 0, "n_comp": 0, "n_tickers": 0, "mean_old": np.nan, "mean_comp": np.nan,
            "effect": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p_one_sided": np.nan, "p_two_sided": np.nan,
            "trimmed_effect": np.nan, "loto_holds": np.nan, "loto_weakest": np.nan, "loqo_holds": np.nan,
            "loqo_weakest": np.nan, "note": note}


# ---- one row per test --------------------------------------------------------------------------------------
def _spec(test: str, group: str, category: str, weights: str, cutoff: float, bucket: str, horizon: str,
          otm: int, metric: str, comparison: str, text: str = "full") -> dict:
    return {"test": test, "prediction": PREDICTION[test], "group": group, "category": category,
            "weights": weights, "text": text, "cutoff": float(cutoff), "bucket": bucket, "horizon": norm_horizon(horizon),
            "otm": int(otm), "metric": metric, "comparison": comparison}


def _finish(row: dict, n_perm: int, n_boot: int) -> dict:
    row["p"] = row["p_two_sided"] if row["prediction"] == "no difference" else row["p_one_sided"]
    row["q_bh"] = np.nan
    row["descriptive"] = bool(row["n"] < MIN_N)
    row.update(n_perm=n_perm, n_boot=n_boot, seed=SEED)
    return row


def _contrast_row(inp: Inputs, test: str, group: str, category: str, weights: str, cutoff: float, bucket: str,
                  horizon: str, otm: int, n_perm: int, n_boot: int, text: str = "full") -> dict:
    """H1 and P: old minus surprise of (event Y minus mean Y of its matched ordinary days). `text` picks the word
    score source: "full" (primary), "excerpt" or "full4"."""
    row = _spec(test, group, category, weights, cutoff, bucket, horizon, otm,
                "y_minus_matched_nulls", "surprise events", text)
    c = inp.classified
    ev = c.loc[universe(c, group, category)]
    ev = ev[ev[cl.s_col(weights, text)].notna()]                       # a text variant needs its own word score
    y = cell(inp, bucket, horizon, otm)
    ev = ev[ev["row_id"].isin(y.index)]
    E = ev["row_id"].map(y).to_numpy(float)
    if len(ev) == 0:
        return _finish({**row, **_empty_row("no usable events")}, n_perm, n_boot)
    d, keep = stats.matched_diffs(E, matched_nulls(inp, ev["row_id"], y))
    ev = ev[keep]
    old = ev[cl.old_col(weights, cutoff, text)].to_numpy(bool)
    n_old, n_sur = int(old.sum()), int((~old).sum())
    base = {**_empty_row(""), "n": len(d), "n_old": n_old, "n_comp": n_sur, "n_tickers": ev["ticker"].nunique()}
    if n_old == 0 or n_sur == 0:
        return _finish({**row, **base, "note": "one group is empty"}, n_perm, n_boot)
    obs, perm = _label_perm(d, old, n_perm, np.random.default_rng(SEED))
    p1, p2 = _p_values(obs, perm, "less")
    rng = np.random.default_rng(SEED)
    boot = _boot_means(d[old], n_boot, rng) - _boot_means(d[~old], n_boot, rng)
    loto = _loo_diff(d, old, ev["ticker"].to_numpy())
    loqo = _loo_diff(d, old, _quarters(ev["t_0"]))
    return _finish({**row, **base, "mean_old": float(d[old].mean()), "mean_comp": float(d[~old].mean()),
                    "effect": obs, "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5)),
                    "p_one_sided": p1, "p_two_sided": p2,
                    "trimmed_effect": stats.trimmed_mean(d[old], TRIM) - stats.trimmed_mean(d[~old], TRIM),
                    "loto_holds": loto[0], "loto_weakest": loto[1], "loqo_holds": loqo[0], "loqo_weakest": loqo[1]},
                   n_perm, n_boot)


def _h1b_row(inp: Inputs, category: str, weights: str, cutoff: float, bucket: str, horizon: str, otm: int,
             n_perm: int, n_boot: int, text: str = "full") -> dict:
    """Old events' Y minus the mean Y of every usable ordinary day whose gap move is at least the event's."""
    row = _spec("H1b", "people", category, weights, cutoff, bucket, horizon, otm,
                "y_minus_gap_pool", "ordinary days with gap_move >= event's", text)
    c = inp.classified
    y = cell(inp, bucket, horizon, otm)
    ev = c.loc[universe(c, "people", category) & c[cl.old_col(weights, cutoff, text)]]
    ev = ev[ev["row_id"].isin(y.index)]
    g = inp.gap
    pool = pd.Index(inp.nulls["row_id"].unique()).intersection(g.index[g["gap_usable"]]).intersection(y.index)
    gm = g.loc[pool, "gap_move"].to_numpy(float)
    pool, gm = pool[np.isfinite(gm)], gm[np.isfinite(gm)]
    if len(ev) == 0 or len(pool) == 0:
        return _finish({**row, **_empty_row("no usable old events or ordinary days")}, n_perm, n_boot)
    E = ev["row_id"].map(y).to_numpy(float)
    ge = ev["gap_move"].to_numpy(float)
    N = np.where(gm[None, :] >= ge[:, None], y.loc[pool].to_numpy(float)[None, :], np.nan)
    d, keep = stats.matched_diffs(E, N)
    dropped = int((~keep).sum())
    note = f"{dropped} old events dropped: no ordinary day with a gap move that large" if dropped else ""
    if keep.sum() == 0:
        return _finish({**row, **_empty_row(note)}, n_perm, n_boot)
    ev, E, N = ev[keep], E[keep], N[keep]
    obs, perm = _set_perm(E, N, n_perm, np.random.default_rng(SEED))
    p1, p2 = _p_values(obs, perm, "less")
    boot = _boot_means(d, n_boot, np.random.default_rng(SEED))
    tickers, quarters = ev["ticker"].to_numpy(), _quarters(ev["t_0"])
    loto, loqo = stats.leave_one_out_sign(d, tickers), stats.leave_one_out_sign(d, quarters)
    return _finish({**row, "n": len(d), "n_old": len(d), "n_comp": int(np.median((~np.isnan(N)).sum(axis=1))),
                    "n_tickers": len(set(tickers)), "mean_old": float(E.mean()), "mean_comp": float((E - d).mean()),
                    "effect": obs, "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5)),
                    "p_one_sided": p1, "p_two_sided": p2, "trimmed_effect": stats.trimmed_mean(d, TRIM),
                    "loto_holds": loto[0], "loto_weakest": loto[1], "loqo_holds": loqo[0], "loqo_weakest": loqo[1],
                    "note": note}, n_perm, n_boot)


def _row(inp: Inputs, test: str, n_perm: int, n_boot: int, **kw) -> dict:
    s = {**PRIMARY, **kw}
    if test == "H1b":
        return _h1b_row(inp, s["category"], s["weights"], s["cutoff"], s["bucket"], s["horizon"], s["otm"],
                        n_perm, n_boot, s["text"])
    group, category = ("placebo", "all") if test == "P" else ("people", s["category"])
    return _contrast_row(inp, test, group, category, s["weights"], s["cutoff"], s["bucket"], s["horizon"],
                         s["otm"], n_perm, n_boot, s["text"])


# ---- ledger ------------------------------------------------------------------------------------------------
def _log(table: pd.DataFrame, log: Log | None) -> pd.DataFrame:
    log = log or Log()
    t = table
    led = pd.DataFrame({
        "kind": "oldnews_" + t["test"], "level": t["weights"], "group": t["group"], "subset": t["category"],
        "filter": "S>=" + t["cutoff"].map("{:g}".format) + t["text"].map(lambda v: "" if v == "full" else f"|text={v}")
        + t["period"].map(lambda v: "" if v == "pooled" else f"|year={v}"), "metric": t["metric"], "strategy": "",
        "bucket": t["bucket"], "horizon": t["horizon"], "otm": t["otm"], "entry": "t_0", "n_sets": t["n"],
        "n_tickers": t["n_tickers"], "event_mean": t["mean_old"], "null_mean": t["mean_comp"], "diff": t["effect"],
        "p_perm": t["p"], "q_bh": t["q_bh"], "trimmed_diff": t["trimmed_effect"], "loto_holds": t["loto_holds"],
        "loto_weakest": t["loto_weakest"], "loqo_holds": t["loqo_holds"], "loqo_weakest": t["loqo_weakest"],
        "low_sample": t["descriptive"], "n_perm": t["n_perm"], "seed": t["seed"]})
    ledger.append(Path(log.path), led, log.run_id, {}, log.head)
    return table


# ---- the tests ---------------------------------------------------------------------------------------------
def h1(inp: Inputs, log: Log | None = None, n_perm: int = N_PERM, n_boot: int = N_BOOT, period: str = "pooled",
       **kw) -> pd.DataFrame:
    """Primary test (defaults = the committed spec). Keyword overrides: weights, cutoff, bucket, horizon, otm,
    category. `period` only labels the rows ("pooled", or a t_0 year when `inp` was cut with `restrict_year`)."""
    return _log(pd.DataFrame([_row(inp, "H1", n_perm, n_boot, **kw)]).assign(period=str(period)), log)


def h1b(inp: Inputs, log: Log | None = None, n_perm: int = N_PERM, n_boot: int = N_BOOT, period: str = "pooled",
        **kw) -> pd.DataFrame:
    """Old events against ordinary days whose gap move was at least as large (prediction: negative)."""
    return _log(pd.DataFrame([_row(inp, "H1b", n_perm, n_boot, **kw)]).assign(period=str(period)), log)


def placebo(inp: Inputs, log: Log | None = None, n_perm: int = N_PERM, n_boot: int = N_BOOT, period: str = "pooled",
            **kw) -> pd.DataFrame:
    """H1 computed on the placebo filings (prediction: no difference)."""
    return _log(pd.DataFrame([_row(inp, "P", n_perm, n_boot, **kw)]).assign(period=str(period)), log)


def horizon_profile(inp: Inputs, log: Log | None = None, n_perm: int = N_PERM, n_boot: int = N_BOOT,
                    period: str = "pooled") -> pd.DataFrame:
    """H1, H1b and P at every fixed horizon; BH q-values across the nine horizons within each test."""
    parts = []
    for test in ("H1", "H1b", "P"):
        t = pd.DataFrame([_row(inp, test, n_perm, n_boot, horizon=h) for h in HORIZONS])
        t["q_bh"] = stats.benjamini_hochberg(t["p"].to_numpy(float))
        parts.append(t)
    return _log(pd.concat(parts, ignore_index=True).assign(period=str(period)), log)


def sensitivity(inp: Inputs, log: Log | None = None, n_perm: int = N_PERM, n_boot: int = N_BOOT) -> pd.DataFrame:
    """H1 with one setting changed at a time from the primary spec (horizons are in horizon_profile). The word
    score text source is varied too: the excerpt T and the full-text-plus-exhibit T_full4, each for every weight
    set at the primary cutoff."""
    specs = [("primary", {})]
    specs += [("weights", {"weights": w}) for w in cl.WEIGHT_SETS if w != PRIMARY["weights"]]
    specs += [("cutoff", {"cutoff": c}) for c in cl.CUTOFFS if c != PRIMARY["cutoff"]]
    specs += [("bucket", {"bucket": b}) for b in BUCKETS if b != PRIMARY["bucket"]]
    specs += [("otm", {"otm": o}) for o in OTMS if o != PRIMARY["otm"]]
    specs += [("category", {"category": k}) for k in CATEGORIES if k != PRIMARY["category"]]
    specs += [("text_excerpt", {"weights": w, "text": "excerpt"}) for w in cl.WEIGHT_SETS]    # excerpt T, every weight set
    specs += [("text_full4", {"weights": w, "text": "full4"}) for w in cl.WEIGHT_SETS]        # full text + exhibit flag
    t = pd.DataFrame([{"dimension": dim, **_row(inp, "H1", n_perm, n_boot, **kw)} for dim, kw in specs])
    return _log(t.assign(period="pooled"), log)


def counts(inp: Inputs) -> pd.DataFrame:
    """Every exclusion, step by step, for the primary cell (people and placebo sets)."""
    c = inp.classified
    y = cell(inp, PRIMARY["bucket"], PRIMARY["horizon"], PRIMARY["otm"])
    has_null = set(inp.nulls.loc[inp.nulls["row_id"].isin(y.index), "event_row_id"])
    cols = {}
    for group, category in (("people", "all8"), ("placebo", "all")):
        m = category_mask(c, group, category)
        steps = [("filings in the set", m)]
        steps.append(("late (event date >= 1 business day before acceptance)", m := m & c["late"]))
        steps.append(("no earnings filing within 5 sessions", m := m & ~c["earnings_excluded"]))
        steps.append(("scored (gap move exists, T present)", m := m & c["scored"]))
        steps.append(("outcome usable (1m, h=10, 3% put)", m := m & c["row_id"].isin(y.index)))
        steps.append(("at least one usable matched ordinary day", m := m & c["row_id"].isin(has_null)))
        steps.append(("  of which old news", m & c["old"]))
        steps.append(("  of which surprise news", m & ~c["old"]))
        cols[group] = {k: int(v.sum()) for k, v in steps}
    out = pd.DataFrame(cols).rename_axis("step").reset_index()
    pre = c[c["late"] & ~c["earnings_excluded"] & ~c["scored"]]
    reasons = pre.groupby(["unscored_reason", "group"]).size().unstack(fill_value=0) if len(pre) else pd.DataFrame()
    if len(reasons):
        reasons = reasons.reindex(columns=["people", "placebo"], fill_value=0)
        reasons.index = "  not scored: " + reasons.index.astype(str)
        out = pd.concat([out, reasons.rename_axis("step").reset_index()], ignore_index=True)
    return out


def text_source(inp: Inputs) -> pd.DataFrame:
    """Where the word score T came from, for the late, earnings-excluded filings that were scored, by group:
    the full text, or the excerpt as a flagged fallback (by full_text_status). Also how often the full text
    changes T against the excerpt and how often the exhibit flag is set. Everything here is a count of cues."""
    c = inp.classified
    b = c["late"] & ~c["earnings_excluded"] & c["scored"]
    cols = {}
    for group in ("people", "placebo"):
        s = c[b & (c["group"] == group)]
        full = ~cl.as_bool(s["t_fallback"])
        rows = [("scored filings", len(s)), ("word score T from the full text (T_full)", int(full.sum()))]
        fb = s.loc[~full, "t_source"].value_counts()
        rows += [("word score T from the excerpt, fallback: " + k.replace("excerpt (", "").rstrip(")"), int(v)) for k, v in fb.items()]
        rows += [("fallbacks in all", int((~full).sum())),
                 ("  T_full differs from the excerpt T (full text used)", int((full & (s["T"] != s["T_excerpt"])).sum())),
                 ("  exhibit flag set (cue_exhibit_dated_prior_full)", int((full & (s["cue_exhibit_dated_prior_full"] == 1)).sum())),
                 ("  T_full4 differs from T_full (full text used)", int((full & (s["T4"] != s["T"])).sum()))]
        cols[group] = dict(rows)
    return pd.DataFrame(cols).fillna(0).rename_axis("what").reset_index()


def coverage(inp: Inputs) -> pd.DataFrame:
    """How well the gap inputs are covered, for the late, earnings-excluded filings of each set: how many are
    scored and why the rest are not, how many inputs M used, which inputs are missing, which expiry bucket gave
    the spot at each end of the gap, how many gap-start marks were stale and how many baseline sessions the
    volume input had. Everything here comes from the gap table, never from outcomes."""
    c = inp.classified
    base = c["late"] & ~c["earnings_excluded"]
    cols = {}
    for group in ("people", "placebo"):
        b = base & (c["group"] == group)
        s = c[b & c["scored"]]
        n = lambda m: int(m.sum())                                                      # noqa: E731
        rows = [("late, earnings-excluded filings", n(b)), ("scored: gap move exists", len(s))]
        rows += [(f"not scored: {r}", int(k)) for r, k in c.loc[b & ~c["scored"], "unscored_reason"].value_counts().items()]
        rows += [(f"scored with {k} of 3 inputs", n(s["n_inputs"] == k)) for k in (3, 2, 1)]
        rows += [("  implied-vol change missing (no 1m mark at both ends)", n(s["d_iv_gap"].isna())),
                 ("  volume input missing", n(s["log_vol_gap_ratio"].isna())),
                 ("    of which volume ratio = 0 (log undefined)", n(s["vol_zero"].astype(bool)))]
        bk = s["spot_bucket"].astype(str).str.split("|", expand=True).reindex(columns=[0, 1])
        for i, what in ((0, "gap start"), (1, "t_pre")):
            rows += [(f"spot at {what} from the {lab} pair", n(bk[i] == lab)) for lab in BUCKETS]
        stale = pd.to_numeric(s["stale_sessions"], errors="coerce")
        rows += [("gap-start mark fresh (0 sessions old)", n(stale == 0)),
                 ("gap-start mark stale (1 to 3 sessions old)", n(stale > 0))]
        nb = pd.to_numeric(s["n_baseline"], errors="coerce")
        rows += [(f"volume baseline from {k} of the 5 sessions before the gap", n(nb == k)) for k in (5, 4, 3, 2)]
        rows += [("volume baseline from fewer than 2 sessions (volume missing)", n(nb < 2))]
        rows += [("median sessions in the gap-move formula (n_eff)",
                  float(pd.to_numeric(s["n_eff"], errors="coerce").median()) if len(s) else np.nan)]
        cols[group] = dict(rows)
    return pd.DataFrame(cols).fillna(0).rename_axis("what").reset_index()


# ---- pooled primary result, and each t_0 year as a robustness check ------------------------------------------
YEAR_SPLIT = {"insample": (2024, 2025)}          # labels that also report each year separately (amended test plan)


def restrict_year(inp: Inputs, year: int) -> Inputs:
    """The same inputs cut to the filings whose entry day t_0 falls in `year`, with only the ordinary days drawn
    for those filings. Nothing is recomputed: labels and outcomes are the pooled run's."""
    c = inp.classified
    keep = pd.to_datetime(c["t_0"]).dt.year == int(year)
    ids = set(c.loc[keep, "row_id"])
    return Inputs(c[keep], inp.nulls[inp.nulls["event_row_id"].isin(ids)], inp.gap, inp.outcome)


def by_year(inp: Inputs, years: tuple[int, ...], log: Log | None = None, n_perm: int = N_PERM,
            n_boot: int = N_BOOT) -> dict[str, pd.DataFrame]:
    """H1, H1b, P and the horizon profile for each year separately, all logged (period = the year). Returns
    `by_year` (H1, H1b, P rows), `profile_by_year` and `counts_by_year` (n per step and group, columns per year)."""
    main, prof, cnt = [], [], None
    for y in years:
        iy = restrict_year(inp, y)
        main += [h1(iy, log, n_perm, n_boot, period=str(y)), h1b(iy, log, n_perm, n_boot, period=str(y)),
                 placebo(iy, log, n_perm, n_boot, period=str(y))]
        prof.append(horizon_profile(iy, log, n_perm, n_boot, period=str(y)))
        ct = counts(iy).rename(columns={"people": f"people_{y}", "placebo": f"placebo_{y}"})
        cnt = ct if cnt is None else cnt.merge(ct, on="step", how="outer")
    return {"by_year": pd.concat(main, ignore_index=True), "profile_by_year": pd.concat(prof, ignore_index=True),
            "counts_by_year": cnt.fillna(0) if cnt is not None else pd.DataFrame()}


# ---- the whole run -----------------------------------------------------------------------------------------
def _md(df: pd.DataFrame) -> str:
    """A plain Markdown table (numbers to at most 4 decimals)."""
    def cell_text(v) -> str:
        if isinstance(v, (float, np.floating)) and np.isfinite(v):
            return f"{v:.4f}".rstrip("0").rstrip(".")
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(cell_text(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def _fmt(r: pd.Series) -> str:
    if not np.isfinite(r["effect"]):
        return f"not computed ({r['note']}); n = {r['n']}"
    s = (f"effect {r['effect']:+.4f} (95% CI {r['ci_lo']:+.4f} to {r['ci_hi']:+.4f}), one-sided p = "
         f"{r['p_one_sided']:.4f}, two-sided p = {r['p_two_sided']:.4f}; n = {r['n']} "
         f"(old {r['n_old']}, comparison {r['n_comp']}{' median per event' if r['test'] == 'H1b' else ''}), "
         f"{r['n_tickers']} tickers")
    return s + ("  **DESCRIPTIVE: fewer than 30 events qualify.**" if r["descriptive"] else "")


def _summary(label: str, t: dict, cap: tuple | None, log: Log, n_perm: int, n_boot: int, added: int,
             dropped: pd.DataFrame, ref: pd.DataFrame, years: tuple[int, ...] = (), NB: dict | None = None) -> str:
    h, b, p = t["h1"].iloc[0], t["h1b"].iloc[0], t["placebo"].iloc[0]
    h0, h1 = cl.holdout_window(NB)
    lines = [f"# Old news vs surprise news: {label}", "",
             f"Run `{log.run_id}`, git `{log.head[:10]}`, seed {SEED}, {n_perm:,} permutations, {n_boot:,} "
             f"bootstrap resamples. Dates: "
             + (f"every date inside [{cap[0].date()}, {cap[1].date()}) and outside the sealed window "
                f"{h0.date()}..{h1.date()}."
                if cap is not None else "no date cap (the label is switched by RUN_HOLDOUT / RUN_OOS in the notebook).")
             + f" Outcome = log(RV/IV0) minus the mean over the event's matched ordinary days. Math inputs are "
             f"standardised with the frozen {ref.attrs['source']} constants (volume as log of the ratio; M is the mean of the "
             f"available inputs, gap move required; {Path(ref.attrs['path']).name}; {ref.attrs['n_null_rows']} "
             f"ordinary days, {ref.attrs['window_t_0']['first']} to {ref.attrs['window_t_0']['last']}, sha256 "
             f"{ref.attrs['sha256'][:12]}), applied unchanged to every event of this run.", "",
             *([f"**Primary result: pooled over t_0 years {', '.join(map(str, years))}.** Each year is reported "
                "separately below as a robustness check (same rules, n per group in every row).", ""] if years else []),
             "## Counts", "", _md(t["counts"]), "",
             "## Coverage of the gap inputs (late, earnings-excluded filings)", "", _md(t["coverage"]), "",
             "## Word score text source (scored late, earnings-excluded filings)", "",
             "T in S is the full-text T_full; a filing whose full text is not ok falls back to the excerpt T and is "
             "counted here. The sensitivity table repeats H1 with the excerpt T and with T_full4 (exhibit flag added).",
             "", _md(t["text_source"]), "",
             *(["Dropped at the window edge before anything was computed:", "", _md(dropped), ""]
               if len(dropped) else []),
             "## H1 (primary): old minus surprise, h = 10, 1-month bucket. Prediction: negative", "", _fmt(h)]
    if np.isfinite(h["effect"]):
        lines.append(f"Sign matches the prediction: {'yes' if h['effect'] < 0 else 'no'}. "
                     f"One-sided p below {ALPHA}: {'yes' if h['p_one_sided'] < ALPHA else 'no'}.")
    lines += ["", "## H1b: old events minus ordinary days with a gap move at least as large. Prediction: negative",
              "", _fmt(b), "", "## P (placebo filings): H1 on the placebo set. Prediction: no difference", "",
              _fmt(p), "", "## Horizon profile (BH q across the nine horizons, within each test)", ""]
    cols = ["test", "horizon", "n", "n_old", "n_comp", "effect", "ci_lo", "ci_hi", "p", "q_bh", "descriptive"]
    lines += [_md(t["profile"][cols]), "", "## Sensitivity (H1, one change at a time)", ""]
    cols = ["dimension", "weights", "text", "cutoff", "bucket", "otm", "category", "n", "n_old", "n_comp", "effect",
            "ci_lo", "ci_hi", "p_one_sided", "descriptive"]
    lines += [_md(t["sensitivity"][cols]), ""]
    if years:
        lines += ["## By year (robustness check, split by the year of the entry day t_0)", "",
                  "n_old is the old-news group, n_comp the comparison group (surprise events; for H1b the median "
                  "number of ordinary days per event). A year with fewer than 30 events is descriptive.", ""]
        for y in years:
            sel = t["by_year"][t["by_year"]["period"] == str(y)]
            lines += [f"### {y}", ""] + [f"- **{r['test']}** ({r['prediction']}): {_fmt(r)}" for _, r in sel.iterrows()] + [""]
        cols = ["period", "test", "horizon", "n", "n_old", "n_comp", "effect", "ci_lo", "ci_hi", "p", "q_bh", "descriptive"]
        lines += ["### Horizon profile by year (BH q across the nine horizons, within each test and year)", "",
                  _md(t["profile_by_year"][cols]), "", "### Counts by year", "", _md(t["counts_by_year"]), ""]
    lines += ["## Not computed here", "",
              "- H2 and the trade record: see `trade_<label>/` (src/oldnews/trade.py).",
              "- Entry session t_0 + 1 and full-text word cues: not in the input tables.", "",
              f"Ledger: this run added {added} rows; {ledger.variant_count(Path(log.path))} rows in total."]
    return "\n".join(lines) + "\n"


def run(label: str, data_dir: Path = DATA_DIR, NB: dict | None = None, n_perm: int = N_PERM,
        n_boot: int = N_BOOT, log: Log | None = None, zref_source: str | None = None, zref: Path | None = None,
        years: tuple[int, ...] | None = None) -> dict[str, pd.DataFrame]:
    """Run every test for `label` from data_dir/{events,nulls,gap,outcome}_<label>.csv and write
    data_dir/results_<label>/ (tables and summary.md). Rows past the end of the window are dropped and counted,
    then dates are checked (classify.guard_dates). Pass NB (the notebook namespace) so the switches and
    HOLDOUT_START / HOLDOUT_END can be seen.

    Labels: insample; holdout (only when RUN_HOLDOUT is True in NB); oos (only when RUN_OOS is True). discovery
    and dryrun are retired and refuse. Constants: always src/oldnews/zref_frozen_insample.json (`zref` may name
    it explicitly; the retired 2022-23 file is refused). A missing file is an error, never a fallback.
    The pooled run is the primary result. For "insample" (or when `years` is given) H1, H1b, P and the horizon
    profile are also reported for each t_0 year, as `by_year`, `profile_by_year` and `counts_by_year`, and logged
    with the year in the ledger's filter column."""
    data_dir = Path(data_dir)
    cl.check_label(label, NB)                 # retired labels refuse; holdout and oos only behind their switch
    ref = cl.load_frozen(zref, zref_source)   # refuses, before anything is read, if the 2024-25 constants are missing
    years = YEAR_SPLIT.get(label, ()) if years is None else tuple(years)
    frames, dropped = cl.load_tables(label, data_dir, NB, ("events", "nulls", "gap", "outcome"))
    cap = cl.window(label)
    classified = cl.save(label, data_dir, cl.classify(frames["events"], frames["gap"], ref), dropped)
    inp = prepare(frames["events"], frames["nulls"], frames["gap"], frames["outcome"], classified=classified)
    log = log or Log(data_dir / "ledger.csv")
    before = ledger.variant_count(Path(log.path))
    t = {"counts": counts(inp), "coverage": coverage(inp), "text_source": text_source(inp), "h1": h1(inp, log, n_perm, n_boot), "h1b": h1b(inp, log, n_perm, n_boot),
         "placebo": placebo(inp, log, n_perm, n_boot), "profile": horizon_profile(inp, log, n_perm, n_boot),
         "sensitivity": sensitivity(inp, log, n_perm, n_boot)}
    if years:
        t.update(by_year(inp, years, log, n_perm, n_boot))
    out = data_dir / f"results_{label}"
    out.mkdir(parents=True, exist_ok=True)
    dropped.to_csv(out / "dropped.csv", index=False)
    for k, v in t.items():
        v.to_csv(out / f"{k}.csv", index=False)
    added = ledger.variant_count(Path(log.path)) - before
    (out / "summary.md").write_text(_summary(label, t, cap, log, n_perm, n_boot, added, dropped, ref, years, NB), encoding="utf-8")
    print(f"[{label}] H1: {_fmt(t['h1'].iloc[0])}")
    return t
