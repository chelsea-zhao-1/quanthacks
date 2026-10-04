"""Exploratory diagnostics, added after the primary result; they do not change it.

The committed test (docs/test_plan.md) was run once on 2024-25 and came out null (data/oldnews/results_insample/).
These four small analyses explain what that null does and does not rule out. They are exploratory, labelled as
such in every output, logged in the test ledger with kind "diagnostic", and never feed back into any rule.

1. `power`       minimum detectable effect (MDE) for H1 at h = 10 (one-sided 5%, 80% power) from the observed n per
                 group and standard deviations; n per group needed for effects of 0.10 and 0.20 in log(RV/IV); the
                 effect sizes the observed result rules out (one-sided 5%).
2. `separation`  old vs surprise on the pre-entry inputs only (gap move, implied-vol change, volume, T, and M):
                 means and medians. `agreement`: the math-only and words-only labels as a 2x2 table, Cohen's kappa.
3. `influence`   H1 at h = 10 with each ticker left out in turn (range of the difference, sign flips) and the
                 10% trimmed mean difference.
4. `levels`      mean y (event Y minus the mean Y of its matched ordinary days) for old and surprise separately at
                 every fixed horizon, with bootstrap 95% intervals, plus the raw event and ordinary-day levels.

The event sample is rebuilt exactly as tests._contrast_row builds it (same universe, same matched ordinary days,
same frozen constants), and the H1 effect it gives is checked against the primary run's. Window guard: label
"insample" only, read through classify.load_tables (every date in [2024-01-01, 2026-01-01), outside the sealed
placeholder); nothing reads data/oldnews/_unused_2022_23/; offline, no network. Nothing here writes outside
data/oldnews/results_insample/diagnostics/ except the appended ledger rows.

Run from the repo root:
    .venv/Scripts/python.exe -c "import sys; sys.path.insert(0, 'src'); from oldnews import diagnostics as D; D.run()"
"""
from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from oldnews import classify as cl
from oldnews import tests as T
from playground import ledger, stats

LABEL = "insample"
SEED = T.SEED
N_BOOT = T.N_BOOT
ALPHA = 0.05
POWER = 0.80
TARGETS = (0.10, 0.20)            # effect sizes in log(RV/IV) for the n-needed question
TRIM10 = 0.10                     # trimmed-mean fraction (each tail) for the influence check
MIN_N = T.MIN_N
OUT_DIR = cl.DATA_DIR / "results_insample" / "diagnostics"
HEADER = "Exploratory diagnostics, added after the primary result; they do not change it."
PRE_ENTRY = (                     # (column in the classified table, description); all known at t_pre
    ("gap_move", "gap move |r_gap - r_mkt| / (sigma * sqrt(n))"),
    ("d_iv_gap", "1m ATM implied-vol change over the gap"),
    ("vol_gap_ratio", "ATM-pair volume in the gap / 5-session baseline (raw ratio)"),
    ("log_vol_gap_ratio", "log of the volume ratio (the input used in M)"),
    ("T", "word score T (full text, 0 to 3)"),
    ("M", "math score M (mean of the standardised inputs)"),
)


# ---- inputs ------------------------------------------------------------------------------------------------
def load(data_dir: Path = cl.DATA_DIR, label: str = LABEL, NB: dict | None = None,
         zref: Path | None = None) -> T.Inputs:
    """Read the 2024-25 tables through the window guard and label them in memory with the frozen constants.
    Only "insample" is allowed; nothing is written."""
    if label != LABEL:
        raise PermissionError(f"the diagnostics run on the 2024-25 label {LABEL!r} only, not {label!r}")
    frames, _ = cl.load_tables(label, Path(data_dir), NB, ("events", "nulls", "gap", "outcome"))
    c = cl.classify(frames["events"], frames["gap"], cl.load_frozen(zref))
    return T.prepare(frames["events"], frames["nulls"], frames["gap"], frames["outcome"], classified=c)


def h1_sample(inp: T.Inputs, horizon: str = "10", weights: str = "equal", cutoff: float = 1.0,
              bucket: str = "1m", otm: int = 3) -> pd.DataFrame:
    """The events behind H1 at one horizon, selected exactly as tests._contrast_row selects them. One row per
    event: its classified columns plus y_event, y_null (mean Y of its usable matched ordinary days), d (their
    difference) and is_old (the label under `weights` and `cutoff`)."""
    c = inp.classified
    ev = c.loc[T.universe(c, "people", "all8")]
    ev = ev[ev[cl.s_col(weights)].notna()]
    y = T.cell(inp, bucket, horizon, otm)
    ev = ev[ev["row_id"].isin(y.index)]
    if len(ev) == 0:                                   # e.g. h = 42, 63: the 1-month option has expired
        return ev.assign(y_event=[], y_null=[], d=[], is_old=pd.Series([], dtype=bool))
    E = ev["row_id"].map(y).to_numpy(float)
    d, keep = stats.matched_diffs(E, T.matched_nulls(inp, ev["row_id"], y))
    out = ev[keep].copy()
    out["y_event"] = E[keep]
    out["y_null"] = E[keep] - d
    out["d"] = d
    out["is_old"] = out[cl.old_col(weights, cutoff)].astype(bool).to_numpy()
    return out


# ---- 1. power ----------------------------------------------------------------------------------------------
def power(d: np.ndarray, old: np.ndarray, alpha: float = ALPHA, pw: float = POWER,
          targets: tuple[float, ...] = TARGETS) -> pd.DataFrame:
    """Normal-approximation power for a difference in two means (Welch standard error), one-sided.

    MDE = (z_{1-alpha} + z_{power}) * sqrt(s_old^2 / n_old + s_sur^2 / n_sur). The n per group needed for an
    effect delta (equal groups) is (z_{1-alpha} + z_{power})^2 * (s_old^2 + s_sur^2) / delta^2. The bound
    `rules_out_below` = observed - z_{1-alpha} * SE: a true old-minus-surprise effect below it (more negative)
    is rejected at one-sided 5% by these data. Also the power the observed n had for each target effect."""
    d, old = np.asarray(d, float), np.asarray(old, bool)
    a, b = d[old], d[~old]
    n1, n2 = len(a), len(b)
    s1, s2 = float(a.std(ddof=1)), float(b.std(ddof=1))
    za, zb = NormalDist().inv_cdf(1 - alpha), NormalDist().inv_cdf(pw)
    se = math.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    obs = float(a.mean() - b.mean())
    rows = [("n_old", n1), ("n_surprise", n2), ("sd_old", s1), ("sd_surprise", s2), ("se_difference", se),
            ("observed_effect", obs), ("z_alpha_one_sided", za), ("z_power", zb),
            ("mde", (za + zb) * se),
            ("rules_out_below", obs - za * se)]
    for t in targets:
        rows.append((f"n_per_group_for_{t:.2f}", int(math.ceil((za + zb) ** 2 * (s1 ** 2 + s2 ** 2) / t ** 2))))
        rows.append((f"power_at_observed_n_for_{t:.2f}", NormalDist().cdf(t / se - za)))
    return pd.DataFrame(rows, columns=["quantity", "value"])


# ---- 2. does the classifier separate what it claims? -------------------------------------------------------
def separation(sample: pd.DataFrame) -> pd.DataFrame:
    """Old vs surprise on the pre-entry inputs only: n with the input, mean and median per group."""
    rows = []
    for col, what in PRE_ENTRY:
        r = {"input": col, "what": what}
        for name, m in (("old", sample["is_old"]), ("surprise", ~sample["is_old"])):
            x = pd.to_numeric(sample.loc[m, col], errors="coerce")
            x = x[np.isfinite(x)]
            r.update({f"n_{name}": len(x), f"mean_{name}": float(x.mean()) if len(x) else np.nan,
                      f"median_{name}": float(x.median()) if len(x) else np.nan})
        r["mean_diff"] = r["mean_old"] - r["mean_surprise"]
        rows.append(r)
    return pd.DataFrame(rows)


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa for two binary labels: (observed agreement - chance agreement) / (1 - chance agreement)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = float((a == b).mean())
    pa, pb = float(a.mean()), float(b.mean())
    pe = pa * pb + (1 - pa) * (1 - pb)
    return np.nan if pe == 1 else (po - pe) / (1 - pe)


def agreement(sample: pd.DataFrame, cutoff: float = 1.0) -> tuple[pd.DataFrame, dict]:
    """2x2 table of the math-only label (M >= cutoff) against the words-only label (T >= cutoff), with how many
    in each cell the primary (equal-weight) rule calls old; and the agreement statistics."""
    m = sample[cl.old_col("math_only", cutoff)].astype(bool).to_numpy()
    w = sample[cl.old_col("words_only", cutoff)].astype(bool).to_numpy()
    p = sample["is_old"].to_numpy(bool)
    rows = []
    for mv in (True, False):
        for wv in (True, False):
            k = (m == mv) & (w == wv)
            rows.append({"math_only": "old" if mv else "surprise", "words_only": "old" if wv else "surprise",
                         "n": int(k.sum()), "of_which_primary_old": int((k & p).sum())})
    stats_ = {"n": len(m), "math_only_old": int(m.sum()), "words_only_old": int(w.sum()),
              "agreement": float((m == w).mean()) if len(m) else np.nan, "kappa": cohen_kappa(m, w)}
    return pd.DataFrame(rows), stats_


# ---- 3. influence ------------------------------------------------------------------------------------------
def influence(sample: pd.DataFrame, trim: float = TRIM10) -> tuple[pd.DataFrame, dict]:
    """H1 difference with each ticker left out in turn, and the trimmed-mean difference (`trim` cut from each
    tail of each group). Sign flips count against the full-sample sign; `n_negative` counts how many
    leave-one-out differences have the predicted (negative) sign."""
    d, old, tick = sample["d"].to_numpy(float), sample["is_old"].to_numpy(bool), sample["ticker"].to_numpy()
    full = float(d[old].mean() - d[~old].mean())
    rows = []
    for t in np.unique(tick):
        k = tick != t
        if (k & old).any() and (k & ~old).any():
            rows.append({"ticker_left_out": t, "n_events_left_out": int((~k).sum()),
                         "n_old_left_out": int((~k & old).sum()), "diff": float(d[k & old].mean() - d[k & ~old].mean())})
    loto = pd.DataFrame(rows, columns=["ticker_left_out", "n_events_left_out", "n_old_left_out", "diff"])
    loto["change"] = loto["diff"] - full
    v = loto["diff"].to_numpy(float)
    summ = {"full_effect": full, "n_tickers": int(len(np.unique(tick))), "n_left_out_runs": len(v),
            "loto_min": float(v.min()) if len(v) else np.nan, "loto_max": float(v.max()) if len(v) else np.nan,
            "sign_flips": int((np.sign(v) != np.sign(full)).sum()), "n_negative": int((v < 0).sum()),
            "trim_each_tail": trim,
            "trimmed_effect": stats.trimmed_mean(d[old], trim) - stats.trimmed_mean(d[~old], trim),
            "median_old_minus_median_surprise": float(np.median(d[old]) - np.median(d[~old]))}
    return loto.sort_values("change", key=np.abs, ascending=False).reset_index(drop=True), summ


# ---- 4. the levels behind the difference -------------------------------------------------------------------
def levels(inp: T.Inputs, n_boot: int = N_BOOT, horizons: tuple[str, ...] = T.HORIZONS) -> pd.DataFrame:
    """At every fixed horizon (1-month bucket, 3% strike, primary rule): for old and for surprise separately,
    n, mean y = event Y minus its matched ordinary days' mean Y (bootstrap 95% interval, seed fixed), and the
    raw means of the event Y and of the ordinary-day Y behind it. Groups under 30 events are descriptive."""
    rows = []
    for h in horizons:
        s = h1_sample(inp, h)
        for name, m in (("old", s["is_old"]), ("surprise", ~s["is_old"])):
            g = s[m]
            r = {"horizon": h, "group": name, "n": len(g), "n_tickers": g["ticker"].nunique()}
            if len(g):
                boot = T._boot_means(g["d"].to_numpy(float), n_boot, np.random.default_rng(SEED))
                r.update(mean_y=float(g["d"].mean()), ci_lo=float(np.percentile(boot, 2.5)),
                         ci_hi=float(np.percentile(boot, 97.5)), median_y=float(g["d"].median()),
                         mean_y_event=float(g["y_event"].mean()), mean_y_ordinary=float(g["y_null"].mean()))
            rows.append(r)
    out = pd.DataFrame(rows).reindex(columns=["horizon", "group", "n", "n_tickers", "mean_y", "ci_lo", "ci_hi",
                                              "median_y", "mean_y_event", "mean_y_ordinary"])
    out["descriptive"] = out["n"] < MIN_N
    return out


# ---- ledger ------------------------------------------------------------------------------------------------
def _ledger_rows(pw: pd.DataFrame, sep: pd.DataFrame, agr: dict, inf: dict, lev: pd.DataFrame,
                 n_tickers: int) -> pd.DataFrame:
    """One ledger row per reported diagnostic number, kind "diagnostic"; the value is in `diff` unless noted."""
    base = {"kind": "diagnostic", "level": "equal", "group": "people", "subset": "all8", "strategy": "",
            "bucket": "1m", "horizon": "10", "otm": 3, "entry": "t_0", "seed": SEED, "n_perm": 0}
    P = dict(zip(pw["quantity"], pw["value"]))
    n = int(P["n_old"] + P["n_surprise"])
    rows = [{**base, "filter": "S>=1|diag=power", "metric": f"power_{q}", "n_sets": n, "n_tickers": n_tickers,
             "diff": float(P[q])} for q in pw["quantity"] if q.startswith(("mde", "rules_out", "n_per_group", "power_at"))]
    rows += [{**base, "filter": "S>=1|diag=separation", "metric": f"pre_entry_{r['input']}_old_minus_surprise",
              "n_sets": int(r["n_old"] + r["n_surprise"]), "n_tickers": n_tickers, "event_mean": r["mean_old"],
              "null_mean": r["mean_surprise"], "diff": r["mean_diff"]} for _, r in sep.iterrows()]
    rows.append({**base, "filter": "S>=1|diag=agreement", "metric": "cohen_kappa_math_only_vs_words_only",
                 "n_sets": agr["n"], "n_tickers": n_tickers, "diff": agr["kappa"]})
    rows.append({**base, "filter": "S>=1|diag=influence", "metric": "loto_and_trimmed10", "n_sets": n,
                 "n_tickers": inf["n_tickers"], "diff": inf["full_effect"], "trimmed_diff": inf["trimmed_effect"],
                 "loto_holds": inf["sign_flips"] == 0,
                 "loto_weakest": inf["loto_min"] if inf["full_effect"] > 0 else inf["loto_max"]})
    rows += [{**base, "filter": f"S>=1|diag=levels|group={r['group']}", "metric": "y_minus_matched_nulls_level",
              "horizon": r["horizon"], "n_sets": r["n"], "n_tickers": r["n_tickers"], "event_mean": r["mean_y_event"],
              "null_mean": r["mean_y_ordinary"], "diff": r["mean_y"], "low_sample": bool(r["descriptive"])}
             for _, r in lev.iterrows()]
    return pd.DataFrame(rows)


# ---- report ------------------------------------------------------------------------------------------------
def _f(v, k: int = 3) -> str:
    return f"{v:+.{k}f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v)


def _headlines(pw: pd.DataFrame, sep: pd.DataFrame, agr: dict, inf: dict, lev: pd.DataFrame) -> list[str]:
    P = dict(zip(pw["quantity"], pw["value"]))
    S = sep.set_index("input")
    l10 = lev[lev["horizon"] == "10"].set_index("group")
    t1, t2 = (f"{t:.2f}" for t in TARGETS)
    out = [
        f"- **Power (H1, h = 10).** With {int(P['n_old'])} old and {int(P['n_surprise'])} surprise events and the "
        f"observed standard deviations ({P['sd_old']:.3f} and {P['sd_surprise']:.3f}), the minimum detectable "
        f"effect (one-sided 5%, 80% power) is {P['mde']:.3f} in log(RV/IV). Detecting {t1} would need about "
        f"{int(P[f'n_per_group_for_{t1}'])} events per group; detecting {t2} about {int(P[f'n_per_group_for_{t2}'])}. "
        f"The power we had was {P[f'power_at_observed_n_for_{t1}']:.0%} for {t1} and "
        f"{P[f'power_at_observed_n_for_{t2}']:.0%} for {t2}.",
        f"- **What the null rules out.** The observed difference is {_f(P['observed_effect'])} "
        f"({'predicted' if P['observed_effect'] < 0 else 'wrong'} sign). At one-sided 5% it rules out a true "
        f"old-minus-surprise effect more negative than {_f(P['rules_out_below'])}"
        + (f", so a true effect of -{t1} or more negative is ruled out. It does not rule out a small negative effect "
           f"between {_f(P['rules_out_below'])} and 0, nor a positive one."
           if P['rules_out_below'] > -TARGETS[0] else
           f"; effects between {_f(P['rules_out_below'])} and 0 are not ruled out."),
        f"- **Separation (old vs surprise, pre-entry).** Gap move mean {S.at['gap_move', 'mean_old']:.2f} vs "
        f"{S.at['gap_move', 'mean_surprise']:.2f} (median {S.at['gap_move', 'median_old']:.2f} vs "
        f"{S.at['gap_move', 'median_surprise']:.2f}); implied-vol change mean {_f(S.at['d_iv_gap', 'mean_old'], 4)} vs "
        f"{_f(S.at['d_iv_gap', 'mean_surprise'], 4)}; log volume ratio mean {_f(S.at['log_vol_gap_ratio', 'mean_old'])} vs "
        f"{_f(S.at['log_vol_gap_ratio', 'mean_surprise'])}; T mean {S.at['T', 'mean_old']:.2f} vs "
        f"{S.at['T', 'mean_surprise']:.2f}. The math-only and words-only labels agree on {agr['agreement']:.0%} of "
        f"{agr['n']} events (math-only old {agr['math_only_old']}, words-only old {agr['words_only_old']}); "
        f"Cohen's kappa = {agr['kappa']:.3f}.",
        f"- **Influence.** Leaving out one ticker at a time ({inf['n_left_out_runs']} runs) the difference ranges "
        f"from {_f(inf['loto_min'])} to {_f(inf['loto_max'])}; the sign flips {inf['sign_flips']} times "
        f"({inf['n_negative']} runs have the predicted negative sign). 10% trimmed difference: "
        f"{_f(inf['trimmed_effect'])}; median old minus median surprise: {_f(inf['median_old_minus_median_surprise'])}.",
    ]
    if {"old", "surprise"} <= set(l10.index):
        o, s = l10.loc["old"], l10.loc["surprise"]
        out.append(
            f"- **Levels (h = 10).** Old events: mean y {_f(o['mean_y'])} (95% CI {_f(o['ci_lo'])} to {_f(o['ci_hi'])}), "
            f"surprise events: {_f(s['mean_y'])} ({_f(s['ci_lo'])} to {_f(s['ci_hi'])}), where y is event log(RV/IV) "
            f"minus its matched ordinary days'. Raw levels: event log(RV/IV) {_f(o['mean_y_event'])} (old) and "
            f"{_f(s['mean_y_event'])} (surprise) against {_f(o['mean_y_ordinary'])} and {_f(s['mean_y_ordinary'])} on "
            f"their ordinary days. Every horizon is in the table below.")
    return out


def report(pw, sep, agr_t, agr, loto, inf, lev, run_id: str, head: str, check: dict) -> str:
    md = T._md
    lines = [f"# Diagnostics: old news vs surprise news, insample (2024-25)", "",
             f"**{HEADER}**", "",
             f"Run `{run_id}`, git `{head[:10]}`, seed {SEED}, {N_BOOT:,} bootstrap resamples (levels). Label insample only "
             "(every date in [2024-01-01, 2026-01-01), outside the sealed placeholder). Event sample rebuilt exactly as the "
             "primary H1 (late people-news 8-Ks, earnings excluded, scored, 1-month bucket, 3% strike, at least one usable "
             f"matched ordinary day). Check against the primary run: H1 at h = 10 here is {_f(check['effect'], 4)} with "
             f"{check['n_old']} old and {check['n_sur']} surprise events (primary summary: +0.0966, 45 and 101). "
             "Logged in data/oldnews/ledger.csv with kind \"diagnostic\".", "",
             "## Headlines", "", *_headlines(pw, sep, agr, inf, lev), "",
             "## 1. Power and minimum detectable effect (H1, h = 10)", "",
             "Normal approximation, Welch standard error, one-sided 5%, 80% power. n per group assumes equal groups.", "",
             md(pw), "",
             "## 2. Does the classifier separate what it claims? (pre-entry inputs only)", "",
             "Same 146-event H1 sample. Every input is known at t_pre; no outcome is used.", "",
             md(sep), "",
             "Math-only label (M >= 1) against words-only label (T >= 1); `of_which_primary_old` = events the primary "
             "equal-weight rule (M + T >= 1) calls old.", "",
             md(agr_t), "",
             f"Agreement {agr['agreement']:.3f}, Cohen's kappa {agr['kappa']:.3f} (n = {agr['n']}).", "",
             "## 3. Influence (H1, h = 10)", "",
             md(pd.DataFrame([inf]).T.reset_index().set_axis(["quantity", "value"], axis=1)), "",
             "Leave-one-ticker-out, sorted by the size of the change (top 15 shown; full table in loto.csv):", "",
             md(loto.head(15)), "",
             "## 4. The levels behind the difference (every horizon, 1-month bucket, 3% strike)", "",
             "mean_y = mean over events of (event log(RV/IV0) minus the mean over its matched ordinary days). Negative "
             "means realised volatility fell short of implied by more after the filing than on the same ticker's "
             "ordinary days. mean_y_event and mean_y_ordinary are the raw levels behind it. Groups under 30 are "
             "descriptive. Horizons 42 and 63 have no usable rows (the 1-month option expires first).", "",
             md(lev), ""]
    return "\n".join(lines)


# ---- the whole run -----------------------------------------------------------------------------------------
def run(data_dir: Path = cl.DATA_DIR, out_dir: Path | None = None, log: T.Log | None = None, n_boot: int = N_BOOT,
        NB: dict | None = None, zref: Path | None = None) -> dict:
    """Run the four diagnostics on insample, write the tables and diagnostics.md to out_dir (default
    data/oldnews/results_insample/diagnostics/) and append the ledger rows."""
    data_dir = Path(data_dir)
    out_dir = Path(out_dir) if out_dir is not None else data_dir / "results_insample" / "diagnostics"
    inp = load(data_dir, LABEL, NB, zref)
    s = h1_sample(inp, "10")
    d, old = s["d"].to_numpy(float), s["is_old"].to_numpy(bool)
    pw = power(d, old)
    sep = separation(s)
    agr_t, agr = agreement(s)
    loto, inf = influence(s)
    lev = levels(inp, n_boot)
    check = {"effect": float(d[old].mean() - d[~old].mean()), "n_old": int(old.sum()), "n_sur": int((~old).sum())}
    log = log or T.Log(data_dir / "ledger.csv")
    ledger.append(Path(log.path), _ledger_rows(pw, sep, agr, inf, lev, int(s["ticker"].nunique())), log.run_id, {},
                  log.head)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {"power": pw, "separation": sep, "agreement": agr_t, "loto": loto,
              "influence": pd.DataFrame([inf]), "levels": lev}
    for k, v in tables.items():
        v.to_csv(out_dir / f"{k}.csv", index=False)
    (out_dir / "diagnostics.md").write_text(report(pw, sep, agr_t, agr, loto, inf, lev, log.run_id, log.head, check),
                                             encoding="utf-8")
    return {**tables, "agreement_stats": agr, "check": check, "sample": s}
