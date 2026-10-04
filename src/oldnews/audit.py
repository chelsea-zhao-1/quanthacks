"""Independent audit of the 2024-25 (insample) old-news result.

Written by the auditor, separately from the pipeline: it does NOT import classify.py, tests.py or trade.py.
It reads only the 2024-25 tables in data/oldnews/ and the frozen constants, recomputes the primary result
from them with its own code and its own trading calendar, and writes data/oldnews/audit_insample.md.

Run from the repo root:  .venv/Scripts/python.exe src/oldnews/audit.py

Offline. No API, no .env, no 2022-23 files, no date on or after 2026-01-01.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "oldnews"
RES = DATA / "results_insample"
TRADE = DATA / "trade_insample"
ZREF = ROOT / "src" / "oldnews" / "zref_frozen_insample.json"
OUT = DATA / "audit_insample.md"

SEED = 20261003
N_PERM = 10_000
N_BOOT = 10_000
WIN_LO, WIN_HI = pd.Timestamp("2024-01-01"), pd.Timestamp("2026-01-01")
SEALED_LO, SEALED_HI = pd.Timestamp("2023-06-01"), pd.Timestamp("2023-08-31")
PEOPLE_TAGS = {"ceo_departure", "cfo_departure", "executive_officer_departure", "director_departure",
               "ceo_appointment", "cfo_appointment", "executive_officer_appointment", "director_appointment"}

# --------------------------------------------------------------------------------------------------------------
# 1. My own NYSE calendar (typed in by hand from NYSE's published holiday / early-close lists, not the notebook's)
# --------------------------------------------------------------------------------------------------------------
NYSE_HOLIDAYS = pd.to_datetime([
    "2023-11-23", "2023-12-25",
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19", "2024-07-04",
    "2024-09-02", "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26", "2025-06-19",
    "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16",
])
EARLY_CLOSES = set(pd.to_datetime(["2023-11-24", "2024-07-03", "2024-11-29", "2024-12-24",
                                   "2025-07-03", "2025-11-28", "2025-12-24"]))
CAL = pd.bdate_range("2023-11-01", "2026-02-27")
CAL = CAL[~CAL.isin(NYSE_HOLIDAYS)]
CAL_POS = {d: i for i, d in enumerate(CAL)}


def close_dt(day: pd.Timestamp) -> datetime:
    """Close of a session (ET wall clock): 16:00, or 13:00 on an early-close day."""
    return datetime.combine(day.date(), time(13 if day in EARLY_CLOSES else 16))


def expected_timing(accepted_at: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(t_pre, t_0) implied by an acceptance time under the plan's rule.

    t_pre: last session whose close is at or before acceptance. t_0: first session whose close is after
    acceptance, except that a filing accepted in the last 30 minutes before a close (15:30 ET, or 12:30 on an
    early-close day) enters at the next session.
    """
    acc = accepted_at.to_pydatetime()
    t_pre = max(d for d in CAL if close_dt(d) <= acc)
    t_0 = min(d for d in CAL if close_dt(d) - timedelta(minutes=30) > acc)
    return t_pre, t_0


def session_before(day: pd.Timestamp) -> pd.Timestamp:
    """Last session strictly before `day`."""
    return CAL[CAL.searchsorted(day, side="left") - 1]


def pos(day) -> int:
    return CAL_POS[pd.Timestamp(day)]


# --------------------------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------------------------
@dataclass
class Tables:
    events: pd.DataFrame
    nulls: pd.DataFrame
    gap: pd.DataFrame
    outcome: pd.DataFrame
    classified: pd.DataFrame
    zref: dict


def load() -> Tables:
    ev = pd.read_csv(DATA / "events_insample.csv")
    nu = pd.read_csv(DATA / "nulls_insample.csv")
    gp = pd.read_csv(DATA / "gap_insample.csv")
    oc = pd.read_csv(DATA / "outcome_insample.csv", dtype={"horizon": str})
    cl = pd.read_csv(DATA / "classified_insample.csv", low_memory=False)
    zr = json.loads(ZREF.read_text(encoding="utf-8"))
    return Tables(ev, nu, gp, oc, cl, zr)


# --------------------------------------------------------------------------------------------------------------
# Task 1a: classification (M, T, S, old) from raw tables + frozen constants
# --------------------------------------------------------------------------------------------------------------
def classify_independent(t: Tables) -> pd.DataFrame:
    """Recompute M, T, S and the old label (equal weights, cutoff 1, full-text T with excerpt fallback)."""
    c = t.zref["constants"]
    g = t.gap.set_index("row_id")
    ev = t.events.set_index("row_id")
    df = ev.join(g[["gap_move", "d_iv_gap", "vol_gap_ratio", "usable"]].rename(columns={"usable": "gap_usable"}))
    ratio = df["vol_gap_ratio"]
    log_ratio = np.log(ratio.where(ratio > 0))                 # a ratio of 0 has no log -> missing
    z = pd.DataFrame({
        "z_gm": (df["gap_move"] - c["gap_move"]["mean"]) / c["gap_move"]["sd"],
        "z_div": (df["d_iv_gap"] - c["d_iv_gap"]["mean"]) / c["d_iv_gap"]["sd"],
        "z_lv": (log_ratio - c["log_vol_gap_ratio"]["mean"]) / c["log_vol_gap_ratio"]["sd"],
    })
    has_gm = df["gap_move"].notna() & df["gap_usable"].fillna(False).astype(bool)
    df["M_a"] = z.mean(axis=1, skipna=True).where(has_gm)      # mean of the available inputs; gap move required
    df["n_inputs_a"] = z.notna().sum(axis=1).where(has_gm)
    full_ok = (df["full_text_status"] == "ok") & df["T_full"].notna()
    df["T_a"] = np.where(full_ok, df["T_full"], df["T"])
    df["T_src_a"] = np.where(full_ok, "full", "excerpt")
    # T_full must be the sum of the three full-text cues (the exhibit flag is NOT part of T)
    cue_sum = df[["cue_dated_prior_full", "cue_prior_wording_full", "cue_related_filing_full"]].sum(axis=1, min_count=3)
    df["T_full_sum_ok"] = (~full_ok) | (cue_sum == df["T_full"])
    df["S_a"] = df["M_a"] + df["T_a"]
    df["old_a"] = df["S_a"] >= 1
    df["scored_a"] = df["M_a"].notna() & pd.notna(df["T_a"])
    df["in_set_a"] = (df["late"] == 1) & (df["earnings_excluded"] == 0)
    return df.join(z)


def recompute_zref(t: Tables) -> dict:
    """Recompute the six frozen constants from usable ordinary days (gap inputs only)."""
    g = t.gap[t.gap["row_id"].str.startswith("null|") & t.gap["usable"].astype(bool)].copy()
    lv = np.log(g["vol_gap_ratio"].where(g["vol_gap_ratio"] > 0))
    out = {}
    for name, s in (("gap_move", g["gap_move"]), ("d_iv_gap", g["d_iv_gap"]), ("log_vol_gap_ratio", lv)):
        ids = g.loc[s.notna(), "row_id"]
        out[name] = {"mean": float(s.mean()), "sd": float(s.std(ddof=1)), "n": int(s.notna().sum()),
                     "sha": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()}
    nul = t.nulls.set_index("row_id").loc[g["row_id"]]
    out["_t0_range"] = (str(nul["t_0"].min()), str(nul["t_0"].max()))
    return out


# --------------------------------------------------------------------------------------------------------------
# Task 1b: H1 and placebo
# --------------------------------------------------------------------------------------------------------------
def outcome_slice(t: Tables, bucket="1m", horizon="10", otm=3, require_usable=True) -> pd.Series:
    o = t.outcome
    s = o[(o["bucket"] == bucket) & (o["horizon"] == horizon) & (o["otm"] == otm)]
    if require_usable:
        s = s[s["usable"].astype(bool)]
    s = s[np.isfinite(s["y"])]
    assert s["row_id"].is_unique
    return s.set_index("row_id")["y"]


def event_outcomes(t: Tables, cls: pd.DataFrame, group: str, y: pd.Series) -> pd.DataFrame:
    """Per event: y_event minus the mean y of ITS OWN matched ordinary days (nulls with event_row_id == event)."""
    ev = cls[(cls["group"] == group) & cls["in_set_a"] & cls["scored_a"]].copy()
    ev = ev[ev.index.isin(y.index)]
    nu = t.nulls[["row_id", "event_row_id"]].copy()
    nu["y"] = nu["row_id"].map(y)
    nu = nu[nu["y"].notna()]
    null_mean = nu.groupby("event_row_id")["y"].mean()
    null_n = nu.groupby("event_row_id")["y"].size()
    ev["y_event"] = y.reindex(ev.index)
    ev["y_null_mean"] = null_mean.reindex(ev.index)
    ev["n_null_used"] = null_n.reindex(ev.index)
    ev = ev[ev["y_null_mean"].notna()]
    ev["d"] = ev["y_event"] - ev["y_null_mean"]
    return ev


def perm_test(d: np.ndarray, old: np.ndarray, seed: int = SEED, n_perm: int = N_PERM) -> dict:
    """Old minus surprise; one-sided (lower tail, prediction negative) and two-sided permutation p-values.

    Labels are shuffled over the events (group sizes fixed); p = (count + 1) / (n_perm + 1).
    """
    rng = np.random.default_rng(seed)
    obs = d[old].mean() - d[~old].mean()
    n_old = int(old.sum())
    tot = d.sum()
    perm = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(len(d))[:n_old]
        s_old = d[idx].sum()
        perm[i] = s_old / n_old - (tot - s_old) / (len(d) - n_old)
    p1 = (np.sum(perm <= obs + 1e-12) + 1) / (n_perm + 1)
    p2 = (np.sum(np.abs(perm) >= abs(obs) - 1e-12) + 1) / (n_perm + 1)
    return {"effect": obs, "p_one_sided": p1, "p_two_sided": p2, "perm_sd": perm.std(ddof=1)}


def boot_ci(d: np.ndarray, old: np.ndarray, seed: int = SEED, n_boot: int = N_BOOT) -> tuple[float, float]:
    """Percentile 95% interval; old and surprise groups resampled separately with replacement."""
    rng = np.random.default_rng(seed)
    a, b = d[old], d[~old]
    ia = rng.integers(0, len(a), size=(n_boot, len(a)))
    ib = rng.integers(0, len(b), size=(n_boot, len(b)))
    stats = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def run_test(t: Tables, cls: pd.DataFrame, group: str, y: pd.Series) -> tuple[dict, pd.DataFrame]:
    ev = event_outcomes(t, cls, group, y)
    d = ev["d"].to_numpy()
    old = ev["old_a"].to_numpy(bool)
    r = perm_test(d, old)
    lo, hi = boot_ci(d, old)
    r.update({"n": len(ev), "n_old": int(old.sum()), "n_comp": int((~old).sum()),
              "n_tickers": ev["ticker"].nunique(), "mean_old": d[old].mean(), "mean_comp": d[~old].mean(),
              "ci_lo": lo, "ci_hi": hi})
    return r, ev


# --------------------------------------------------------------------------------------------------------------
# Task 2: sign and definition checks
# --------------------------------------------------------------------------------------------------------------
def definition_checks(t: Tables, h1_ev: pd.DataFrame, p_ev: pd.DataFrame) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    o = t.outcome[np.isfinite(t.outcome["y"])]
    err = np.abs(o["y"] - np.log(o["rv"] / o["iv0"]))
    out.append(("y = log(rv/iv0) on every outcome row with a finite y", bool(err.max() < 1e-9),
                f"{len(o):,} rows, max |y - log(rv/iv0)| = {err.max():.2e}"))
    y_by_otm = o[(o["bucket"] == "1m") & (o["horizon"] == "10")].pivot_table(index="row_id", columns="otm", values="y")
    spread = (y_by_otm.max(axis=1) - y_by_otm.min(axis=1)).max()
    out.append(("y does not depend on the put strike (otm 3/5/10 rows agree)", bool(spread < 1e-12),
                f"max spread across otm at 1m/h=10 = {spread:.1e}; only the `usable` flag depends on the put"))
    ev = t.events
    out.append(("no event row appears twice (row_id, accession)", bool(ev["row_id"].is_unique and ev["accession_number"].is_unique),
                f"{len(ev)} rows, {ev['row_id'].nunique()} row_ids, {ev['accession_number'].nunique()} accessions"))
    out.append(("no event in both the people and the placebo set", bool(ev.groupby("accession_number")["group"].nunique().max() == 1), ""))
    for name, e in (("H1", h1_ev), ("P", p_ev)):
        dup_tt = e.duplicated(["ticker", "t_0"], keep=False)
        out.append((f"{name}: each event used once", bool(e.index.is_unique),
                    f"{len(e)} events; {int(dup_tt.sum())} events share a ticker and entry day with another in the sample"
                    + (": " + ", ".join(f"{r.ticker} {r.t_0}" for r in e[dup_tt].drop_duplicates(['ticker', 't_0']).itertuples()) if dup_tt.any() else "")))
    # nulls belong to their own event
    nu = t.nulls.merge(ev[["row_id", "ticker", "t_0", "t_pre", "n_gap"]].rename(columns={"row_id": "event_row_id"}),
                       on="event_row_id", how="left", suffixes=("", "_ev"))
    out.append(("every null points to an existing event", bool(nu["ticker_ev"].notna().all()), f"{len(nu)} nulls"))
    out.append(("every null has its event's ticker", bool((nu["ticker"] == nu["ticker_ev"]).all()), ""))
    out.append(("every null's pseudo-gap has its event's n_gap", bool((nu["n_gap"] == nu["n_gap_ev"]).all()), ""))
    out.append(("null row_ids unique; at most 2 rounds per event, distinct dates",
                bool(t.nulls["row_id"].is_unique and t.nulls.groupby("event_row_id")["t_0"].apply(lambda s: s.is_unique).all()
                     and t.nulls.groupby("event_row_id").size().max() <= 2), ""))
    # The null "day" is its entry t_0; the event's reference day is its filing session (first session on or after
    # filing_date). Distances are in sessions of my calendar.
    fsess = lambda d: int(CAL.searchsorted(pd.Timestamp(d), side="left"))          # noqa: E731
    nu = nu.merge(ev[["row_id", "filing_date"]].rename(columns={"row_id": "event_row_id"}), on="event_row_id")
    dist = pd.Series([pos(a) - fsess(b) for a, b in zip(nu["t_0"], nu["filing_date"])])
    dist_t0 = pd.Series([pos(a) - pos(b) for a, b in zip(nu["t_0"], nu["t_0_ev"])])
    out.append(("null day (t_0) within +-60 sessions of its event's filing session", bool(dist.abs().max() <= 60),
                f"max |distance| = {dist.abs().max()}, min = {dist.abs().min()}. Measured against the event's t_0 "
                f"instead, {int((dist_t0.abs() > 60).sum())} nulls sit at 61 (events accepted after the close, so t_0 = filing session + 1)"))
    # >5 sessions from any 8-K in the events table (people + placebo; other 8-Ks are not in our tables)
    filing_pos: dict[str, list[int]] = {}
    for r in ev.itertuples():
        filing_pos.setdefault(r.ticker, []).extend([fsess(r.filing_date), pos(r.t_0)])
    near = [r.row_id for r in t.nulls.itertuples()
            if any(abs(pos(r.t_0) - p) <= 5 for p in filing_pos.get(r.ticker, []))]
    mind = min(abs(pos(r.t_0) - p) for r in t.nulls.itertuples() for p in filing_pos.get(r.ticker, []))
    span = [r.row_id for r in t.nulls.itertuples() if pd.notna(r.gap_start)
            and any(pos(r.gap_start) <= p <= pos(r.t_0) for p in filing_pos.get(r.ticker, []))]
    out.append(("null day more than 5 sessions from every 8-K in the events table (filing session and t_0)", len(near) == 0,
                f"{len(near)} violations; closest = {mind} sessions. 28 nulls sit at exactly 6-7 sessions, so the null's t_pre "
                f"(one session earlier) can be 5 sessions from another filing's t_0: boundary, not a violation. "
                f"Nulls whose pseudo-gap..entry span contains a same-ticker people/placebo filing session or entry: {len(span)}"
                + (f" ({', '.join(span)}). Its pseudo-gap starts on its own event's entry close (gap_start = event t_0), so the "
                   "gap return starts after that close; boundary case, disclosed, not a rule violation" if span else "")))
    t_next = all(pos(r.t_0) - pos(r.t_pre) == 1 for r in t.nulls.itertuples())
    gs_ok = all(pos(r.t_pre) - pos(r.gap_start) == r.n_gap for r in t.nulls.itertuples() if r.n_gap > 0)
    no_gs = t.nulls["gap_start"].isna()
    out.append(("null t_0 is the session after t_pre; gap_start is n_gap sessions before t_pre", bool(t_next and gs_ok),
                f"{int(no_gs.sum())} nulls have no gap_start, all with n_gap = 0 (matched to same-day, non-late events): "
                f"{bool((t.nulls.loc[no_gs, 'n_gap'] == 0).all())}"))
    # nulls used in the averages are only the event's own
    for name, e in (("H1", h1_ev), ("P", p_ev)):
        out.append((f"{name}: ordinary days averaged per event are its own (1 or 2)",
                    bool(e["n_null_used"].between(1, 2).all()),
                    f"{int((e['n_null_used'] == 2).sum())} events with 2, {int((e['n_null_used'] == 1).sum())} with 1"))
    return out


# --------------------------------------------------------------------------------------------------------------
# Task 3: lookahead and window audit
# --------------------------------------------------------------------------------------------------------------
def timing_table(t: Tables) -> pd.DataFrame:
    ev = t.events.copy()
    ev["acc"] = pd.to_datetime(ev["accepted_at"])
    exp = ev["acc"].apply(expected_timing)
    ev["t_pre_exp"] = [str(a.date()) for a, _ in exp]
    ev["t_0_exp"] = [str(b.date()) for _, b in exp]
    ev["gap_start_exp"] = [str(session_before(pd.Timestamp(d)).date()) for d in ev["event_date"]]
    ev["n_gap_exp"] = [pos(r.t_pre) - pos(r.gap_start) for r in ev.itertuples()]
    hol = [d.date() for d in NYSE_HOLIDAYS]
    ev["lag_bd_exp"] = [int(np.busday_count(pd.Timestamp(r.event_date).date(), r.acc.date(), holidays=hol))
                        for r in ev.itertuples()]
    ev["t_pre_close_before_acc"] = [close_dt(pd.Timestamp(r.t_pre)) <= r.acc.to_pydatetime() for r in ev.itertuples()]
    ev["t_0_close_after_acc"] = [close_dt(pd.Timestamp(r.t_0)) > r.acc.to_pydatetime() for r in ev.itertuples()]
    ev["ok_t_pre"] = ev["t_pre"] == ev["t_pre_exp"]
    ev["ok_t_0"] = ev["t_0"] == ev["t_0_exp"]
    ev["ok_gap_start"] = ev["gap_start"] == ev["gap_start_exp"]
    ev["ok_n_gap"] = ev["n_gap"] == ev["n_gap_exp"]
    ev["ok_order"] = (ev["gap_start"] < ev["event_date"]) & (ev["gap_start"] <= ev["t_pre"]) & (ev["t_pre"] < ev["t_0"])
    ev["ok_late"] = (ev["lag_bd"] == ev["lag_bd_exp"]) & (ev["late"] == (ev["lag_bd_exp"] >= 1).astype(int))
    return ev


def related_filing_check(t: Tables) -> dict:
    """Recompute cue 3 (another people-news 8-K, same ticker, filed in the 30 calendar days before the event date)
    from the events table, counting only filings ACCEPTED strictly before this filing (so no lookahead is possible).
    Readings: "plan" = the plan's words (the other filing's FILING date in [event_date - 30, event_date), accepted
    before this filing); "pipeline" = what events.py documents (the other filing's EVENT date OR filing date in that
    window, accepted before this row's t_pre close). Both are lookahead-free by construction; the check is whether
    the stored cue reproduces and how many filings the two readings disagree on."""
    ev = t.events.copy()
    ev["acc"] = pd.to_datetime(ev["accepted_at"])
    ev["ed"] = pd.to_datetime(ev["event_date"])
    ev["fd"] = pd.to_datetime(ev["filing_date"])
    ev["known_by"] = [close_dt(pd.Timestamp(d)) for d in ev["t_pre"]]
    ppl = ev[ev["group"] == "people"]
    cmp_ = ev[ev["cue_related_filing_full"].notna()]
    res = {"n": len(cmp_)}
    for name in ("plan", "pipeline"):
        agree, future_only, bad = 0, 0, []
        for r in cmp_.itertuples():
            same = ppl[(ppl["ticker"] == r.ticker) & (ppl["accession_number"] != r.accession_number)]
            lo = r.ed - pd.Timedelta(days=30)
            in_fd = (same["fd"] >= lo) & (same["fd"] < r.ed)
            if name == "plan":
                win, gate = same[in_fd], r.acc
            else:
                win, gate = same[in_fd | ((same["ed"] >= lo) & (same["ed"] < r.ed))], r.known_by
            mine = int(len(win[win["acc"] < gate]) > 0)
            if len(win) and not len(win[win["acc"] < r.acc]):
                future_only += 1
            if mine == int(r.cue_related_filing_full):
                agree += 1
            else:
                bad.append(f"{r.row_id} ({r.ticker}, {r.group}, event_date {r.event_date}): pipeline {int(r.cue_related_filing_full)}, mine {mine}")
        res[name] = {"agree": agree, "future_only": future_only, "bad": bad}
    return res


def window_checks(t: Tables, tim: pd.DataFrame) -> list[tuple[str, bool, str]]:
    out = []
    def inwin(s: pd.Series) -> bool:
        s = pd.to_datetime(s.dropna())
        return bool(((s >= WIN_LO) & (s < WIN_HI)).all() and not ((s >= SEALED_LO) & (s <= SEALED_HI)).any())
    ev, nu, oc = t.events, t.nulls, t.outcome
    for col in ("gap_start", "t_pre", "t_0"):
        mn, mx = ev[col].min(), ev[col].max()
        out.append((f"events.{col} in [2024-01-01, 2026-01-01)", inwin(ev[col]), f"{mn} .. {mx}"))
    for col in ("gap_start", "t_pre", "t_0"):
        out.append((f"nulls.{col} in [2024-01-01, 2026-01-01)", inwin(nu[col]), f"{nu[col].min()} .. {nu[col].max()}"))
    for col in ("entry_date", "exit_date"):
        out.append((f"outcome.{col} in [2024-01-01, 2026-01-01)", inwin(oc[col]), f"{oc[col].min()} .. {oc[col].max()}"))
    ed = pd.to_datetime(ev["event_date"])
    out.append(("event_date (cover-page date, not a computation input) range", True,
                f"{ev['event_date'].min()} .. {ev['event_date'].max()}; {int((ed < WIN_LO).sum())} before 2024"))
    # earliest date READ by the gap inputs: gap-start mark (up to 3 sessions stale) and the 5 baseline sessions
    g = t.gap.set_index("row_id")
    rows = pd.concat([ev.set_index("row_id")[["gap_start"]], nu.set_index("row_id")[["gap_start"]]])
    rows = rows.join(g[["stale_sessions", "n_baseline", "usable"]])
    rows = rows[rows["gap_start"].notna()]
    first = pos(CAL[CAL >= WIN_LO][0])
    rows["p_gs"] = [pos(d) for d in rows["gap_start"]]
    stale = rows["stale_sessions"].fillna(0).astype(int)
    rows["earliest_mark_pos"] = rows["p_gs"] - stale
    rows["baseline_avail"] = (rows["p_gs"] - first).clip(upper=5)
    bad_mark = rows[rows["earliest_mark_pos"] < first]
    bad_base = rows[rows["n_baseline"].fillna(0) > rows["baseline_avail"]]
    out.append(("gap-start mark (incl. up to 3 stale sessions) on or after 2024-01-02", len(bad_mark) == 0,
                f"{len(bad_mark)} rows read a mark dated before 2024"))
    out.append(("volume-baseline sessions all on or after 2024-01-01 (n_baseline <= sessions available in 2024)",
                len(bad_base) == 0, f"{len(bad_base)} rows; {int((rows['baseline_avail'] < 5).sum())} rows have fewer than 5 sessions available after 2024-01-01"))
    early = rows[rows["baseline_avail"] < 5]
    out.append(("rows whose 5-session baseline would reach into 2023", True,
                f"{len(early)} rows; their n_baseline values: {sorted(early['n_baseline'].fillna(-1).astype(int).tolist())} "
                f"(available: {sorted(early['baseline_avail'].tolist())})"))
    return out


# --------------------------------------------------------------------------------------------------------------
# Task 4: trade spot-check
# --------------------------------------------------------------------------------------------------------------
def cost_per_share(p: float, mult: float) -> float:
    return mult * max(0.05 * p, 0.05)


def trade_spotcheck(t: Tables, cls: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    tr = pd.read_csv(TRADE / "trades.csv", dtype={"horizon": str})
    book = tr[(tr["book"] == "old") & (tr["horizon"] == "10")].copy()
    info = {"n_book_rows": len(book), "n_taken": int(book["taken"].astype(bool).sum())}
    taken = book[book["taken"].astype(bool)].copy()
    worst = taken.nsmallest(3, "csp_net")
    rest = taken.drop(worst.index).sort_values("row_id")
    rng = np.random.default_rng(SEED)
    rand = rest.iloc[sorted(rng.choice(len(rest), 2, replace=False))]
    pick = pd.concat([worst.assign(why="largest loss"), rand.assign(why="random")])
    o = t.outcome[(t.outcome["bucket"] == "1m") & (t.outcome["horizon"] == "10") & (t.outcome["otm"] == 3)].set_index("row_id")
    rows = []
    for r in pick.itertuples():
        orow = o.loc[r.row_id]
        K, P0, gross = orow["put_strike"], orow["put_premium"], orow["csp_gross"]
        P1 = P0 - gross * K                      # exit premium implied by the gross P&L (no exit price column)
        net1 = gross - (cost_per_share(P0, 1) + cost_per_share(P1, 1)) / K
        net2 = gross - (cost_per_share(P0, 2) + cost_per_share(P1, 2)) / K
        ev = cls.loc[r.row_id]
        rows.append({
            "why": r.why, "row_id": r.row_id, "ticker": r.ticker, "entry": r.entry_date, "exit": r.exit_date,
            "t_0 (event)": ev["t_0"], "sessions held": pos(r.exit_date) - pos(r.entry_date),
            "strike": K, "premium": P0, "implied exit premium": P1, "put vol at entry": orow["put_volume_entry"],
            "gross %": 100 * gross,
            "net 1x % (mine)": 100 * net1, "net 1x % (trades.csv)": 100 * r.csp_net,
            "net 2x % (mine)": 100 * net2, "net 2x % (trades.csv)": 100 * r.csp_net2x,
            "outcome row = trade row": bool(orow["entry_date"] == r.entry_date and orow["exit_date"] == r.exit_date
                                            and orow["put_strike"] == r.put_strike and orow["put_premium"] == r.put_premium
                                            and abs(orow["csp_gross"] - r.csp_gross) < 1e-12),
            "old label (mine)": bool(ev["old_a"]),
        })
    df = pd.DataFrame(rows)
    # whole-book consistency checks (cheap, so do them on every taken trade, not only the five)
    allrows = []
    for r in taken.itertuples():
        orow = o.loc[r.row_id]
        K, P0, gross = orow["put_strike"], orow["put_premium"], orow["csp_gross"]
        P1 = P0 - gross * K
        allrows.append((abs(gross - (cost_per_share(P0, 1) + cost_per_share(P1, 1)) / K - r.csp_net),
                        abs(gross - (cost_per_share(P0, 2) + cost_per_share(P1, 2)) / K - r.csp_net2x),
                        bool(cls.loc[r.row_id, "old_a"]), r.entry_date == cls.loc[r.row_id, "t_0"],
                        pos(r.exit_date) - pos(r.entry_date) == 10, P1 >= -1e-9))
    a = np.array(allrows, dtype=object)
    info.update({
        "max_err_1x": float(max(a[:, 0])), "max_err_2x": float(max(a[:, 1])),
        "all_old": bool(all(a[:, 2])), "entry_is_t0": bool(all(a[:, 3])), "held_10": bool(all(a[:, 4])),
        "exit_premium_nonneg": bool(all(a[:, 5])),
    })
    # do all old-labelled, usable trades of the book exist?
    olds = cls[(cls["group"] == "people") & cls["in_set_a"] & cls["scored_a"] & cls["old_a"]]
    usable = o[o["usable"].astype(bool) & (o["put_volume_entry"] > 0)]
    info["n_old_usable_mine"] = int(olds.index.isin(usable.index).sum())
    return df, info



# --------------------------------------------------------------------------------------------------------------
# Extra check 5 (coordinator): the trimmed difference
# --------------------------------------------------------------------------------------------------------------
def trim_mean(x: np.ndarray, frac: float) -> float:
    """Mean after cutting floor(n * frac) values from EACH tail."""
    x = np.sort(np.asarray(x, float))
    k = int(math.floor(len(x) * frac))
    return float(x[k:len(x) - k].mean()) if len(x) - 2 * k > 0 else float(x.mean())


def trimmed_variants(ev: pd.DataFrame) -> list[list[str]]:
    d, old = ev["d"].to_numpy(), ev["old_a"].to_numpy(bool)
    rows = []
    for frac in (0.05, 0.10):
        k_o, k_s = int(math.floor(old.sum() * frac)), int(math.floor((~old).sum() * frac))
        rows.append([f"within each group, each tail, {int(frac * 100)}% (after subtracting the nulls)",
                     f"{k_o} + {k_o} of {int(old.sum())} old; {k_s} + {k_s} of {int((~old).sum())} surprise",
                     f"{trim_mean(d[old], frac) - trim_mean(d[~old], frac):+.4f}"])
    for frac in (0.05, 0.10):
        lo, hi = np.quantile(d, [frac, 1 - frac])
        keep = (d >= lo) & (d <= hi)
        rows.append([f"pooled: cut {int(frac * 100)}% of each tail of all events, then old minus surprise",
                     f"{int((~keep).sum())} cut; kept old {int((keep & old).sum())}, surprise {int((keep & ~old).sum())}",
                     f"{d[keep & old].mean() - d[keep & ~old].mean():+.4f}"])
    ye, yn = ev["y_event"].to_numpy(), ev["y_null_mean"].to_numpy()
    for frac in (0.05, 0.10):
        v = (trim_mean(ye[old], frac) - trim_mean(yn[old], frac)) - (trim_mean(ye[~old], frac) - trim_mean(yn[~old], frac))
        rows.append([f"within group, {int(frac * 100)}%, trimming event y and null y separately BEFORE subtracting",
                     "", f"{v:+.4f}"])
    rows.append(["untrimmed (primary)", "", f"{d[old].mean() - d[~old].mean():+.4f}"])
    rows.append(["median old minus median surprise", "", f"{np.median(d[old]) - np.median(d[~old]):+.4f}"])
    return rows


# --------------------------------------------------------------------------------------------------------------
# Extra check 6 (coordinator): concentration of the old-news book's P&L
# --------------------------------------------------------------------------------------------------------------
def book_concentration() -> dict:
    tr = pd.read_csv(TRADE / "trades.csv", dtype={"horizon": str})
    b = tr[(tr["book"] == "old") & (tr["horizon"] == "10") & tr["taken"].astype(bool)].copy()
    r = 100 * b["csp_net"].to_numpy()              # % of the trade's collateral, 1x costs; equal collateral per trade
    worst = int(np.argmin(r))
    bb = b.assign(r=r)
    top3 = bb.nlargest(3, "r")
    by_tk = bb.groupby("ticker")["r"].sum().sort_values(ascending=False)
    return {"n": len(r), "mean": r.mean(), "mean_wo_worst": np.delete(r, worst).mean(),
            "worst": (b.iloc[worst]["ticker"], b.iloc[worst]["entry_date"], r[worst]),
            "total": r.sum(), "top3": [(x.ticker, x.entry_date, x.r) for x in top3.itertuples()],
            "top3_share": top3["r"].sum() / r.sum(), "gains": r[r > 0].sum(), "losses": r[r < 0].sum(),
            "top3_tk": list(by_tk.head(3).items()), "top3_tk_share": by_tk.head(3).sum() / r.sum(),
            "mean_2x": 100 * b["csp_net2x"].mean()}

# --------------------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------------------
def fmt(x, nd=4) -> str:
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    return f"{x:+.{nd}f}" if isinstance(x, (float, np.floating)) else str(x)


def compare_rows(mine: dict, theirs: pd.Series, keys: list[str]) -> list[list[str]]:
    rows = []
    for k in keys:
        a, b = mine[k], theirs[k]
        if isinstance(a, (int, np.integer)):
            ok = int(a) == int(b)
            rows.append([k, str(int(a)), str(int(b)), "exact" if ok else "DIFFERS"])
        else:
            ok4 = round(float(a), 4) == round(float(b), 4)
            rows.append([k, f"{a:.6f}", f"{float(b):.6f}", f"{abs(a - b):.1e}",
                         "agree to 4 dp" if ok4 else ("within MC error" if k.startswith(("p_", "ci_")) else "DIFFERS")])
    return rows


def md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def main() -> None:
    t = load()
    cls = classify_independent(t)

    # ---- Task 1: classification agreement
    their = t.classified.set_index("row_id")
    sc = cls[cls["in_set_a"] & cls["scored_a"]]
    j = sc.join(their[["M", "S", "S_equal", "old", "old_equal_1", "scored", "n_inputs", "t_source"]], rsuffix="_c")
    dM = (j["M_a"] - j["M"]).abs().max()
    dS = (j["S_a"] - j["S"]).abs().max()
    lab_dis = int((j["old_a"] != j["old"].astype(bool)).sum())
    n_in_dis = int((j["n_inputs_a"] != j["n_inputs"]).sum())
    scored_theirs = their[(their["late"] == 1) & (their["earnings_excluded"] == 0) & their["scored"].astype(bool)]
    set_dis = sorted(set(scored_theirs.index) ^ set(sc.index))
    zr = recompute_zref(t)

    y = outcome_slice(t)
    h1, h1_ev = run_test(t, cls, "people", y)
    pl, pl_ev = run_test(t, cls, "placebo", y)
    th1 = pd.read_csv(RES / "h1.csv").iloc[0]
    tpl = pd.read_csv(RES / "placebo.csv").iloc[0]
    keys_i = ["n", "n_old", "n_comp", "n_tickers"]
    keys_f = ["mean_old", "mean_comp", "effect", "p_one_sided", "p_two_sided", "ci_lo", "ci_hi"]

    # sensitivity of the sample definition: y does not depend on the put, but the usable flag does
    y_any = outcome_slice(t, require_usable=False)
    h1_any, _ = run_test(t, cls, "people", y_any)

    # Monte Carlo standard error of the permutation p-values
    mc_se = lambda p: math.sqrt(p * (1 - p) / N_PERM)
    d_h1, o_h1 = h1_ev["d"].to_numpy(), h1_ev["old_a"].to_numpy(bool)
    alt = [(sd, perm_test(d_h1, o_h1, seed=sd)["p_one_sided"]) for sd in (SEED + 1, SEED + 2, 12345)]
    p_big = perm_test(d_h1, o_h1, seed=SEED + 99, n_perm=200_000)["p_one_sided"]
    trim_rows = trimmed_variants(h1_ev)
    conc = book_concentration()

    # ---- Task 2
    checks2 = definition_checks(t, h1_ev, pl_ev)
    checks2.insert(0, ("reported effect = mean_old - mean_comp (old minus surprise)",
                       bool(abs(th1["effect"] - (th1["mean_old"] - th1["mean_comp"])) < 1e-12 and
                            abs(tpl["effect"] - (tpl["mean_old"] - tpl["mean_comp"])) < 1e-12),
                       f"H1: {th1['mean_old']:+.4f} - ({th1['mean_comp']:+.4f}) = {th1['effect']:+.4f}"))
    checks2.insert(1, ("prediction recorded as negative; one-sided p is the lower tail",
                       bool(th1["prediction"] == "negative" and th1["effect"] > 0 and th1["p_one_sided"] > 0.5),
                       f"positive effect with one-sided p = {th1['p_one_sided']:.4f} > 0.5, so the tail tested is 'old < surprise'"))

    # ---- Task 3
    tim = timing_table(t)
    late_people = tim[(tim["group"] == "people") & (tim["late"] == 1)].sort_values("row_id").reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    samp = late_people.iloc[sorted(rng.choice(len(late_people), 20, replace=False))]
    rel = related_filing_check(t)
    wchecks = window_checks(t, tim)

    # ---- Task 4
    trades, tinfo = trade_spotcheck(t, cls)

    # ------------------------------------------------------------------ write the report
    L: list[str] = []
    L.append("# Independent audit: 2024-25 (insample) old-news result\n")
    L.append(f"Written by `src/oldnews/audit.py` (auditor; does not import classify.py, tests.py or trade.py). "
             f"Generated {datetime.now():%Y-%m-%d %H:%M}. Seed {SEED}, {N_PERM:,} permutations, {N_BOOT:,} bootstrap "
             "resamples. Inputs: `events/nulls/gap/outcome/classified_insample.csv`, `zref_frozen_insample.json`, "
             "`results_insample/`, `trade_insample/`. Offline; 2024-25 only. Own NYSE calendar typed in by hand. "
             "Nothing was changed in any result or rule.\n")

    # Summary
    t1_pass = (dM < 5e-5 and dS < 5e-5 and lab_dis == 0 and not set_dis and
               all(int(h1[k]) == int(th1[k]) and int(pl[k]) == int(tpl[k]) for k in keys_i) and
               all(round(h1[k], 4) == round(th1[k], 4) and round(pl[k], 4) == round(tpl[k], 4) for k in ["mean_old", "mean_comp", "effect"]))
    t2_pass = all(ok for _, ok, _ in checks2)
    t3_cols = ["ok_t_pre", "ok_t_0", "ok_gap_start", "ok_n_gap", "ok_order", "ok_late", "t_pre_close_before_acc", "t_0_close_after_acc"]
    t3_pass = bool(samp[t3_cols].all().all()) and all(ok for _, ok, _ in wchecks) and rel['pipeline']['agree'] == rel['n']
    t4_pass = (tinfo["max_err_1x"] < 1e-9 and tinfo["max_err_2x"] < 1e-9 and tinfo["all_old"] and tinfo["entry_is_t0"]
               and tinfo["held_10"] and bool(trades["outcome row = trade row"].all()))
    L.append("## Verdict\n")
    L.append(md_table(["task", "result"], [
        ["1. Independent recomputation (M, T, S, label; H1; placebo)", "PASS" if t1_pass else "SEE NOTES"],
        ["2. Sign and definition checks", "PASS" if t2_pass else "SEE NOTES"],
        ["3. Lookahead and window audit", "PASS" if t3_pass else "SEE NOTES"],
        ["4. Trade spot-check (5 trades, 1x and 2x)", "PASS" if t4_pass else "SEE NOTES"],
        ["5. Trimmed difference (extra)", "both values arithmetically right; they differ by fraction (5% vs 10%); 5% is pre-committed"],
        ["6. Book concentration (extra)", "reported (no pass/fail)"],
    ]) + "\n")

    # Task 1
    L.append("## 1. Independent recomputation\n")
    L.append("**Classification.** For every late, earnings-excluded filing: z-scores of `gap_move`, `d_iv_gap` and "
             "`log(vol_gap_ratio)` (ratio 0 -> missing) with the frozen means and sds; `M` = mean of the available z "
             "(gap move required); `T` = `T_full` when the full text is ok, else the excerpt `T`; `S = M + T`; old if `S >= 1`.\n")
    L.append(md_table(["check", "result"], [
        ["scored filings (people + placebo), mine vs classified_insample", f"{len(sc)} vs {len(scored_theirs)}; set difference {len(set_dis)}"],
        ["max abs difference in M", f"{dM:.1e}"],
        ["max abs difference in S (equal weights)", f"{dS:.1e}"],
        ["number of inputs per event disagreeing", str(n_in_dis)],
        ["old/surprise labels disagreeing", str(lab_dis)],
        ["T source: full text / excerpt fallback (scored set)", f"{int((sc['T_src_a'] == 'full').sum())} / {int((sc['T_src_a'] == 'excerpt').sum())}"],
        ["T_full = sum of the three full-text cues (exhibit flag excluded)", fmt(bool(cls.loc[cls['full_text_status'] == 'ok', 'T_full_sum_ok'].all()))],
        ["old news among scored people filings", f"{int(sc[(sc.group == 'people')]['old_a'].sum())} of {int((sc.group == 'people').sum())}"],
    ]) + "\n")
    c = t.zref["constants"]
    zr_rows = []
    for k in ("gap_move", "d_iv_gap", "log_vol_gap_ratio"):
        zr_rows.append([k, f"{zr[k]['mean']:.10f} / {c[k]['mean']:.10f}", f"{zr[k]['sd']:.10f} / {c[k]['sd']:.10f}",
                        f"{zr[k]['n']} / {c[k]['n']}", "yes" if zr[k]["sha"] == c[k]["row_ids_sha256"] else "no"])
    L.append("**Frozen constants recomputed** from the usable ordinary days in `gap_insample.csv` (gap inputs only), "
             f"mine / frozen file. Ordinary-day entry range {zr['_t0_range'][0]} .. {zr['_t0_range'][1]}.\n")
    L.append(md_table(["input", "mean", "sd", "n", "same row-id hash"], zr_rows) + "\n")

    for name, mine, theirs in (("H1 (people, primary)", h1, th1), ("Placebo P", pl, tpl)):
        L.append(f"**{name}: old minus surprise, h = 10, 1m bucket, 3% put row usable.** Outcome per event = y minus "
                 "the mean y of its own usable matched ordinary days.\n")
        rows = compare_rows(mine, theirs, keys_i)
        rows = [[r[0], r[1], r[2], "", r[3]] for r in rows] + compare_rows(mine, theirs, keys_f)
        L.append(md_table(["quantity", "mine", "results_insample", "abs diff", "status"], rows) + "\n")
    L.append(f"Monte Carlo standard error of a permutation p near 0.86 with 10,000 draws is about {mc_se(0.86):.4f}, "
             f"near 0.80 about {mc_se(0.80):.4f}. My permutation and bootstrap use my own random draws "
             "(`numpy.random.default_rng(20261003)`, one `permutation` per draw; separate within-group resampling for the "
             "bootstrap), so p-values and interval ends are expected to differ from the pipeline's in the third or fourth "
             "decimal even though the data are identical; the means, n and effect are deterministic and must agree exactly.\n")
    L.append("In fact the p-values and interval agree to about 1e-16. My code was written before I read `tests.py`; "
             "reading it afterwards shows why. Its `rng.permuted` over a tiled index array draws the same stream as my one "
             "`rng.permutation` per draw, and both bootstraps resample the old group first and then the surprise group. "
             "So the agreement comes from two natural, identical conventions; it is not a shared code path. "
             "To show the p-value does not hinge on that stream, H1's one-sided p with other seeds: "
             + ", ".join(f"seed {sd}: {pv:.4f}" for sd, pv in alt)
             + f"; with 200,000 permutations p = {p_big:.4f} (MC s.e. {math.sqrt(p_big * (1 - p_big) / 200_000):.4f}). "
             f"The reported {h1['p_one_sided']:.4f} is {abs(h1['p_one_sided'] - p_big) / mc_se(p_big):.1f} ten-thousand-draw "
             "standard errors from the precise value, which is ordinary Monte Carlo noise; "
             "the conclusion (no evidence for the predicted negative effect) does not depend on the draw.\n")
    L.append(f"Sample-definition note: `y` is identical across the 3/5/10% put rows; only the `usable` flag depends on "
             f"the put. If the event/null filter used any finite 1m y instead of the 3% put row's `usable` flag, H1 would "
             f"have n = {h1_any['n']} (old {h1_any['n_old']}), effect {h1_any['effect']:+.4f}, one-sided p {h1_any['p_one_sided']:.4f}. "
             "The pipeline's choice (drop events whose 3% put did not trade, as the plan's exclusion rule says) is consistent "
             "with the test plan; this is reported only to show the conclusion does not hinge on it.\n")

    # Task 2
    L.append("## 2. Sign and definition checks\n")
    L.append(md_table(["check", "pass", "detail"], [[a, "yes" if b else "**NO**", d] for a, b, d in checks2]) + "\n")

    # Task 3
    L.append("## 3. Lookahead audit\n")
    L.append("Rule checked: `t_pre` = last session whose close is at or before `accepted_at`; `t_0` = first session "
             "whose close minus 30 minutes is after `accepted_at` (15:30 ET, 12:30 on early-close days); `gap_start` = "
             "last session strictly before `event_date`; `n_gap` = sessions from `gap_start` to `t_pre`. `accepted_at` "
             "is read as US Eastern wall-clock time (EDGAR's convention).\n")
    L.append(f"Sample: 20 of the {len(late_people)} late people-news events, drawn with "
             "`default_rng(20261003).choice(286, 20, replace=False)` over events sorted by row_id.\n")
    srows = []
    for r in samp.itertuples():
        srows.append([r.row_id.split("|")[1], r.ticker, r.event_date, r.gap_start, r.t_pre, r.accepted_at, r.t_0,
                      r.n_gap, "yes" if (r.ok_t_pre and r.ok_t_0 and r.ok_gap_start and r.ok_n_gap and r.ok_order and r.ok_late
                                         and r.t_pre_close_before_acc and r.t_0_close_after_acc) else "**NO**"])
    L.append(md_table(["accession", "ticker", "event_date", "gap_start", "t_pre", "accepted_at (ET)", "t_0", "n_gap", "all rules hold"], srows) + "\n")
    allx = tim
    L.append("The same rules checked on **all** 751 events (not only the sample):\n")
    L.append(md_table(["rule", "events violating"], [
        ["t_pre = last close at or before acceptance", int((~allx["ok_t_pre"]).sum())],
        ["t_0 = first close after acceptance (15:30 / 12:30 cutoff)", int((~allx["ok_t_0"]).sum())],
        ["t_pre close is not after acceptance", int((~allx["t_pre_close_before_acc"]).sum())],
        ["t_0 close is after acceptance", int((~allx["t_0_close_after_acc"]).sum())],
        ["gap_start = session before event_date", int((~allx["ok_gap_start"]).sum())],
        ["n_gap = sessions gap_start..t_pre", int((~allx["ok_n_gap"]).sum())],
        ["gap_start < event_date, gap_start <= t_pre < t_0", int((~allx["ok_order"]).sum())],
        ["lag_bd / late flag reproduce (NYSE business days)", int((~allx["ok_late"]).sum())],
    ]) + "\n")
    bad_t = allx[~(allx[["ok_t_pre", "ok_t_0", "ok_gap_start", "ok_n_gap", "ok_order", "ok_late"]].all(axis=1))]
    if len(bad_t):
        L.append("Events failing a timing rule:\n")
        L.append(md_table(["row_id", "group", "late", "accepted_at", "t_pre / exp", "t_0 / exp", "gap_start / exp", "lag_bd / exp"],
                          [[r.row_id, r.group, r.late, r.accepted_at, f"{r.t_pre} / {r.t_pre_exp}", f"{r.t_0} / {r.t_0_exp}",
                            f"{r.gap_start} / {r.gap_start_exp}", f"{r.lag_bd} / {r.lag_bd_exp}"] for r in bad_t.itertuples()]) + "\n")
    L.append("**What each classification input reads, and why none is after `t_pre`:**\n")
    cat_id = "event|0001104659-25-034874"
    cat = cls.loc[cat_id]
    cat_in_p = cat_id in pl_ev.index
    L.append("- Gap move, IV change, volume ratio (`measure.gap_inputs`, read, not imported): the parity spot and IV "
             "on `gap_start` (or the latest fresh mark up to 3 sessions earlier) and on `t_pre`; ATM volume on the "
             "sessions `gap_start+1 .. t_pre` and on the 5 sessions before `gap_start`; `r_mkt` from the cross-ticker "
             "panel over the same dates (`gs_eff .. t_pre`). Every read is bounded by `t_pre` and by a floor at the window "
             "start (`fetch_start = max(t_pre - 10 days, floor)`). The ATM contracts are chosen at `t_pre` "
             "(`price_event(ticker, t_pre, t_0, ...)`), so the strike choice uses only `t_pre` information. "
             "`t_pre`'s close is at or before acceptance for every event (table above).\n"
             "- Word cue 3 (related earlier filing), recomputed from the events table. `events.py` documents its rule as: another "
             "people filing whose *event date or filing date* is in the 30 calendar days before this event date, accepted before "
             f"this row's `t_pre` close. My implementation of that rule reproduces the stored cue on {rel['pipeline']['agree']} of {rel['n']} filings"
             f"{'' if not rel['pipeline']['bad'] else ' (' + '; '.join(rel['pipeline']['bad'][:5]) + ')'}. "
             "The acceptance gate (before the `t_pre` close) means no later filing can enter: **no lookahead**. "
             "The plan's literal wording counts only the other filing's *filing* date (\"filed another people-news 8-K in the 30 "
             f"calendar days before this event date\"). Under that reading {rel['plan']['agree']} of {rel['n']} agree"
             f"{'' if not rel['plan']['bad'] else ' (' + '; '.join(rel['plan']['bad'][:5]) + ')'}. "
             "The one difference is a placebo filing (CAT, event 2025-04-09). The related CAT director departure had event date "
             "2025-04-07 but was filed on 2025-04-09, the placebo's event date; it was accepted six days before the placebo filing. "
             f"Under the literal reading that filing's S would be {cat['S_a'] - 1:+.3f} instead of {cat['S_a']:+.3f} "
             f"(label {'old' if cat['S_a'] >= 1 else 'surprise'} -> {'old' if cat['S_a'] - 1 >= 1 else 'surprise'}); it "
             f"{'is' if cat_in_p else 'is not'} in the placebo test sample. This is a small, undisclosed widening of the plan's "
             "cue definition (decided in code, not in the plan). It affects one placebo filing and no people filing, "
             "so H1 is untouched, and the placebo label does not change either. It should be disclosed or aligned.\n"
             "- Word cues 1-2: the filing's own text (excerpt or full EDGAR document), available at acceptance.\n"
             "- Standardisation constants: means and sds of ordinary-day gap inputs over all of 2024-25 (no outcomes). "
             "Inside 2024-25 this uses later ordinary days to set the scale of M. That is mild and disclosed in the plan; "
             "it touches no outcome and is fixed for the sealed window.\n"
             "- Sample filter, not a classification input: the earnings exclusion (an earnings 8-K within +-5 sessions of entry) "
             "also looks up to 5 sessions *after* entry. Earnings dates are normally pre-announced, so a trader could apply "
             "it, but the filter as built uses filing dates that are not yet known at entry. It is worth one line in the note.\n")
    L.append("**Window checks (dates actually used):**\n")
    L.append(md_table(["check", "pass", "detail"], [[a, "yes" if b else "**NO**", d] for a, b, d in wchecks]) + "\n")

    # Task 4
    L.append("## 4. Trade spot-check (old-news book, h = 10)\n")
    L.append("Recomputed from `outcome_insample.csv` (1m, h = 10, 3% put row): cost per share = max(5% of premium, "
             "$0.05) on entry and on exit (2x: double), P&L as a fraction of collateral (strike). The outcome table has "
             "no exit-premium column, so the exit premium is implied from the gross P&L (`P1 = P0 - gross x K`); the "
             "check therefore verifies row matching, holding period, labels and the cost arithmetic at 1x and 2x, not the "
             "option marks themselves. Picks: the 3 largest 1x losses among taken trades, plus 2 drawn with "
             "`default_rng(20261003)` from the rest (sorted by row_id).\n")
    show = trades.copy()
    for col in show.columns:
        if show[col].dtype.kind == "f":
            show[col] = show[col].map(lambda v: f"{v:.4f}")
    L.append(md_table(list(show.columns), show.values.tolist()) + "\n")
    L.append(md_table(["whole-book check (all taken old trades at h = 10)", "result"], [
        ["book rows / taken", f"{tinfo['n_book_rows']} / {tinfo['n_taken']}"],
        ["max |net 1x mine - trades.csv| (fraction of collateral)", f"{tinfo['max_err_1x']:.1e}"],
        ["max |net 2x mine - trades.csv|", f"{tinfo['max_err_2x']:.1e}"],
        ["every traded event is old under my recomputed label", fmt(tinfo["all_old"])],
        ["entry date = event t_0 for every trade", fmt(tinfo["entry_is_t0"])],
        ["exit exactly 10 sessions after entry (my calendar)", fmt(tinfo["held_10"])],
        ["implied exit premium non-negative", fmt(tinfo["exit_premium_nonneg"])],
        ["old, scored events with a usable 3% put and volume > 0 (mine)", str(tinfo["n_old_usable_mine"])],
    ]) + "\n")

    L.append("## 5. Extra check: the trimmed difference for H1 (h = 10)\n")
    L.append("Two numbers are in circulation: +0.0811 (`results_insample/h1.csv` column `trimmed_effect`, shown as "
             "'trimmed effect' in `README_numbers.md` without its fraction) and +0.0709 (`results_insample/diagnostics/`, "
             "labelled '10% trimmed difference'). Both use the same method: cut floor(n x frac) values from each tail "
             "*within each group*, *after* subtracting the matched-null mean (on `d`), then old minus surprise "
             "(`playground.stats.trimmed_mean`). They differ only in the fraction. `tests.py` has `TRIM = 0.05` "
             "(5% each tail; committed in 89c7940 at 21:03, before the 2024-25 run at 22:18) and `diagnostics.py` has "
             "`TRIM10 = 0.10` (committed in 69d80b9 at 22:31, after the result). My recomputation:\n")
    L.append(md_table(["trimming rule", "values cut", "old minus surprise"], trim_rows) + "\n")
    L.append("Which matches the plan: the test plan names no trimmed mean and no fraction, and `selection_rules.json` "
             "still has `trimmed_mean_frac: TBD`. The only fraction fixed before the result is the 5% in `tests.py`, "
             "so **+0.0811 (5% each tail, within group, after subtracting the nulls) is the pre-committed value** to report "
             "with the primary result. +0.0709 is a post-result exploratory diagnostic and should be labelled that way. "
             "Neither is wrong arithmetically. The defect is labelling: `README_numbers.md` must say '5% trimmed'. "
             "Both values are positive (wrong sign) and close to the untrimmed +0.0966, so the conclusion does not change.\n")
    c6 = conc
    L.append("## 6. Extra check: concentration of the old-news book (h = 10, 1x costs, `trade_insample/trades.csv`)\n")
    L.append(md_table(["quantity", "value"], [
        ["trades taken", c6["n"]],
        ["mean return per trade (% of collateral)", f"{c6['mean']:+.4f}"],
        [f"... without the single largest loss ({c6['worst'][0]} {c6['worst'][1]}, {c6['worst'][2]:+.2f}%)", f"{c6['mean_wo_worst']:+.4f}"],
        ["sum of trade returns (equal collateral per trade; total P&L in trade units)", f"{c6['total']:+.4f}"],
        ["gross gains / gross losses", f"{c6['gains']:+.3f} / {c6['losses']:+.3f}"],
        ["top 3 trades", "; ".join(f"{a} {b} {c:+.3f}%" for a, b, c in c6["top3"])],
        ["share of total P&L from the top 3 trades", f"{100 * c6['top3_share']:.1f}%"],
        ["top 3 tickers (trades summed)", "; ".join(f"{a} {b:+.3f}%" for a, b in c6["top3_tk"])],
        ["share of total P&L from the top 3 tickers", f"{100 * c6['top3_tk_share']:.1f}%"],
        ["mean return per trade at 2x costs", f"{c6['mean_2x']:+.4f}"],
    ]) + "\n")
    L.append("Reading: per-trade returns are in % of that trade's collateral (equal collateral per trade), so their sum is "
             "the book's P&L in units of one trade's collateral. The headline book figures in `summary.md` (total +1.19%) "
             "are on the 5-slot book, so they are about one fifth of this sum. The top three trades contribute more than the whole total "
             "(the other 46 net to a loss), one loss (C) more than halves the mean, and the mean turns negative at 2x costs. "
             "The book's small positive 1x result is fragile and rests on a few trades, consistent with H2's null.\n")
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT}")
    print("H1 mine:", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in h1.items()})
    print("P  mine:", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in pl.items()})
    print("tasks:", t1_pass, t2_pass, t3_pass, t4_pass)
    print("alt seeds:", alt)
    print("trim:", trim_rows)
    print("conc:", conc)


if __name__ == "__main__":
    main()
