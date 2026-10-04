"""Gap inputs and outcomes for the old-news test (docs/test_plan.md), from the cached option data only.

    gap, outcome = measure(rows, "discovery")

`rows` is the events table and the nulls table stacked (columns row_id, kind, ticker, t_pre, t_0, gap_start,
n_gap). Every row is priced offline with the notebook's own `price_event`; nothing calls the API. The function
writes data/oldnews/gap_<label>.csv, outcome_<label>.csv (formats in .claude/ctx/00_shared.md) and
drops_<label>.csv (every bucket the notebook could not price, with its reason, so each exclusion is counted).

Conventions
- Parity spot, S = K e^(-rT) + C - P on the bucket's ATM pair (the notebook's `synthetic_spot`). Default mode
  "fresh": taken only on sessions when both ATM legs traded, so a stale close never enters a spot, a return or
  an IV. Mode "notebook": the notebook's marks (a leg's last close up to MAX_STALE_SESSIONS old).
- Entry rule (test plan): an outcome row is usable only if both ATM legs and its put traded at entry.
- IV is the notebook's straddle proxy, (C + P) / (0.8 S sqrt(T)), from `playground.measure.iv_proxy`.
- RV over [entry, exit] is sqrt(252 * sum of squared log returns between consecutive fresh sessions / sessions
  spanned). A session with no fresh spot merges into the next return, so no variance is lost. The last fresh
  session must be within MAX_STALE_SESSIONS of the exit, as for the notebook's marks.
- Gap inputs follow the test plan's "Coverage rules for the gap inputs": spot from the 1m ATM pair, else 2m,
  else 3-6m on the day; at the gap start, if no pair has a mark, the latest mark up to 3 sessions earlier, with
  n_eff = sessions from that mark to t_pre (r_mkt uses the same dates); sigma from the gap-start spot's bucket;
  volume baseline = mean over the cached sessions among the 5 before gap_start (at least 2); d_iv_gap needs a
  1m mark at both ends. usable = gap_move exists. Option bars start BARS_FROM_DAYS calendar days before t_pre
  (price_event), so a day earlier than that has no data and is never read as zero volume.
- Dates (hard_stop_for): every t_pre, t_0 and gap_start must be before the window's hard stop, and no session on
  or after it is ever read (exits and panel dates stop the session before). discovery and dryrun: 2024-01-01;
  insample: 2026-01-01; holdout only when the notebook's RUN_HOLDOUT is True and oos only when RUN_OOS is True
  (then up to LAST_SESSION). Any other label needs an explicit hard stop on or before 2026-01-01.
- Judges' path (dryrun, holdout, oos): the r_mkt panel is the run's own rows plus at most 400 extra ticker-dates
  from the window's other filings, a seeded sample (seed 20261003), so the run's cost stays bounded.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from playground.measure import CacheMiss, go_offline, iv_proxy   # noqa: E402

OUT_DIR = ROOT / "data" / "oldnews"
NEVER = pd.Timestamp("2026-01-01")                     # out-of-sample starts here: never computed on
STOPS = {"discovery": "2024-01-01", "dryrun": "2024-01-01", "insample": "2026-01-01"}   # fixed hard stops
SWITCHED = {"holdout": "RUN_HOLDOUT", "oos": "RUN_OOS"}   # windows that run only when the notebook switch is on
PANEL_CAP = {"dryrun": 400, "holdout": 400, "oos": 400}   # judges' path: at most this many extra panel keys
SEED = 20261003
BARS_FROM_DAYS = 10            # price_event fetches option bars from t_pre - 10 calendar days
BASELINE_SESSIONS = 5          # volume baseline: the 5 sessions before gap_start (test plan)
MIN_BASELINE = 2               # ... of which at least 2 must be cached (coverage rules)
MAX_GAP_STALE = 3              # gap-start spot: the latest mark at most 3 sessions before gap_start (coverage rules)
GAP_RULES = "coverage-rules-2026-10-03"    # stored with resumable results, so older gap results are never mixed in
MIN_TICK_COST = 0.05           # cost floor, $ per share, on entry and on exit (test plan)
MIN_MKT_TICKERS = 10           # r_mkt needs at least this many other tickers with a return over the same dates
GAP_COLS = ["row_id", "r_gap", "r_mkt", "sigma_d", "gap_move", "iv_gap_start", "iv_tpre", "d_iv_gap",
            "vol_gap_ratio", "usable", "reason", "spot_bucket", "n_eff", "stale_sessions", "n_baseline"]
OUTCOME_COLS = ["row_id", "bucket", "horizon", "otm", "entry_date", "exit_date", "iv0", "rv", "y", "put_strike",
                "put_premium", "put_volume_entry", "csp_gross", "csp_net", "csp_net2x", "usable"]


# ---- small pure pieces ------------------------------------------------------------------------------------

def atm_marks(pe, day: pd.Timestamp, mode: str = "fresh") -> dict | None:
    """ATM call and put prices on `day`, or None. mode "fresh": that session's closes, only if both legs traded
    that session. mode "notebook": the notebook's Leg.mark (last close at most MAX_STALE_SESSIONS old)."""
    if mode == "notebook":
        m = {k: pe.legs[k].mark(day) for k in ("C_K", "P_K")}
        return m if np.isfinite(m["C_K"]) and np.isfinite(m["P_K"]) else None
    if mode != "fresh":
        raise ValueError(f"unknown mark mode {mode!r}")
    c, p = pe.legs["C_K"].bars, pe.legs["P_K"].bars
    if day in c.index and day in p.index:
        return {"C_K": float(c.at[day, "close"]), "P_K": float(p.at[day, "close"])}
    return None


def atm_traded(pe, day: pd.Timestamp) -> bool:
    """Both ATM legs traded on `day` (the test plan's entry rule)."""
    return pe.legs["C_K"].volume_on(day) > 0 and pe.legs["P_K"].volume_on(day) > 0


def parity_spot(pe, day: pd.Timestamp, mode: str = "fresh") -> tuple[float, float]:
    """(parity spot, ATM IV proxy) on `day` from the notebook's synthetic_spot and iv_proxy; nan if no marks."""
    m = atm_marks(pe, day, mode)
    if m is None:
        return np.nan, np.nan
    S = pe.synthetic_spot(day, m)
    return S, iv_proxy(m, S, pe.expiry, day)


def spot_path(pe, days: pd.DatetimeIndex, mode: str = "fresh") -> pd.Series:
    """Parity spots on `days` (sessions without marks are left out)."""
    vals = {d: parity_spot(pe, d, mode)[0] for d in days}
    s = pd.Series(vals, dtype=float)
    return s[np.isfinite(s) & (s > 0)]


def realized_vol(path: pd.Series, entry: pd.Timestamp, exit_: pd.Timestamp, cal: pd.DatetimeIndex,
                 max_stale: int) -> float:
    """Annualised RV of log parity-spot returns from entry to exit. NaN if the entry spot is missing, no return
    exists, or the last fresh spot is more than `max_stale` sessions before the exit."""
    p = path[(path.index >= entry) & (path.index <= exit_)]
    if entry not in p.index or len(p) < 2:
        return np.nan
    pos = cal.get_indexer(p.index)
    if pos[-1] < cal.get_loc(exit_) - max_stale:
        return np.nan
    r = np.diff(np.log(p.to_numpy()))
    return float(np.sqrt(252 * np.sum(r ** 2) / (pos[-1] - pos[0])))


def cost(premium: float, haircut: float, mult: float = 1.0) -> float:
    """Per-share cost of one trade: the larger of `haircut` x premium or MIN_TICK_COST, times `mult`."""
    return mult * max(haircut * premium, MIN_TICK_COST)


def csp_pnl(p0: float, px: float, strike: float, haircut: float) -> tuple[float, float, float]:
    """Short put sold at p0, bought back at px: (gross, net, net at 2x costs) as fractions of collateral
    (strike x 100 per contract, so per share: / strike)."""
    if not (np.isfinite(p0) and np.isfinite(px) and strike > 0):
        return np.nan, np.nan, np.nan
    gross = p0 - px
    return (gross / strike, (gross - cost(p0, haircut) - cost(px, haircut)) / strike,
            (gross - cost(p0, haircut, 2) - cost(px, haircut, 2)) / strike)


# ---- per bucket -------------------------------------------------------------------------------------------

def spot_on(priced: dict, day: pd.Timestamp, mode: str = "fresh") -> tuple[str | None, float, float]:
    """(bucket, parity spot, IV proxy) from the first bucket in `priced` order (1m, 2m, 3-6m) whose ATM pair has a
    mark on `day`; (None, nan, nan) if none has."""
    for b, pe in priced.items():
        S, iv = parity_spot(pe, day, mode)
        if np.isfinite(S) and S > 0:
            return b, S, iv
    return None, np.nan, np.nan


def gap_inputs(priced: dict, gap_start: pd.Timestamp, t_pre: pd.Timestamp, n_gap: int, cal: pd.DatetimeIndex,
               min_baseline: int = MIN_BASELINE, mode: str = "fresh") -> dict:
    """Pre-entry gap inputs under the test plan's coverage rules (r_mkt and gap_move come later, from the panel).
    `priced` maps bucket -> PricedEvent in preference order. Returns the inputs, the diagnostics spot_bucket
    ("<gap-start bucket>|<t_pre bucket>"), n_eff, stale_sessions, n_baseline, the effective gap start gs_eff
    (r_mkt is measured from it) and the reasons any input is missing."""
    out = {"r_gap": np.nan, "sigma_d": np.nan, "iv_gap_start": np.nan, "iv_tpre": np.nan, "d_iv_gap": np.nan,
           "vol_gap_ratio": np.nan, "spot_bucket": "", "n_eff": np.nan, "stale_sessions": np.nan, "n_baseline": 0,
           "gs_eff": pd.NaT, "reasons": []}
    why = out["reasons"]
    fetch_start = t_pre - pd.Timedelta(days=BARS_FROM_DAYS)
    i_gs, i_tp = cal.get_loc(gap_start), cal.get_loc(t_pre)
    gap_ok = n_gap >= 1 and i_tp - i_gs == n_gap
    if not gap_ok:
        why.append(f"no gap (n_gap={n_gap}, sessions gap_start..t_pre={i_tp - i_gs})")

    # spot: 1m, else 2m, else 3-6m on the day; at the gap start, else the latest mark up to 3 sessions earlier
    b_tp, S_tp, _ = spot_on(priced, t_pre, mode)
    b_gs, S_gs, iv_gs, k = None, np.nan, np.nan, 0
    for k in range(MAX_GAP_STALE + 1):
        if i_gs - k < 0 or cal[i_gs - k] < fetch_start:
            break
        b_gs, S_gs, iv_gs = spot_on(priced, cal[i_gs - k], mode)
        if b_gs:
            break
    out["spot_bucket"] = f"{b_gs or '-'}|{b_tp or '-'}"
    if b_gs is None:
        why.append(f"no ATM pair has a {mode} mark on gap_start or the {MAX_GAP_STALE} sessions before"
                   + (" (before the cached bars)" if gap_start < fetch_start else ""))
    if b_tp is None:
        why.append(f"no ATM pair has a {mode} mark on t_pre")
    if b_gs and b_tp:
        out["r_gap"] = float(np.log(S_tp / S_gs))
    if b_gs:
        out.update(sigma_d=iv_gs / np.sqrt(252), stale_sessions=k, gs_eff=cal[i_gs - k],
                   n_eff=n_gap + k if gap_ok else np.nan)

    # implied-volatility change: a 1m mark at both ends
    pe1 = priced.get("1m")
    if pe1 is not None:
        out["iv_gap_start"] = parity_spot(pe1, gap_start, mode)[1]
        out["iv_tpre"] = parity_spot(pe1, t_pre, mode)[1]
        out["d_iv_gap"] = out["iv_tpre"] - out["iv_gap_start"]
    if not np.isfinite(out["d_iv_gap"]):
        why.append("d_iv_gap: no 1m mark at both ends")

    # volume: the 1m ATM pair over the gap / its mean over the cached sessions among the 5 before gap_start
    base_days = [d for d in cal[max(i_gs - BASELINE_SESSIONS, 0):i_gs] if d >= fetch_start]
    gap_days = cal[i_gs + 1:i_tp + 1]
    out["n_baseline"] = len(base_days)
    if pe1 is None:
        why.append("volume: no 1m bucket")
    elif not gap_ok or len(gap_days) == 0 or gap_days[0] < fetch_start:
        why.append("volume: gap sessions not cached")
    elif len(base_days) < min_baseline:
        why.append(f"volume baseline: {len(base_days)} of {BASELINE_SESSIONS} sessions before gap_start are cached")
    else:
        vol = lambda d: pe1.legs["C_K"].volume_on(d) + pe1.legs["P_K"].volume_on(d)   # noqa: E731
        base = np.mean([vol(d) for d in base_days])
        if base > 0:
            out["vol_gap_ratio"] = float(sum(vol(d) for d in gap_days) / base)
        else:
            why.append("volume: no ATM volume in the baseline sessions")
    return out


def bucket_outcomes(pe, row_id: str, entry: pd.Timestamp, cal: pd.DatetimeIndex, horizons: list[int],
                    otm_grid: list[float], last_day: pd.Timestamp, max_stale: int, haircut: float,
                    mode: str = "fresh") -> list[dict]:
    """Outcome rows for one bucket: every horizon whose exit is on or before the bucket's expiry and last_day."""
    if entry > last_day or entry > pe.expiry_session:
        return []
    i0 = cal.get_loc(entry)
    exits = [(h, cal[i0 + h]) for h in horizons if i0 + h < len(cal) and cal[i0 + h] <= pe.expiry_session]
    exits = [(h, d) for h, d in exits + [("expiry", pe.expiry_session)] if d <= last_day]
    if not exits:
        return []
    S0, iv0 = parity_spot(pe, entry, mode)
    atm_ok = atm_traded(pe, entry) and np.isfinite(S0)
    path = spot_path(pe, cal[(cal >= entry) & (cal <= max(d for _, d in exits))], mode)
    puts = {}
    for otm in otm_grid:
        leg = pe.legs[f"P_L{otm}"]
        puts[otm] = (pe.strikes[f"L{otm}"], leg.mark(entry), leg.volume_on(entry), leg)
    rows = []
    for h, x in exits:
        rv = realized_vol(path, entry, x, cal, max_stale)
        y = float(np.log(rv / iv0)) if np.isfinite(rv) and np.isfinite(iv0) and rv > 0 and iv0 > 0 else np.nan
        for otm, (K, p0, v0, leg) in puts.items():
            gross, net, net2 = csp_pnl(p0, leg.mark(x), K, haircut)
            rows.append({"row_id": row_id, "bucket": pe.bucket, "horizon": str(h), "otm": int(round(otm * 100)),
                         "entry_date": entry, "exit_date": x, "iv0": iv0, "rv": rv, "y": y, "put_strike": K,
                         "put_premium": p0, "put_volume_entry": v0, "csp_gross": gross, "csp_net": net,
                         "csp_net2x": net2,
                         "usable": bool(atm_ok and v0 > 0 and np.isfinite(y))})
    return rows


# ---- r_mkt ------------------------------------------------------------------------------------------------

def market_returns(series: list[tuple[str, pd.Series]], queries: list[tuple[str, pd.Timestamp, pd.Timestamp]],
                   min_tickers: int = MIN_MKT_TICKERS) -> tuple[np.ndarray, np.ndarray]:
    """For each (ticker, d1, d2): the median across other tickers of each ticker's log spot return d1 -> d2
    (the median over that ticker's series that have a fresh spot on both dates). Returns (r_mkt, n_tickers);
    r_mkt is NaN when fewer than `min_tickers` tickers qualify."""
    r_mkt, n_tk = np.full(len(queries), np.nan), np.zeros(len(queries), dtype=int)
    series = [(t, s) for t, s in series if len(s)]
    if not series:
        return r_mkt, n_tk
    long = pd.concat([np.log(s).rename("v").to_frame().assign(sid=i) for i, (_, s) in enumerate(series)])
    wide = long.reset_index(names="day").pivot(index="day", columns="sid", values="v")
    L, pos = wide.to_numpy(), {d: i for i, d in enumerate(wide.index)}
    tick = np.array([series[c][0] for c in wide.columns])
    for q, (tk, d1, d2) in enumerate(queries):
        if d1 not in pos or d2 not in pos:
            continue
        diff = L[pos[d2]] - L[pos[d1]]
        ok = np.isfinite(diff) & (tick != tk)
        if not ok.any():
            continue
        per = pd.Series(diff[ok]).groupby(tick[ok]).median()
        n_tk[q] = len(per)
        if len(per) >= min_tickers:
            r_mkt[q] = float(per.median())
    return r_mkt, n_tk


# ---- driver -----------------------------------------------------------------------------------------------

def window_of(label: str) -> str | None:
    """The window a label belongs to ("discovery_nbmarks" -> "discovery"), or None."""
    base = label.split("_")[0]
    return base if base in STOPS or base in SWITCHED else None


def hard_stop_for(label: str, NB: dict, hard_stop: str | None = None) -> pd.Timestamp:
    """The first date a run under `label` may never read. An explicit hard_stop can only tighten it."""
    w = window_of(label)
    if w in SWITCHED:
        if NB.get(SWITCHED[w]) is not True:
            raise ValueError(f"label {label!r} runs only when the notebook's {SWITCHED[w]} is True; refused")
        if w == "oos":
            print(NB.get("OOS_WARNING", "WARNING: THE OUT-OF-SAMPLE SECTION IS ON."), file=sys.stderr, flush=True)
        stop = pd.Timestamp(NB["LAST_SESSION"]) + pd.Timedelta(days=1)
    elif w is not None:
        stop = pd.Timestamp(STOPS[w])
    else:
        if hard_stop is None or pd.Timestamp(hard_stop) > NEVER:
            raise ValueError(f"label {label!r} is not a known window: pass a hard_stop on or before {NEVER.date()}")
        stop = pd.Timestamp(hard_stop)
    return min(stop, pd.Timestamp(hard_stop)) if hard_stop is not None else stop


def sample_panel(keys, cap: int | None, seed: int = SEED) -> list[tuple]:
    """The sorted unique panel keys, or a seeded sample of `cap` of them (deterministic for a given input)."""
    keys = sorted(set(keys))
    if cap is None or len(keys) <= cap:
        return keys
    idx = np.sort(np.random.default_rng(seed).choice(len(keys), size=cap, replace=False))
    return [keys[i] for i in idx]


READ_CSV = {"keep_default_na": False, "na_values": [""]}    # "null" (the kind) and "NA" (a ticker) stay strings


def read_table(path) -> pd.DataFrame:
    """Read an events, nulls or gap CSV without turning the strings "null" or "NA" into missing values."""
    return pd.read_csv(path, **READ_CSV)


def load_offline_nb() -> dict:
    """The notebook namespace loaded with a dummy key (the real key is never read), then go_offline."""
    import os
    import nb
    old = os.environ.get("MASSIVE_API_KEY")
    os.environ["MASSIVE_API_KEY"] = "offline"
    try:
        NB = nb.load()
    finally:
        if old is None:
            os.environ.pop("MASSIVE_API_KEY", None)
        else:
            os.environ["MASSIVE_API_KEY"] = old
    go_offline(NB)
    return NB


def _dates(df: pd.DataFrame, cols: list[str], stop: pd.Timestamp, cal: pd.DatetimeIndex, strict: bool
           ) -> pd.DataFrame:
    """Parse date columns; hard-fail (strict) or drop (not strict) any date on or after the hard stop; add
    <col>_ok = the date is a trading session."""
    df = df.copy()
    for c in cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")
        bad = df[c] >= stop
        if bad.any():
            if strict:
                raise ValueError(f"{int(bad.sum())} rows have {c} on or after {stop.date()}; refused")
            df = df[~bad]
    for c in cols:
        df[c + "_ok"] = df[c].isin(cal)
    return df


class _Parts:
    """Resumable storage: one pickle per finished batch in <out_dir>/_parts_<label>/, plus the run settings.
    Results are keyed by row_id and a fingerprint of the row's inputs, so a row whose inputs change is
    recomputed and a finished one is skipped. In memory only when out_dir is None."""

    def __init__(self, out_dir: Path | None, label: str, settings: dict):
        import json
        self.dir = None if out_dir is None else Path(out_dir) / f"_parts_{label}"
        self.per_row, self.series = {}, {}
        if self.dir is None:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        cfg = self.dir / "settings.json"
        if cfg.exists() and json.loads(cfg.read_text()) != settings:
            raise ValueError(f"{self.dir} holds results for other settings ({cfg.read_text()}); delete it to restart")
        cfg.write_text(json.dumps(settings))
        for f in sorted(self.dir.glob("part_*.pkl")):
            self._merge(pd.read_pickle(f))

    def _merge(self, part: dict) -> None:
        self.per_row.update(part["per_row"])
        self.series.update(part["series"])

    def save(self, part: dict) -> None:
        import os
        import pickle
        self._merge(part)
        if self.dir is None:
            return
        n = len(list(self.dir.glob("part_*.pkl")))
        tmp = self.dir / f"part_{n:05d}.tmp"
        tmp.write_bytes(pickle.dumps(part))
        os.replace(tmp, self.dir / f"part_{n:05d}.pkl")


def measure(rows: pd.DataFrame, label: str, NB: dict | None = None, panel_rows: pd.DataFrame | None = None,
            hard_stop: str | None = None, entry_shift: int = 0, min_baseline: int = MIN_BASELINE,
            mode: str = "fresh", workers: int = 8, out_dir: Path | None = OUT_DIR, batch_size: int = 100,
            log_path: Path | None = None, max_panel: int | None | str = "auto", outcomes: bool = True
            ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Price every row offline and return (gap, outcome); with out_dir, also write gap_<label>.csv,
    outcome_<label>.csv and drops_<label>.csv there. Resumable: finished batches are kept in
    <out_dir>/_parts_<label>/ and a rerun skips every row whose inputs have not changed.

    panel_rows: extra (ticker, t_pre, t_0) rows priced only to widen the r_mkt spot panel (e.g. every cached
    discovery event and ordinary day), so r_mkt approaches the test plan's "all cached TOP_100 tickers".
    max_panel: at most this many panel-only (ticker, t_pre) keys, a seeded sample; "auto" = 400 on the judges'
    path (dryrun, holdout, oos), unbounded otherwise; None = unbounded.
    hard_stop: required for a label outside the known windows; otherwise it can only tighten the window's stop.
    entry_shift: enter this many sessions after t_0 (test plan sensitivity: t_0 + 1).
    min_baseline: cached sessions (of the 5 before gap_start) the volume baseline needs (coverage rules: 2).
    outcomes: False computes, writes and returns only the gap table (outcome is None, no outcome file is touched);
    its resumable results go to <out_dir>/_parts_<label>_gaponly/.
    mode: "fresh" (default) prices a spot or IV only from same-session closes of both ATM legs; "notebook" uses
    the notebook's marks, which may be up to MAX_STALE_SESSIONS old.
    A row that cannot be priced gets usable=False and a reason; the run continues.
    """
    if mode not in ("fresh", "notebook"):
        raise ValueError(f"unknown mark mode {mode!r}")
    NB = NB if NB is not None else load_offline_nb()
    if "SESSION" in NB:
        go_offline(NB)
    stop = hard_stop_for(label, NB, hard_stop)
    cal, horizons = NB["CAL"], list(NB["HORIZONS"])
    otm_grid, buckets = list(NB["OTM_GRID"]), NB["EXPIRY_BUCKETS"]
    max_stale, haircut = int(NB["MAX_STALE_SESSIONS"]), float(NB["COST_HAIRCUT"])
    last_day = min(cal[cal.searchsorted(stop) - 1], NB["LAST_SESSION"])

    rows = _dates(rows, ["t_pre", "t_0", "gap_start"], stop, cal, strict=True).reset_index(drop=True)
    if rows["row_id"].duplicated().any():
        raise ValueError(f"row_id is not unique ({int(rows.row_id.duplicated().sum())} duplicates)")
    rows["n_gap"] = pd.to_numeric(rows["n_gap"], errors="coerce")
    fmt = lambda t: "" if pd.isna(t) else f"{t:%Y-%m-%d}"   # noqa: E731
    rows["fp"] = [f"{r.ticker}|{fmt(r.t_pre)}|{fmt(r.t_0)}|{fmt(r.gap_start)}|{r.n_gap}" for r in rows.itertuples()]
    extra = pd.DataFrame({"ticker": [], "t_pre": pd.to_datetime([]), "t_0": pd.to_datetime([]), "t_pre_ok": []})
    if panel_rows is not None and len(panel_rows):
        extra = _dates(panel_rows[["ticker", "t_pre", "t_0"]], ["t_pre", "t_0"], stop, cal, strict=False)
    settings = {"hard_stop": f"{stop:%Y-%m-%d}", "entry_shift": entry_shift, "min_baseline": min_baseline,
                "mode": mode, "bars_from_days": BARS_FROM_DAYS, "horizons": horizons, "otm_grid": otm_grid,
                "gap_rules": GAP_RULES, "outcomes": outcomes}
    store = _Parts(out_dir, label if outcomes else f"{label}_gaponly", settings)
    log_path = log_path or (Path(out_dir) / "measure_progress.log" if out_dir is not None else None)

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {label}: {msg}"
        print(line, flush=True)
        if log_path is not None:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def bucket_of(note: str) -> str:
        head = note.split(":")[0]
        return head if head in buckets else "all"

    def blank(r, why: str | None) -> dict:
        return {"fp": r.fp, "outcome": [], "drops": [],
                "gap": {"row_id": r.row_id, "ticker": r.ticker, "gap_start": r.gap_start, "t_pre": r.t_pre,
                        "n_gap": r.n_gap, "reasons": [why] if why else []}}

    # rows still to do (new, or inputs changed); a row whose t_pre is not a session cannot be priced
    todo = rows[[store.per_row.get(rid, {}).get("fp") != fp for rid, fp in zip(rows.row_id, rows.fp)]]
    bad = todo[~todo.t_pre_ok]
    if len(bad):
        store.save({"per_row": {r.row_id: blank(r, "t_pre missing or not a trading session")
                                for r in bad.itertuples()}, "series": {}})
    todo = todo[todo.t_pre_ok]
    by_key = {k: list(g.index) for k, g in todo.groupby(["ticker", "t_pre"], sort=False)}
    ok_extra = extra[extra.t_pre_ok.astype(bool)]
    t0_of = (dict(zip(zip(ok_extra.ticker, ok_extra.t_pre), ok_extra.t_0))
             | dict(zip(zip(todo.ticker, todo.t_pre), todo.t_0)))
    row_keys = list(dict.fromkeys(zip(rows.ticker[rows.t_pre_ok], rows.t_pre[rows.t_pre_ok])))
    cap = PANEL_CAP.get(window_of(label)) if max_panel == "auto" else max_panel
    panel_keys = sample_panel(set(zip(ok_extra.ticker, ok_extra.t_pre)) - set(row_keys), cap)
    all_keys = row_keys + panel_keys
    keys = [k for k in all_keys if k in by_key or k not in store.series]

    def work(key) -> dict:
        ticker, t_pre = key
        out = {"key": key, "series": None, "per_row": {}}
        try:
            t0 = t0_of.get(key, t_pre)
            try:
                priced, notes = NB["price_event"](ticker, t_pre, t0, t0, buckets, otm_grid)
            except CacheMiss as e:
                priced, notes = [], [f"cache miss ({e})"]
            by_bucket = {b: pe for b in buckets for pe in priced if pe.bucket == b}     # 1m, 2m, 3-6m order
            pe1 = by_bucket.get("1m")
            if pe1 is not None:                   # the 1m spot series for the r_mkt panel, before the hard stop
                days = cal[(cal >= t_pre - pd.Timedelta(days=BARS_FROM_DAYS))
                           & (cal <= min(pe1.expiry_session, last_day))]
                out["series"] = (ticker, spot_path(pe1, days, mode))
            for i in by_key.get(key, []):
                r = rows.loc[i]
                rec = blank(r, None)
                rec["drops"] = [{"row_id": r.row_id, "bucket": bucket_of(n), "reason": n} for n in notes]
                if pe1 is None:
                    why = "; ".join(n for n in notes if bucket_of(n) in ("1m", "all")) or "no 1m expiry"
                    rec["gap"]["reasons"].append(f"1m bucket not priced ({why})")
                if by_bucket and (not r.gap_start_ok or pd.isna(r.n_gap)):
                    rec["gap"]["reasons"].append("gap_start or n_gap missing or not a trading session")
                elif by_bucket:
                    g = gap_inputs(by_bucket, r.gap_start, t_pre, int(r.n_gap), cal, min_baseline, mode)
                    rec["gap"]["reasons"] += g.pop("reasons")
                    rec["gap"].update(g)
                i0 = cal.get_loc(r.t_0) + entry_shift if r.t_0_ok else -1
                if outcomes and 0 <= i0 < len(cal) and cal[i0] < stop:
                    for pe in priced:
                        rec["outcome"] += bucket_outcomes(pe, r.row_id, cal[i0], cal, horizons, otm_grid, last_day,
                                                          max_stale, haircut, mode)
                elif outcomes:
                    rec["drops"].append({"row_id": r.row_id, "bucket": "all",
                                         "reason": "entry session missing, not a session, or past the hard stop"})
                out["per_row"][r.row_id] = rec
        except Exception as e:                    # flag and continue; the error is written to the log
            why = f"error: {type(e).__name__}: {e}"
            out["per_row"] = {rows.loc[i].row_id: blank(rows.loc[i], why) for i in by_key.get(key, [])}
            out["error"] = why
        return out

    t_start, n_todo, rows_done, errors = time.time(), len(todo) + len(bad), len(bad), 0
    log(f"{len(rows):,} rows ({len(rows) - n_todo:,} already done), {len(keys):,} keys to price "
        f"({len(all_keys):,} in all, panel included), mode={mode}, hard stop {stop.date()}")
    with ThreadPoolExecutor(max(1, workers)) as ex:
        for b in range(0, len(keys), batch_size):
            res = list(ex.map(work, keys[b:b + batch_size]))
            store.save({"per_row": {k: v for r in res for k, v in r["per_row"].items()},
                        "series": {r["key"]: r["series"] for r in res}})
            rows_done += sum(len(r["per_row"]) for r in res)
            for r in (r for r in res if "error" in r):
                errors += 1
                log(f"  {r['key'][0]} {r['key'][1]:%Y-%m-%d}: {r['error']}")
            el, k = time.time() - t_start, min(b + batch_size, len(keys))
            log(f"keys {k:,}/{len(keys):,}, rows {rows_done:,}/{n_todo:,}, {el:.0f}s "
                f"({el / k:.2f}s per key, ETA {el / k * (len(keys) - k) / 60:.1f} min), errors {errors}")

    # finalize on the current rows only (results for changed or removed rows are ignored)
    recs = [store.per_row[rid] for rid in rows.row_id]
    assert all(rec["fp"] == fp for rec, fp in zip(recs, rows.fp)), "a row was not measured"
    num = ["r_gap", "sigma_d", "iv_gap_start", "iv_tpre", "d_iv_gap", "vol_gap_ratio", "n_eff", "stale_sessions",
           "n_baseline"]
    gap = pd.DataFrame([rec["gap"] for rec in recs]).reindex(
        columns=["row_id", "ticker", "gap_start", "t_pre", "n_gap", "reasons", "spot_bucket", "gs_eff"] + num)
    gap[num] = gap[num].apply(pd.to_numeric, errors="coerce")
    gap["n_baseline"] = gap.n_baseline.fillna(0).astype(int)
    gap["spot_bucket"] = gap.spot_bucket.fillna("")
    gap["gs_eff"] = pd.to_datetime(gap.gs_eff)
    series = [store.series[k] for k in all_keys if store.series.get(k) is not None]
    r_mkt, n_tk = market_returns(series, list(zip(gap.ticker, gap.gs_eff, gap.t_pre)))     # same dates as r_gap
    gap["r_mkt"] = r_mkt
    mkt_why = [[] if np.isfinite(v) or pd.isna(d) else [f"r_mkt: {n} other tickers (need {MIN_MKT_TICKERS})"]
               for v, n, d in zip(r_mkt, n_tk, gap.gs_eff)]
    gap["reason"] = ["; ".join(a + b) for a, b in zip(gap.reasons, mkt_why)]
    gap["gap_move"] = (gap.r_gap - gap.r_mkt).abs() / (gap.sigma_d * np.sqrt(gap.n_eff.where(gap.n_eff > 0)))
    gap["usable"] = np.isfinite(gap.gap_move.to_numpy(dtype=float))          # coverage rules: gap_move exists
    gap = gap[GAP_COLS]
    outcome = None
    if outcomes:
        outcome = pd.DataFrame([o for rec in recs for o in rec["outcome"]], columns=OUTCOME_COLS)
        if len(outcome):
            assert (outcome.exit_date < stop).all() and (outcome.entry_date < stop).all(), "a date reached the stop"
    drops = pd.DataFrame([d for rec in recs for d in rec["drops"]], columns=["row_id", "bucket", "reason"])
    if out_dir is not None:
        out_dir = Path(out_dir)
        gap.to_csv(out_dir / f"gap_{label}.csv", index=False)
        if outcomes:
            outcome.to_csv(out_dir / f"outcome_{label}.csv", index=False, date_format="%Y-%m-%d")
            drops.to_csv(out_dir / f"drops_{label}.csv", index=False)
    has_dates = rows.gap_start.notna().to_numpy()
    nt = n_tk[has_dates] if has_dates.any() else np.zeros(1, dtype=int)
    per_day = pd.Series(dtype=float)
    if series:                                   # tickers with a spot on each session of the rows' gap span
        days = pd.concat([s.rename("v").to_frame().assign(tk=t) for t, s in series]).reset_index(names="day")
        per_day = days.groupby("day").tk.nunique()
        span = pd.concat([rows.gap_start, rows.t_pre])
        per_day = per_day[(per_day.index >= span.min()) & (per_day.index <= span.max())]
    pd_med, pd_min = (per_day.median(), per_day.min()) if len(per_day) else (0, 0)
    msg = f"done in {time.time() - t_start:.0f}s: gap usable {int(gap.usable.sum()):,}/{len(gap):,}; errors {errors}"
    if outcomes:
        h10 = outcome[(outcome.bucket == "1m") & (outcome.otm == 3) & (outcome.horizon == "10")]
        msg += f"; outcome rows {len(outcome):,}; 1m/3%/h10 usable {int(h10.usable.sum()):,}/{len(rows):,}"
    log(msg)
    log(f"r_mkt panel: {len(series):,} series ({len(panel_keys):,} panel-only keys, cap {cap}); tickers per session "
        f"median {pd_med:.0f}, min {pd_min:.0f}; other tickers per row's gap median {np.median(nt):.0f}, min {nt.min()}")
    return gap, outcome


def main(argv: list[str] | None = None) -> int:
    """Run measure() on data/oldnews/events_<label>.csv + nulls_<label>.csv, offline, resumable.
    Example:  .venv/Scripts/python.exe src/oldnews/measure.py discovery --panel"""
    import argparse
    import os
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("label", choices=sorted(STOPS))   # holdout and oos run only from the notebook, with its switch on
    ap.add_argument("--mode", default="fresh", choices=["fresh", "notebook"])
    ap.add_argument("--suffix", default="", help="output name suffix: gap_<label>_<suffix>.csv")
    ap.add_argument("--panel", action="store_true",
                    help="also price every cached playground event and ordinary day of this label (r_mkt panel)")
    ap.add_argument("--entry-shift", type=int, default=0)
    ap.add_argument("--min-baseline", type=int, default=MIN_BASELINE)
    ap.add_argument("--gap-only", action="store_true", help="compute and write only gap_<label>.csv")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=100)
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    NB = load_offline_nb()
    rows = pd.concat([read_table(OUT_DIR / f"events_{a.label}.csv"), read_table(OUT_DIR / f"nulls_{a.label}.csv")],
                     ignore_index=True)
    panel = None
    if a.panel:
        pg = ROOT / "data" / "playground"
        panel = pd.concat([read_table(pg / f"events_{a.label}.csv"), read_table(pg / f"nulls_{a.label}.csv")],
                          ignore_index=True)[["ticker", "t_pre", "t_0"]]
    name = a.label + (f"_{a.suffix}" if a.suffix else "")
    measure(rows, name, NB=NB, panel_rows=panel, entry_shift=a.entry_shift,
            min_baseline=a.min_baseline, mode=a.mode, workers=a.workers, batch_size=a.batch, outcomes=not a.gap_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
