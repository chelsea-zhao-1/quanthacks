"""Measurement layer: one long table from the cached download, built with the notebook's own pricing and P&L.

Rows: (event or null day) x bucket x entry (pre/post) x OTM level x horizon, as the notebook's evaluate().
Added columns:
  costs      cost_<s> = premium of strategy s's option legs at entry / spot x COST_HAIRCUT x 2 (the notebook's
             section 11 rule: a round trip at the entry premium); net_<s> and net2x_<s> (doubled cost).
  measures   iv_entry / iv_exit / iv_change (ATM straddle volatility proxy, sigma ~ straddle / (0.8 S sqrt T)),
             skew_entry / skew_exit / skew_change ((OTM put - OTM call) / spot at this row's OTM level),
             plus the notebook's realized, implied_scaled and ratio (|realized| / implied_scaled).
  features   all known at entry: timing (from the acceptance time), n_tags, earnings_cofiled, iv_pre,
             iv_term_slope (3-6m vs 1m at t_pre), atm_volume_ratio (t_pre vs the 5 sessions before),
             pre_drift (5 sessions to t_pre), quarter.

Offline by default: a cache miss raises instead of calling the API (pass allow_fetch=True to permit it).
"""
import numpy as np
import pandas as pd

from . import HARD_STOP

LEGS = {"stock": lambda o: [], "long_call": lambda o: ["C_K"], "covered_call": lambda o: [f"C_U{o}"],
        "protective_put": lambda o: [f"P_L{o}"], "collar": lambda o: [f"C_U{o}", f"P_L{o}"],
        "cash_secured_put": lambda o: [f"P_L{o}"]}


class CacheMiss(RuntimeError):
    pass


def go_offline(NB) -> None:
    def refuse(url, *a, **k):
        raise CacheMiss(f"not in cache: {str(url).split('?')[0]}")
    NB["SESSION"].get = refuse


def iv_proxy(m: dict, S: float, expiry: pd.Timestamp, day: pd.Timestamp) -> float:
    T = (expiry - day).days / 365
    if T <= 0 or not np.isfinite(S) or S <= 0:
        return np.nan
    return (m["C_K"] + m["P_K"]) / (0.8 * S * np.sqrt(T))


def timing_label(accepted_at, cutoff: str, close_time) -> str:
    if pd.isna(accepted_at):
        return "unknown"
    t = accepted_at - accepted_at.normalize()
    cut = pd.Timedelta(hours=16) - pd.to_timedelta(cutoff + ":00")
    close = close_time(accepted_at)
    if t < pd.Timedelta(hours=9, minutes=30):
        return "pre_open"
    if t < close - cut:
        return "intraday"
    if t < close:
        return "late_session"
    return "after_close"


def measure_priced(NB, pe, meta: dict) -> pd.DataFrame:
    """Every evaluate() row for one PricedEvent, with costs, extra measures and entry-time features."""
    res = NB["evaluate"]([pe])
    if res.empty:
        return res
    CAL, sessions_between = NB["CAL"], NB["sessions_between"]
    marks = {}

    def mk(day):
        if day not in marks:
            m = pe.marks(day)
            marks[day] = (m, pe.synthetic_spot(day, m))
        return marks[day]

    # features at t_pre for this bucket
    m_pre, S_pre = mk(pe.t_pre)
    i_pre = CAL.get_loc(pe.t_pre)
    before = CAL[max(i_pre - 5, 0):i_pre]
    vol_now = pe.legs["C_K"].volume_on(pe.t_pre) + pe.legs["P_K"].volume_on(pe.t_pre)
    vol_before = np.mean([pe.legs["C_K"].volume_on(d) + pe.legs["P_K"].volume_on(d) for d in before]) if len(before) else np.nan
    S_5 = mk(CAL[i_pre - 5])[1] if i_pre >= 5 else np.nan
    res = res.assign(
        iv_pre=iv_proxy(m_pre, S_pre, pe.expiry, pe.t_pre),
        atm_volume_ratio=vol_now / vol_before if vol_before and vol_before > 0 else np.nan,
        pre_drift=S_pre / S_5 - 1 if np.isfinite(S_5) and S_5 > 0 else np.nan,
        **meta)

    cost_haircut = NB["COST_HAIRCUT"]
    cols = {k: [] for k in ("iv_entry", "iv_exit", "skew_entry", "skew_exit")}
    cost_cols = {s: [] for s in LEGS}
    for r in res.itertuples(index=False):
        m_e, S_e = mk(r.entry_date)
        m_x, S_x = mk(r.exit_date)
        cols["iv_entry"].append(iv_proxy(m_e, S_e, pe.expiry, r.entry_date))
        cols["iv_exit"].append(iv_proxy(m_x, S_x, pe.expiry, r.exit_date))
        cols["skew_entry"].append((m_e[f"P_L{r.otm}"] - m_e[f"C_U{r.otm}"]) / S_e)
        cols["skew_exit"].append((m_x[f"P_L{r.otm}"] - m_x[f"C_U{r.otm}"]) / S_x if S_x else np.nan)
        for s, legs in LEGS.items():
            cost_cols[s].append(sum(abs(m_e[l]) for l in legs(r.otm)) / S_e * cost_haircut * 2)
    res = res.assign(**cols)
    res["iv_change"] = res["iv_exit"] - res["iv_entry"]
    res["skew_change"] = res["skew_exit"] - res["skew_entry"]
    for s in LEGS:
        res[f"cost_{s}"] = cost_cols[s]
        res[f"net_{s}"] = res[s] - res[f"cost_{s}"]
        res[f"net2x_{s}"] = res[s] - 2 * res[f"cost_{s}"]
    return res


def check_dates(rows: pd.DataFrame, hard_stop: str = HARD_STOP) -> None:
    """Hard fail: no entry or ordinary day at or after hard_stop (discovery: 2024-01-01; in-sample: 2026-01-01)."""
    for col in ("t_0", "t_pre"):
        bad = rows[col] >= pd.Timestamp(hard_stop)
        if bad.any():
            raise ValueError(f"{int(bad.sum())} rows have {col} on or after {hard_stop}; the playground refuses them")


def clip_to(NB, before: str) -> None:
    """Never fetch or compute on dates on or after `before` (CLAUDE.md rule 8 for the in-sample window):
    option bars stop the day before, and evaluate() treats the last session before it as the last one, so
    horizons and expiries that would end on or after it are simply absent."""
    cut = pd.Timestamp(before) - pd.Timedelta(days=1)
    if NB.get("_clipped_to") == before:
        return
    orig = NB["option_bars"]
    NB["option_bars"] = lambda tk, start, end: orig(tk, start, min(pd.Timestamp(end), cut))
    NB["LAST_SESSION"] = min(NB["LAST_SESSION"], NB["CAL"][NB["CAL"].searchsorted(pd.Timestamp(before)) - 1])
    NB["_clipped_to"] = before


def build(NB, events: pd.DataFrame, nulls: pd.DataFrame, earnings_tags: list[str], allow_fetch: bool = False,
          progress_every: int = 250, hard_stop: str = HARD_STOP, clip_before: str | None = None
          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The measurement table for every event and null row, plus a log of rows that could not be priced."""
    check_dates(events, hard_stop)
    check_dates(nulls, hard_stop)
    if clip_before:
        clip_to(NB, clip_before)
    if not allow_fetch:
        go_offline(NB)
    earnings = set(earnings_tags)
    ev = events.assign(set_id=events["ticker"] + "|" + events["filing_date"].dt.strftime("%Y-%m-%d"))
    nu = nulls.assign(set_id=nulls["ticker"] + "|" + nulls["filing_date"].dt.strftime("%Y-%m-%d"))
    jobs = [("event", r.ticker, r.t_pre, r.t_0, r.filing_date, r.set_id, {
                "tags": r.tags, "n_tags": len(r.tags.split("|")),
                "earnings_cofiled": bool(earnings & set(r.tags.split("|"))),
                "timing": timing_label(r.accepted_at, NB["ENTRY_CUTOFF"], NB["close_time"])})
            for r in ev.itertuples(index=False)]
    jobs += [("null", r.ticker, r.t_pre, r.t_0, r.t_0, r.set_id, {"round": r.round}) for r in nu.itertuples(index=False)]

    parts, failures = [], []
    for i, (kind, ticker, t_pre, t_0, event_date, set_id, extra) in enumerate(jobs, 1):
        try:
            priced, notes = NB["price_event"](ticker, t_pre, t_0, event_date, NB["EXPIRY_BUCKETS"], NB["OTM_GRID"])
        except CacheMiss as e:
            failures.append({"kind": kind, "set_id": set_id, "t_0": t_0, "reason": str(e)})
            continue
        if not priced:
            failures.append({"kind": kind, "set_id": set_id, "t_0": t_0, "reason": "; ".join(notes)})
        meta = {"kind": kind, "set_id": set_id, "quarter": f"{t_0.year}Q{(t_0.month - 1) // 3 + 1}", **extra}
        parts += [measure_priced(NB, pe, meta) for pe in priced]
        if progress_every and i % progress_every == 0:
            print(f"  measured {i:,}/{len(jobs):,} rows", flush=True)
    table = pd.concat([p for p in parts if len(p)], ignore_index=True) if parts else pd.DataFrame()
    if clip_before and len(table):
        assert (table["exit_date"] < pd.Timestamp(clip_before)).all(), f"an exit reached {clip_before}"
    if len(table):
        slope = (table[table.entry == "post"].drop_duplicates(["kind", "set_id", "t_0", "bucket"])
                 .pivot_table(index=["kind", "set_id", "t_0"], columns="bucket", values="iv_pre"))
        if {"1m", "3-6m"} <= set(slope.columns):
            table = table.merge((slope["3-6m"] / slope["1m"] - 1).rename("iv_term_slope").reset_index(),
                                on=["kind", "set_id", "t_0"], how="left")
    return table, pd.DataFrame(failures, columns=["kind", "set_id", "t_0", "reason"])
