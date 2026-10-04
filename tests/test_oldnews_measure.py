"""Synthetic tests for src/oldnews/measure.py (no real data, no network, no API key).
Run:  .venv/Scripts/python.exe tests/test_oldnews_measure.py"""
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402

from oldnews import measure as M   # noqa: E402

FAILS = []
R, SIGMA, SEED = 0.04, 0.30, 20261003
CAL = pd.bdate_range("2023-12-01", "2025-12-31")   # Dec 2023 precedes the window


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{': ' + detail if detail else ''}")
    if not ok:
        FAILS.append(name)


def sessions_between(a, b) -> int:
    return int(CAL.searchsorted(pd.Timestamp(b), side="right") - CAL.searchsorted(pd.Timestamp(a), side="right"))


# ---- a fake of the notebook's pricing objects (same interface as Leg and PricedEvent) ----------------------

@dataclass
class Leg:
    bars: pd.DataFrame

    def mark(self, day):
        b = self.bars.loc[: pd.Timestamp(day)]
        if b.empty or sessions_between(b.index[-1], day) > 3:
            return np.nan
        return float(b["close"].iloc[-1])

    def volume_on(self, day):
        return float(self.bars["volume"].get(pd.Timestamp(day), 0.0))


@dataclass
class PE:
    bucket: str
    expiry: pd.Timestamp
    expiry_session: pd.Timestamp
    strikes: dict
    legs: dict

    def synthetic_spot(self, day, m):
        T = max((self.expiry - pd.Timestamp(day)).days, 0) / 365
        return self.strikes["K"] * np.exp(-R * T) + m["C_K"] - m["P_K"]


def spot_series(ticker: str) -> pd.Series:
    """A deterministic daily spot path per ticker: alternating +/- moves whose size depends on the ticker."""
    k = sum(map(ord, ticker)) % 7 + 1
    r = np.where(np.arange(len(CAL)) % 2 == 0, 0.004 * k, -0.003 * k)
    return pd.Series(100 * np.exp(np.cumsum(r)), index=CAL)


def make_pe(ticker, t_pre, bucket="1m", days_out=30, missing=None, vol=None, put_volume=10.0, sigma=SIGMA):
    """ATM pair priced so parity gives the true spot and the straddle proxy gives SIGMA exactly."""
    S = spot_series(ticker)
    expiry = t_pre + pd.Timedelta(days=days_out)
    exp_s = CAL[CAL.searchsorted(expiry, side="right") - 1]
    days = CAL[(CAL >= t_pre - pd.Timedelta(days=M.BARS_FROM_DAYS)) & (CAL <= exp_s)]
    K = float(round(S[t_pre]))
    T = np.maximum((expiry - days).days.to_numpy(), 0) / 365
    s = S[days].to_numpy()
    A = 0.8 * s * sigma * np.sqrt(T)                  # C + P
    B = s - K * np.exp(-R * T)                        # C - P
    C, P = (A + B) / 2, (A - B) / 2
    vol = vol if vol is not None else {}
    v = np.array([vol.get(d, 50.0) for d in days])
    legs = {"C_K": Leg(pd.DataFrame({"close": C, "volume": v}, index=days)),
            "P_K": Leg(pd.DataFrame({"close": P, "volume": v}, index=days))}
    strikes = {"K": K}
    for otm in (0.03, 0.05, 0.10):
        L = round(S[t_pre] * (1 - otm))
        strikes[f"L{otm}"] = float(L)
        pv = np.full(len(days), put_volume)
        legs[f"P_L{otm}"] = Leg(pd.DataFrame({"close": np.maximum(L - s, 0) + 0.02 * s * np.sqrt(T + 1e-9),
                                              "volume": pv}, index=days))
        legs[f"C_U{otm}"] = legs["C_K"]
    for d in (missing or []):                         # the put leg did not trade on these days
        legs["P_K"].bars = legs["P_K"].bars.drop(index=d, errors="ignore")
    return PE(bucket, expiry, exp_s, strikes, legs), S


def fake_nb(no_put_volume=()):
    def price_event(ticker, t_pre, t_0, event_date, buckets, otm_pcts):
        if ticker == "MISS":
            raise M.CacheMiss("not in cache: /v3/reference/options/contracts")
        pv = 0.0 if ticker in no_put_volume else 10.0
        return [make_pe(ticker, t_pre, "1m", 30, put_volume=pv)[0],
                make_pe(ticker, t_pre, "2m", 60, put_volume=pv)[0]], []
    return {"CAL": CAL, "HORIZONS": [1, 2, 3, 5, 10, 21, 42, 63], "OTM_GRID": [0.03, 0.05, 0.10],
            "EXPIRY_BUCKETS": {"1m": (21, 45, 30), "2m": (46, 80, 60)}, "MAX_STALE_SESSIONS": 3,
            "COST_HAIRCUT": 0.05, "LAST_SESSION": CAL[-1], "RUN_OOS": False, "price_event": price_event}


# ---- tests ------------------------------------------------------------------------------------------------

def test_read_table():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "nulls.csv"
        pd.DataFrame({"row_id": ["null|NA|1", "null|X|2"], "kind": ["null", "null"], "ticker": ["NA", "X"],
                      "gap_start": ["2022-03-01", ""]}).to_csv(f, index=False)
        plain, safe = pd.read_csv(f), M.read_table(f)
        check("read_table keeps kind 'null' and ticker 'NA' (plain read_csv loses them)",
              (safe.kind == "null").all() and safe.ticker.iloc[0] == "NA" and plain.kind.isna().all())
        check("read_table still reads an empty cell as missing", safe.gap_start.isna().iloc[1])


def test_parity_and_iv():
    t_pre = CAL[40]
    pe, S = make_pe("AAA", t_pre, missing=[CAL[42]])
    s, iv = M.parity_spot(pe, CAL[41])
    check("parity spot recovers the true spot", abs(s - S[CAL[41]]) < 1e-9, f"{s:.6f} vs {S[CAL[41]]:.6f}")
    check("IV proxy recovers sigma", abs(iv - SIGMA) < 1e-9, f"{iv:.6f}")
    check("fresh mode: no spot when a leg did not trade", np.isnan(M.parity_spot(pe, CAL[42])[0]))
    s_nb = M.parity_spot(pe, CAL[42], "notebook")[0]
    check("notebook mode: a stale leg gives a (wrong) spot", np.isfinite(s_nb) and abs(s_nb - S[CAL[42]]) > 1e-6)


def test_realized_vol():
    days = CAL[10:21]
    r = np.where(np.arange(10) % 2 == 0, 0.01, -0.01)
    path = pd.Series(100 * np.exp(np.concatenate([[0], np.cumsum(r)])), index=days)
    rv = M.realized_vol(path, days[0], days[-1], CAL, 3)
    check("RV of +/-1% daily moves is 1% x sqrt(252)", abs(rv - 0.01 * np.sqrt(252)) < 1e-12, f"{rv:.6f}")
    gap = path.drop(days[3])                          # a missing session merges two returns into one
    r2 = np.diff(np.log(gap.to_numpy()))
    want = np.sqrt(252 * np.sum(r2 ** 2) / 10)
    check("RV with a missing session uses the merged return over the same span",
          abs(M.realized_vol(gap, days[0], days[-1], CAL, 3) - want) < 1e-12)
    check("RV is NaN without an entry spot", np.isnan(M.realized_vol(path.drop(days[0]), days[0], days[-1], CAL, 3)))
    check("RV is NaN when the last spot is too stale", np.isnan(M.realized_vol(path[:5], days[0], days[-1], CAL, 3)))
    check("RV of one return is defined", np.isfinite(M.realized_vol(path, days[0], days[1], CAL, 3)))


def test_csp_costs():
    g, n, n2 = M.csp_pnl(2.0, 1.0, 100.0, 0.05)
    check("CSP gross is (p0 - px) / strike", abs(g - 0.01) < 1e-12)
    check("CSP net: costs max(5%, $0.05) each way", abs(n - (1 - 0.10 - 0.05) / 100) < 1e-12, f"{n}")
    check("CSP 2x costs doubles both", abs(n2 - (1 - 0.20 - 0.10) / 100) < 1e-12, f"{n2}")
    check("CSP is NaN without an exit mark", np.isnan(M.csp_pnl(2.0, np.nan, 100.0, 0.05)[0]))


def test_gap_inputs():
    t_pre, gs = CAL[60], CAL[59]
    vol = {d: 10.0 for d in CAL[50:60]}               # baseline sessions 54..58 trade 10 a leg (20 a pair)
    vol.update({CAL[60]: 80.0})                       # the gap session 60: a pair volume of 160
    pe, S = make_pe("BBB", t_pre, vol=vol)
    g = M.gap_inputs({"1m": pe}, gs, t_pre, 1, CAL)
    check("r_gap = log(S_tpre / S_gap_start)", abs(g["r_gap"] - np.log(S[t_pre] / S[gs])) < 1e-9)
    check("sigma_d = 1m IV at gap_start / sqrt(252)", abs(g["sigma_d"] - SIGMA / np.sqrt(252)) < 1e-9)
    check("d_iv_gap = IV(t_pre) - IV(gap_start)", abs(g["d_iv_gap"]) < 1e-9)
    check("vol_gap_ratio = gap total / baseline mean", abs(g["vol_gap_ratio"] - 160 / 20) < 1e-9,
          f"{g['vol_gap_ratio']}")
    check("clean row: 1m spot at both ends, no stale, n_eff = n_gap, 5 baseline sessions, no reasons",
          g["spot_bucket"] == "1m|1m" and g["stale_sessions"] == 0 and g["n_eff"] == 1 and g["n_baseline"] == 5
          and g["reasons"] == [], f"{g['spot_bucket']} {g['reasons']}")
    check("n_gap that disagrees with the calendar: no n_eff, flagged",
          np.isnan(M.gap_inputs({"1m": pe}, gs, t_pre, 3, CAL)["n_eff"]))
    check("n_gap = 0: no n_eff, flagged", np.isnan(M.gap_inputs({"1m": pe}, t_pre, t_pre, 0, CAL)["n_eff"]))

    # baseline: the mean over the cached sessions among the 5 before gap_start, at least 2
    fetch = t_pre - pd.Timedelta(days=M.BARS_FROM_DAYS)
    cached_before = lambda i: sum(d >= fetch for d in CAL[i - 5:i])                     # noqa: E731
    i2 = next(i for i in range(59, 40, -1) if cached_before(i) == 2)
    i1 = next(i for i in range(59, 40, -1) if cached_before(i) == 1)
    g2 = M.gap_inputs({"1m": pe}, CAL[i2], t_pre, 60 - i2, CAL)
    check("baseline with 2 cached sessions is used", g2["n_baseline"] == 2 and np.isfinite(g2["vol_gap_ratio"]))
    g1 = M.gap_inputs({"1m": pe}, CAL[i1], t_pre, 60 - i1, CAL)
    check("baseline with 1 cached session: volume missing, row still usable for the gap move",
          g1["n_baseline"] == 1 and np.isnan(g1["vol_gap_ratio"]) and np.isfinite(g1["r_gap"]))

    # 1m has no mark on gap_start: the 2m pair gives the spot and sigma, d_iv_gap is missing
    pe1, _ = make_pe("BBB", t_pre, "1m", 30, missing=[gs], vol=vol)
    pe2, _ = make_pe("BBB", t_pre, "2m", 60, sigma=0.4)
    g = M.gap_inputs({"1m": pe1, "2m": pe2}, gs, t_pre, 1, CAL)
    check("2m fallback at gap_start: spot_bucket 2m|1m, sigma from 2m, d_iv_gap missing",
          g["spot_bucket"] == "2m|1m" and abs(g["sigma_d"] - 0.4 / np.sqrt(252)) < 1e-9 and np.isnan(g["d_iv_gap"])
          and abs(g["r_gap"] - np.log(S[t_pre] / S[gs])) < 1e-9, f"{g['spot_bucket']} {g['sigma_d']}")
    g = M.gap_inputs({"1m": pe, "2m": pe2}, gs, t_pre, 1, CAL)
    check("1m preferred when it has a mark", g["spot_bucket"] == "1m|1m")

    # no pair has a mark on gap_start: the latest mark up to 3 sessions earlier, n_eff counts from it
    gs4 = CAL[58]
    pe1, _ = make_pe("BBB", t_pre, "1m", 30, missing=[gs4, CAL[57]])
    pe2, _ = make_pe("BBB", t_pre, "2m", 60, missing=[gs4, CAL[57]])
    g = M.gap_inputs({"1m": pe1, "2m": pe2}, gs4, t_pre, 2, CAL)
    check("stale fallback: mark 2 sessions before gap_start, n_eff = n_gap + 2, r_gap from that day",
          g["stale_sessions"] == 2 and g["n_eff"] == 4 and g["gs_eff"] == CAL[56]
          and abs(g["r_gap"] - np.log(S[t_pre] / S[CAL[56]])) < 1e-9, f"{g['stale_sessions']} {g['n_eff']}")
    pe1, _ = make_pe("BBB", t_pre, "1m", 30, missing=list(CAL[54:59]))
    g = M.gap_inputs({"1m": pe1}, gs4, t_pre, 2, CAL)
    check("no mark within 3 sessions: no spot, no n_eff", np.isnan(g["r_gap"]) and np.isnan(g["n_eff"])
          and g["spot_bucket"] == "-|1m")
    pe1, _ = make_pe("BBB", t_pre, "1m", 30, missing=[t_pre])
    g = M.gap_inputs({"1m": pe1, "2m": pe2}, gs, t_pre, 1, CAL)
    check("t_pre falls back to 2m too", g["spot_bucket"] == "1m|2m" and np.isfinite(g["r_gap"]))


def test_market_returns():
    d1, d2 = CAL[5], CAL[9]
    series, want = [], {}
    for i in range(12):
        tk = f"T{i}"
        s = pd.Series([100.0, 100 * np.exp(0.01 * i)], index=[d1, d2])
        series.append((tk, s))
        want[tk] = 0.01 * i
    series.append(("T3", pd.Series([100.0, 100 * np.exp(0.5)], index=[d1, d2])))   # T3's second series
    series.append(("T4", pd.Series([100.0], index=[d1])))                             # no d2: ignored
    r, n = M.market_returns(series, [("T0", d1, d2), ("ZZ", d1, d2), ("T0", d1, CAL[30])], min_tickers=10)
    per = {k: v for k, v in want.items() if k != "T0"}
    per["T3"] = np.median([0.03, 0.5])
    check("r_mkt = median over other tickers of each ticker's median return",
          abs(r[0] - np.median(list(per.values()))) < 1e-12 and n[0] == 11, f"{r[0]} n={n[0]}")
    check("r_mkt for an outside ticker uses all 12", n[1] == 12)
    check("r_mkt NaN when no ticker has both dates", np.isnan(r[2]))
    r2, _ = M.market_returns(series, [("T0", d1, d2)], min_tickers=12)
    check("r_mkt NaN below min_tickers", np.isnan(r2[0]))


def rows_table(n_tickers=12, n_gap=1):
    """Two events and two ordinary days per ticker, close in time across tickers so r_mkt has a panel."""
    rows = []
    for i in range(n_tickers):
        tk = f"T{i:02d}"
        for j, base in enumerate((40, 120)):
            for kind, b in (("event", base), ("null", base + 30)):
                b += i % 3
                rows.append({"row_id": f"{kind}|{tk}|{j}", "kind": kind, "ticker": tk, "t_pre": CAL[b],
                             "t_0": CAL[b + 1], "gap_start": CAL[b - n_gap], "n_gap": n_gap})
    return pd.DataFrame(rows)


def test_measure_end_to_end():
    rows = rows_table()
    rows.loc[len(rows)] = {"row_id": "event|MISS|0", "kind": "event", "ticker": "MISS", "t_pre": CAL[50],
                           "t_0": CAL[51], "gap_start": CAL[48], "n_gap": 2}
    rows.loc[len(rows)] = {"row_id": "null|T00|nogap", "kind": "null", "ticker": "T00", "t_pre": CAL[170],
                           "t_0": CAL[171], "gap_start": pd.NaT, "n_gap": 0}
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        gap, oc = M.measure(rows, "insample_test", NB=fake_nb(no_put_volume={"T05"}), hard_stop="2024-10-01",
                            out_dir=out, batch_size=7, workers=4)
        check("gap file columns exactly as specified", list(pd.read_csv(out / "gap_insample_test.csv").columns) == M.GAP_COLS)
        check("outcome file columns exactly as specified",
              list(pd.read_csv(out / "outcome_insample_test.csv").columns) == M.OUTCOME_COLS)
        check("one gap row per row_id", len(gap) == len(rows) and gap.row_id.is_unique)
        check("null rows are kept", (gap.row_id.str.startswith("null|")).sum() == 25)
        ok = gap[gap.row_id.str.startswith(("event|T", "null|T")) & (gap.row_id != "null|T00|nogap")]
        check("clean synthetic rows are usable", ok.usable.all(), ok[~ok.usable].reason.head(3).tolist().__str__())
        check("usable means gap_move exists", (gap.usable == gap.gap_move.notna()).all())
        check("new diagnostic columns filled", (ok.spot_bucket == "1m|1m").all() and (ok.stale_sessions == 0).all()
              and (ok.n_eff == ok.row_id.map(rows.set_index("row_id").n_gap)).all() and (ok.n_baseline >= 2).all())
        miss = gap.set_index("row_id").loc["event|MISS|0"]
        check("a cache miss is flagged, not fatal", (not miss.usable) and "cache miss" in miss.reason, miss.reason)
        check("a row with no gap is flagged", not gap.set_index("row_id").loc["null|T00|nogap"].usable)
        r0 = gap.set_index("row_id").loc["event|T00|0"]
        S = spot_series("T00")
        check("end-to-end r_gap", abs(r0.r_gap - np.log(S[CAL[40]] / S[CAL[39]])) < 1e-9)
        check("gap_move formula", abs(r0.gap_move - abs(r0.r_gap - r0.r_mkt) / (r0.sigma_d * np.sqrt(r0.n_eff))) < 1e-12)
        check("no outcome date on or after the hard stop", (oc.exit_date < pd.Timestamp("2024-10-01")).all())
        check("horizons stop at the bucket's expiry",
              set(oc[oc.bucket == "1m"].horizon) <= {"1", "2", "3", "5", "10", "21", "expiry"})
        h = oc[(oc.bucket == "1m") & (oc.otm == 3) & (oc.horizon == "10") & oc.row_id.str.contains("T01")]
        check("y = log(rv / iv0)", np.allclose(h.y, np.log(h.rv / h.iv0)) and np.allclose(h.iv0, SIGMA))
        check("a put with no volume at entry is not usable",
              not oc[oc.row_id.str.contains("T05")].usable.any() and oc[oc.row_id.str.contains("T01")].usable.all())
        check("csp_net below gross, 2x below net", (oc.csp_net < oc.csp_gross).all() and (oc.csp_net2x < oc.csp_net).all())

        # resume: a rerun skips every finished row and gives the same tables
        gap2, oc2 = M.measure(rows, "insample_test", NB=fake_nb(no_put_volume={"T05"}), hard_stop="2024-10-01",
                              out_dir=out, batch_size=7, workers=4)
        start = [ln for ln in (out / "measure_progress.log").read_text().splitlines() if "already done" in ln][-1]
        check("rerun skips finished rows", f"({len(rows)} already done), 0 keys" in start, start)
        check("rerun gives identical tables", gap2.equals(gap) and oc2.equals(oc))
        rows2 = rows.copy()
        rows2.loc[0, "t_0"] = CAL[42]                 # one row's inputs change: only it is recomputed
        gap3, oc3 = M.measure(rows2, "insample_test", NB=fake_nb(no_put_volume={"T05"}), hard_stop="2024-10-01",
                              out_dir=out, batch_size=7, workers=4)
        last = [ln for ln in (out / "measure_progress.log").read_text().splitlines() if "already done" in ln][-1]
        check("a changed row is recomputed alone", f"({len(rows) - 1} already done)" in last, last)
        check("its entry date moved", (oc3[oc3.row_id == rows2.row_id[0]].entry_date == CAL[42]).all())
        try:
            M.measure(rows, "insample_test", NB=fake_nb(), hard_stop="2024-10-01", out_dir=out, mode="notebook")
            check("other settings refuse to mix with saved parts", False)
        except ValueError:
            check("other settings refuse to mix with saved parts", True)


def expect_refusal(name, fn):
    try:
        fn()
        check(name, False)
    except ValueError:
        check(name, True)


def test_gap_only():
    rows = rows_table(12)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        gap, oc = M.measure(rows, "insample_test", NB=fake_nb(), hard_stop="2024-10-01", out_dir=out, outcomes=False)
        check("gap-only: no outcome returned or written", oc is None and not (out / "outcome_insample_test.csv").exists()
              and not (out / "drops_insample_test.csv").exists())
        check("gap-only: gap file written, resumable store kept apart",
              (out / "gap_insample_test.csv").exists() and (out / "_parts_insample_test_gaponly").is_dir()
              and not (out / "_parts_insample_test").exists())
        gap2, _ = M.measure(rows, "insample_test", NB=fake_nb(), hard_stop="2024-10-01", out_dir=out)
        check("gap table is the same with and without outcomes", gap2.equals(gap))


def raises(fn, text: str = "") -> bool:
    try:
        fn()
    except ValueError as e:
        return text in str(e)
    return False


def test_window_guards():
    nb = fake_nb()
    b = M.bounds_for
    check("insample: window is 2024-01-01 .. 2026-01-01",
          b("insample", nb) == (pd.Timestamp("2024-01-01"), pd.Timestamp("2026-01-01")) == b("insample_gaps", nb))
    check("an explicit hard stop only tightens", b("insample", nb, "2025-06-01")[1] == pd.Timestamp("2025-06-01")
          and b("insample", nb, "2026-06-01")[1] == pd.Timestamp("2026-01-01"))
    for label in ("discovery", "dryrun", "discovery_nbmarks", "dryrun_x"):
        check(f"{label} is retired and refused with the stated reason",
              raises(lambda: b(label, nb), "2022-2023 is outside the allowed 2024-2025 window and overlaps the "
                                           "sealed placeholder (2023-06-01..2023-08-31)"))
    rows = rows_table(n_tickers=1)
    for label in ("discovery", "dryrun"):
        check(f"measure refuses {label} (no code path reads it)", raises(lambda: M.measure(rows, label, NB=fake_nb(), out_dir=None, hard_stop="2023-12-31"), "retired"))
    for label in ("x", "test", "insamples", ""):
        check(f"unknown label {label!r} refused", raises(lambda: b(label, nb), "unknown label"))
    check("main() offers only insample (the CLI refuses discovery and dryrun)",
          all(raises_exit(lambda: M.main([lb])) for lb in ("discovery", "dryrun")))

    # insample never overlaps the notebook's sealed window (HOLDOUT_START..HOLDOUT_END)
    for hs, he, refused in [("2023-06-01", "2023-08-31", False), ("2023-06-01", "2023-12-31", False),
                            ("2023-06-01", "2024-01-01", True), ("2024-03-01", "2024-05-31", True),
                            ("2025-12-01", "2026-02-28", True), ("2026-01-01", "2026-03-01", False)]:
        nbh = {**fake_nb(), "HOLDOUT_START": hs, "HOLDOUT_END": he}
        check(f"insample vs sealed window {hs}..{he}: {'refused' if refused else 'allowed'}",
              raises(lambda: b("insample", nbh), "overlaps") == refused)

    # row dates: t_pre and t_0 must lie in [2024-01-01, 2026-01-01); gap_start may not be on or after 2026-01-01
    # and, if before 2024-01-01, the row is only flagged (its bars are never read)
    for col, day, refused in [("t_pre", "2023-12-29", True), ("t_0", "2023-12-29", True), ("gap_start", "2026-01-02", True),
                              ("t_pre", "2026-01-02", True), ("t_0", "2026-01-02", True), ("t_pre", "2025-06-02", False),
                              ("t_0", "2025-06-02", False), ("gap_start", "2023-12-29", False)]:
        r = rows.copy()
        r.loc[0, col] = pd.Timestamp(day)
        check(f"insample: {col} on {day} {'refused' if refused else 'accepted'}",
              raises(lambda: M.measure(r, "insample", NB=fake_nb(), out_dir=None), "refused") == refused)
    r = rows.copy()
    r.loc[0, "gap_start"] = pd.Timestamp("2023-12-29")
    g, _ = M.measure(r, "insample", NB=fake_nb(), out_dir=None)
    check("a gap_start before the window is flagged unusable, not read",
          not g.usable.iloc[0] and "gap reaches before 2024-01-01" in g.reason.iloc[0] and np.isnan(g.r_gap.iloc[0]),
          g.reason.iloc[0])
    r = rows.copy()
    r.loc[0, "t_0"] = pd.Timestamp("2024-11-01")
    check("a t_0 on or after a tightened hard stop is refused",
          raises(lambda: M.measure(r, "insample", NB=fake_nb(), out_dir=None, hard_stop="2024-10-01"), "refused"))

    # holdout and oos: only with the notebook switch
    check("holdout refused without RUN_HOLDOUT", raises(lambda: M.measure(rows, "holdout", NB=fake_nb(), out_dir=None)))
    check("holdout refused when RUN_HOLDOUT is False",
          raises(lambda: M.measure(rows, "holdout", NB={**fake_nb(), "RUN_HOLDOUT": False}, out_dir=None)))
    gap, _ = M.measure(rows_table(12), "holdout", NB={**fake_nb(), "RUN_HOLDOUT": True}, out_dir=None)
    check("holdout runs when RUN_HOLDOUT is True (no floor: any dates the judges choose)",
          len(gap) == 48 and gap.usable.all() and b("holdout", {**fake_nb(), "RUN_HOLDOUT": True})[0] is None)
    check("oos refused when RUN_OOS is False", raises(lambda: M.measure(rows, "oos", NB=fake_nb(), out_dir=None)))
    import contextlib
    import io
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        warning = "WARNING: THE OUT-OF-SAMPLE SECTION IS ON.\nsigned, lalitha"
        gap, _ = M.measure(rows, "oos", NB={**fake_nb(), "RUN_OOS": True, "OOS_WARNING": warning}, out_dir=None)
    check("oos runs only with RUN_OOS True, and prints the warning", len(gap) == len(rows)
          and err.getvalue().strip().endswith("signed, lalitha"))


def raises_exit(fn) -> bool:
    import contextlib
    import io
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            fn()
        except SystemExit as e:
            return e.code not in (0, None)
    return False


def test_window_bars():
    """The notebook's option_bars is clipped to [2024-01-01, 2026-01-01) for the run and put back after."""
    calls, seen = [], {}

    def raw_bars(tk, start, end):
        calls.append((pd.Timestamp(start), pd.Timestamp(end)))
        idx = pd.bdate_range("2023-12-01", end)
        return pd.DataFrame({"close": 1.0, "volume": 1.0}, index=pd.DatetimeIndex(idx, name="session"))

    nb = fake_nb()
    nb["option_bars"] = raw_bars
    inner = nb["price_event"]

    def price_event(ticker, t_pre, t_0, event_date, buckets, otm):      # asks for bars the way price_event does
        seen[(ticker, t_pre)] = nb["option_bars"]("X", t_pre - pd.Timedelta(days=10), pd.Timestamp("2026-03-20"))
        return inner(ticker, t_pre, t_0, event_date, buckets, otm)

    nb["price_event"] = price_event
    rows = pd.DataFrame([{"row_id": "event|A", "kind": "event", "ticker": "T00", "t_pre": pd.Timestamp("2024-01-04"),
                          "t_0": pd.Timestamp("2024-01-05"), "gap_start": pd.Timestamp("2024-01-03"), "n_gap": 1}])
    M.measure(rows, "insample", NB=nb, out_dir=None)
    bars = seen[("T00", pd.Timestamp("2024-01-04"))]
    check("bars returned to the pricing code start at 2024-01-01 (nothing from 2023)", bars.index.min() >= pd.Timestamp("2024-01-01"),
          str(bars.index.min().date()))
    check("bars end before 2026-01-01 and the request ends at 2025-12-31 (cache URLs of the 2024-25 download)",
          bars.index.max() < pd.Timestamp("2026-01-01") and max(c[1] for c in calls) == pd.Timestamp("2025-12-31"))
    check("option_bars is restored after the run", nb["option_bars"] is raw_bars)
    nb2 = {**fake_nb(), "RUN_HOLDOUT": True, "option_bars": raw_bars}
    with M.window_bars(nb2, None, pd.Timestamp("2026-01-01")):
        check("no clipping without a floor (judges' window)", nb2["option_bars"] is raw_bars)


def test_floor_in_gap_inputs():
    floor = pd.Timestamp("2024-01-01")
    t_pre, gs = pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-02")
    pe, S = make_pe("BBB", t_pre, missing=[pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-01")])
    g0 = M.gap_inputs({"1m": pe}, gs, t_pre, 2, CAL)
    check("without a floor the stale mark from 2023-12-29 is used", g0["stale_sessions"] == 2 and g0["n_baseline"] == 5,
          f"{g0['stale_sessions']} {g0['n_baseline']}")
    g1 = M.gap_inputs({"1m": pe}, gs, t_pre, 2, CAL, floor=floor)
    check("with the floor no 2023 mark is used for the gap-start spot", g1["spot_bucket"] == "-|1m" and np.isnan(g1["r_gap"])
          and np.isnan(g1["n_eff"]), g1["spot_bucket"])
    check("with the floor the volume baseline counts only sessions from 2024-01-01",
          g1["n_baseline"] == 1 and np.isnan(g1["vol_gap_ratio"]), str(g1["n_baseline"]))
    pe2, S = make_pe("BBB", t_pre)
    g2 = M.gap_inputs({"1m": pe2}, pd.Timestamp("2023-12-29"), t_pre, 3, CAL, floor=floor)
    check("a gap_start before the floor gives no spot, sigma or IV from 2023",
          np.isnan(g2["r_gap"]) and np.isnan(g2["sigma_d"]) and np.isnan(g2["iv_gap_start"]) and np.isnan(g2["d_iv_gap"]))


def test_panel_window():
    """The insample r_mkt panel uses only 2024-2025 rows: 2022-23 and later rows are dropped, not priced."""
    rows = rows_table(n_tickers=12)
    mk = lambda tk, t_pre: {"ticker": tk, "t_pre": pd.Timestamp(t_pre), "t_0": pd.Timestamp(t_pre) + pd.offsets.BDay(1)}   # noqa: E731
    panel = pd.DataFrame([mk(f"A{i}", d) for i, d in enumerate(["2022-03-01", "2022-06-15", "2023-05-31", "2023-06-01",
                                                                "2023-08-31", "2023-12-29"])]            # 2022-23 playground rows
                         + [mk(f"B{i}", CAL[60 + i]) for i in range(20)]                              # 2024 rows
                         + [mk("C0", "2025-12-30"), mk("C1", "2026-01-02"), mk("C2", "2026-03-02")])    # edge and 2026 rows
    priced = []
    nb = fake_nb()
    inner = nb["price_event"]

    def price_event(ticker, t_pre, t_0, event_date, buckets, otm):
        priced.append((ticker, pd.Timestamp(t_pre)))
        return inner(ticker, t_pre, t_0, event_date, buckets, otm)

    nb["price_event"] = price_event
    with tempfile.TemporaryDirectory() as d:
        M.measure(rows, "insample", NB=nb, panel_rows=panel, out_dir=Path(d))
        log = (Path(d) / "measure_progress.log").read_text()
    extra = [(t, d) for t, d in priced if t[0] in "ABC" and t not in set(rows.ticker)]
    check("2022-23 panel rows are never priced", not any(t.startswith("A") for t, _ in extra), str(extra[:3]))
    check("panel rows from 2026 are never priced", not any(t in ("C1", "C2") for t, _ in extra))
    check("2024-25 panel rows are priced", len([1 for t, _ in extra if t.startswith("B")]) == 20
          and ("C0", pd.Timestamp("2025-12-30")) in extra)
    check("every priced date lies in 2024-01-01 .. 2025-12-31",
          all(pd.Timestamp("2024-01-01") <= d < pd.Timestamp("2026-01-01") for _, d in priced))
    check("the log counts the dropped panel rows", "offered 29, kept 21 inside the window (2024-01-01..2025-12-31), dropped 8" in log,
          [ln for ln in log.splitlines() if "panel rows offered" in ln][-1:].__str__())


def test_panel_cap():
    keys = [(f"T{j}", CAL[i]) for j in range(5) for i in range(200)]
    a, b = M.sample_panel(keys, 400), M.sample_panel(list(reversed(keys)), 400)
    check("panel sample: 400 keys, deterministic, from the input", len(a) == 400 and a == b and set(a) <= set(keys))
    check("panel sample: all keys when under the cap", M.sample_panel(keys[:10], 400) == sorted(keys[:10]))
    check("judges' path caps the panel at 400", M.PANEL_CAP == {"holdout": 400, "oos": 400})
    rows = rows_table(n_tickers=12)
    panel = pd.DataFrame({"ticker": [f"P{i}" for i in range(30)], "t_pre": [CAL[45 + i] for i in range(30)],
                          "t_0": [CAL[46 + i] for i in range(30)]})
    with tempfile.TemporaryDirectory() as d:
        for label, cap, want in [("holdout", 5, 5), ("holdout", "auto", 30), ("insample", "auto", 30)]:
            nb = {**fake_nb(), "RUN_HOLDOUT": True}
            M.measure(rows, label, NB=nb, panel_rows=panel, out_dir=Path(d), max_panel=cap)
            line = [ln for ln in (Path(d) / "measure_progress.log").read_text().splitlines() if "r_mkt panel" in ln][-1]
            check(f"{label} with max_panel={cap}: {want} panel-only keys, tickers per session printed",
                  f"({want} panel-only keys" in line and "tickers per session median" in line, line)
            import shutil
            shutil.rmtree(Path(d) / f"_parts_{label}")


if __name__ == "__main__":
    for fn in [test_read_table, test_parity_and_iv, test_realized_vol, test_csp_costs, test_gap_inputs,
               test_market_returns, test_measure_end_to_end, test_gap_only, test_window_guards, test_window_bars,
               test_floor_in_gap_inputs, test_panel_window, test_panel_cap]:
        fn()
    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    sys.exit(1 if FAILS else 0)
