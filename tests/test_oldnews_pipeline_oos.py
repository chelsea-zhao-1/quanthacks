"""Synthetic tests of the pipeline's one-time 2026 out-of-sample path (label "oos"): no real data, no network.

A fake notebook namespace and stand-in events and measure steps (with the 00_shared.md columns) drive the real
classify, tests and trade modules in a temporary folder that is deleted afterwards. Checks:
  1  the oos window: refused unless RUN_OOS is True, and outside OOS_START..OOS_END
  2  full text: "fetch" for oos with allow_fetch, "cache" without
  3  the request estimate is printed before any fetch and a run above the cap (default 40,000) is refused before
     anything is fetched
  4  the whole chain runs with RUN_OOS True and allow_fetch: events get full_text="fetch", the estimate is checked
     again before the option download, and *_oos files, results_oos/ and trade_oos/ are written
  5  the command line loads the notebook through nb.load() and refuses unless its RUN_OOS is True
Run:  .venv/Scripts/python.exe tests/test_oldnews_pipeline_oos.py
"""
import contextlib
import io
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
from pandas.tseries.offsets import CustomBusinessDay  # noqa: E402

from oldnews import pipeline as P  # noqa: E402
import oldnews.events as E  # noqa: E402
import oldnews.measure as M  # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


TMP = Path(tempfile.mkdtemp(prefix="oldnews_oos_test_"))
P.OUT, P.RAW, P.PLAYGROUND = TMP / "oldnews", TMP / "oldnews" / "raw", TMP / "playground"
CAL = pd.bdate_range("2023-06-01", "2027-12-31", freq=CustomBusinessDay())
LAST = CAL[CAL.searchsorted(pd.Timestamp("2026-10-02"))]
TOP = [f"T{i:02d}" for i in range(30)]
PEOPLE = {"ceo_departure": "CEO", "director_appointment": "director"}
PLACEBO = {"dividend_declaration", "bylaw_amendment"}
TAGS = sorted(set(PEOPLE) | PLACEBO | {"quarterly_earnings"})
OOS = ("2026-01-01", "2026-08-31")
CALLS = {"api": 0, "price": 0}


class Session:
    def get(self, *a, **k):
        raise AssertionError("network used")


def session_on_or_after(d):
    return CAL[CAL.searchsorted(pd.Timestamp(d))]


def build_events(tag, start, end, universe):
    CALLS["api"] += 1
    days = CAL[(CAL >= pd.Timestamp(start)) & (CAL <= pd.Timestamp(end))]
    r = np.random.default_rng(sum(map(ord, tag)))
    n = max(4, len(days) // 4)
    fd = pd.DatetimeIndex(sorted(r.choice(days, n)))
    tick = r.choice(universe, n)
    return pd.DataFrame({"cik": [sum(map(ord, t)) for t in tick], "filing_date": fd, "ticker": tick,
                         "accession_number": [f"{tag}-{i}" for i in range(n)], "filing_url": "u", "supporting_text": "x",
                         "accepted_at": fd + pd.to_timedelta(r.integers(8, 20, n), unit="h"),
                         "t_0": [session_on_or_after(d + pd.Timedelta(days=1)) for d in fd],
                         "t_pre": [CAL[CAL.searchsorted(d) - 1] for d in fd]})


def api_get_all(path, params=None):
    CALLS["api"] += 1
    return [{"tertiary_category": t} for t in TAGS]


def price_event(tk, tp, t0, ed, b, o):
    CALLS["price"] += 1
    return [], ["stand-in"]


def fake_nb(**over):
    nb = {"CAL": CAL, "LAST_SESSION": LAST, "session_on_or_after": session_on_or_after, "SESSION": Session(),
          "OOS_START": OOS[0], "OOS_END": OOS[1], "RUN_OOS": False, "RUN_HOLDOUT": False, "OOS_WARNING": "W",
          "HOLDOUT_START": "2023-06-01", "HOLDOUT_END": "2023-08-31", "TOP_100": TOP, "requests": requests,
          "option_bars": lambda *a: None, "fetch_acceptance_time": lambda u: None, "CACHE_DIR": TMP,
          "api_get_all": api_get_all, "build_events": build_events, "EXPIRY_BUCKETS": {"1m": (21, 45, 30)},
          "OTM_GRID": [0.03], "price_event": price_event,
          "fetch_disclosures": lambda tag, s, e: pd.DataFrame({"filing_date": pd.to_datetime([])})}
    nb.update(over)
    return nb


SEEN = {}


def s_events(label, NB=None, out_dir=None, playground=None, write=True, window=None, full_text=None):
    SEEN["events"] = {"label": label, "window": window, "full_text": full_text}
    raw = pd.read_csv(Path(playground) / f"events_{label}.csv", parse_dates=["filing_date", "accepted_at", "t_0", "t_pre"])
    drawn = pd.read_csv(Path(playground) / f"nulls_{label}.csv", parse_dates=["filing_date", "t_0", "t_pre"])
    rng = np.random.default_rng(7)
    tags = raw.tags.str.split("|").map(set)
    e = raw[tags.map(lambda t: bool(t & (set(PEOPLE) | PLACEBO)))].reset_index(drop=True)
    e["row_id"], e["kind"] = "event|" + e.accession_number, "event"
    is_people = e.tags.map(lambda t: bool(set(t.split("|")) & set(PEOPLE)))
    e["group"] = np.where(is_people, "people", "placebo")
    e["role"] = np.where(is_people, "CEO", "placebo")
    e["event_date"] = e.filing_date - pd.to_timedelta(rng.integers(1, 5, len(e)), unit="D")
    e["gap_start"] = [CAL[CAL.searchsorted(d) - 1] for d in e.event_date]
    e["n_gap"] = [CAL.searchsorted(b) - CAL.searchsorted(a) for a, b in zip(e.gap_start, e.t_pre)]
    e["lag_bd"], e["late"] = e["n_gap"], (e["n_gap"] >= 1).astype(int)
    e["earnings_excluded"] = (rng.random(len(e)) < 0.1).astype(int)
    for c in ("cue_dated_prior", "cue_prior_wording", "cue_related_filing"):
        e[c] = (rng.random(len(e)) < 0.3).astype(int)
    e["T"] = e[["cue_dated_prior", "cue_prior_wording", "cue_related_filing"]].sum(axis=1)
    e["text_source"] = "excerpt"
    e = e[e.gap_start >= pd.Timestamp(OOS[0])].reset_index(drop=True)
    n = e[["row_id", "accession_number", "ticker", "filing_date", "n_gap"]].rename(columns={"row_id": "event_row_id"})
    n = n.merge(drawn[["ticker", "filing_date", "round", "t_0", "t_pre"]], on=["ticker", "filing_date"])
    n["gap_start"] = [CAL[CAL.searchsorted(t) - k] for t, k in zip(n.t_pre, n.n_gap)]
    n["row_id"] = ("null|" + n.ticker + "|" + n.t_pre.dt.strftime("%Y-%m-%d") + "|" + n.t_0.dt.strftime("%Y-%m-%d")
                   + "|" + n["round"].astype(str) + "|" + n.accession_number)
    n["kind"] = "null"
    n = n[n.gap_start >= pd.Timestamp(OOS[0])][["row_id", "kind", "event_row_id", "ticker", "t_pre", "t_0", "gap_start",
                                                 "n_gap", "round"]]
    if write:
        e.to_csv(Path(out_dir) / f"events_{label}.csv", index=False)
        n.to_csv(Path(out_dir) / f"nulls_{label}.csv", index=False)
    return e, n


def s_measure(rows, label, NB=None, panel_rows=None, hard_stop=None, out_dir=None):
    SEEN["measure"] = {"label": label, "hard_stop": hard_stop}
    rng = np.random.default_rng(11)
    gap = pd.DataFrame({"row_id": rows.row_id, "r_gap": 0.0, "r_mkt": 0.0, "sigma_d": 0.015,
                        "gap_move": rng.gamma(2, 0.3, len(rows)), "iv_gap_start": 0.25, "iv_tpre": 0.26,
                        "d_iv_gap": rng.normal(0.0, 0.04, len(rows)), "vol_gap_ratio": rng.gamma(2, 2.0, len(rows)),
                        "usable": True, "reason": ""})
    out = []
    for r in rows.itertuples():
        i0 = CAL.searchsorted(r.t_0)
        for h in ("1", "2", "3", "5", "10", "21", "42", "63", "expiry"):
            exit_ = CAL[i0 + (21 if h == "expiry" else int(h))]
            if exit_ > LAST:
                continue
            for otm in (3, 5, 10):
                pnl = rng.normal(0.002, 0.01)
                out.append({"row_id": r.row_id, "bucket": "1m", "horizon": h, "otm": otm, "entry_date": r.t_0,
                            "exit_date": exit_, "iv0": 0.25, "rv": 0.2, "y": rng.normal(0, 0.3), "put_strike": 97.0,
                            "put_premium": 1.2, "put_volume_entry": float(rng.integers(1, 500)), "csp_gross": pnl,
                            "csp_net": pnl - 0.0012, "csp_net2x": pnl - 0.0024, "usable": True})
    gap.to_csv(Path(out_dir) / f"gap_{label}.csv", index=False)
    pd.DataFrame(out).to_csv(Path(out_dir) / f"outcome_{label}.csv", index=False)
    return gap, pd.DataFrame(out)


def refused(fn, text: str) -> bool:
    try:
        fn()
    except (PermissionError, RuntimeError) as e:
        return text in str(e)
    return False


real = P.default_steps()
steps = P.Steps(events=s_events, measure=s_measure, classify=real.classify,
                tests=lambda label, data_dir, NB=None: real.tests(label, data_dir, NB, n_perm=500, n_boot=500),
                trade=real.trade)

try:
    # ---- 1 the window ------------------------------------------------------------------------------------------
    check("oos refused without RUN_OOS", refused(lambda: P.window_rules(fake_nb(), *OOS, "oos"), "RUN_OOS = True"))
    on = fake_nb(RUN_OOS=True)
    check("oos refused outside OOS_START..OOS_END",
          refused(lambda: P.window_rules(on, "2025-12-01", OOS[1], "oos"), "runs on")
          and refused(lambda: P.window_rules(on, OOS[0], "2026-09-30", "oos"), "runs on"))
    check("oos window accepted with RUN_OOS", P.window_rules(on, *OOS, "oos").hard_stop is None)

    # ---- 2 full text -----------------------------------------------------------------------------------------------
    check("full text: fetch with allow_fetch, cache without",
          P.full_text_mode("oos", True) == "fetch" and P.full_text_mode("oos", False) == "cache"
          and P.full_text_mode("insample", True) == "cache")

    # ---- 3 the cap, before any fetch -----------------------------------------------------------------------------------------------
    orig_e, orig_m = getattr(E, "estimate_requests", None), getattr(M, "estimate_requests", None)
    E.estimate_requests = lambda window, label, NB, cache_dir=None: {
        "massive_pages": 120, "sec_headers": 30_000, "sec_full_text": 15_000, "massive_more_pages_possible": [],
        "unknown_filings": False}
    CALLS.update(api=0, price=0)
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        ok = refused(lambda: P.run_oldnews(*OOS, "oos", allow_fetch=True, NB=fake_nb(RUN_OOS=True), steps=steps,
                                           out_dir=TMP / "cap"), "exceeds the cap of 40,000")
    check("default cap 40,000 refuses before anything is fetched", ok and CALLS == {"api": 0, "price": 0},
          f"calls {CALLS}")
    check("the estimate is printed before the refusal", "request estimate before any fetch" in out.getvalue()
          and "~45,120" in out.getvalue(), out.getvalue()[-300:])
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        ok = refused(lambda: P.run_oldnews(*OOS, "oos", allow_fetch=True, NB=fake_nb(RUN_OOS=True), steps=steps,
                                           max_requests=10_000, out_dir=TMP / "cap2"), "exceeds the cap of 10,000")
    check("a lower cap argument is honoured", ok)

    # ---- 4 the whole chain ---------------------------------------------------------------------------------------------------------
    E.estimate_requests = lambda window, label, NB, cache_dir=None: {
        "massive_pages": 130, "sec_headers": 500, "sec_full_text": 200, "massive_more_pages_possible": [],
        "unknown_filings": False}
    M.estimate_requests = lambda rows, label, NB, panel_rows=None, max_panel="auto": {
        "keys": 900, "cached": 0, "uncached": 900, "requests_estimate": 900 * 25}
    CALLS.update(api=0, price=0)
    out_dir = TMP / "run"
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        res = P.run_oldnews(*OOS, "oos", allow_fetch=True, NB=fake_nb(RUN_OOS=True), steps=steps, out_dir=out_dir,
                            figures=False)
    text = out.getvalue()
    check("events got label oos, the OOS window and full_text='fetch'",
          SEEN["events"] == {"label": "oos", "window": OOS, "full_text": "fetch"}, str(SEEN.get("events")))
    check("measure ran for oos with no hard stop", SEEN["measure"] == {"label": "oos", "hard_stop": None})
    check("estimate checked before any fetch and again before the option download",
          "before any fetch" in text and "before the option download" in text and "measure.estimate_requests" in text)
    check("the option download ran after the second check (stand-in price_event)", CALLS["price"] > 0)
    check("every stage ran", all(res["stages"].get(s) == "ran" for s in ("inputs", "events", "measure", "classify",
                                                                         "tests", "trade")), str(res["stages"]))
    files = ["events_oos.csv", "nulls_oos.csv", "gap_oos.csv", "outcome_oos.csv", "classified_oos.csv",
             "results_oos/h1.csv", "results_oos/summary.md", "trade_oos/summary.md", "run_oos.json"]
    missing = [f for f in files if not (out_dir / f).exists()]
    check("*_oos files, results_oos/ and trade_oos/ written", not missing, str(missing))
    check("nothing written to data/oldnews", not (ROOT / "data" / "oldnews" / "events_oos.csv").exists())
    if orig_e is None:
        del E.estimate_requests
    else:
        E.estimate_requests = orig_e
    if orig_m is None:
        del M.estimate_requests
    else:
        M.estimate_requests = orig_m

    # ---- 5 the command line ---------------------------------------------------------------------------------------------------------
    calls = []
    saved_run, saved_nb, saved_argv = P.run_oldnews, sys.modules.get("nb"), sys.argv
    P.run_oldnews = lambda *a, **k: calls.append((a, k)) or {"label": "oos", "start": a[0], "end": a[1]}
    try:
        sys.modules["nb"] = types.SimpleNamespace(load=lambda: fake_nb(RUN_OOS=False))
        sys.argv = ["pipeline.py", "--label", "oos", "--allow-fetch", "--max-requests", "30000"]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = P.main()
        check("CLI refuses when the notebook's RUN_OOS is not True", code == 2 and not calls)
        sys.modules["nb"] = types.SimpleNamespace(load=lambda: fake_nb(RUN_OOS=True))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = P.main()
        a, k = calls[0] if calls else ((), {})
        check("CLI with RUN_OOS runs OOS_START..OOS_END with the cap and allow_fetch",
              code == 0 and a[:3] == (*OOS, "oos") and k.get("allow_fetch") is True and k.get("max_requests") == 30000
              and k.get("NB", {}).get("RUN_OOS") is True, str((a, {x: y for x, y in k.items() if x != "NB"})))
    finally:
        P.run_oldnews, sys.argv = saved_run, saved_argv
        if saved_nb is None:
            sys.modules.pop("nb", None)
        else:
            sys.modules["nb"] = saved_nb
        os.chdir(ROOT)
finally:
    shutil.rmtree(TMP, ignore_errors=True)

print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
