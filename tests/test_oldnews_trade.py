"""Synthetic tests for src/oldnews/trade.py (no real data, no network). Run:
    .venv/Scripts/python.exe tests/test_oldnews_trade.py
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402

from oldnews import trade     # noqa: E402

FAILS = []
DAYS = pd.bdate_range("2024-01-02", "2025-12-31")


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{': ' + detail if detail else ''}")
    if not ok:
        FAILS.append(name)


def synthetic(n_events=120, effect=0.0, seed=0):
    """Events, nulls and outcome tables with the exact column names of 00_shared.md, plus a labels table.
    `effect` is added to the old-news events' put P&L (every horizon, bucket, otm)."""
    rng = np.random.default_rng(seed)
    ev, nu = [], []
    for i in range(n_events):
        p = int(rng.integers(5, len(DAYS) - 90))
        tk = f"T{i % 12}"
        ev.append({"row_id": f"event|{i:06d}", "kind": "event", "accession_number": f"{i:06d}", "ticker": tk,
                   "filing_date": DAYS[p].date(), "t_pre": DAYS[p - 1].date(), "t_0": DAYS[p].date(), "pos": p,
                   "late": rng.random() < 0.8, "group": "people" if rng.random() < 0.85 else "placebo",
                   "earnings_excluded": rng.random() < 0.1, "old": rng.random() < 0.5})
        for rnd in (1, 2):
            q = int(np.clip(p + rng.integers(-40, 40), 5, len(DAYS) - 90))
            nu.append({"row_id": f"null|{tk}|{DAYS[q - 1].date()}|{DAYS[q].date()}|{rnd}", "kind": "null",
                       "event_row_id": f"event|{i:06d}", "ticker": tk, "t_pre": DAYS[q - 1].date(),
                       "t_0": DAYS[q].date(), "n_gap": 3, "round": rnd, "pos": q})
    events, nulls = pd.DataFrame(ev), pd.DataFrame(nu)
    old_ids = set(events.loc[events["old"], "row_id"])
    rows = []
    for r in pd.concat([events, nulls]).itertuples():
        vol = int(rng.choice([0, 4, 30, 250, 2000]))
        for bucket in ("1m", "2m"):
            for otm in (3, 5, 10):
                strike, prem = 100 * (1 - otm / 100), 1.5 - otm / 10
                cost = max(0.05 * prem, 0.05) / strike             # one side, fraction of collateral
                for hz in trade.HORIZONS:
                    k = 17 if hz == "expiry" else int(hz)
                    gross = rng.normal(0.003, 0.01) + (effect if r.row_id in old_ids else 0.0)
                    rows.append({"row_id": r.row_id, "bucket": bucket, "horizon": hz, "otm": otm,
                                 "entry_date": DAYS[r.pos].date(), "exit_date": DAYS[r.pos + k].date(),
                                 "iv0": 0.3, "rv": 0.25, "y": np.log(0.25 / 0.3), "put_strike": strike,
                                 "put_premium": prem, "put_volume_entry": vol, "csp_gross": gross,
                                 "csp_net": gross - 2 * cost, "csp_net2x": gross - 4 * cost,
                                 "usable": rng.random() < 0.95})
    labels = events[["row_id", "old"]].copy()
    return events.drop(columns=["pos", "old"]), nulls.drop(columns="pos"), pd.DataFrame(rows), labels


def write(tmp: Path, label: str, events, nulls, outcome):
    for name, df in (("events", events), ("nulls", nulls), ("outcome", outcome)):
        df.to_csv(tmp / f"{name}_{label}.csv", index=False)


def books_for(events, nulls, outcome, labels):
    ev, _ = trade.eligible_events(events, labels)
    return ev, trade.build_trades(ev, nulls, trade.prep_outcome(outcome))


# ------------------------------------------------------------------ filters

def test_filters():
    events, nulls, outcome, labels = synthetic()
    ev, (trades, counts) = books_for(events, nulls, outcome, labels)
    lab = events.merge(labels, on="row_id")
    want_ev = lab[(lab["group"] == "people") & lab["late"] & ~lab["earnings_excluded"]]
    o = outcome[(outcome["bucket"] == "1m") & (outcome["otm"] == 3) & (outcome["horizon"] == "10")
                & outcome["usable"] & (outcome["put_volume_entry"] > 0)]
    want_old = set(want_ev.loc[want_ev["old"], "row_id"]) & set(o["row_id"])
    got = trades[(trades["book"] == "old") & (trades["horizon"] == "10")]
    check("old book = old, late, people, no earnings, usable, volume > 0", set(got["row_id"]) == want_old,
          f"{len(got)} rows")
    want_all = set(want_ev["row_id"]) & set(o["row_id"])
    got_all = trades[(trades["book"] == "all_late") & (trades["horizon"] == "10")]
    check("all_late book = every eligible event, old or not", set(got_all["row_id"]) == want_all)
    nul = trades[(trades["book"] == "null_r1") & (trades["horizon"] == "10")]
    check("null books only hold ordinary days of tradeable old-news events",
          set(nul["event_row_id"]) <= want_old and len(nul) > 0)
    check("null book rows also need usable outcome and volume > 0", set(nul["row_id"]) <= set(o["row_id"]))
    check("placebo filings never traded", not trades["row_id"].isin(events.loc[events["group"] == "placebo", "row_id"]).any())
    check("every horizon present", set(trades["horizon"]) == set(trade.HORIZONS))
    c = counts[(counts["book"] == "old") & (counts["horizon"] == "10")].set_index("step")["kept"]
    check("counts record each step", c["put volume at entry > 0"] == len(got) and c["rows in book"] >= len(got))


def test_gap_and_label_filters():
    events, nulls, outcome, labels = synthetic(n_events=30)
    gap = pd.DataFrame({"row_id": events["row_id"], "usable": [i % 3 != 0 for i in range(len(events))]})
    labels = labels.astype({"old": object})
    labels.loc[labels.index[:4], "old"] = np.nan                       # unclassified events
    ev, counts = trade.eligible_events(events, labels, gap=gap)
    bad_gap = set(gap.loc[~gap["usable"], "row_id"])
    check("events with unusable gap dropped", not ev["row_id"].isin(bad_gap).any())
    check("unclassified events dropped", not ev["row_id"].isin(labels["row_id"][:4]).any())
    check("label counts include 'of which old news'", counts[-1]["step"] == "of which old news")


# ------------------------------------------------------------------ the book

def _toy(entries, exits, pnl, ids=None):
    n = len(entries)
    return pd.DataFrame({"row_id": ids or [f"r{i}" for i in range(n)], "ticker": "X",
                         "entry_date": pd.to_datetime(entries), "exit_date": pd.to_datetime(exits),
                         "put_strike": 97.0, "put_premium": 1.0, "put_volume_entry": 100,
                         "csp_net": pnl, "csp_net2x": [p - 0.01 for p in pnl]})


def test_cap():
    t = _toy(["2024-01-03"] * 7, ["2024-01-10"] * 7, [0.01] * 7, ids=[f"r{i}" for i in (6, 5, 4, 3, 2, 1, 0)])
    taken = trade.simulate(t)
    check("cap: 7 same-day entries -> 5 taken", taken.sum() == 5)
    check("cap: ties taken by row_id", set(t.loc[taken, "row_id"]) == {"r0", "r1", "r2", "r3", "r4"})
    t2 = pd.concat([t, _toy(["2024-01-10", "2024-01-07"], ["2024-01-20", "2024-01-20"], [0.0, 0.0],
                            ids=["a_frees", "b_full"])], ignore_index=True)
    tk2 = trade.simulate(t2)
    check("cap: an exit frees its slot for an entry the same day", bool(tk2[t2["row_id"] == "a_frees"].iloc[0]))
    check("cap: no slot while 5 are open", not bool(tk2[t2["row_id"] == "b_full"].iloc[0]))

    events, nulls, outcome, labels = synthetic(n_events=200, seed=3)
    _, (trades, _) = books_for(events, nulls, outcome, labels)
    g = trades[(trades["book"] == "all_late") & (trades["horizon"] == "21")]
    tk = g[trade.simulate(g)]
    worst_open = max(((tk["entry_date"] <= d) & (tk["exit_date"] > d)).sum() for d in DAYS)
    check("cap: never more than 5 open on real-sized synthetic book", worst_open <= 5, f"max open {worst_open}")
    check("cap binds on the synthetic book", len(tk) < len(g), f"{len(tk)} of {len(g)}")
    shuffled = g.assign(csp_net=np.random.default_rng(1).permutation(g["csp_net"].to_numpy()))
    check("selection ignores P&L (same trades at any P&L, so at 1x and 2x)",
          trade.simulate(shuffled).equals(trade.simulate(g)))


def test_equity_and_metrics():
    t = _toy(["2024-01-03", "2024-01-04"], ["2024-01-14", "2024-01-18"], [-0.10, 0.05])
    marks = pd.DataFrame({"row_id": ["r0", "r0", "r1"], "mark_date": pd.to_datetime(["2024-01-05", "2024-01-07",
                                                                                     "2024-01-06"]),
                          "csp_net": [-0.20, -0.15, 0.02], "csp_net2x": [-0.21, -0.16, 0.01]})
    c = trade.equity(t, marks, "csp_net")
    check("closed equity ends at sum(pnl)/5", np.isclose(c["closed_pct"].iloc[-1], 100 * (-0.05) / 5))
    check("MTM equity ends where closed equity ends", np.isclose(c["mtm_pct"].iloc[-1], c["closed_pct"].iloc[-1]))
    on = c.set_index("date")
    check("MTM carries the latest mark", np.isclose(on.loc["2024-01-06", "mtm_pct"], 100 * (-0.20 + 0.02) / 5))
    check("n_open counts open positions", on.loc["2024-01-04", "n_open"] == 2 and on["n_open"].iloc[-1] == 0)
    m = trade.book_metrics(t, c, "csp_net")
    check("max drawdown closed = -2% (one -10% trade in a 5-slot book)", np.isclose(m["max_dd_closed_pct"], -2.0))
    check("max drawdown MTM is deeper (-4% at the worst mark)", np.isclose(m["max_dd_mtm_pct"], -4.0)
          and m["max_dd_mtm_pct"] < m["max_dd_closed_pct"])
    check("hit rate 50%", m["hit_rate_pct"] == 50.0)
    check("total return -1%", np.isclose(m["total_return_pct"], -1.0))
    check("drawdown is measured from the peak", np.isclose(trade.max_drawdown(pd.Series([10.0, 0.0])), -100 / 11))
    w = trade.worst(_toy(["2024-01-03"] * 7, ["2024-01-10"] * 7, [0.03, -0.2, 0.01, -0.05, 0.0, -0.01, 0.02]),
                    "csp_net")
    check("five worst, worst first", len(w) == 5 and w["pnl_pct"].is_monotonic_increasing
          and np.isclose(w["pnl_pct"].iloc[0], -20.0))
    empty = trade.equity(t.iloc[:0], None, "csp_net")
    check("empty book gives NaN metrics, no crash", np.isnan(trade.book_metrics(t.iloc[:0], empty, "csp_net")["max_dd_closed_pct"]))


def test_capacity():
    t = _toy(["2024-01-03"] * 3, ["2024-01-10"] * 3, [0.01, 0.02, -0.01])
    t["put_volume_entry"] = [25, 5, 1000]
    cap = trade.capacity(t)
    check("contracts = floor(10% of volume)", cap["max_contracts"].tolist() == [2, 0, 100])
    check("collateral = contracts x strike x 100", np.allclose(cap["max_collateral_usd"], [2 * 9700, 0, 100 * 9700]))
    s = trade.capacity_summary(cap)
    check("capacity summary: median, total, zero-capacity count",
          s["median_collateral_usd"] == 19400 and s["total_collateral_usd"] == 102 * 9700 and s["n_zero_contracts"] == 1)


# ------------------------------------------------------------------ H2

def test_h2():
    events, nulls, outcome, labels = synthetic(n_events=300, effect=0.02, seed=5)
    _, (trades, _) = books_for(events, nulls, outcome, labels)
    h = trade.h2(trades).set_index(["comparison", "horizon", "cost"])
    r = h.loc[("old_minus_null", "10", "1x")]
    check("H2 vs nulls finds a planted +2% edge", 1.0 < r["diff_pct"] < 3.0 and r["ci_lo_pct"] > 0 and r["p_boot"] < 0.01,
          f"diff {r['diff_pct']:.2f}% CI [{r['ci_lo_pct']:.2f}, {r['ci_hi_pct']:.2f}] p {r['p_boot']:.4f}")
    a = h.loc[("old_minus_all_late", "10", "1x")]
    check("H2 vs all late events is positive with a planted edge", a["diff_pct"] > 0 and a["ci_lo_pct"] > 0)
    check("2x costs give the same paired difference here (costs equal per row)",
          np.isclose(h.loc[("old_minus_null", "10", "2x"), "diff_pct"], r["diff_pct"]))
    check("q-values present and >= p", (h["q_bh"].dropna() >= h["p_boot"].dropna() - 1e-12).all())

    events, nulls, outcome, labels = synthetic(n_events=300, effect=0.0, seed=6)
    _, (trades0, _) = books_for(events, nulls, outcome, labels)
    h0 = trade.h2(trades0)
    r0 = h0.set_index(["comparison", "horizon", "cost"]).loc[("old_minus_null", "10", "1x")]
    check("H2 with no edge: interval covers 0", r0["ci_lo_pct"] < 0 < r0["ci_hi_pct"])
    check("H2 is reproducible (seeded)", trade.h2(trades0).equals(h0))


# ------------------------------------------------------------------ guards and run

def expect_refused(name, fn, exc=PermissionError, match=""):
    try:
        fn()
    except exc as e:
        check(name, match in str(e), str(e)[:70])
    else:
        check(name, False, "no error raised")


def test_window_guard():
    events, nulls, outcome, labels = synthetic(n_events=40, seed=7)
    o = trade.prep_outcome(outcome)
    ok = {"events": events, "nulls": nulls, "outcome": o}
    trade.check_dates(ok, "insample", {})
    check("insample: clean 2024-25 tables pass", True)

    def bump(frame, col, value, usable=None):
        f = {k: v.copy() for k, v in ok.items()}
        f[frame].loc[f[frame].index[0], col] = pd.Timestamp(value) if frame == "outcome" else value
        if usable is not None:
            f[frame].loc[f[frame].index[0], "usable"] = usable
        return f

    expect_refused("insample: outcome entry in 2023 refused", lambda: trade.check_dates(
        bump("outcome", "entry_date", "2023-12-29"), "insample", {}), match="before 2024-01-01")
    expect_refused("insample: event gap_start in 2023 refused", lambda: trade.check_dates(
        bump("events", "gap_start", "2023-12-29"), "insample", {}), match="before 2024-01-01")
    expect_refused("insample: ordinary-day t_pre in 2026 refused", lambda: trade.check_dates(
        bump("nulls", "t_pre", "2026-01-02"), "insample", {}), match="on or after 2026-01-01")
    expect_refused("insample: usable exit in 2026 refused", lambda: trade.check_dates(
        bump("outcome", "exit_date", "2026-01-05", usable=True), "insample", {}), match="on or after 2026-01-01")
    trade.check_dates(bump("outcome", "exit_date", "2026-01-05", usable=False), "insample", {})
    check("insample: an unusable row's exit is never used", True)
    sealed = {"HOLDOUT_START": "2024-06-03", "HOLDOUT_END": "2024-08-30"}
    expect_refused("insample: dates inside the notebook's HOLDOUT window refused", lambda: trade.check_dates(
        bump("outcome", "entry_date", "2024-07-15"), "insample", sealed), match="sealed window")
    trade.check_dates(ok, "insample", {"HOLDOUT_START": "2023-06-01", "HOLDOUT_END": "2023-08-31"})
    check("insample: a holdout window outside 2024-25 does not interfere", True)
    check("placeholder holdout is 2023-06-01..2023-08-31",
          trade.holdout_window({}) == (pd.Timestamp("2023-06-01"), pd.Timestamp("2023-08-31")))
    check("holdout window is read from the notebook namespace",
          trade.holdout_window(sealed) == (pd.Timestamp("2024-06-03"), pd.Timestamp("2024-08-30")))

    for lab in ("discovery", "dryrun"):
        expect_refused(f"{lab}: retired, check_label refuses", lambda lab=lab: trade.check_label(lab, {}),
                       match="outside the allowed 2024-2025 window")
        expect_refused(f"{lab}: load refuses before touching any file", lambda lab=lab: trade.load(lab),
                       match="2023-06-01..2023-08-31")
        expect_refused(f"{lab}: run refuses", lambda lab=lab: trade.run(lab, labels, NB={"RUN_HOLDOUT": True}),
                       match="overlaps the sealed placeholder")
    expect_refused("unknown label refused", lambda: trade.check_label("anything", {}), match="unknown label")
    expect_refused("holdout: refused when RUN_HOLDOUT is missing", lambda: trade.check_label("holdout", {}),
                   match="RUN_HOLDOUT")
    expect_refused("holdout: refused when RUN_HOLDOUT is False",
                   lambda: trade.check_label("holdout", {"RUN_HOLDOUT": False}), match="RUN_HOLDOUT")
    expect_refused("holdout: refused with only a truthy RUN_HOLDOUT",
                   lambda: trade.check_label("holdout", {"RUN_HOLDOUT": 1}), match="RUN_HOLDOUT")
    trade.check_label("holdout", {"RUN_HOLDOUT": True})
    check("holdout: allowed when RUN_HOLDOUT is True", True)
    expect_refused("holdout: RUN_OOS does not unlock it", lambda: trade.check_label("holdout", {"RUN_OOS": True}),
                   match="RUN_HOLDOUT")
    expect_refused("oos: refused unless RUN_OOS is True", lambda: trade.check_label("oos", {"RUN_HOLDOUT": True}),
                   match="RUN_OOS")
    trade.check_label("oos", {"RUN_OOS": True})
    check("oos: allowed when RUN_OOS is True", True)
    expect_refused("oos: refused with the script's own namespace (no switch)", lambda: trade.check_label("oos"),
                   match="RUN_OOS")
    with tempfile.TemporaryDirectory() as d:                      # refusal happens before any file is read
        expect_refused("oos: run refuses before reading files",
                       lambda: trade.run("oos", labels, data_dir=Path(d), NB={}), match="RUN_OOS")
        expect_refused("holdout: run refuses before reading files",
                       lambda: trade.run("holdout", labels, data_dir=Path(d)), match="RUN_HOLDOUT")
    # holdout with the switch on may use any dates (the judges choose them)
    ev2, nu2, out2, lab2 = synthetic(n_events=60, seed=8)
    shift = pd.DateOffset(years=-2)
    for df, cols in ((ev2, ["filing_date", "t_pre", "t_0"]), (nu2, ["t_pre", "t_0"]), (out2, ["entry_date", "exit_date"])):
        for c in cols:
            df[c] = pd.to_datetime(df[c]) + shift
    with tempfile.TemporaryDirectory() as d:
        write(Path(d), "holdout", ev2, nu2, out2)
        tables = trade.run("holdout", lab2, data_dir=Path(d), NB={"RUN_HOLDOUT": True}, log=False)
        check("holdout: runs on judges' dates when RUN_HOLDOUT is True", len(tables["trades"]) > 0)


def test_run_insample():
    events, nulls, outcome, labels = synthetic(n_events=60, seed=7)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        write(tmp, "insample", events, nulls, outcome)
        tables = trade.run("insample", labels, data_dir=tmp)
        out = tmp / "trade_insample"
        names = ["counts", "trades", "summary", "by_year", "by_quarter", "worst", "equity", "capacity", "h2", "metrics"]
        check("run writes every table and summary.md",
              all((out / f"{n}.csv").exists() for n in names) and (out / "summary.md").exists())
        s = tables["summary"]
        check("summary has every book x horizon x cost",
              len(s) == s["book"].nunique() * len(trade.HORIZONS) * 2
              and {"old", "all_late", "null_r1", "null_r2"} <= set(s["book"]))
        hd = s[(s["book"] == "old") & (s["horizon"] == "10")].set_index("cost")
        check("2x costs never beat 1x", hd.loc["2x", "total_return_pct"] < hd.loc["1x", "total_return_pct"])
        check("same trades at 1x and 2x", hd.loc["2x", "n_trades"] == hd.loc["1x", "n_trades"])
        y = tables["by_year"]
        yo = y[(y["book"] == "old") & (y["horizon"] == "10") & (y["cost"] == "1x")]
        check("2024 and 2025 reported separately and add up to the pooled book", set(yo["year"]) == {2024, 2025}
              and yo["n_trades"].sum() == hd.loc["1x", "n_trades"]
              and np.isclose(yo["total_return_pct"].sum(), hd.loc["1x", "total_return_pct"]))
        check("by-year table covers every book and both costs",
              set(y["book"]) == set(s["book"]) and set(y["cost"]) == {"1x", "2x"} and set(y["year"]) == {2024, 2025})
        q = tables["by_quarter"]
        qo = q[(q["book"] == "old") & (q["horizon"] == "10") & (q["cost"] == "1x")]
        check("by-quarter table: quarters lie in 2024Q1..2025Q4 and add up to the pooled book",
              set(qo["quarter"]) <= {f"{yr}Q{k}" for yr in (2024, 2025) for k in (1, 2, 3, 4)}
              and np.isclose(qo["total_return_pct"].sum(), hd.loc["1x", "total_return_pct"]))
        wq = trade.worst_quarter(q).set_index("cost")
        check("worst quarter is the lowest quarter at each cost level",
              np.isclose(wq.loc["1x", "total_return_pct"], qo["total_return_pct"].min())
              and set(wq.index) == {"1x", "2x"})
        t = tables["trades"]
        lo, hi = pd.Timestamp("2024-01-01"), pd.Timestamp("2026-01-01")
        used = pd.concat([pd.to_datetime(t["entry_date"]), pd.to_datetime(t["exit_date"])])
        check("every trade date lies in [2024-01-01, 2026-01-01) (no panel or 2022-23 rows)",
              ((used >= lo) & (used < hi)).all())
        led = pd.read_csv(tmp / "ledger.csv")
        check("every book and H2 variant logged to the ledger",
              len(led) == len(s) // 2 + len(tables["h2"]) // 2 + len(tables["metrics"]) // 2
              and set(led["kind"]) == {"trade_book", "H2", "trade_risk"}
              and (led["subset"] == "insample").all())
        md = (out / "summary.md").read_text(encoding="utf-8")
        check("summary.md: pooled 2024-25 is the primary result, by-year alongside",
              "pooled 2024-25 (primary)" in md and "by entry year" in md and "worst calendar quarter" in md
              and "Five worst" in md)
        check("summary.md has no 2022 stress section", "2022" not in md and "discovery" not in md.lower())
        tables2 = trade.run("insample", labels, data_dir=tmp, log=False)
        check("run is reproducible", tables2["summary"].equals(tables["summary"]) and tables2["h2"].equals(tables["h2"]))

        # the pipeline calls run(label): labels come from classify's classified_<label>.csv, unscored dropped
        cl = labels.assign(scored=[i % 5 != 0 for i in range(len(labels))])
        cl.loc[~cl["scored"], "old"] = False                   # classify's convention for unscored events
        cl.to_csv(tmp / "classified_insample.csv", index=False)
        tables3 = trade.run("insample", data_dir=tmp, log=False)
        unscored = set(cl.loc[~cl["scored"], "row_id"])
        check("run(label) reads classified_<label>.csv and drops unscored events",
              len(tables3["trades"]) > 0 and not tables3["trades"]["row_id"].isin(unscored).any())
        trade.run("insample", data_dir=tmp, log=False, label_col="old", otm=5)
        check("sensitivity runs write to their own folder",
              (tmp / "trade_insample_1m_otm5_old" / "summary.md").exists())

        # a table with a 2023 date is refused, not silently trimmed
        bad = outcome.copy()
        bad.loc[bad.index[0], "entry_date"] = "2023-12-29"
        bad.to_csv(tmp / "outcome_insample.csv", index=False)
        expect_refused("run refuses an outcome table with a 2023 entry",
                       lambda: trade.run("insample", labels, data_dir=tmp, log=False), match="before 2024-01-01")


def _curve_from_returns(dates, r, n_open=1):
    wealth = np.cumprod(1 + np.asarray(r, float))
    return pd.DataFrame({"date": dates, "closed_pct": 0.0, "mtm_pct": 100 * (wealth - 1), "n_open": n_open})


def test_risk_metrics():
    dates = pd.bdate_range("2024-01-02", periods=300)
    rng = np.random.default_rng(3)
    r = rng.normal(0.0004, 0.01, len(dates))
    r[0] = 0.0                                                   # the first day has no prior wealth change
    curve = _curve_from_returns(dates, r)
    taken = _toy([d.strftime("%Y-%m-%d") for d in dates[:10:2]], [d.strftime("%Y-%m-%d") for d in dates[-10::2]],
                 [0.02, -0.10, 0.01, 0.03, 0.0])
    taken.loc[0, "entry_date"], taken.loc[4, "exit_date"] = dates[0], dates[-1]
    m = trade.risk_metrics(taken, curve, "csp_net")
    want = r.mean() / r.std(ddof=1) * np.sqrt(252)
    check("Sharpe = mean / sd(ddof=1) x sqrt(252) of daily MTM returns", np.isclose(m["sharpe"], want),
          f"{m['sharpe']:.3f} vs {want:.3f}")
    years = (dates[-1] - dates[0]).days / 365.25
    check("turnover = n_trades / 5 / years", np.isclose(m["turnover"], 5 / 5 / years))
    check("turnover_deployed divides by average positions open", np.isclose(m["turnover_deployed"], 5 / years / 1.0))
    check("skew of per-trade P&L (pandas)", np.isclose(m["skew_trade"], taken["csp_net"].skew()))
    check("largest single-position loss = worst trade / 5", np.isclose(m["largest_loss_pct_book"], -2.0))
    w = pd.Series(np.cumprod(1 + r), index=dates).resample("ME").last()
    mret = w / w.shift(1).fillna(1.0) - 1
    check("worst calendar month from MTM wealth", m["worst_month"] == mret.idxmin().strftime("%Y-%m")
          and np.isclose(m["worst_month_pct"], 100 * mret.min()))
    check("exposure: average and maximum open positions", m["avg_open"] == 1.0 and m["max_open"] == 1)
    check("no market series -> beta blank", np.isnan(m["beta"]))
    sparse = curve.iloc[::5]                                     # marks every 5 days: idle days carry the mark
    d = trade.daily_returns(sparse)
    check("daily returns fill business days and carry marks flat", len(d) == len(pd.bdate_range(sparse.date.min(), sparse.date.max()))
          and (d["r"] == 0).sum() >= 0.7 * len(d))
    mkt = pd.Series(rng.normal(0, 0.01, len(dates)), index=dates, name="r_mkt")
    rb = 0.0002 + 0.5 * mkt.to_numpy()
    rb[0] = 0.0
    mb = trade.risk_metrics(taken, _curve_from_returns(dates, rb), "csp_net", mkt)
    check("beta recovers a planted 0.5", abs(mb["beta"] - 0.5) < 0.02 and mb["r2"] > 0.95,
          f"beta {mb['beta']:.3f} R2 {mb['r2']:.3f}")
    check("alpha annualised = intercept x 252", abs(mb["alpha_ann_pct"] - 100 * 252 * 0.0002) < 1.0)
    e = trade.risk_metrics(taken.iloc[:0], curve.iloc[:0], "csp_net", mkt)
    check("empty book -> NaN metrics", np.isnan(e["sharpe"]) and np.isnan(e["beta"]))


def _write_parts(folder: Path, n_tickers=15, seed=0):
    import pickle
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2023-12-01", "2025-12-31")
    series = {}
    for k in range(n_tickers):
        px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days)))), index=days)
        for j, start in enumerate(range(0, len(days) - 40, 60)):
            series[(f"T{k}", days[start])] = (f"T{k}", px.iloc[start:start + 70])
    series[("UNPRICED", days[0])] = None                     # the measure step stores None for unpriced rows
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "part_00000.pkl").write_bytes(pickle.dumps({"per_row": {}, "series": series}))
    return days


def test_panel_and_beta():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _write_parts(tmp / "_parts_insample")
        panel = trade.spot_panel("insample", tmp, NB={})
        check("panel: only 2024-25 dates for insample (2023 rows dropped)",
              panel.index.min() >= pd.Timestamp("2024-01-01") and panel.index.max() < pd.Timestamp("2026-01-01"))
        p2 = trade.spot_panel("insample", tmp, NB={"HOLDOUT_START": "2024-06-03", "HOLDOUT_END": "2024-08-30"})
        check("panel: dates inside the sealed window dropped",
              not ((p2.index >= pd.Timestamp("2024-06-03")) & (p2.index <= pd.Timestamp("2024-08-30"))).any())
        check("panel: never reads another label's parts", trade.spot_panel("holdout", tmp, NB={"RUN_HOLDOUT": True}) is None)
        expect_refused("panel: retired labels refuse", lambda: trade.spot_panel("discovery", tmp), match="2024-2025")
        m = trade.market_returns(panel)
        lr = np.log(panel).diff()
        day = m.dropna().index[5]
        check("market return = median one-session log return, as a simple return",
              np.isclose(m[day], np.expm1(lr.loc[day].median())))
        few = trade.market_returns(panel.iloc[:, :trade.MARKET_MIN_TICKERS - 1])
        check("market return blank with fewer than the minimum tickers", few.isna().all())

        events, nulls, outcome, labels = synthetic(n_events=60, seed=7)
        write(tmp, "insample", events, nulls, outcome)
        plain = tempfile.mkdtemp()
        write(Path(plain), "insample", events, nulls, outcome)
        a = trade.run("insample", labels, data_dir=tmp, log=False)
        b = trade.run("insample", labels, data_dir=Path(plain), log=False)
        check("risk reporting never changes a trade or its P&L",
              a["trades"].equals(b["trades"]) and a["summary"].equals(b["summary"]) and a["h2"].equals(b["h2"]))
        mt = a["metrics"]
        check("metrics for old and ordinary-day books at 1x and 2x",
              {"old", "null_r1", "null_r2"} <= set(mt["book"]) and set(mt["cost"]) == {"1x", "2x"}
              and mt["sharpe"].notna().all() and mt["beta"].notna().all())
        check("without a panel beta is blank but Sharpe is reported",
              b["metrics"]["beta"].isna().all() and b["metrics"]["sharpe"].notna().all())
        md = (tmp / "trade_insample" / "summary.md").read_text(encoding="utf-8")
        check("summary.md has the risk section and its formulas", "Sharpe = mean(r) / sd(r, ddof=1)" in md
              and "Risk, turnover and market beta" in md and "Market panel: 15 tickers" in md)


def test_extra_stats():
    events, nulls, outcome, labels = synthetic(n_events=60, seed=11)
    rng = np.random.default_rng(3)
    events["lag_bd"] = rng.integers(0, 5, len(events))
    nxt = rng.random(len(events)) < 0.4                        # these filings enter the session after the filing date
    days = list(DAYS)
    events["t_0"] = [days[days.index(pd.Timestamp(f)) + 1].date() if n else f
                     for f, n in zip(events["filing_date"], nxt)]
    cl = events.merge(labels, on="row_id").assign(scored=True)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        write(tmp, "insample", events, nulls, outcome)
        cl.to_csv(tmp / "classified_insample.csv", index=False)
        tables = trade.run("insample", data_dir=tmp, log=False)
        before = {p.name: p.read_bytes() for p in (tmp / "trade_insample").iterdir()}
        ex = trade.extra_stats("insample", data_dir=tmp)
        res = tmp / "results_insample"
        check("extra_stats writes extra_stats.md and its three CSVs",
              all((res / f).exists() for f in ("extra_stats.md", "extra_stats_pnl.csv", "extra_stats_costs.csv",
                                                "extra_stats_lag.csv")))
        check("extra_stats leaves every trade table byte-identical",
              before == {p.name: p.read_bytes() for p in (tmp / "trade_insample").iterdir()})
        led = pd.read_csv(tmp / "ledger.csv")
        check("one ledger row per statistic family, kind report",
              len(led) == 3 and set(led["kind"]) == {"report"}
              and set(led["group"]) == {"pnl_by_horizon", "cost_bps", "filing_lag"})

        p = ex["pnl"]
        check("P&L table: every group x scope x horizon x cost", len(p) == 2 * 2 * len(trade.HORIZONS) * 2
              and set(p["horizon"]) == set(trade.HORIZONS))
        t = tables["trades"]
        old10 = t[(t["book"] == "old") & (t["horizon"] == "10")]
        r = p.set_index(["group", "scope", "horizon", "cost"])
        e1, e2 = r.loc[("old", "eligible", "10", "1x")], r.loc[("old", "eligible", "10", "2x")]
        check("old eligible mean equals the trades' mean, CI brackets it",
              e1["n"] == len(old10) and np.isclose(e1["mean_pct"], 100 * old10["csp_net"].mean())
              and e1["ci_lo_pct"] <= e1["mean_pct"] <= e1["ci_hi_pct"])
        check("2x costs lower the mean", e2["mean_pct"] < e1["mean_pct"])
        tk = r.loc[("old", "taken", "10", "1x")]
        check("taken scope counts only the book's trades", tk["n"] == int(old10["taken"].sum()))
        nul10 = t[t["book"].str.startswith("null_r") & (t["horizon"] == "10")]
        check("ordinary-day group pools both rounds", r.loc[("null", "eligible", "10", "1x"), "n"] == len(nul10))
        clipped = trade.pnl_by_horizon(t[t["horizon"] != "63"])
        c63 = clipped[clipped["horizon"] == "63"]
        check("a horizon with no usable exit reports n = 0 and no mean", (c63["n"] == 0).all()
              and c63["mean_pct"].isna().all() and "| 63 | 1x | 0 | n/a |" in trade._extra_md(
                  "insample", clipped, ex["costs"], ex["lag"]))
        check("bootstrap is seeded (same CI twice)", trade.pnl_by_horizon(t).equals(p))

        c = ex["costs"].set_index(["scope", "cost"])
        # synthetic 3% row: strike 97, premium 1.2, cost max(5% x 1.2, 0.05) = 0.06/share each way
        check("cost in bps of collateral and of premium (1x)",
              np.isclose(c.loc[("taken", "1x"), "median_bps_collateral"], 1e4 * 0.12 / 97)
              and np.isclose(c.loc[("taken", "1x"), "median_bps_premium"], 1e4 * 0.12 / 1.2))
        check("2x cost is double", np.isclose(c.loc[("eligible", "2x"), "mean_bps_collateral"],
                                              2 * c.loc[("eligible", "1x"), "mean_bps_collateral"]))

        lag = ex["lag"].set_index("subset")
        a = lag.loc["all classified"]
        check("filing lag median and IQR from lag_bd", a["n"] == len(cl)
              and np.isclose(a["median_lag_bd"], cl["lag_bd"].median())
              and np.isclose(a["p25_lag_bd"], cl["lag_bd"].quantile(.25))
              and np.isclose(a["p75_lag_bd"], cl["lag_bd"].quantile(.75)))
        check("share entering the next session", a["n_next_session"] == int(nxt.sum())
              and np.isclose(a["share_next_session_pct"], 100 * nxt.mean()))
        check("trade-eligible subset is reported",
              any(s.startswith("trade-eligible") for s in lag.index))

        expect_refused("extra_stats refuses a retired label", lambda: trade.extra_stats("discovery", data_dir=tmp),
                       match="2022-2023")
        bad = pd.read_csv(tmp / "trade_insample" / "trades.csv")
        bad.loc[bad.index[0], "exit_date"] = "2026-01-05"
        bad.to_csv(tmp / "trade_insample" / "trades.csv", index=False)
        expect_refused("extra_stats refuses a trade exiting in 2026",
                       lambda: trade.extra_stats("insample", data_dir=tmp, log=False), match="2026-01-01")


def test_note_numbers():
    """report.add_entry_shift patches sensitivity.csv and summary.md once; note_numbers reads the trade tables and
    never stops a run when a part is missing."""
    from oldnews import report as R

    events, nulls, outcome, labels = synthetic(n_events=60, seed=5)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        write(tmp, "insample", events, nulls, outcome)
        trade.run("insample", labels, data_dir=tmp, log=False)
        res = tmp / "results_insample"
        res.mkdir()
        sens = pd.DataFrame([{"dimension": "primary", "weights": "equal", "text": "full", "cutoff": 1.0, "bucket": "1m",
                              "otm": 3, "category": "all8", "n": 10, "n_old": 4, "n_comp": 6, "effect": 0.1,
                              "ci_lo": -0.1, "ci_hi": 0.3, "p_one_sided": 0.8, "descriptive": True, "period": "pooled"}])
        sens.to_csv(res / "sensitivity.csv", index=False)
        md = ["# t", "", "## Sensitivity (H1, one change at a time)", "", "| dimension | x |", "|---|---|",
              "| primary | 1 |", "", "## By year", ""]
        (res / "summary.md").write_text("\n".join(md), encoding="utf-8")
        fake = sens.assign(dimension=R.ENTRY_SHIFT_DIM, effect=0.05)
        real_row = R.entry_shift_row
        R.entry_shift_row = lambda *a, **k: fake
        try:
            R.add_entry_shift("insample", tmp, log=False)
            R.add_entry_shift("insample", tmp, log=False)              # second call: no duplicate row
        finally:
            R.entry_shift_row = real_row
        s2 = pd.read_csv(res / "sensitivity.csv")
        check("entry t0+1 row appended once to sensitivity.csv",
              list(s2["dimension"]) == ["primary", R.ENTRY_SHIFT_DIM])
        text = (res / "summary.md").read_text(encoding="utf-8").split("\n")
        i = text.index("| primary | 1 |")
        check("summary.md: row after the table, then the note line, then the next section",
              text[i + 1].startswith("| entry t0+1 |") and text[i + 2] == ""
              and text[i + 3] == R.ENTRY_SHIFT_NOTE.format(label="insample") and text[i + 5] == "## By year")

        t = pd.read_csv(tmp / "trade_insample" / "trades.csv", dtype={"horizon": str})
        b = t[(t["book"] == "old") & (t["horizon"] == "10") & t["taken"]]
        r = 100 * b["csp_net"].to_numpy()
        c = R._book_concentration(t)
        check("book concentration: mean, mean without the largest loss, top-3 share",
              c["n"] == len(r) and np.isclose(c["mean"], r.mean())
              and np.isclose(c["mean_wo_worst"], np.delete(r, r.argmin()).mean())
              and np.isclose(c["top3_share"], np.sort(r)[-3:].sum() / r.sum()))
        nn = R.note_numbers("insample", tmp, NB={})
        v = nn.set_index("item")["value"]
        check("note numbers: book and beta parts read the trade tables",
              v["trade: mean without the largest loss"] == f"{c['mean_wo_worst']:+.3f}"
              and "trade: R^2" in v.index)
        check("note numbers: a missing input gives n/a, not an error",
              (nn.loc[nn["item"] == "trimmed", "value"] == "n/a").all() and "diagnostics" in set(nn["item"]))
        # the diagnostics part on a stand-in sample: 35 of 45 old calls are math-only surprise; power n from D.power
        from oldnews import diagnostics as D
        rng = np.random.default_rng(1)
        mo = np.r_[np.ones(10), np.zeros(136)].astype(bool)
        wo = np.r_[np.ones(8), np.zeros(2), np.ones(60), np.zeros(76)].astype(bool)
        po = np.r_[np.ones(10), np.ones(35), np.zeros(25), np.zeros(76)].astype(bool)
        samp = pd.DataFrame({"d": rng.normal(0, 0.5, 146), "is_old": po, "ticker": [f"T{i % 9}" for i in range(146)],
                             "old_math_only_1": mo, "old_words_only_1": wo})
        real_inp, real_s = R._inputs, D.h1_sample
        R._inputs, D.h1_sample = (lambda *a, **k: None), (lambda *a, **k: samp)
        try:
            v2 = R.note_numbers("insample", tmp, NB={}).set_index("item")["value"]
        finally:
            R._inputs, D.h1_sample = real_inp, real_s
        need = D.power(samp["d"].to_numpy(), po).set_index("quantity")["value"]["n_per_group_for_0.10"]
        check("note numbers: old calls that math alone calls surprise, and n per group for 0.10",
              v2["diagnostics: old calls that math alone calls surprise"] == "35 of 45"
              and v2["diagnostics: events per group to detect 0.10"] == f"{int(need):,}")
        R.write_note_numbers("insample", tmp, NB={})
        check("note_numbers.md and .csv written", (res / "note_numbers.md").exists() and (res / "note_numbers.csv").exists())


if __name__ == "__main__":
    for fn in (test_filters, test_gap_and_label_filters, test_cap, test_equity_and_metrics, test_capacity, test_h2,
               test_window_guard, test_run_insample, test_risk_metrics, test_panel_and_beta, test_extra_stats,
               test_note_numbers):
        print(f"--- {fn.__name__}")
        fn()
    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)
