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
DAYS = pd.bdate_range("2022-01-03", "2023-12-29")


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
    t = _toy(["2022-01-03"] * 7, ["2022-01-10"] * 7, [0.01] * 7, ids=[f"r{i}" for i in (6, 5, 4, 3, 2, 1, 0)])
    taken = trade.simulate(t)
    check("cap: 7 same-day entries -> 5 taken", taken.sum() == 5)
    check("cap: ties taken by row_id", set(t.loc[taken, "row_id"]) == {"r0", "r1", "r2", "r3", "r4"})
    t2 = pd.concat([t, _toy(["2022-01-10", "2022-01-07"], ["2022-01-20", "2022-01-20"], [0.0, 0.0],
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
    t = _toy(["2022-01-03", "2022-01-04"], ["2022-01-14", "2022-01-18"], [-0.10, 0.05])
    marks = pd.DataFrame({"row_id": ["r0", "r0", "r1"], "mark_date": pd.to_datetime(["2022-01-05", "2022-01-07",
                                                                                     "2022-01-06"]),
                          "csp_net": [-0.20, -0.15, 0.02], "csp_net2x": [-0.21, -0.16, 0.01]})
    c = trade.equity(t, marks, "csp_net")
    check("closed equity ends at sum(pnl)/5", np.isclose(c["closed_pct"].iloc[-1], 100 * (-0.05) / 5))
    check("MTM equity ends where closed equity ends", np.isclose(c["mtm_pct"].iloc[-1], c["closed_pct"].iloc[-1]))
    on = c.set_index("date")
    check("MTM carries the latest mark", np.isclose(on.loc["2022-01-06", "mtm_pct"], 100 * (-0.20 + 0.02) / 5))
    check("n_open counts open positions", on.loc["2022-01-04", "n_open"] == 2 and on["n_open"].iloc[-1] == 0)
    m = trade.book_metrics(t, c, "csp_net")
    check("max drawdown closed = -2% (one -10% trade in a 5-slot book)", np.isclose(m["max_dd_closed_pct"], -2.0))
    check("max drawdown MTM is deeper (-4% at the worst mark)", np.isclose(m["max_dd_mtm_pct"], -4.0)
          and m["max_dd_mtm_pct"] < m["max_dd_closed_pct"])
    check("hit rate 50%", m["hit_rate_pct"] == 50.0)
    check("total return -1%", np.isclose(m["total_return_pct"], -1.0))
    check("drawdown is measured from the peak", np.isclose(trade.max_drawdown(pd.Series([10.0, 0.0])), -100 / 11))
    w = trade.worst(_toy(["2022-01-03"] * 7, ["2022-01-10"] * 7, [0.03, -0.2, 0.01, -0.05, 0.0, -0.01, 0.02]),
                    "csp_net")
    check("five worst, worst first", len(w) == 5 and w["pnl_pct"].is_monotonic_increasing
          and np.isclose(w["pnl_pct"].iloc[0], -20.0))
    empty = trade.equity(t.iloc[:0], None, "csp_net")
    check("empty book gives NaN metrics, no crash", np.isnan(trade.book_metrics(t.iloc[:0], empty, "csp_net")["max_dd_closed_pct"]))


def test_capacity():
    t = _toy(["2022-01-03"] * 3, ["2022-01-10"] * 3, [0.01, 0.02, -0.01])
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

def test_guards_and_run():
    events, nulls, outcome, labels = synthetic(n_events=60, seed=7)
    o = trade.prep_outcome(outcome)
    bad = o.copy()
    bad.loc[bad.index[0], "entry_date"] = pd.Timestamp("2024-01-02")
    try:
        trade.check_dates(bad, "discovery")
        check("discovery entry in 2024 refused", False)
    except ValueError:
        check("discovery entry in 2024 refused", True)
    late = o.copy()
    late.loc[late.index[0], ["exit_date", "usable"]] = [pd.Timestamp("2026-01-05"), True]
    try:
        trade.check_dates(late, "insample")
        check("usable exit in 2026 refused", False)
    except ValueError:
        check("usable exit in 2026 refused", True)
    trade.check_dates(late, "holdout")
    check("2026 allowed only for the gated holdout/oos labels", True)
    try:
        trade.check_dates(bad, "some_new_label")
        check("unknown label gets discovery date rules", False)
    except ValueError:
        check("unknown label gets discovery date rules", True)
    trade.check_dates(bad, "insample")
    check("insample allows 2024 entries", True)
    try:
        trade.build_trades(trade.eligible_events(events, labels)[0], nulls, o, otm=0.03)
        check("otm in the wrong units refused", False)
    except ValueError:
        check("otm in the wrong units refused", True)

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        write(tmp, "discovery", events, nulls, outcome)
        tables = trade.run("discovery", labels, data_dir=tmp)
        out = tmp / "trade_discovery"
        names = ["counts", "trades", "summary", "by_year", "worst", "equity", "capacity", "h2"]
        check("run writes every table and summary.md",
              all((out / f"{n}.csv").exists() for n in names) and (out / "summary.md").exists())
        s = tables["summary"]
        check("summary has every book x horizon x cost",
              len(s) == s["book"].nunique() * len(trade.HORIZONS) * 2 and {"old", "all_late", "null_r1", "null_r2"} <= set(s["book"]))
        hd = s[(s["book"] == "old") & (s["horizon"] == "10")].set_index("cost")
        check("2x costs never beat 1x", hd.loc["2x", "total_return_pct"] < hd.loc["1x", "total_return_pct"])
        check("same trades at 1x and 2x", hd.loc["2x", "n_trades"] == hd.loc["1x", "n_trades"])
        y = tables["by_year"]
        y = y[(y["book"] == "old") & (y["horizon"] == "10") & (y["cost"] == "1x")]
        check("2022 reported separately and years add up", 2022 in set(y["year"])
              and y["n_trades"].sum() == hd.loc["1x", "n_trades"]
              and np.isclose(y["total_return_pct"].sum(), hd.loc["1x", "total_return_pct"]))
        led = pd.read_csv(tmp / "ledger.csv")
        check("every book and H2 variant logged to the ledger",
              len(led) == len(s) // 2 + len(tables["h2"]) // 2 and set(led["kind"]) == {"trade_book", "H2"})
        md = (out / "summary.md").read_text(encoding="utf-8")
        check("summary.md has the headline sections", all(k in md for k in ("Books at h = 10", "by year", "H2", "Capacity")))
        tables2 = trade.run("discovery", labels, data_dir=tmp, log=False)
        check("run is reproducible", tables2["summary"].equals(tables["summary"]) and tables2["h2"].equals(tables["h2"]))

        # the pipeline calls run(label): labels come from classify's classified_<label>.csv, unscored dropped
        cl = labels.assign(scored=[i % 5 != 0 for i in range(len(labels))])
        cl.loc[~cl["scored"], "old"] = False                   # classify's convention for unscored events
        cl.to_csv(tmp / "classified_discovery.csv", index=False)
        tables3 = trade.run("discovery", data_dir=tmp, log=False)
        unscored = set(cl.loc[~cl["scored"], "row_id"])
        check("run(label) reads classified_<label>.csv and drops unscored events",
              len(tables3["trades"]) > 0 and not tables3["trades"]["row_id"].isin(unscored).any())
        trade.run("discovery", data_dir=tmp, log=False, label_col="old", otm=5)
        check("sensitivity runs write to their own folder", (tmp / "trade_discovery_1m_otm5_old" / "summary.md").exists())


if __name__ == "__main__":
    for fn in (test_filters, test_gap_and_label_filters, test_cap, test_equity_and_metrics, test_capacity, test_h2,
               test_guards_and_run):
        print(f"--- {fn.__name__}")
        fn()
    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)
