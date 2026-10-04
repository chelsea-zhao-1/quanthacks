"""Trading record for the old-news hypothesis: a cash-secured put book, its risk, its capacity and test H2.

Every rule comes from docs/test_plan.md ("Trade" and test H2). None is tuned on results.

- Trade: sell the 1-month put about 3% below spot at t_0 on old-news, late, people-news events (earnings
  excluded, outcome usable, put volume at entry > 0) and close it after 10 sessions. All other fixed
  horizons are reported too.
- Book: at most MAX_OPEN = 5 positions open at once, taken in order of entry (ties broken by row_id). A
  position that closes on a date frees its slot for one that opens at that date's close. Each position
  gets equal collateral, so each trade adds csp_net / 5 to the book's return. P&L is a fraction of
  collateral (strike x 100) and is net of costs: the larger of 5% of premium or $0.05/share, on entry and on
  exit (`csp_net`). Every result is repeated at double costs (`csp_net2x`). The selection uses dates only,
  so the same trades are taken at 1x and 2x.
- Risk: the equity curve is reported two ways. "Closed" books each trade at its exit. "MTM" also marks
  open trades at their liquidation value (the same put closed at an earlier fixed horizon, net of the exit
  cost), carried forward between marks. Max drawdown, the five worst trades and each calendar year (so 2022
  separately) are reported at both cost levels.
- Benchmarks for H2: the same put on the matched ordinary days of the old-news events (one book per null
  round), and on all late people-news events (old and surprise together).
- Capacity: contracts per trade = floor(10% of the put's entry-day volume), and the dollar collateral
  that allows.

Inputs (data/oldnews/): events_<label>.csv, nulls_<label>.csv, outcome_<label>.csv, gap_<label>.csv
(optional), and classify's labelled table classified_<label>.csv (`row_id`, `scored`, `old` = primary rule).
Outputs: data/oldnews/trade_<label>/ (CSV tables and summary.md). Every book and H2 row is appended to the
test ledger, data/oldnews/ledger.csv.
"""
from __future__ import annotations

import heapq
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from playground import ledger
from playground.stats import benjamini_hochberg

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "oldnews"

SEED = 20261003
HORIZONS = ("1", "2", "3", "5", "10", "21", "42", "63", "expiry")   # fixed by the organisers
HEADLINE_HORIZON = "10"
BUCKET, OTM = "1m", 3.0
MAX_OPEN = 5
CAPACITY_SHARE = 0.10
N_BOOT = 10_000
LOW_SAMPLE = 30
COSTS = {"1x": "csp_net", "2x": "csp_net2x"}
DISCOVERY_END = pd.Timestamp("2024-01-01")
HARD_STOP = pd.Timestamp("2026-01-01")
ENTRY_CAPS = {"discovery": DISCOVERY_END, "dryrun": DISCOVERY_END, "insample": HARD_STOP}   # as pipeline.WINDOWS
GATED_UPSTREAM = ("holdout", "oos")   # pipeline.window_rules runs these only after RUN_HOLDOUT / RUN_OOS is set

TRADE_COLS = ["row_id", "entry_date", "exit_date", "put_strike", "put_premium", "put_volume_entry",
              "csp_gross", "csp_net", "csp_net2x"]


# ---------------------------------------------------------------- loading and guards

def _flag(s: pd.Series) -> pd.Series:
    """0/1, True/False or 'True'/'False' as bool; missing counts as False."""
    if s.dtype == bool:
        return s
    txt = s.astype("string").str.strip().str.lower()
    return txt.isin(["true", "1", "1.0", "yes"]).fillna(False).astype(bool)


def _horizon(s: pd.Series) -> pd.Series:
    """Horizon labels as strings: 10, 10.0 and '10' all become '10'; 'expiry' stays."""
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def prep_outcome(outcome: pd.DataFrame) -> pd.DataFrame:
    """Typed copy of outcome_<label>.csv."""
    o = outcome.copy()
    o["horizon"] = _horizon(o["horizon"])
    o["bucket"] = o["bucket"].astype(str)
    o["otm"] = pd.to_numeric(o["otm"])
    o["usable"] = _flag(o["usable"])
    for c in ("entry_date", "exit_date"):
        o[c] = pd.to_datetime(o[c])
    for c in ("put_strike", "put_premium", "put_volume_entry", "csp_gross", "csp_net", "csp_net2x"):
        o[c] = pd.to_numeric(o[c])
    return o


def check_dates(outcome: pd.DataFrame, label: str) -> None:
    """Hard stop before any computation, with the pipeline's date rules. Entries: before 2024-01-01 for
    discovery, the dry run and any unknown label; before 2026-01-01 for insample. Usable exits: before
    2026-01-01. Only "holdout" (judges' sealed window) and "oos" (the one-time human run) skip this; the
    pipeline lets those labels run only after RUN_HOLDOUT or RUN_OOS has been switched on."""
    if label in GATED_UPSTREAM:
        return
    cap = ENTRY_CAPS.get(label, DISCOVERY_END)
    if (outcome["entry_date"] >= cap).any():
        raise ValueError(f"label {label!r}: outcome has entries on or after {cap.date()}; refusing to run")
    if (outcome["usable"] & (outcome["exit_date"] >= HARD_STOP)).any():
        raise ValueError(f"label {label!r}: usable outcome rows exit on or after {HARD_STOP.date()}; refusing to run")


def _read(path: Path) -> pd.DataFrame:
    """read_csv where only empty cells are missing: pandas' defaults would turn kind="null" (and a ticker
    such as "NA") into NaN."""
    ids = {"row_id": str, "event_row_id": str, "ticker": str, "accession_number": str}
    return pd.read_csv(path, dtype=ids, keep_default_na=False, na_values=["", "nan", "NaN"])


def load(label: str, data_dir: Path = DATA) -> dict[str, pd.DataFrame]:
    """The input tables for one label (gap is optional)."""
    out = {name: _read(data_dir / f"{name}_{label}.csv") for name in ("events", "nulls", "outcome")}
    gap = data_dir / f"gap_{label}.csv"
    out["gap"] = _read(gap) if gap.exists() else None
    return out


# ---------------------------------------------------------------- which trades exist

def eligible_events(events: pd.DataFrame, labels: pd.DataFrame, label_col: str = "old",
                    gap: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Late people-news events with earnings excluded and a classification, with a bool `old` column.
    Returns the events and one count row per filtering step (every exclusion is counted)."""
    ev = events[events["kind"].eq("event")] if "kind" in events else events
    counts = [{"step": "events in table", "kept": len(ev)}]

    def step(name: str, mask: pd.Series) -> pd.DataFrame:
        counts.append({"step": name, "kept": int(mask.sum()), "dropped": int((~mask).sum())})
        return ev[mask.to_numpy()]

    ev = step("people-news group", ev["group"].eq("people"))
    ev = step("late (event date >= 1 business day before acceptance)", _flag(ev["late"]))
    ev = step("not earnings-excluded", ~_flag(ev["earnings_excluded"]))
    if gap is not None:
        ok = set(gap.loc[_flag(gap["usable"]), "row_id"])
        ev = step("gap endpoints usable", ev["row_id"].isin(ok))
    keep = ["row_id", label_col] + (["scored"] if "scored" in labels else [])
    lab = labels[keep].dropna(subset=[label_col]).drop_duplicates("row_id")
    ev = ev.drop(columns=[c for c in keep[1:] + ["old"] if c in ev], errors="ignore").merge(lab, on="row_id", how="left")
    ev = step(f"classified ({label_col} present)", ev[label_col].notna())
    if "scored" in ev:                                   # classify sets old=False for unscored events
        ev = step("scored by classify (gap usable, M and T finite)", _flag(ev["scored"]))
    ev = ev.assign(old=_flag(ev[label_col]))
    counts.append({"step": "of which old news", "kept": int(ev["old"].sum())})
    return ev, counts


def _tradeable(rows: pd.DataFrame, ob: pd.DataFrame, horizon: str, book: str,
               counts: list[dict]) -> pd.DataFrame:
    """The rows' outcome at one horizon, kept if usable and the put traded at entry. Counts each step."""
    t = rows.merge(ob[ob["horizon"] == horizon][TRADE_COLS + ["usable"]], on="row_id", how="inner")
    n0 = len(rows)
    use = t["usable"] & t["csp_net"].notna() & t["csp_net2x"].notna()
    vol = use & (t["put_volume_entry"] > 0)
    for name, kept in (("rows in book", n0), ("has outcome row", len(t)), ("outcome usable", int(use.sum())),
                       ("put volume at entry > 0", int(vol.sum()))):
        counts.append({"book": book, "horizon": horizon, "step": name, "kept": kept})
    return t[vol.to_numpy()].drop(columns="usable")


def build_trades(ev: pd.DataFrame, nulls: pd.DataFrame, outcome: pd.DataFrame, *, bucket: str = BUCKET,
                 otm: float = OTM) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every tradeable row for every book and horizon (long table, column `book`), and the step counts.

    Books: `old` (the strategy), `all_late` (all eligible events), `null_r<k>` (round-k ordinary days of the
    old-news events that are tradeable at that horizon)."""
    ob = outcome[(outcome["bucket"] == bucket) & np.isclose(outcome["otm"], otm)]
    if ob.empty:
        raise ValueError(f"no outcome rows for bucket={bucket!r}, otm={otm}; otm is in percent (3 means 3%)")
    ev_rows = ev[["row_id", "ticker", "old"]]
    nu_rows = nulls[["row_id", "ticker", "event_row_id", "round"]]       # the nulls table holds only nulls
    rounds = sorted(nu_rows["round"].dropna().unique())
    parts, counts = [], []
    for hz in HORIZONS:
        old = _tradeable(ev_rows[ev_rows["old"]], ob, hz, "old", counts)
        books = {"old": old, "all_late": _tradeable(ev_rows, ob, hz, "all_late", counts)}
        for rnd in rounds:
            name = f"null_r{int(rnd) if float(rnd).is_integer() else rnd}"
            rows = nu_rows[(nu_rows["round"] == rnd) & nu_rows["event_row_id"].isin(old["row_id"])]
            books[name] = _tradeable(rows, ob, hz, name, counts)
        parts += [t.assign(book=name, horizon=hz) for name, t in books.items()]
    return pd.concat(parts, ignore_index=True), pd.DataFrame(counts)


# ---------------------------------------------------------------- the book

def simulate(trades: pd.DataFrame, max_open: int = MAX_OPEN) -> pd.Series:
    """Which trades the book takes: in order of entry (ties by row_id), skipped while `max_open` positions
    are open. A position exiting on a date frees its slot for one entering that date. Uses dates, never P&L."""
    taken = pd.Series(False, index=trades.index)
    exits: list[pd.Timestamp] = []                       # min-heap of open positions' exit dates
    for i in trades.sort_values(["entry_date", "row_id"]).index:
        entry = trades.at[i, "entry_date"]
        while exits and exits[0] <= entry:
            heapq.heappop(exits)
        if len(exits) < max_open:
            taken.at[i] = True
            heapq.heappush(exits, trades.at[i, "exit_date"])
    return taken


def equity(taken: pd.DataFrame, marks: pd.DataFrame | None, pnl: str, max_open: int = MAX_OPEN) -> pd.DataFrame:
    """Book equity by date, in % of total collateral (max_open equal slots).

    closed_pct: realised P&L, each trade booked at its exit. mtm_pct: open trades also marked at their latest
    liquidation value from `marks` (row_id, mark_date, P&L columns), carried forward. n_open: positions open
    after that date's close."""
    if taken.empty:
        return pd.DataFrame(columns=["date", "closed_pct", "mtm_pct", "n_open"])
    final = taken[["row_id", "exit_date", pnl]].rename(columns={"exit_date": "date", pnl: "value"})
    path = final
    if marks is not None and not marks.empty:
        m = marks.merge(taken[["row_id", "exit_date"]], on="row_id")
        m = m[m["mark_date"] < m["exit_date"]][["row_id", "mark_date", pnl]].dropna()
        path = pd.concat([m.rename(columns={"mark_date": "date", pnl: "value"}), final], ignore_index=True)
    path = path.sort_values(["row_id", "date"])
    path["inc"] = path["value"] - path.groupby("row_id")["value"].shift(1).fillna(0.0)

    dates = pd.DatetimeIndex(sorted(set(taken["entry_date"]) | set(path["date"])))
    closed = final.groupby("date")["value"].sum().reindex(dates, fill_value=0.0).cumsum()
    mtm = path.groupby("date")["inc"].sum().reindex(dates, fill_value=0.0).cumsum()
    n_open = (taken.groupby("entry_date").size().reindex(dates, fill_value=0)
              - taken.groupby("exit_date").size().reindex(dates, fill_value=0)).cumsum()
    return pd.DataFrame({"date": dates, "closed_pct": 100 * closed.to_numpy() / max_open,
                         "mtm_pct": 100 * mtm.to_numpy() / max_open, "n_open": n_open.to_numpy()})


def max_drawdown(ret_pct: pd.Series) -> float:
    """Largest peak-to-trough fall of wealth (1 + return), in %, starting from 1. NaN when empty."""
    if len(ret_pct) == 0:
        return float("nan")
    wealth = 1 + np.asarray(ret_pct, float) / 100
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    return float(100 * (wealth / peak - 1).min())


def book_metrics(taken: pd.DataFrame, curve: pd.DataFrame, pnl: str, max_open: int = MAX_OPEN) -> dict:
    """Return, annualised return (over first entry to last exit), hit rate, trade stats and drawdowns."""
    x = taken[pnl]
    n = len(x)
    total = x.sum() / max_open
    years = (taken["exit_date"].max() - taken["entry_date"].min()).days / 365.25 if n else float("nan")
    ann = (1 + total) ** (1 / years) - 1 if n and years > 0 and 1 + total > 0 else float("nan")
    nan = float("nan")
    return {"n_trades": n, "total_return_pct": 100 * total if n else nan, "annualised_pct": 100 * ann,
            "hit_rate_pct": 100 * (x > 0).mean() if n else nan, "mean_trade_pct": 100 * x.mean(),
            "median_trade_pct": 100 * x.median(), "worst_trade_pct": 100 * x.min(),
            "max_dd_closed_pct": max_drawdown(curve["closed_pct"]), "max_dd_mtm_pct": max_drawdown(curve["mtm_pct"]),
            "first_entry": taken["entry_date"].min(), "last_exit": taken["exit_date"].max(), "years": years}


def worst(taken: pd.DataFrame, pnl: str, k: int = 5) -> pd.DataFrame:
    """The k worst trades by P&L (% of collateral)."""
    cols = ["row_id", "ticker", "entry_date", "exit_date", "put_strike", "put_premium"]
    w = taken.nsmallest(k, pnl)
    return w[cols].assign(pnl_pct=100 * w[pnl])


# ---------------------------------------------------------------- capacity

def capacity(taken: pd.DataFrame) -> pd.DataFrame:
    """Per trade: contracts allowed by 10% of the put's entry-day volume, and the collateral and premium that is."""
    c = np.floor(CAPACITY_SHARE * taken["put_volume_entry"])
    return taken[["row_id", "ticker", "entry_date", "exit_date", "put_strike", "put_premium", "put_volume_entry",
                  "csp_net"]].assign(max_contracts=c, max_collateral_usd=c * taken["put_strike"] * 100,
                                     max_premium_usd=c * taken["put_premium"] * 100,
                                     pnl_at_capacity_usd=c * taken["put_strike"] * 100 * taken["csp_net"])


def capacity_summary(cap: pd.DataFrame) -> dict:
    """Median and total dollar collateral the 10%-of-volume limit allows (plus the spread of it)."""
    c = cap["max_collateral_usd"]
    return {"n_trades": len(cap), "n_zero_contracts": int((cap["max_contracts"] == 0).sum()),
            "min_collateral_usd": c.min(), "p10_collateral_usd": c.quantile(0.10),
            "median_collateral_usd": c.median(), "total_collateral_usd": c.sum(),
            "book_at_median_usd": MAX_OPEN * c.median(), "pnl_at_capacity_usd": cap["pnl_at_capacity_usd"].sum()}


# ---------------------------------------------------------------- H2

def _boot(stat, n: int, rng: np.random.Generator) -> np.ndarray:
    """N_BOOT replicates of stat(index matrix), drawn in blocks to bound memory."""
    out = []
    for start in range(0, N_BOOT, 1000):
        out.append(stat(rng.integers(0, n, size=(min(1000, N_BOOT - start), n))))
    return np.concatenate(out)


def _h2_row(comparison: str, hz: str, cost: str, diff: float, old_m: float, base_m: float, n: int,
            n_tickers: int, boot: np.ndarray) -> dict:
    b = boot[~np.isnan(boot)]
    ok = n > 0 and len(b) > 0
    return {"comparison": comparison, "horizon": hz, "cost": cost, "n": n, "n_tickers": n_tickers,
            "old_mean_pct": 100 * old_m, "base_mean_pct": 100 * base_m, "diff_pct": 100 * diff,
            "ci_lo_pct": 100 * np.percentile(b, 2.5) if ok else np.nan,
            "ci_hi_pct": 100 * np.percentile(b, 97.5) if ok else np.nan,
            "p_boot": (1 + (b <= 0).sum()) / (1 + len(b)) if ok else np.nan,
            "descriptive": n < LOW_SAMPLE}


def h2(trades: pd.DataFrame) -> pd.DataFrame:
    """H2 at every horizon and cost, on every eligible trade (before the 5-position cap).

    old_minus_null: per old-news event, its put P&L minus the mean of its matched ordinary days' put P&L.
    old_minus_all_late: mean old-news put P&L minus the mean over all late people-news events.
    95% percentile-bootstrap intervals over events (N_BOOT, seed SEED; the same draws at 1x and 2x),
    one-sided p = share of bootstrap means <= 0 (prediction: positive), BH q-values across horizons."""
    rows = []
    for hz in HORIZONS:
        t = trades[trades["horizon"] == hz]
        old, allr = t[t["book"] == "old"], t[t["book"] == "all_late"]
        nul = t[t["book"].str.startswith("null_r")]
        for cost, pnl in COSTS.items():
            nm = (nul.groupby("event_row_id")[pnl].mean() if len(nul) else pd.Series(dtype=float)).rename("null_mean")
            pair = old.set_index("row_id")[[pnl, "ticker"]].join(nm, how="inner")
            d = (pair[pnl] - pair["null_mean"]).to_numpy()
            boot = (_boot(lambda idx: d[idx].mean(1), len(d), np.random.default_rng(SEED)) if len(d)
                    else np.array([]))
            rows.append(_h2_row("old_minus_null", hz, cost, d.mean() if len(d) else np.nan, pair[pnl].mean(),
                                pair["null_mean"].mean(), len(d), pair["ticker"].nunique(), boot))

            x, o = allr[pnl].to_numpy(), allr["old"].to_numpy(bool)

            def stat(idx: np.ndarray) -> np.ndarray:
                xo, oo = x[idx], o[idx]
                with np.errstate(invalid="ignore", divide="ignore"):
                    return (xo * oo).sum(1) / oo.sum(1) - xo.mean(1)

            boot = _boot(stat, len(x), np.random.default_rng(SEED)) if len(x) and o.any() else np.array([])
            diff = x[o].mean() - x.mean() if o.any() else np.nan
            rows.append(_h2_row("old_minus_all_late", hz, cost, diff, x[o].mean() if o.any() else np.nan,
                                x.mean() if len(x) else np.nan, int(o.sum()), allr["ticker"].nunique(), boot))
    out = pd.DataFrame(rows)
    out["q_bh"] = np.nan
    for _, g in out.groupby(["comparison", "cost"]):
        out.loc[g.index, "q_bh"] = benjamini_hochberg(g["p_boot"].to_numpy())
    return out


# ---------------------------------------------------------------- run

def _git_head() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else f"{v:,.2f}"
    if isinstance(v, pd.Timestamp):
        return v.date().isoformat()
    return str(v)


def _md(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(_fmt(v) for v in r) + " |" for r in df.itertuples(index=False)])


def run(label: str, labels: pd.DataFrame | None = None, *, data_dir: Path = DATA, out_dir: Path | None = None,
        bucket: str = BUCKET, otm: float = OTM, label_col: str = "old", log: bool = True) -> dict[str, pd.DataFrame]:
    """Build the trading record for one label and write data/oldnews/trade_<label>/ (tables and summary.md).

    `labels` is the labelled event table from classify (`row_id`, `label_col`, `scored`); by default it is read
    from classified_<label>.csv. `bucket`, `otm` (percent) and `label_col` (e.g. old_math_only_1) are for the
    test plan's sensitivity runs; the defaults are the committed headline, and any other choice writes to its
    own folder trade_<label>_<bucket>_otm<otm>_<label_col>/."""
    data_dir = Path(data_dir)
    inp = load(label, data_dir)
    outcome = prep_outcome(inp["outcome"])
    check_dates(outcome, label)
    if labels is None:
        labels = _read(data_dir / f"classified_{label}.csv")
    ev, ev_counts = eligible_events(inp["events"], labels, label_col, inp["gap"])
    trades, counts = build_trades(ev, inp["nulls"], outcome, bucket=bucket, otm=otm)
    counts = pd.concat([pd.DataFrame(ev_counts).assign(book="events", horizon=""), counts], ignore_index=True)

    marks = outcome[(outcome["bucket"] == bucket) & np.isclose(outcome["otm"], otm) & outcome["usable"]
                    & outcome["horizon"].isin([h for h in HORIZONS if h.isdigit()])]
    marks = marks[["row_id", "exit_date", *COSTS.values()]].rename(columns={"exit_date": "mark_date"})

    trades["taken"] = False
    summary, years, worst_rows, curves = [], [], [], []
    for (book, hz), g in trades.groupby(["book", "horizon"], sort=False):
        trades.loc[g.index, "taken"] = simulate(g)
        tk = trades.loc[g.index][trades.loc[g.index, "taken"]]
        mk = marks[marks["row_id"].isin(tk["row_id"])]
        for cost, pnl in COSTS.items():
            key = {"book": book, "horizon": hz, "cost": cost}
            curve = equity(tk, mk, pnl)
            summary.append({**key, "n_eligible": len(g), "n_skipped_cap": len(g) - len(tk),
                            **book_metrics(tk, curve, pnl)})
            curves.append(curve.assign(**key))
            worst_rows.append(worst(tk, pnl).assign(**key))
            for yr, ty in tk.groupby(tk["entry_date"].dt.year):
                years.append({**key, "year": int(yr), **book_metrics(ty, equity(ty, mk, pnl), pnl)})

    summary, years = pd.DataFrame(summary), pd.DataFrame(years)
    worst_df, curves = pd.concat(worst_rows, ignore_index=True), pd.concat(curves, ignore_index=True)
    head = trades[(trades["book"] == "old") & (trades["horizon"] == HEADLINE_HORIZON) & trades["taken"]]
    cap = capacity(head)
    h2_df = h2(trades)

    headline = (bucket, float(otm), label_col) == (BUCKET, OTM, "old")
    out_dir = out_dir or data_dir / (f"trade_{label}" if headline else f"trade_{label}_{bucket}_otm{otm:g}_{label_col}")
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {"counts": counts, "trades": trades, "summary": summary, "by_year": years, "worst": worst_df,
              "equity": curves, "capacity": cap, "h2": h2_df}
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    n_exit_2024 = int((head["exit_date"] >= DISCOVERY_END).sum()) if label == "discovery" else 0
    (out_dir / "summary.md").write_text(_summary_md(label, tables, capacity_summary(cap), n_exit_2024, bucket, otm,
                                                    label_col), encoding="utf-8")
    if log:
        _log(label, summary, h2_df, bucket, otm, label_col, data_dir / "ledger.csv")
    return tables


def _log(label: str, summary: pd.DataFrame, h2_df: pd.DataFrame, bucket: str, otm: float, label_col: str,
         path: Path) -> None:
    """Append every book and H2 variant to the test ledger (CLAUDE.md rule 13)."""
    base = {"subset": label, "strategy": "cash_secured_put", "bucket": bucket, "otm": otm, "entry": "t_0",
            "filter": f"late people news, earnings excluded, usable, put volume > 0, label={label_col}",
            "seed": SEED}
    s1 = summary[summary["cost"] == "1x"].set_index(["book", "horizon"])
    s2 = summary[summary["cost"] == "2x"].set_index(["book", "horizon"])
    books = pd.DataFrame([{**base, "kind": "trade_book", "group": b, "horizon": hz,
                           "metric": "mean csp_net per taken trade (fraction of collateral), 5-position book",
                           "n_sets": r["n_trades"], "event_mean": r["mean_trade_pct"] / 100,
                           "event_mean_2x": s2.loc[(b, hz), "mean_trade_pct"] / 100,
                           "low_sample": r["n_trades"] < LOW_SAMPLE} for (b, hz), r in s1.iterrows()])
    t1 = h2_df[h2_df["cost"] == "1x"].set_index(["comparison", "horizon"])
    t2 = h2_df[h2_df["cost"] == "2x"].set_index(["comparison", "horizon"])
    tests = pd.DataFrame([{**base, "kind": "H2", "group": c, "horizon": hz,
                           "metric": "csp_net (fraction of collateral); p_z holds the one-sided bootstrap p",
                           "n_sets": r["n"], "n_tickers": r["n_tickers"], "event_mean": r["old_mean_pct"] / 100,
                           "null_mean": r["base_mean_pct"] / 100, "diff": r["diff_pct"] / 100, "p_z": r["p_boot"],
                           "q_bh": r["q_bh"], "event_mean_2x": t2.loc[(c, hz), "old_mean_pct"] / 100,
                           "diff_2x": t2.loc[(c, hz), "diff_pct"] / 100, "low_sample": r["descriptive"],
                           "n_perm": N_BOOT} for (c, hz), r in t1.iterrows()])
    ledger.append(path, pd.concat([books, tests], ignore_index=True), ledger.new_run_id(), {}, _git_head())


def _summary_md(label: str, t: dict[str, pd.DataFrame], cap: dict, n_exit_2024: int, bucket: str, otm: float,
                label_col: str) -> str:
    s, hz = t["summary"], HEADLINE_HORIZON
    head = s[s["horizon"] == hz][["book", "cost", "n_eligible", "n_trades", "n_skipped_cap", "total_return_pct",
                                  "annualised_pct", "hit_rate_pct", "mean_trade_pct", "max_dd_closed_pct",
                                  "max_dd_mtm_pct"]]
    yr = t["by_year"]
    yr = yr[(yr["book"] == "old") & (yr["horizon"] == hz)][["year", "cost", "n_trades", "total_return_pct",
                                                              "hit_rate_pct", "max_dd_closed_pct", "max_dd_mtm_pct"]]
    w = t["worst"]
    w = w[(w["book"] == "old") & (w["horizon"] == hz) & (w["cost"] == "1x")].drop(columns=["book", "horizon", "cost"])
    h = t["h2"][t["h2"]["horizon"] == hz].drop(columns="horizon")
    hor = s[(s["book"] == "old")][["horizon", "cost", "n_trades", "total_return_pct", "mean_trade_pct",
                                   "max_dd_mtm_pct"]]
    n_old = int(head.loc[(head["book"] == "old") & (head["cost"] == "1x"), "n_trades"].sum())
    ev = t["counts"][t["counts"]["book"] == "events"][["step", "kept"]]
    lines = [
        f"# Trade record: {label}",
        "",
        f"Cash-secured put, {bucket} bucket, strike {otm:g}% below spot, sold at t_0 on old-news late people-news "
        f"events (label `{label_col}`), closed after {hz} sessions (headline). At most {MAX_OPEN} positions open, "
        "taken in order of entry, equal collateral; P&L in % of book collateral, net of max(5% of premium, "
        "$0.05/share) on entry and exit; 2x = double costs. Rules: docs/test_plan.md.",
        "",
        f"**{'Descriptive only' if n_old < LOW_SAMPLE else 'Sample size'}: {n_old} old-news trades at h = {hz}"
        f"{' (fewer than 30)' if n_old < LOW_SAMPLE else ''}.**",
        "",
        "## Events", _md(ev), "",
        f"## Books at h = {hz}", _md(head), "",
        "Books: `old` = the strategy; `null_r<k>` = the same put on round-k matched ordinary days; `all_late` = "
        "all late people-news events. MTM drawdown marks open trades at the fixed horizons only.", "",
        f"## Old-news book by year (2022 separately), h = {hz}", _md(yr), "",
        f"## Five worst old-news trades, h = {hz}, 1x costs", _md(w), "",
        f"## H2 at h = {hz} (every eligible trade, before the cap; prediction: positive)", _md(h), "",
        f"## Capacity (10% of entry-day put volume), old-news trades taken at h = {hz}",
        _md(pd.DataFrame([cap])), "",
        "`book_at_median_usd` = 5 slots x the median per-trade capacity: the book size at which half the "
        "trades fit inside the limit. `pnl_at_capacity_usd` = dollar P&L (1x costs) if every trade were sized "
        "at its capacity.", "",
        "## Old-news book at every horizon", _md(hor), "",
    ]
    if n_exit_2024:
        lines += [f"Note: {n_exit_2024} headline discovery trades exit on or after 2024-01-01.", ""]
    return "\n".join(lines)
