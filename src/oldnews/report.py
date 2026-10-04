"""Every number the note and the README quote, written from the results tables, so they cannot disagree.

    .venv/Scripts/python.exe src/oldnews/report.py --label insample [--data data/oldnews] [--out PATH]

reads data/oldnews/results_<label>/*.csv (src/oldnews/tests.py) and data/oldnews/trade_<label>/*.csv
(src/oldnews/trade.py), plus the frozen constants and the test ledger, and writes data/oldnews/README_numbers.md
(for label insample; other labels write README_numbers_<label>.md). The label is required: the retired labels
(discovery, dryrun) are refused, because nothing from 2022-2023 may reach a deliverable.
It computes nothing new: values are copied from the tables and formatted (the capacity summary is
trade.capacity_summary on trade_<label>/capacity.csv, the same call trade.py makes). The only derived words are
"sign matches the prediction" and "one-sided p below 0.05", exactly as tests.py's summary.md states them. Copy
numbers from the output; never retype them. The file sits in data/, which is git-ignored (derived from licensed
data), so rerun the script after every run of the tests.

A section is skipped, with a line saying so, when its tables do not exist yet (for example before the trade run).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent.parent
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATA = ROOT / "data" / "oldnews"
FROZEN = SRC / "oldnews" / "zref_frozen_insample.json"      # the old zref_frozen.json is a retired archive, never read
HEADLINE_H = "10"
ALPHA = 0.05


# ---- formatting ---------------------------------------------------------------------------------------------
def _num(x, d: int = 3, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x):
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def _int(x) -> str:
    return "n/a" if pd.isna(x) else f"{int(round(float(x))):,}"


def _yes(x) -> str:
    if pd.isna(x):
        return "n/a"
    return "yes" if str(x).strip().lower() in {"true", "1", "1.0", "yes"} else "no"


def _ci(lo, hi, d: int = 3) -> str:
    return "n/a" if pd.isna(lo) or pd.isna(hi) else f"[{lo:+.{d}f}, {hi:+.{d}f}]"


def table(rows: list[dict]) -> str:
    """A markdown table from a list of dicts that share their keys."""
    if not rows:
        return "(no rows)\n"
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out) + "\n"


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=["", "nan", "NaN"], dtype={"horizon": str})


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


# ---- sections -----------------------------------------------------------------------------------------------
def sources(files: list[Path], root: Path) -> str:
    rows = []
    for f in files:
        rel = f.relative_to(root.parent.parent) if root.parent.parent in f.parents else f
        n = len(pd.read_csv(f)) if f.suffix == ".csv" else ""
        rows.append({"file": f"`{rel.as_posix()}`", "rows": n, "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime)),
                     "sha256 (12)": _sha(f)})
    return table(rows)


def method_line(summary: Path) -> str:
    """The run line tests.py prints under its title (run id, git head, seed, permutations, dates, constants)."""
    if not summary.exists():
        return ""
    for block in summary.read_text(encoding="utf-8").split("\n\n"):
        if block.startswith("Run `"):
            return block.strip() + "\n"
    return ""


def sample(res: Path) -> str:
    c = _read(res / "counts.csv")
    rows = [{"step": r["step"], "people": _int(r["people"]), "placebo": _int(r["placebo"])} for _, r in c.iterrows()]
    return table(rows)


def headline(res: Path) -> str:
    rows, notes = [], []
    for name in ("h1", "h1b", "placebo"):
        f = res / f"{name}.csv"
        if not f.exists():
            continue
        r = _read(f).iloc[0]
        rows.append({
            "test": r["test"], "prediction": r["prediction"], "n": _int(r["n"]), "n old": _int(r["n_old"]),
            "n comparison": _int(r["n_comp"]), "tickers": _int(r["n_tickers"]), "effect": _num(r["effect"], 4, True),
            "95% CI": _ci(r["ci_lo"], r["ci_hi"], 4), "p one-sided": _num(r["p_one_sided"], 4),
            "p two-sided": _num(r["p_two_sided"], 4), "trimmed effect": _num(r["trimmed_effect"], 4, True),
            "leave-one-ticker-out holds": _yes(r["loto_holds"]), "leave-one-quarter-out holds": _yes(r["loqo_holds"]),
            "descriptive (n < 30)": _yes(r["descriptive"])})
        if r["test"] == "H1":
            neg = r["prediction"] == "negative"
            sign = pd.notna(r["effect"]) and ((r["effect"] < 0) == neg)
            notes.append(f"H1 sign matches the prediction ({r['prediction']}): {'yes' if sign else 'no'}. "
                         f"One-sided p below {ALPHA}: {'yes' if pd.notna(r['p_one_sided']) and r['p_one_sided'] < ALPHA else 'no'}.")
    return table(rows) + "\n" + "\n".join(notes) + "\n"


def profile(res: Path) -> str:
    f = res / "profile.csv"
    if not f.exists():
        return "(profile.csv not written)\n"
    p = _read(f)
    out = []
    for test, g in p.groupby("test", sort=False):
        out.append(f"**{test}** (prediction: {g['prediction'].iloc[0]})\n\n")
        out.append(table([{"horizon": r["horizon"], "n": _int(r["n"]), "n old": _int(r["n_old"]), "n comparison": _int(r["n_comp"]),
                           "mean old": _num(r["mean_old"], 4, True), "mean comparison": _num(r["mean_comp"], 4, True),
                           "effect": _num(r["effect"], 4, True), "95% CI": _ci(r["ci_lo"], r["ci_hi"], 4),
                           "p": _num(r["p"], 4), "q (BH)": _num(r["q_bh"], 4), "descriptive": _yes(r["descriptive"])}
                          for _, r in g.iterrows()]))
    out.append("`mean old` and `mean comparison` are the group means of the outcome (event Y minus the mean Y of its matched "
               "ordinary days); `effect` is old minus comparison. p is one-sided for H1 and H1b, two-sided for P. "
               "Horizons 42 and 63 have no observations in a 1-month bucket (the option expires first).\n")
    return "\n".join(out)


def sensitivity(res: Path) -> str:
    f = res / "sensitivity.csv"
    if not f.exists():
        return "(sensitivity.csv not written)\n"
    s = _read(f)
    return table([{"dimension": r["dimension"], "weights": r["weights"], "cutoff": _num(r["cutoff"], 2), "bucket": r["bucket"],
                   "otm": _int(r["otm"]), "category": r["category"], "n": _int(r["n"]), "n old": _int(r["n_old"]),
                   "effect": _num(r["effect"], 4, True), "95% CI": _ci(r["ci_lo"], r["ci_hi"], 4),
                   "p one-sided": _num(r["p_one_sided"], 4), "descriptive": _yes(r["descriptive"])} for _, r in s.iterrows()])


def trade(tr: Path) -> str:
    out = []
    if not (tr / "summary.csv").exists():
        return "(the trade tables are not written yet)\n"
    s = _read(tr / "summary.csv")
    h = s[s["horizon"] == HEADLINE_H]
    n_old = h[(h["book"] == "old") & (h["cost"] == "1x")]["n_trades"]
    out.append(f"**Headline: {_int(n_old.iloc[0]) if len(n_old) else 'n/a'} old-news trades at h = {HEADLINE_H}** "
               "(cash-secured put, 1m bucket, 3% below spot, at most 5 open, costs max(5% of premium, $0.05/share) each way).\n")
    out.append(f"*Books at h = {HEADLINE_H}*\n\n")
    out.append(table([{"book": r["book"], "cost": r["cost"], "eligible": _int(r["n_eligible"]), "taken": _int(r["n_trades"]),
                       "skipped (cap)": _int(r["n_skipped_cap"]), "total return %": _num(r["total_return_pct"], 2, True),
                       "annualised %": _num(r["annualised_pct"], 2, True), "hit rate %": _num(r["hit_rate_pct"], 1),
                       "mean trade %": _num(r["mean_trade_pct"], 2, True), "max drawdown % (closed)": _num(r["max_dd_closed_pct"], 2),
                       "max drawdown % (marked)": _num(r["max_dd_mtm_pct"], 2)} for _, r in h.iterrows()]))
    if (tr / "by_year.csv").exists():
        y = _read(tr / "by_year.csv")
        y = y[(y["book"] == "old") & (y["horizon"] == HEADLINE_H)]
        out.append(f"\n*Old-news book by year, h = {HEADLINE_H}*\n\n")
        out.append(table([{"year": str(int(r["year"])), "cost": r["cost"], "trades": _int(r["n_trades"]),
                           "total return %": _num(r["total_return_pct"], 2, True), "hit rate %": _num(r["hit_rate_pct"], 1),
                           "max drawdown % (marked)": _num(r["max_dd_mtm_pct"], 2)} for _, r in y.sort_values(["cost", "year"], ascending=[False, True]).iterrows()]))
    if (tr / "worst.csv").exists():
        w = _read(tr / "worst.csv")
        w = w[(w["book"] == "old") & (w["horizon"] == HEADLINE_H) & (w["cost"] == "1x")]
        out.append(f"\n*Five worst old-news trades, h = {HEADLINE_H}, 1x costs*\n\n")
        out.append(table([{"ticker": r["ticker"], "entry": r["entry_date"], "exit": r["exit_date"], "strike": _num(r["put_strike"], 2),
                           "premium": _num(r["put_premium"], 2), "P&L %": _num(r["pnl_pct"], 2, True)} for _, r in w.iterrows()]))
    if (tr / "h2.csv").exists():
        h2 = _read(tr / "h2.csv")
        h2 = h2[h2["horizon"] == HEADLINE_H]
        out.append(f"\n*H2 at h = {HEADLINE_H} (prediction: positive; every eligible trade, before the position cap)*\n\n")
        out.append(table([{"comparison": r["comparison"], "cost": r["cost"], "n": _int(r["n"]), "tickers": _int(r["n_tickers"]),
                           "old mean %": _num(r["old_mean_pct"], 2, True), "base mean %": _num(r["base_mean_pct"], 2, True),
                           "difference %": _num(r["diff_pct"], 2, True), "95% CI %": _ci(r["ci_lo_pct"], r["ci_hi_pct"], 2),
                           "p (bootstrap)": _num(r["p_boot"], 3), "q (BH)": _num(r["q_bh"], 3), "descriptive": _yes(r["descriptive"])}
                          for _, r in h2.iterrows()]))
    if (tr / "capacity.csv").exists():
        try:
            from oldnews import trade as trade_module
            c = trade_module.capacity_summary(pd.read_csv(tr / "capacity.csv"))
            out.append("\n*Capacity (contracts limited to 10% of the put's entry-day volume), old-news trades taken at h = "
                       f"{HEADLINE_H}*\n\n")
            out.append(table([{k: (f"{v:,.0f}" if isinstance(v, (int, float, np.integer, np.floating)) else v)
                               for k, v in c.items()}]))
        except Exception as e:  # noqa: BLE001 (the capacity summary is optional; say why it is missing)
            out.append(f"\n(capacity summary unavailable: {type(e).__name__}: {e})\n")
    a = s[(s["book"] == "old")]
    out.append("\n*Old-news book at every horizon*\n\n")
    out.append(table([{"horizon": r["horizon"], "cost": r["cost"], "trades": _int(r["n_trades"]),
                       "total return %": _num(r["total_return_pct"], 2, True), "mean trade %": _num(r["mean_trade_pct"], 2, True),
                       "max drawdown % (marked)": _num(r["max_dd_mtm_pct"], 2)} for _, r in a.iterrows()]))
    return "".join(out)


def frozen_file() -> Path:
    """The frozen constants file classify reads (zref_frozen_insample.json); never the retired archive."""
    try:
        from oldnews import classify
        return Path(classify.frozen_path("insample"))
    except (ImportError, AttributeError):
        return FROZEN


def frozen() -> str:
    path = frozen_file()
    if not path.exists():
        return f"({path.name} does not exist yet)\n"
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = [{"input": k, "mean": _num(v["mean"], 4, True), "sd": _num(v["sd"], 4), "n": _int(v.get("n"))}
            for k, v in d["constants"].items()]
    first = d["window_t_0"]["first"]
    warn = ("**WARNING: these constants were computed from dates before 2024-01-01, outside the allowed 2024-2025 "
            "window. Re-freeze them (classify.freeze) before this file is used for the note or the README.**\n\n"
            if pd.Timestamp(first) < pd.Timestamp("2024-01-01") else "")
    return (f"{warn}Computed once from {d['n_null_rows']:,} usable ordinary days with t_0 from {first} to "
            f"{d['window_t_0']['last']} (gap inputs only, no outcomes), as stored in the file; file sha256 "
            f"`{_sha(path)}`.\n\n" + table(rows))


def ledger(data: Path) -> str:
    f = data / "ledger.csv"
    if not f.exists():
        return "(no ledger yet)\n"
    L = pd.read_csv(f, usecols=["run_id", "timestamp", "git_head", "kind"], keep_default_na=False)
    by = L.groupby("run_id", sort=False).agg(rows=("kind", "size"), first=("timestamp", "min"), git=("git_head", "first"))
    return (f"**{len(L):,} variants logged in {len(by)} runs.** Disclose this count with the results.\n\n"
            + table([{"run": f"`{i}`", "rows": _int(r["rows"]), "time": r["first"], "git": f"`{str(r['git'])[:10]}`"}
                     for i, r in by.iterrows()]))


# ---- computed in the pipeline run (pipeline.run_oldnews calls these for every label) ------------------------
# The functions above copy numbers. The ones below compute the few numbers the note quotes that tests.py and
# trade.py do not write, using the committed code unchanged (tests._row, the diagnostics functions), so that the
# notebook run reproduces them for any label. They change no existing number.

ENTRY_SHIFT_DIM = "entry t0+1"
ENTRY_SHIFT_NOTE = ("The row `entry t0+1` uses outcome_{label}_entry1.csv (entry one session after t_0, matched ordinary "
                    "days shifted the same way); labels unchanged; added after the primary run, logged to the ledger with "
                    "entry t_0+1.")
SENS_COLS = ["dimension", "weights", "text", "cutoff", "bucket", "otm", "category", "n", "n_old", "n_comp", "effect",
             "ci_lo", "ci_hi", "p_one_sided", "descriptive"]


def _inputs(label: str, data_dir: Path, NB: dict | None):
    """tests.Inputs from data_dir's tables through classify's window guard, labelled in memory with the frozen
    constants exactly as tests.run and diagnostics.load label them (nothing is written)."""
    from oldnews import classify as cl, tests as T

    frames, _ = cl.load_tables(label, data_dir, NB, ("events", "nulls", "gap", "outcome"))
    c = cl.classify(frames["events"], frames["gap"], cl.load_frozen())
    return T.prepare(frames["events"], frames["nulls"], frames["gap"], frames["outcome"], classified=c)


def entry_shift_row(label: str, data_dir: Path = DATA, NB: dict | None = None) -> pd.DataFrame:
    """The H1 sensitivity row for entry one session after t_0 (test plan, "Sensitivity"): the primary spec on
    outcome_<label>_entry1.csv, labels unchanged, the same permutations, bootstrap and seed as tests.sensitivity."""
    from oldnews import classify as cl, tests as T

    data_dir = Path(data_dir)
    frames, _ = cl.load_tables(label, data_dir, NB, ("events", "nulls", "gap"))
    o1 = cl.read_table(data_dir / f"outcome_{label}_entry1.csv")
    cl.guard_dates(label, {"outcome": o1}, NB)
    c = cl.classify(frames["events"], frames["gap"], cl.load_frozen())          # labels unchanged (frozen constants)
    inp = T.prepare(frames["events"], frames["nulls"], frames["gap"], o1, classified=c)
    row = {"dimension": ENTRY_SHIFT_DIM, **T._row(inp, "H1", T.N_PERM, T.N_BOOT)}
    return pd.DataFrame([row]).assign(period="pooled")


def _entry_shift_ledger(row: pd.Series, path: Path) -> None:
    """The ledger row for the entry t0+1 sensitivity, in tests._log's layout with the entry marked t_0+1."""
    from oldnews import tests as T
    from playground import ledger as L

    led = pd.DataFrame([{
        "kind": "oldnews_" + row["test"], "level": row["weights"], "group": row["group"], "subset": row["category"],
        "filter": f"S>={row['cutoff']:g}|entry=t0+1", "metric": row["metric"], "strategy": "", "bucket": row["bucket"],
        "horizon": row["horizon"], "otm": row["otm"], "entry": "t_0+1", "n_sets": row["n"], "n_tickers": row["n_tickers"],
        "event_mean": row["mean_old"], "null_mean": row["mean_comp"], "diff": row["effect"], "p_perm": row["p"],
        "q_bh": row["q_bh"], "trimmed_diff": row["trimmed_effect"], "loto_holds": row["loto_holds"],
        "loto_weakest": row["loto_weakest"], "loqo_holds": row["loqo_holds"], "loqo_weakest": row["loqo_weakest"],
        "low_sample": row["descriptive"], "n_perm": row["n_perm"], "seed": row["seed"]}])
    L.append(Path(path), led, L.new_run_id(), {}, T.git_head())


def add_entry_shift(label: str, data_dir: Path = DATA, NB: dict | None = None, log: bool = True) -> pd.DataFrame:
    """Append the entry t0+1 row to results_<label>/sensitivity.csv and to the sensitivity table in summary.md
    (with one line saying where it comes from), and log it. Does nothing if the row is already there."""
    from oldnews import tests as T

    res = Path(data_dir) / f"results_{label}"
    sens = pd.read_csv(res / "sensitivity.csv", keep_default_na=False, na_values=["", "nan", "NaN"])
    if (sens["dimension"] == ENTRY_SHIFT_DIM).any():
        return sens
    row = entry_shift_row(label, data_dir, NB)
    sens = pd.read_csv(res / "sensitivity.csv")
    out = pd.concat([sens, row], ignore_index=True)
    out.to_csv(res / "sensitivity.csv", index=False)
    md = res / "summary.md"
    if md.exists():
        lines = md.read_text(encoding="utf-8").split("\n")
        i = lines.index("## Sensitivity (H1, one change at a time)") + 3        # title, blank, header: rule next
        while lines[i].startswith("|"):
            i += 1                                                          # i: the blank line after the table
        body = T._md(row[SENS_COLS]).split("\n")[2:]
        lines[i:i + 1] = body + ["", ENTRY_SHIFT_NOTE.format(label=label), ""]
        md.write_text("\n".join(lines), encoding="utf-8")
    if log:
        _entry_shift_ledger(row.iloc[0], Path(data_dir) / "ledger.csv")
    return out


def _book_concentration(trades: pd.DataFrame) -> dict:
    """Old-news book at h = 10, trades taken, 1x costs, in % of each trade's collateral (as audit.py computes it):
    mean, mean without the largest loss, and the three largest trades' share of the total P&L."""
    t = trades[(trades["book"] == "old") & (trades["horizon"].astype(str) == HEADLINE_H)
               & trades["taken"].astype(str).str.lower().isin(["true", "1"])]
    r = 100 * pd.to_numeric(t["csp_net"]).to_numpy(float)
    if len(r) == 0:
        return {"n": 0, "mean": np.nan, "mean_wo_worst": np.nan, "top3_share": np.nan}
    return {"n": len(r), "mean": r.mean(), "mean_wo_worst": np.delete(r, int(np.argmin(r))).mean() if len(r) > 1 else np.nan,
            "top3_share": np.sort(r)[::-1][:3].sum() / r.sum() if r.sum() != 0 else np.nan}


def note_numbers(label: str, data_dir: Path = DATA, NB: dict | None = None) -> pd.DataFrame:
    """The note's numbers that the main tables do not print: the entry t0+1 sensitivity row, the 5% trimmed H1
    difference, the diagnostics (kappa, MDE, leave-one-ticker-out range), the old-news book's concentration, and
    its market beta and R^2. One row per number: item, value (formatted), detail. A part whose inputs are missing
    or too small says so instead of stopping the run."""
    from oldnews import diagnostics as D

    data_dir = Path(data_dir)
    res, tr = data_dir / f"results_{label}", data_dir / f"trade_{label}"
    rows: list[dict] = []

    def add(item: str, value: str, detail: str = "") -> None:
        rows.append({"item": item, "value": value, "detail": detail})

    def part(name: str, fn) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 (a small sealed window may leave a part empty; say why)
            add(name, "n/a", f"{type(e).__name__}: {str(e)[:120]}")

    def sens():
        s = _read(res / "sensitivity.csv")
        r = s[s["dimension"] == ENTRY_SHIFT_DIM]
        if r.empty:
            add("sensitivity entry t0+1", "n/a", "row not computed (no outcome_<label>_entry1.csv)")
        else:
            r = r.iloc[0]
            add("sensitivity entry t0+1: effect", _num(r["effect"], 4, True),
                f"95% CI {_ci(r['ci_lo'], r['ci_hi'], 4)}, one-sided p {_num(r['p_one_sided'], 4)}, "
                f"old/comparison {_int(r['n_old'])}/{_int(r['n_comp'])}")
        add("sensitivity settings (rows other than primary)", _int(len(s) - 1), "")

    def trimmed():
        h = _read(res / "h1.csv").iloc[0]
        add("H1 5% trimmed difference", _num(h["trimmed_effect"], 4, True), f"untrimmed {_num(h['effect'], 4, True)}")

    def diagnostics():
        s = D.h1_sample(_inputs(label, data_dir, NB), "10")
        d, old = s["d"].to_numpy(float), s["is_old"].to_numpy(bool)
        pw = D.power(d, old).set_index("quantity")["value"]
        agr_t, agr = D.agreement(s)
        _, inf = D.influence(s)
        n_old = int(agr_t["of_which_primary_old"].sum())
        math_sur = int(agr_t.loc[agr_t["math_only"] == "surprise", "of_which_primary_old"].sum())
        add("diagnostics: old calls that math alone calls surprise", f"{math_sur} of {n_old}",
            "primary equal-weight old labels whose math-only label (M >= 1) is surprise")
        add("diagnostics: events per group to detect 0.10", _int(pw["n_per_group_for_0.10"]),
            f"one-sided 5%, 80% power; for 0.20: {_int(pw['n_per_group_for_0.20'])}")
        add("diagnostics: Cohen's kappa (math-only vs words-only)", _num(agr["kappa"], 3),
            f"agreement {_num(agr['agreement'], 3)}, n {agr['n']}")
        add("diagnostics: MDE (one-sided 5%, 80% power)", _num(pw["mde"], 4),
            f"rules out effects below {_num(pw['rules_out_below'], 4, True)}")
        add("diagnostics: leave-one-ticker-out range", f"{_num(inf['loto_min'], 4, True)}..{_num(inf['loto_max'], 4, True)}",
            f"{inf['n_left_out_runs']} runs, {inf['sign_flips']} sign flips")

    def book():
        c = _book_concentration(_read(tr / "trades.csv"))
        add("trade: mean per taken trade, h = 10, 1x (% of collateral)", _num(c["mean"], 3, True), f"n {c['n']}")
        add("trade: mean without the largest loss", _num(c["mean_wo_worst"], 3, True), "")
        add("trade: top-3 trades' share of total P&L", f"{_num(100 * c['top3_share'], 0)}%", "")

    def beta():
        m = _read(tr / "metrics.csv")
        r = m[(m["book"] == "old") & (m["cost"] == "1x")].iloc[0]
        add("trade: market beta (old-news book, h = 10, 1x)", _num(r["beta"], 3), f"R^2 {_num(r['r2'], 3)}, "
            f"{_int(r['n_beta_days'])} days")
        add("trade: R^2", _num(r["r2"], 3), "")

    for name, fn in (("sensitivity", sens), ("trimmed", trimmed), ("diagnostics", diagnostics), ("book", book),
                     ("beta", beta)):
        part(name, fn)
    return pd.DataFrame(rows)


def write_note_numbers(label: str, data_dir: Path = DATA, NB: dict | None = None) -> str:
    """Write results_<label>/note_numbers.md and note_numbers.csv; return the Markdown text."""
    t = note_numbers(label, data_dir, NB)
    res = Path(data_dir) / f"results_{label}"
    res.mkdir(parents=True, exist_ok=True)
    t.to_csv(res / "note_numbers.csv", index=False)
    text = (f"## Numbers in the note: {label}\n\n" + table(t.to_dict("records"))
            + "\nComputed by the pipeline from this run's tables with the committed code (tests._row for the "
              "entry t0+1 row, the diagnostics functions, trade metrics). Exploratory diagnostics do not change the "
              "primary result.\n")
    (res / "note_numbers.md").write_text(text, encoding="utf-8")
    return text
def build(label: str, data: Path = DATA) -> str:
    from oldnews import pipeline

    pipeline.check_label(label)
    data = Path(data)
    res, tr = data / f"results_{label}", data / f"trade_{label}"
    if not (res / "h1.csv").exists():
        raise FileNotFoundError(f"{res / 'h1.csv'} not found: run the tests for {label!r} first "
                                f"(pipeline.py --label {label}, or oldnews.tests.run)")
    files = [p for p in [*sorted(res.glob("*.csv")), *sorted(tr.glob("*.csv")), frozen_file(), data / "ledger.csv"]
             if p.exists() and p.name != "equity.csv" and p.name != "trades.csv"]
    parts = [
        f"# Numbers for the note and the README: {label}\n",
        "Written by `src/oldnews/report.py` from the tables listed in section 0. Do not edit by hand and do not retype "
        "these numbers: rerun the script. Outcome Y = log(RV/IV0) minus the mean over the event's matched ordinary days; "
        "primary test: old minus surprise news, h = 10 sessions, 1-month bucket, 3% put row.\n",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')}.\n",
        "## 0. Sources\n", sources(files, data),
        "## 1. Method line (from tests.py)\n", (method_line(res / "summary.md") or "(summary.md not found)\n"),
        "## 2. Sample\n", sample(res),
        "## 3. Headline tests\n", headline(res),
        "## 4. Every horizon\n", profile(res),
        "## 5. Sensitivity (H1, one change at a time)\n", sensitivity(res),
        "## 6. The trade\n", trade(tr),
        "## 7. Frozen z-score constants\n", frozen(),
        "## 8. Test ledger (variant count)\n", ledger(data),
    ]
    return "\n".join(parts)


def output_path(label: str, data: Path = DATA) -> Path:
    """data/oldnews/README_numbers.md for the test label; README_numbers_<label>.md for any other."""
    return Path(data) / ("README_numbers.md" if label == "insample" else f"README_numbers_{label}.md")


def write(label: str, data: Path = DATA, out: Path | None = None) -> Path:
    """Write the numbers file. Refuses the retired labels and any destination that is the README or a notebook."""
    from oldnews import pipeline

    pipeline.check_label(label)
    out = Path(out) if out else output_path(label, data)
    if out.name.lower() == "readme.md" or out.suffix.lower() == ".ipynb":
        raise PermissionError(f"{out}: this script writes a numbers file for you to copy from, never the README or "
                              "a notebook itself")
    text = build(label, data)
    out.write_text(text, encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Write data/oldnews/README_numbers.md from the results tables.")
    ap.add_argument("--label", required=True, help="which results_<label>/ and trade_<label>/ to read (insample)")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    path = write(args.label, Path(args.data), args.out)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
