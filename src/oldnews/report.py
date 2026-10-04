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


# ---- the document -------------------------------------------------------------------------------------------
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
