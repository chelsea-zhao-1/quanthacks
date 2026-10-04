"""Synthetic tests for src/oldnews/diagnostics.py (no real data, no network, nothing outside a temp folder).
Run from the repo root:  .venv/Scripts/python.exe tests/test_oldnews_diagnostics.py

Every synthetic date lies inside the allowed window (2024-2025); the label guard and the date guard are checked."""
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np            # .venv/Scripts/python.exe
import pandas as pd

from oldnews import classify as cl
from oldnews import diagnostics as D
from oldnews import tests as T

FAILS: list[str] = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def raises(f, exc=PermissionError):
    try:
        f()
    except exc:
        return True
    except Exception:
        return False
    return False


# ---- 1. power: the formulas ----------------------------------------------------------------------------------
rng = np.random.default_rng(1)
a, b = rng.normal(-0.1, 0.4, 40), rng.normal(0.0, 0.5, 90)
pw = dict(zip(*D.power(np.r_[a, b], np.r_[np.ones(40, bool), np.zeros(90, bool)]).to_dict("list").values()))
se = math.sqrt(a.var(ddof=1) / 40 + b.var(ddof=1) / 90)
z = 1.6448536269514722 + 0.8416212335729143
check("power: SE is the Welch standard error and MDE = (z_0.95 + z_0.80) * SE",
      abs(pw["se_difference"] - se) < 1e-12 and abs(pw["mde"] - z * se) < 1e-9)
check("power: n per group for 0.10 and 0.20 = ceil(z^2 (s1^2 + s2^2) / delta^2)",
      pw["n_per_group_for_0.10"] == math.ceil(z ** 2 * (a.var(ddof=1) + b.var(ddof=1)) / 0.01)
      and pw["n_per_group_for_0.20"] == math.ceil(z ** 2 * (a.var(ddof=1) + b.var(ddof=1)) / 0.04))
check("power: the rules-out bound is observed - z_0.95 * SE, and power at an effect equal to the MDE is 80%",
      abs(pw["rules_out_below"] - (a.mean() - b.mean() - 1.6448536269514722 * se)) < 1e-9
      and abs(D.NormalDist().cdf(pw["mde"] / se - 1.6448536269514722) - 0.80) < 1e-9)

# ---- 2. kappa and separation ---------------------------------------------------------------------------------
x = np.r_[np.ones(20), np.ones(5), np.zeros(10), np.zeros(15)].astype(bool)
y = np.r_[np.ones(20), np.zeros(5), np.ones(10), np.zeros(15)].astype(bool)
check("Cohen's kappa on a known 2x2 (20, 5 / 10, 15) is 0.4", abs(D.cohen_kappa(x, y) - 0.4) < 1e-12)
check("Cohen's kappa is 1 for identical labels and 0 for independent-looking ones",
      abs(D.cohen_kappa(x, x) - 1) < 1e-12 and abs(D.cohen_kappa(np.r_[1, 1, 0, 0].astype(bool), np.r_[1, 0, 1, 0].astype(bool))) < 1e-12)
s = pd.DataFrame({"is_old": [True, True, False, False, False], "gap_move": [3.0, 1.0, 0.5, 0.2, np.nan],
                  "d_iv_gap": [0.01, 0.03, 0.0, -0.01, 0.0], "vol_gap_ratio": [2.0, 4.0, 1.0, 1.0, 1.0],
                  "log_vol_gap_ratio": np.log([2.0, 4.0, 1.0, 1.0, 1.0]), "T": [1, 2, 0, 0, 0], "M": [1.5, 0.2, 0.1, -0.2, 0.0]})
sep = D.separation(s).set_index("input")
check("separation: means and medians per group, missing values skipped",
      sep.at["gap_move", "mean_old"] == 2.0 and sep.at["gap_move", "n_surprise"] == 2
      and abs(sep.at["gap_move", "mean_surprise"] - 0.35) < 1e-12 and sep.at["T", "median_old"] == 1.5
      and abs(sep.at["d_iv_gap", "mean_diff"] - (0.02 - (-1 / 300))) < 1e-12)

# ---- 3. influence --------------------------------------------------------------------------------------------
inf_s = pd.DataFrame({"ticker": ["A", "A", "B", "B", "C", "C", "D", "D"], "is_old": [True, False] * 4,
                      "d": [2.0, 0.0, -0.1, 0.0, -0.1, 0.0, -0.1, 0.0]})
loto, inf = D.influence(inf_s)
check("influence: full effect, leave-one-ticker-out range and sign flips",
      abs(inf["full_effect"] - 0.425) < 1e-12 and inf["n_left_out_runs"] == 4 and abs(inf["loto_min"] + 0.1) < 1e-12
      and abs(inf["loto_max"] - 0.6) < 1e-12 and inf["sign_flips"] == 1 and inf["n_negative"] == 1
      and loto.iloc[0]["ticker_left_out"] == "A")
d10 = np.r_[np.arange(10.0), 100.0]
check("influence: the 10% trimmed mean drops one value from each tail of each group (n = 11)",
      abs(D.stats.trimmed_mean(d10, 0.10) - np.arange(1.0, 10.0).mean()) < 1e-12)

# ---- 4. a full run on synthetic files ------------------------------------------------------------------------
TMP = Path(tempfile.mkdtemp(prefix="oldnews_diag_"))


def synth(n_ev=90, n_tick=12, seed=7, gap_start_override=None):
    r = np.random.default_rng(seed)
    days = pd.bdate_range("2024-02-01", "2025-10-31")
    ev_rows, nl_rows, gap_rows, out_rows = [], [], [], []
    for i in range(n_ev):
        t0 = days[r.integers(10, len(days) - 80)]
        rid = f"event|acc{i:04d}"
        tick = f"T{i % n_tick}"
        Tfull = int(r.integers(0, 3))
        ev_rows.append({"row_id": rid, "kind": "event", "accession_number": f"acc{i:04d}", "ticker": tick,
                        "filing_date": (t0 - pd.offsets.BDay(1)).date(), "event_date": (t0 - pd.offsets.BDay(3)).date(),
                        "gap_start": (t0 - pd.offsets.BDay(4)).date(), "t_pre": (t0 - pd.offsets.BDay(1)).date(),
                        "t_0": t0.date(), "n_gap": 3, "late": 1, "group": "people" if i % 5 else "placebo",
                        "tags": "ceo_departure" if i % 5 else "dividend_declaration", "earnings_excluded": 0,
                        "T": Tfull, "full_text_status": "ok", "T_full": Tfull, "T_full4": Tfull,
                        "cue_exhibit_dated_prior_full": 0})
        gm = abs(r.normal(0, 1.5))
        gap_rows.append({"row_id": rid, "usable": True, "gap_move": gm, "d_iv_gap": r.normal(0, 0.02),
                         "vol_gap_ratio": float(np.exp(r.normal(0, 0.5))), "reason": ""})
        old_like = gm + Tfull >= 1.5
        ids = [rid]
        for k in range(2):
            nid = f"null|{tick}|{i}|{k}"
            nt0 = t0 + pd.offsets.BDay(10 * (k + 1))
            nl_rows.append({"row_id": nid, "kind": "null", "event_row_id": rid, "ticker": tick,
                            "t_pre": (nt0 - pd.offsets.BDay(1)).date(), "t_0": nt0.date(),
                            "gap_start": (nt0 - pd.offsets.BDay(4)).date(), "n_gap": 3, "round": k})
            gap_rows.append({"row_id": nid, "usable": True, "gap_move": abs(r.normal(0, 1)), "d_iv_gap": r.normal(0, 0.02),
                             "vol_gap_ratio": float(np.exp(r.normal(0, 0.5))), "reason": ""})
            ids.append(nid)
        for rid_ in ids:
            for h in T.HORIZONS:
                usable = h not in ("42", "63")
                shift = -0.2 if (rid_ == rid and old_like) else 0.0
                out_rows.append({"row_id": rid_, "bucket": "1m", "horizon": h, "otm": 3, "entry_date": t0.date(),
                                 "exit_date": (t0 + pd.offsets.BDay(10)).date(), "y": r.normal(-0.1 + shift, 0.3),
                                 "usable": usable})
    ev = pd.DataFrame(ev_rows)
    if gap_start_override is not None:
        ev.loc[0, "gap_start"] = gap_start_override
    return ev, pd.DataFrame(nl_rows), pd.DataFrame(gap_rows), pd.DataFrame(out_rows)


def write(d: Path, tabs):
    d.mkdir(parents=True, exist_ok=True)
    for name, df in zip(("events", "nulls", "gap", "outcome"), tabs):
        df.to_csv(d / f"{name}_insample.csv", index=False)


ZREF = TMP / "zref.json"
ZREF.write_text(json.dumps({"source": "insample", "uses_outcomes": False, "window_t_0": {"first": "2024-02-01", "last": "2025-10-31"},
                            "n_null_rows": 1, "null_row_ids_sha256": "0" * 64,
                            "constants": {"gap_move": {"mean": 1.0, "sd": 1.0, "n": 10}, "d_iv_gap": {"mean": 0.0, "sd": 0.02, "n": 10},
                                          "log_vol_gap_ratio": {"mean": 0.0, "sd": 0.5, "n": 10}}}), encoding="utf-8")
dd = TMP / "data"
write(dd, synth())
log = T.Log(dd / "ledger.csv")
res = D.run(dd, log=log, n_boot=500, zref=ZREF)
out = dd / "results_insample" / "diagnostics"
check("run writes the six tables and diagnostics.md, labelled exploratory",
      all((out / f"{k}.csv").exists() for k in ("power", "separation", "agreement", "loto", "influence", "levels"))
      and D.HEADER in (out / "diagnostics.md").read_text(encoding="utf-8"))
inp = D.load(dd, zref=ZREF)
h1 = T.h1(inp, T.Log(TMP / "scratch_ledger.csv"), n_perm=50, n_boot=50).iloc[0]
check("the rebuilt sample gives exactly the primary H1 effect and group sizes",
      abs(res["check"]["effect"] - h1["effect"]) < 1e-12 and res["check"]["n_old"] == h1["n_old"]
      and res["check"]["n_sur"] == h1["n_comp"], f"{res['check']} vs {h1['effect']}, {h1['n_old']}, {h1['n_comp']}")
led = pd.read_csv(dd / "ledger.csv")
check("every ledger row has kind 'diagnostic' and one run id", set(led["kind"]) == {"diagnostic"}
      and led["run_id"].nunique() == 1 and len(led) > 20)
lev = res["levels"]
s10 = res["sample"]
check("levels: two groups at each of the nine horizons; at h = 10 the group means match the sample",
      len(lev) == 18 and abs(lev.query("horizon == '10' and group == 'old'")["mean_y"].iloc[0] - s10.loc[s10["is_old"], "d"].mean()) < 1e-12
      and lev.query("horizon == '42'")["n"].sum() == 0)
check("levels: mean_y = mean_y_event - mean_y_ordinary in every non-empty cell",
      np.allclose((lev["mean_y_event"] - lev["mean_y_ordinary"]).dropna(), lev["mean_y"].dropna()))
check("the sample is the people set only (placebo filings never enter)", set(s10["group"]) == {"people"})

# ---- 5. the window guard -------------------------------------------------------------------------------------
for lab in ("discovery", "dryrun", "holdout", "oos", "anything"):
    check(f"label {lab!r} is refused before anything is read", raises(lambda: D.load(dd, lab, zref=ZREF)))
bad = TMP / "bad"
write(bad, synth(gap_start_override="2023-12-29"))
check("a date before 2024-01-01 makes the guard refuse", raises(lambda: D.run(bad, n_boot=10, zref=ZREF, log=T.Log(bad / "l.csv"))))
write(bad, synth(gap_start_override="2023-07-03"))
check("a date inside the sealed placeholder makes the guard refuse",
      raises(lambda: D.run(bad, n_boot=10, zref=ZREF, log=T.Log(bad / "l.csv"))))
src = (ROOT / "src" / "oldnews" / "diagnostics.py").read_text(encoding="utf-8")
check("no code path names the unused 2022-23 folder (only the docstring mentions it)",
      "_unused_2022_23" not in src.replace(D.__doc__, ""))
check("no network: the module imports no HTTP client", not any(w in src for w in ("requests", "urllib", "http.client", "massive")))

shutil.rmtree(TMP, ignore_errors=True)
print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
