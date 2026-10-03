"""Synthetic tests for the playground (no real data, no network). Run:  .venv/Scripts/python tests/test_playground.py"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402

from playground import explore, ledger, measure, rules, scan, stats   # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{': ' + detail if detail else ''}")
    if not ok:
        FAILS.append(name)


RULES = {
    "test_grid": {"levels": ["tertiary", "secondary", "primary"], "earnings_subsets": ["all", "alone", "cofiled"],
                  "strategies": ["long_call", "covered_call", "protective_put", "collar", "cash_secured_put"],
                  "buckets": ["1m", "3-6m"], "horizons": [5, 21, "exp"], "otm": [0.03, 0.05, 0.10], "entry": "post",
                  "measures": ["realized_vs_implied", "iv_change", "skew_change", "drift"]},
    "earnings_tags": ["earn"], "min_sets": 10, "low_sample_n": 30, "permutations": 400, "p_value": "p_z",
    "fdr_family": "all", "fdr_q": 0.10, "headline": {"bucket": ["3-6m"], "otm": [0.05]}, "seed": 1,
    "robustness": {"trimmed_mean_frac": 0.05, "trimmed_mean_same_sign": True, "leave_one_ticker_out": True,
                   "leave_one_quarter_out": True, "plateau_min_same_sign_neighbours": 0},
    "doubled_cost_must_hold": True,
    "selection": {"direction": "event_better_and_net_positive", "allow_low_sample": False, "ranking": "q_then_effect"},
    "max_candidates": 5, "_commit": "synthetic", "_sha256": "synthetic",
}
TAXONOMY = pd.DataFrame({"tertiary_category": ["tagA", "tagB", "tagC", "earn"],
                         "secondary_category": ["famX", "famX", "famY", "famE"],
                         "primary_category": ["P", "P", "P", "Q"]})
STRATS = RULES["test_grid"]["strategies"]


def synthetic_table(n_sets=240, effect=0.0, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_sets):
        tag = ["tagA", "tagB", "tagC"][i % 3]
        cofiled = rng.random() < 0.25
        tags = tag + ("|earn" if cofiled else "")
        sid, q = f"T{rng.integers(40)}|{i}", f"2022Q{rng.integers(1, 5)}"
        for b in ("1m", "3-6m"):
            for h in (5, 21, "exp"):
                for o in (0.03, 0.05, 0.10):
                    for kind, rnd in (("event", None), ("null", 1), ("null", 2)):
                        base = {s: rng.normal(0, 0.02) for s in STRATS}
                        if kind == "event" and tag == "tagA" and (b, h, o) == ("3-6m", 21, 0.05):
                            base["covered_call"] += effect
                        rows.append({"kind": kind, "set_id": sid, "round": rnd, "tags": tags, "earnings_cofiled": cofiled,
                                     "quarter": q, "entry": "post", "bucket": b, "horizon": h, "otm": o,
                                     **{f"net_{s}": v for s, v in base.items()},
                                     **{f"net2x_{s}": v - 0.002 for s, v in base.items()},
                                     "ratio": abs(rng.normal(1, 0.3)), "iv_change": rng.normal(0, 0.02),
                                     "skew_change": rng.normal(0, 0.01), "realized": rng.normal(0, 0.03)})
    return pd.DataFrame(rows)


# ---- stats ---------------------------------------------------------------------------------------------
q = stats.benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.20, np.nan]))
check("BH matches hand computation", np.allclose(q[:4], [0.04, 0.04 * 4 / 3 * 3 / 3, 0.04, 0.20], atol=1e-9) or
      np.allclose(q[:4], [0.04, 0.0533333, 0.0533333, 0.2], atol=1e-6), str(np.round(q, 4)))
rng = np.random.default_rng(3)
ps = [stats.permutation_test(rng.normal(0, 1, 60), rng.normal(0, 1, (60, 2)), 300, rng)[1] for _ in range(300)]
check("permutation p calibrated under the null", 0.02 <= np.mean(np.array(ps) < 0.05) <= 0.09, f"{np.mean(np.array(ps) < 0.05):.3f} below 0.05")
obs, p, n, pz = stats.permutation_test(rng.normal(0.5, 1, 80), rng.normal(0, 1, (80, 2)), 500, rng)
check("permutation test detects a 0.5 sd shift", p < 0.01 and pz < 0.01 and n == 80, f"p={p:.4f}, p_z={pz:.2e}")
E, N = np.array([1.0, np.nan, 2.0, 3.0]), np.array([[0.0, np.nan], [1.0, 1.0], [np.nan, np.nan], [1.0, 2.0]])
d, keep = stats.matched_diffs(E, N)
check("matched diffs drop sets without an event or without any null", list(keep) == [True, False, False, True] and np.allclose(d, [1.0, 1.5]))

# ---- rules gate ------------------------------------------------------------------------------------------
try:
    rules.load()
    check("template rules refuse (TBD left)", False)
except rules.RulesNotReady as e:
    check("template rules refuse (TBD left)", "TBD" in str(e), str(e)[:90])
tmp_rules = ROOT / "data" / "_test_rules.json"
tmp_rules.parent.mkdir(exist_ok=True)
tmp_rules.write_text(json.dumps({k: v for k, v in RULES.items() if not k.startswith("_")}))
try:
    rules.load(tmp_rules)
    check("complete but uncommitted rules refuse", False)
except rules.RulesNotReady as e:
    check("complete but uncommitted rules refuse", "not committed" in str(e), str(e)[:90])
finally:
    tmp_rules.unlink()

# ---- hard stop on 2024 dates -------------------------------------------------------------------------------
try:
    measure.check_dates(pd.DataFrame({"t_0": pd.to_datetime(["2023-12-29", "2024-01-02"]),
                                      "t_pre": pd.to_datetime(["2023-12-28", "2023-12-29"])}))
    check("rows on or after 2024-01-01 hard-fail", False)
except ValueError as e:
    check("rows on or after 2024-01-01 hard-fail", "2024-01-01" in str(e))

# ---- scan: no effect, then a planted effect ----------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    res0 = scan.run_scan(synthetic_table(effect=0.0), TAXONOMY, RULES, tmp / "scan", tmp / "ledger.csv")
    atlas0 = pd.read_csv(res0["out"] / "atlas.csv")
    frac = (atlas0.p_perm < 0.05).mean()
    check("no-effect scan: about 5% of p below 0.05", 0.02 <= frac <= 0.09, f"{frac:.3f} of {len(atlas0)} tests")
    check("no-effect scan: no candidates", res0["candidates"] == 0, f"{res0['candidates']} candidates")
    check("ledger has one row per test", ledger.variant_count(tmp / "ledger.csv") == res0["tests"], f"{res0['tests']} tests")

    res1 = scan.run_scan(synthetic_table(effect=0.03), TAXONOMY, RULES, tmp / "scan", tmp / "ledger.csv")
    cands = pd.read_csv(res1["out"] / "candidates.csv")
    top = cands.iloc[0] if len(cands) else None
    check("planted effect is the top candidate",
          top is not None and top.group in ("tagA", "famX", "P") and top.strategy == "covered_call"
          and top.bucket == "3-6m" and str(top.horizon) == "21" and abs(top.otm - 0.05) < 1e-9,
          "" if top is None else f"{top.level}/{top.group}/{top.subset} {top.strategy} {top.bucket} h={top.horizon} otm={top.otm} q={top.q_bh:.2e}")
    check("ledger accumulates across runs", ledger.variant_count(tmp / "ledger.csv") == res0["tests"] + res1["tests"])
    atlas1 = pd.read_csv(res1["out"] / "atlas.csv")
    check("long call tested once per bucket and horizon (OTM-free)",
          atlas1[(atlas1.strategy == "long_call")].groupby(["level", "group", "subset", "bucket", "horizon"]).size().max() == 1)

    hl = dict(RULES, fdr_family="headline")
    res2 = scan.run_scan(synthetic_table(effect=0.03), TAXONOMY, hl, tmp / "scan", tmp / "ledger.csv")
    a2 = pd.read_csv(res2["out"] / "atlas.csv")
    outside = a2[(a2.bucket != "3-6m")]
    check("headline family: q only inside the headline grid", outside.q_bh.isna().all() and a2[a2.bucket == "3-6m"].q_bh.notna().any())

    # explorer on the same synthetic table, logged to the ledger
    t = synthetic_table(effect=0.03)
    r = explore.query(t, TAXONOMY, RULES, "tertiary", "tagA", "net_covered_call", "3-6m", 21, 0.05)
    check("explorer finds the planted effect", r["n_sets"] == 80 and r["p_perm"] < 0.01 and r["diff"] > 0.02,
          f"n={r['n_sets']}, diff={r['diff']:.4f}, p={r['p_perm']:.4f}")
    set_no = t.set_id.str.split("|").str[1].astype(int)
    t_feat = t.assign(timing=np.where(set_no % 2 == 0, "after_close", "intraday"))
    r2 = explore.query(t_feat, TAXONOMY, RULES, "tertiary", "tagA", "net_covered_call", "3-6m", 21, 0.05,
                       where="timing == 'after_close'")
    check("explorer where-filter narrows the events", 0 < r2["n_sets"] < 80, f"n={r2['n_sets']}")

# ---- confirmation: Holm, verdicts ---------------------------------------------------------------------------
from playground import confirm  # noqa: E402

check("Holm adjustment", np.allclose(confirm.holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06]))
cands = pd.DataFrame({"level": ["tertiary"] * 4, "group": list("abcd"), "subset": ["all"] * 4,
                      "strategy": ["covered_call"] * 4, "bucket": ["3-6m"] * 4, "horizon": [21] * 4,
                      "otm": [0.05] * 4, "diff": [0.01, 0.01, 0.01, 0.01], "q_bh": [0.01] * 4})
res = pd.DataFrame({"n_sets": [40, 40, 40, 3], "event_mean": [0.004, 0.004, -0.001, 0.01],
                    "diff": [0.008, -0.008, 0.008, 0.02], "p_z": [0.002, 0.002, 0.002, 0.001],
                    "event_mean_2x": [0.002, 0.002, -0.003, 0.01], "diff_2x": [0.006, -0.01, 0.006, 0.02]})
conf = {"p_max_one_sided": 0.05, "multiple_testing": "holm", "net_positive": True, "doubled_cost_must_hold": True, "min_sets": 10}
v = confirm.judge(cands, res, conf, "p_z")
check("confirmation verdicts: pass / wrong sign / not net positive / too few sets",
      list(v.verdict) == ["CONFIRMED", "not confirmed", "not confirmed", "untestable (too few in-sample sets)"], str(list(v.verdict)))
check("one-sided p uses the scan direction", np.isclose(v.p_one_sided[0], 0.001) and np.isclose(v.p_one_sided[1], 0.999))

# ---- measurement layer on a synthetic PricedEvent (real notebook pricing classes and evaluate) -----------------
import nb  # noqa: E402

NB = nb.load()
CAL = NB["CAL"]
days = CAL[(CAL >= "2022-02-01") & (CAL <= "2022-06-17")]
t_pre, t_0 = pd.Timestamp("2022-03-01"), pd.Timestamp("2022-03-02")
expiry = pd.Timestamp("2022-06-17")


def bars(start_px, drift):
    px = start_px + drift * np.arange(len(days))
    return pd.DataFrame({"close": px, "volume": 100.0}, index=pd.DatetimeIndex(days, name="session"))


legs = {"C_K": NB["Leg"]("C", "call", 100, bars(6.0, 0.01)), "P_K": NB["Leg"]("P", "put", 100, bars(5.0, -0.01))}
for o in NB["OTM_GRID"]:
    legs[f"C_U{o}"] = NB["Leg"]("CU", "call", 100 * (1 + o), bars(3.0, 0.0))
    legs[f"P_L{o}"] = NB["Leg"]("PL", "put", 100 * (1 - o), bars(2.5, 0.0))
pe = NB["PricedEvent"]("TEST", t_0, t_pre, t_0, "3-6m", expiry, CAL[CAL.searchsorted(expiry, side="right") - 1],
                       100.0, {"K": 100.0}, legs)
m = measure.measure_priced(NB, pe, {"kind": "event", "set_id": "TEST|2022-03-02", "quarter": "2022Q1"})
row = m[(m.entry == "post") & (m.horizon == 21) & (m.otm == 0.05)].iloc[0]
m_e = pe.marks(t_0)
S_e = pe.synthetic_spot(t_0, m_e)
check("long-call cost = ATM call premium / spot x haircut x 2",
      np.isclose(row.cost_long_call, m_e["C_K"] / S_e * NB["COST_HAIRCUT"] * 2), f"{row.cost_long_call:.5f}")
check("collar cost counts both OTM legs",
      np.isclose(row.cost_collar, (m_e["C_U0.05"] + m_e["P_L0.05"]) / S_e * NB["COST_HAIRCUT"] * 2))
check("net and doubled-cost columns", np.isclose(row.net_covered_call, row.covered_call - row.cost_covered_call)
      and np.isclose(row.net2x_covered_call, row.covered_call - 2 * row.cost_covered_call))
T = (expiry - t_0).days / 365
check("IV proxy at entry", np.isclose(row.iv_entry, (m_e["C_K"] + m_e["P_K"]) / (0.8 * S_e * np.sqrt(T))))
check("stock row costs nothing", (m.cost_stock == 0).all())
check("features at t_pre present", {"iv_pre", "atm_volume_ratio", "pre_drift"} <= set(m.columns) and np.isfinite(row.iv_pre))

# ---- clip_to: nothing fetched or evaluated on or after the out-of-sample start ------------------------------------
asked = []
NB2 = nb.load()
NB2["option_bars"] = lambda tk, start, end: asked.append(pd.Timestamp(end)) or pd.DataFrame()
measure.clip_to(NB2, "2026-01-01")
NB2["option_bars"]("X", pd.Timestamp("2025-11-01"), pd.Timestamp("2026-05-15"))
check("clip_to caps bar requests and LAST_SESSION", asked == [pd.Timestamp("2025-12-31")]
      and NB2["LAST_SESSION"] == pd.Timestamp("2025-12-31"), f"asked {asked}, last {NB2['LAST_SESSION'].date()}")

print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
