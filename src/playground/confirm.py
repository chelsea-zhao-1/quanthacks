"""Confirmation: each scan candidate gets exactly one test on the 2024-2025 in-sample window, with the same
matched-null design. Nothing dated on or after the out-of-sample start (2026-01-01) is fetched or computed:
option bars and exits are clipped, so late-2025 events simply lose the horizons that would end in 2026.

Refuses to run until selection_rules.json (including its `confirmation` section) is complete and committed.

Usage:
  python src/playground/confirm.py plan [--scan-run RUN_ID]   tags to fetch, request estimate, the fetch command
  python src/playground/confirm.py test [--scan-run RUN_ID]   measure (offline, clipped), test, apply the rules
"""
import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "playground"

import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402

from playground import ROOT, ledger, rules as rules_mod   # noqa: E402
from playground.scan import memberships, run_test, slice_arrays   # noqa: E402

PG = ROOT / "data" / "playground"
INSAMPLE = PG / "insample"
CENSUS = ROOT / "data" / "census" / "top100_events.csv"
REQUESTS_PER_ROW = 28
RESULT_COLUMNS = ["n_sets", "n_tickers", "event_mean", "null_mean", "diff", "p_perm", "p_z", "trimmed_diff",
                  "loto_holds", "loqo_holds", "event_mean_2x", "diff_2x", "low_sample"]


def latest_scan(run_id: str | None) -> Path:
    runs = sorted((PG / "scan").glob("*/candidates.csv"))
    if run_id:
        return PG / "scan" / run_id
    if not runs:
        raise SystemExit("no scan run found; run src/playground/scan.py first")
    return runs[-1].parent


def member_tags(cands: pd.DataFrame, taxonomy: pd.DataFrame) -> list[str]:
    tags = set()
    for c in cands.itertuples(index=False):
        col = {"tertiary": "tertiary_category", "secondary": "secondary_category", "primary": "primary_category"}[c.level]
        tags |= set(taxonomy.loc[taxonomy[col] == c.group, "tertiary_category"])
    return sorted(tags)


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.maximum.accumulate([(m - i) * p[j] for i, j in enumerate(order)]).clip(max=1.0)
    out = np.empty(m)
    out[order] = adj
    return out


def judge(cands: pd.DataFrame, results: pd.DataFrame, conf: dict, p_col: str) -> pd.DataFrame:
    """Apply the committed confirmation rules. `results` has one row per candidate (same order) from run_test."""
    out = cands.reset_index(drop=True)[["level", "group", "subset", "strategy", "bucket", "horizon", "otm", "diff", "q_bh"]]
    out = out.rename(columns={"diff": "scan_diff", "q_bh": "scan_q"}).join(results.add_prefix("is_"))
    testable = out["is_n_sets"].fillna(0) >= conf["min_sets"]
    same = np.sign(out["is_diff"]) == np.sign(out["scan_diff"])
    p_two = out[f"is_{p_col}"]
    out["p_one_sided"] = np.where(same, p_two / 2, 1 - p_two / 2)
    p = out["p_one_sided"].where(testable)
    adj = {"none": lambda x: x, "bonferroni": lambda x: np.minimum(x * len(x), 1.0), "holm": holm}[conf["multiple_testing"]]
    out["p_adjusted"] = np.nan
    if testable.any():
        out.loc[testable, "p_adjusted"] = adj(p[testable].to_numpy())
    ok = testable & (out["p_adjusted"] <= conf["p_max_one_sided"])
    if conf["net_positive"]:
        ok &= out["is_event_mean"] > 0
    if conf["doubled_cost_must_hold"]:
        ok &= out["is_diff_2x"] * np.sign(out["scan_diff"]) > 0
        if conf["net_positive"]:
            ok &= out["is_event_mean_2x"] > 0
    out["verdict"] = np.where(~testable, "untestable (too few in-sample sets)", np.where(ok, "CONFIRMED", "not confirmed"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["plan", "test"])
    ap.add_argument("--scan-run", default=None)
    args = ap.parse_args()
    rl = rules_mod.load()
    conf = rl["confirmation"]
    run_dir = latest_scan(args.scan_run)
    cands = pd.read_csv(run_dir / "candidates.csv")
    taxonomy = pd.read_csv(PG / "taxonomy.csv")
    if cands.empty:
        print(f"Scan run {run_dir.name} has no candidates: nothing to confirm. Report the null result.")
        return 0
    tags = member_tags(cands, taxonomy)

    import os
    os.chdir(ROOT)
    import nb
    from playground import measure
    NB = nb.load()
    start, end, oos = NB["STUDY_START"], NB["STUDY_END"], NB["OOS_START"]
    assert end < oos

    if args.step == "plan":
        census = pd.read_csv(ROOT / "data" / "census" / "by_tertiary.csv")
        n_ev = int(census[(census.window == "in_sample") & census.tertiary.isin(tags)].events_top100.sum())
        est = n_ev * 3 * REQUESTS_PER_ROW
        print(f"{len(cands)} candidates from scan run {run_dir.name}; {len(tags)} tags; up to {n_ev:,} in-sample events "
              f"(+2 null days each): about {est:,} requests at most (overlaps and cache make it less).")
        print("Ask before running if this is more than a few thousand (CLAUDE.md rule 18). Then:")
        print(f"  .venv\\Scripts\\python src\\discovery_fetch.py --label insample --start {start} --end {end} "
              f"--hard-stop {oos} --clip-before {oos} --tags {','.join(tags)} --busy-csv {CENSUS.relative_to(ROOT)} "
              f"--out {INSAMPLE.relative_to(ROOT)} --max-requests {est}")
        return 0

    events = pd.read_csv(INSAMPLE / "events_insample.csv", parse_dates=["filing_date", "accepted_at", "t_0", "t_pre"])
    nulls = pd.read_csv(INSAMPLE / "nulls_insample.csv", parse_dates=["filing_date", "t_0", "t_pre"])
    # The fetch covered only the candidates' tags; take each filing's full tag set from the census so that
    # "co-filed with earnings" is judged on every tag the filing carries.
    census = pd.read_csv(CENSUS, parse_dates=["filing_date"])
    full = (census[census.window == "in_sample"].groupby(["cik", "filing_date"])["tertiary"]
            .apply(lambda s: "|".join(sorted(set(s)))).rename("census_tags"))
    events = events.merge(full, on=["cik", "filing_date"], how="left")
    events["tags"] = [("|".join(sorted(set(a.split("|")) | set(b.split("|")))) if isinstance(b, str) else a)
                      for a, b in zip(events["tags"], events["census_tags"])]
    table, failures = measure.build(NB, events.drop(columns="census_tags"), nulls, rl["earnings_tags"],
                                    hard_stop=oos, clip_before=oos)
    table.to_pickle(INSAMPLE / "measurements.pkl")
    failures.to_csv(INSAMPLE / "measurement_failures.csv", index=False)

    rng = np.random.default_rng(int(rl["seed"]))
    rows = []
    for c in cands.itertuples(index=False):
        sids = memberships(table, taxonomy, [c.level], [c.subset]).get((c.level, c.group, c.subset), set())
        col = f"net_{c.strategy}"
        sl = slice_arrays(table[(table.bucket == c.bucket) & (table.otm == (c.otm if pd.notna(c.otm) else 0.03))],
                          rl["test_grid"]["entry"], [col, col.replace("net_", "net2x_")])
        key = (c.bucket, c.horizon if c.horizon == "exp" else int(c.horizon), float(c.otm) if pd.notna(c.otm) else 0.03)
        r = run_test(*sl[key], sids, col, col.replace("net_", "net2x_"), rng, dict(rl, min_sets=0)) if key in sl else None
        rows.append(r or {"n_sets": 0})
    results = pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS)
    verdicts = judge(cands, results, conf, rl["p_value"])

    run_id = ledger.new_run_id()
    out = PG / "confirm" / run_id
    out.mkdir(parents=True, exist_ok=True)
    verdicts.to_csv(out / "confirmation.csv", index=False)
    led = verdicts.rename(columns=lambda k: k[3:] if k.startswith("is_") else k).assign(
        kind="confirm", metric="net_pnl", entry=rl["test_grid"]["entry"], filter=f"in-sample {start}..{end}",
        n_perm=rl["permutations"], seed=rl["seed"])
    ledger.append(PG / "ledger.csv", led.loc[:, ~led.columns.duplicated()], run_id, rl, rules_mod.git_head())
    summary = (f"confirmation {run_id} of scan {run_dir.name}: {int((verdicts.verdict == 'CONFIRMED').sum())} confirmed, "
               f"{int((verdicts.verdict == 'not confirmed').sum())} not confirmed, "
               f"{int(verdicts.verdict.str.startswith('untestable').sum())} untestable; "
               f"{conf['multiple_testing']} across {len(verdicts)} candidates at one-sided p <= {conf['p_max_one_sided']}.\n")
    (out / "summary.txt").write_text(summary)
    print(summary + f"Details in {out / 'confirmation.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
