"""Explorer: one quick query on the measurement table, compared with matched null days, logged to the ledger.

Every query is a test, so every query is ledgered (CLAUDE.md rule 13). Refuses to run until the selection
rules are committed, like the scan.

Example:
  python src/playground/explore.py --level tertiary --group cfo_appointment --metric net_covered_call \
      --bucket 3-6m --horizon 21 --otm 0.05 --where "timing == 'after_close'"
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
from playground.scan import MEASURES, memberships, norm_h, run_test, slice_arrays   # noqa: E402

OUT = ROOT / "data" / "playground"


def query(table: pd.DataFrame, taxonomy: pd.DataFrame, rl: dict, level: str, group: str, metric: str, bucket: str,
          horizon, otm: float, subset: str = "all", where: str = "", entry: str = "post") -> dict:
    """One matched-null test. `where` filters the events (entry-time features only), never the nulls."""
    col = MEASURES.get(metric, metric)
    col2x = col.replace("net_", "net2x_") if col.startswith("net_") else None
    members = memberships(table, taxonomy, [level], [subset]).get((level, group, subset), set())
    if where:                                   # features are per bucket, so filter within this bucket and entry
        ev_rows = table[(table.kind == "event") & (table.bucket == bucket) & (table.entry == entry)
                        & table.set_id.isin(members)].query(where)
        members = members & set(ev_rows.set_id)
    sl = slice_arrays(table[(table.bucket == bucket) & (table.otm == otm)], entry, [c for c in (col, col2x) if c])
    key = (bucket, norm_h(horizon), float(otm))
    if key not in sl:
        return {"n_sets": 0}
    rng = np.random.default_rng(int(rl["seed"]))
    r = run_test(*sl[key], members, col, col2x, rng, rl) or {"n_sets": 0}
    return {"kind": "explore", "level": level, "group": group, "subset": subset, "filter": where,
            "metric": metric, "strategy": "", "bucket": bucket, "horizon": norm_h(horizon), "otm": otm, "entry": entry,
            "n_perm": rl["permutations"], "seed": rl["seed"], **{k: v for k, v in r.items() if k != "skipped"}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="tertiary", choices=["tertiary", "secondary", "primary"])
    ap.add_argument("--group", required=True)
    ap.add_argument("--metric", required=True, help="net_<strategy>, net2x_<strategy>, or a measure: " + ", ".join(MEASURES))
    ap.add_argument("--bucket", default="3-6m")
    ap.add_argument("--horizon", default="21")
    ap.add_argument("--otm", type=float, default=0.05)
    ap.add_argument("--subset", default="all", choices=["all", "alone", "cofiled"])
    ap.add_argument("--where", default="", help="pandas query on event features, e.g. \"timing == 'after_close'\"")
    args = ap.parse_args()
    rl = rules_mod.load()
    table = pd.read_pickle(OUT / "measurements.pkl")
    taxonomy = pd.read_csv(OUT / "taxonomy.csv")
    res = query(table, taxonomy, rl, args.level, args.group, args.metric, args.bucket, args.horizon, args.otm,
                args.subset, args.where)
    if res.get("n_sets", 0) == 0 or "p_perm" not in res:
        print(f"Not enough matched sets for this query (n = {res.get('n_sets', 0)}); nothing logged.")
        return 1
    ledger.append(OUT / "ledger.csv", pd.DataFrame([res]), ledger.new_run_id(), rl, rules_mod.git_head())
    pct = lambda v: f"{v * 100:+.2f}%" if args.metric.startswith("net") or args.metric == "drift" else f"{v:+.4f}"  # noqa: E731
    print(f"{args.group} ({args.level}, {args.subset}{', ' + args.where if args.where else ''}) | {args.metric} | "
          f"{args.bucket} h={args.horizon} otm={args.otm}")
    print(f"  matched sets {res['n_sets']} ({res['n_tickers']} tickers){' LOW SAMPLE' if res['low_sample'] else ''}")
    print(f"  event {pct(res['event_mean'])}  null {pct(res['null_mean'])}  difference {pct(res['diff'])}  "
          f"permutation p = {res['p_perm']:.4f}")
    if "diff_2x" in res:
        print(f"  doubled cost: event {pct(res['event_mean_2x'])}, difference {pct(res['diff_2x'])}")
    print(f"  trimmed difference {pct(res['trimmed_diff'])}; sign holds leaving out any ticker: {res['loto_holds']}, "
          f"any quarter: {res['loqo_holds']}")
    print(f"  logged to the ledger ({ledger.variant_count(OUT / 'ledger.csv'):,} tests so far)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
