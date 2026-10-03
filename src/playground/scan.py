"""The scan: every (category or family) x earnings subset x metric x bucket x horizon x OTM level in the grid
of selection_rules.json, each event set compared with its own matched null days.

Per test: matched permutation p-value, BH q-value across the configured family, net-of-cost and doubled-cost
effects, trimmed mean, leave-one-ticker-out, leave-one-quarter-out and plateau (neighbouring horizons and OTM
levels). Every test goes to the ledger. Outputs (data/playground/scan/<run_id>/): atlas.csv (all tests),
candidates.csv (what the committed rules select), summary.txt.

Refuses to run unless selection_rules.json is complete and committed.

Usage:  python src/playground/scan.py [--rebuild]
"""
import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "playground"

import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402

from playground import ROOT, ledger, rules as rules_mod, stats   # noqa: E402

OUT = ROOT / "data" / "playground"
HORIZON_ORDER = [1, 2, 3, 5, 10, 21, 42, 63, "exp"]
MEASURES = {"realized_vs_implied": "ratio", "iv_change": "iv_change", "skew_change": "skew_change", "drift": "realized"}
OTM_FREE = {"long_call", "stock", "realized_vs_implied", "iv_change", "drift"}   # identical across OTM levels


def norm_h(h):
    return h if h == "exp" else int(h)


def memberships(table: pd.DataFrame, taxonomy: pd.DataFrame, levels: list[str], earnings_subsets: list[str]):
    """{(level, group, subset): set_ids} from each event's tags and the taxonomy's own families."""
    ev = table[table.kind == "event"].drop_duplicates("set_id")[["set_id", "tags", "earnings_cofiled"]]
    tax = taxonomy.set_index("tertiary_category")
    out: dict[tuple, set] = {}
    for r in ev.itertuples(index=False):
        tags = [t for t in r.tags.split("|") if t in tax.index]
        names = {"tertiary": set(tags),
                 "secondary": {tax.at[t, "secondary_category"] for t in tags},
                 "primary": {tax.at[t, "primary_category"] for t in tags}}
        for level in levels:
            for g in names[level]:
                for sub in earnings_subsets:
                    if sub == "all" or (sub == "cofiled") == bool(r.earnings_cofiled):
                        out.setdefault((level, g, sub), set()).add(r.set_id)
    return out


def slice_arrays(table: pd.DataFrame, entry: str, cols: list[str]) -> dict:
    """{(bucket, horizon, otm): (event frame by set_id, null values by set_id: col -> n x 2 array)}"""
    t = table[table.entry == entry]
    out = {}
    for (b, h, o), g in t.groupby(["bucket", "horizon", "otm"], sort=False):
        ev = g[g.kind == "event"].drop_duplicates("set_id").set_index("set_id")
        nu = g[g.kind == "null"]
        piv = {c: nu.pivot_table(index="set_id", columns="round", values=c, aggfunc="first") for c in cols}
        out[(b, norm_h(h), float(o))] = (ev, piv)
    return out


def run_test(ev: pd.DataFrame, piv: dict, sids, col: str, col2x: str | None, rng, rl: dict) -> dict | None:
    sids = [s for s in sids if s in ev.index]
    if not sids:
        return None
    E = ev.loc[sids, col].to_numpy(float)
    N = piv[col].reindex(sids).reindex(columns=[1, 2]).to_numpy(float)
    obs, p, n, p_z = stats.permutation_test(E, N, rl["permutations"], rng)
    if n < rl["min_sets"]:
        return {"n_sets": n, "skipped": True}
    d, keep = stats.matched_diffs(E, N)
    kept = np.array(sids)[keep]
    tick = np.array([s.split("|")[0] for s in kept])
    quarters = ev.loc[kept, "quarter"].to_numpy()
    loto = stats.leave_one_out_sign(d, tick)
    loqo = stats.leave_one_out_sign(d, quarters)
    row = {"n_sets": n, "n_tickers": len(set(tick)), "event_mean": float(np.nanmean(E[keep])),
           "null_mean": float(np.nanmean(N[keep])), "diff": obs, "p_perm": p, "p_z": p_z,
           "trimmed_diff": stats.trimmed_mean(d, rl["robustness"]["trimmed_mean_frac"]),
           "loto_holds": loto[0], "loto_weakest": loto[1], "loqo_holds": loqo[0], "loqo_weakest": loqo[1],
           "low_sample": n < rl["low_sample_n"], "skipped": False}
    if col2x:
        E2, N2 = ev.loc[sids, col2x].to_numpy(float), piv[col2x].reindex(sids).reindex(columns=[1, 2]).to_numpy(float)
        d2, k2 = stats.matched_diffs(E2, N2)
        row.update(event_mean_2x=float(np.nanmean(E2[k2])) if k2.any() else np.nan, diff_2x=float(d2.mean()) if len(d2) else np.nan)
    return row


def add_plateau(atlas: pd.DataFrame) -> pd.DataFrame:
    """Neighbours: the same test at the adjacent horizons, and at the adjacent OTM levels where OTM matters."""
    key = ["level", "group", "subset", "metric", "strategy", "bucket"]
    idx = {tuple(r): (d) for r, d in zip(atlas[key + ["horizon", "otm"]].itertuples(index=False, name=None), atlas["diff"])}
    otms = sorted(atlas["otm"].dropna().unique())
    n_list, same_list = [], []
    for r in atlas.itertuples(index=False):
        base = tuple(getattr(r, k) for k in key)
        nb = []
        hi = HORIZON_ORDER.index(r.horizon)
        for j in (hi - 1, hi + 1):
            if 0 <= j < len(HORIZON_ORDER):
                nb.append(idx.get(base + (HORIZON_ORDER[j], r.otm)))
        if pd.notna(r.otm) and r.otm in otms:
            oi = otms.index(r.otm)
            for j in (oi - 1, oi + 1):
                if 0 <= j < len(otms):
                    nb.append(idx.get(base + (r.horizon, otms[j])))
        nb = [x for x in nb if x is not None and pd.notna(x)]
        n_list.append(len(nb))
        same_list.append(sum(np.sign(x) == np.sign(r.diff) for x in nb))
    return atlas.assign(plateau_n=n_list, plateau_same=same_list)


def in_headline(atlas: pd.DataFrame, spec: dict) -> pd.Series:
    """Tests inside the pre-declared headline grid (used when fdr_family is "headline")."""
    m = pd.Series(True, index=atlas.index)
    for col in ("bucket", "horizon", "metric", "level", "subset"):
        if col in spec:
            m &= atlas[col].isin([norm_h(x) if col == "horizon" else x for x in spec[col]])
    if "otm" in spec:
        m &= atlas["otm"].isna() | atlas["otm"].isin([float(x) for x in spec["otm"]])
    return m


def fdr(atlas: pd.DataFrame, rl: dict) -> pd.Series:
    """BH q-values within the configured family; tests outside the headline family get none (exploratory)."""
    family, pcol = rl["fdr_family"], rl["p_value"]
    q = pd.Series(np.nan, index=atlas.index)
    scope = in_headline(atlas, rl["headline"]) if family == "headline" else pd.Series(True, index=atlas.index)
    keys = {"all": "all", "headline": "all", "per_level": atlas["level"],
            "per_metric": atlas["metric"]}[family]
    keys = pd.Series(keys, index=atlas.index)
    for _, ix in atlas[scope].groupby(keys[scope]).groups.items():
        q.loc[ix] = stats.benjamini_hochberg(atlas.loc[ix, pcol].to_numpy())
    return q


def select(atlas: pd.DataFrame, rl: dict) -> pd.DataFrame:
    """Apply the committed selection rules to the atlas: trade candidates only (net P&L tests)."""
    s, rb = rl["selection"], rl["robustness"]
    c = atlas[(atlas.metric == "net_pnl") & (atlas.q_bh <= rl["fdr_q"])]
    if s["direction"] == "event_better_and_net_positive":
        c = c[(c["diff"] > 0) & (c["event_mean"] > 0)]
    elif s["direction"] == "event_better":
        c = c[c["diff"] > 0]
    if rl["doubled_cost_must_hold"]:
        c = c[c["diff_2x"] > 0]
        if s["direction"] == "event_better_and_net_positive":
            c = c[c["event_mean_2x"] > 0]
    if rb["leave_one_ticker_out"]:
        c = c[c.loto_holds]
    if rb["leave_one_quarter_out"]:
        c = c[c.loqo_holds]
    if rb["trimmed_mean_same_sign"]:
        c = c[np.sign(c.trimmed_diff) == np.sign(c["diff"])]
    c = c[c.plateau_same >= rb["plateau_min_same_sign_neighbours"]]
    if not s["allow_low_sample"]:
        c = c[~c.low_sample]
    by, asc = {"q_then_effect": (["q_bh", "abs_diff"], [True, False]),
               "effect_then_q": (["abs_diff", "q_bh"], [False, True])}[s["ranking"]]
    c = c.assign(abs_diff=c["diff"].abs()).sort_values(by, ascending=asc)
    return c.head(int(rl["max_candidates"])).drop(columns="abs_diff")


def run_scan(table: pd.DataFrame, taxonomy: pd.DataFrame, rl: dict, out_dir: Path, ledger_path: Path,
             head: str = "") -> dict:
    """Run every test in the grid; write atlas, candidates, summary and ledger rows. Returns paths and counts."""
    grid = rl["test_grid"]
    rng = np.random.default_rng(int(rl["seed"]))
    strategies = grid["strategies"]
    cols = [f"net_{s}" for s in strategies] + [f"net2x_{s}" for s in strategies] + [MEASURES[m] for m in grid["measures"]]
    members = memberships(table, taxonomy, grid["levels"], grid["earnings_subsets"])
    slices = slice_arrays(table, grid["entry"], cols)
    horizons = [norm_h(h) for h in grid["horizons"]]
    rows, skipped, t0 = [], 0, time.time()
    for (b, h, o), (ev, piv) in slices.items():
        if b not in grid["buckets"] or h not in horizons or o not in [float(x) for x in grid["otm"]]:
            continue
        first_otm = o == min(float(x) for x in grid["otm"])
        metrics = [("net_pnl", s, f"net_{s}", f"net2x_{s}") for s in strategies if first_otm or s not in OTM_FREE]
        metrics += [(m, "", MEASURES[m], None) for m in grid["measures"] if first_otm or m not in OTM_FREE]
        for (level, group, sub), sids in members.items():
            for metric, strat, col, col2x in metrics:
                r = run_test(ev, piv, sids, col, col2x, rng, rl)
                if r is None or r.get("skipped"):
                    skipped += r is not None
                    continue
                rows.append({"kind": "scan", "level": level, "group": group, "subset": sub, "filter": "",
                             "metric": metric, "strategy": strat, "bucket": b, "horizon": h,
                             "otm": np.nan if (strat or metric) in OTM_FREE else o, "entry": grid["entry"],
                             **{k: v for k, v in r.items() if k != "skipped"}})
    atlas = pd.DataFrame(rows)
    if atlas.empty:
        raise RuntimeError("no test had enough matched sets; check min_sets and the measurement table")
    atlas["q_bh"] = fdr(atlas, rl)
    atlas = add_plateau(atlas)
    atlas = atlas.assign(n_perm=rl["permutations"], seed=rl["seed"])
    cands = select(atlas, rl)

    run_id = ledger.new_run_id()
    out = out_dir / run_id
    out.mkdir(parents=True, exist_ok=True)
    atlas.to_csv(out / "atlas.csv", index=False)
    cands.to_csv(out / "candidates.csv", index=False)
    ledger.append(ledger_path, atlas, run_id, rl, head)
    total = ledger.variant_count(ledger_path)
    summary = (f"run {run_id}: {len(atlas):,} tests evaluated ({skipped:,} skipped below min_sets={rl['min_sets']}) "
               f"in {time.time() - t0:.0f}s; {int((atlas.q_bh <= rl['fdr_q']).sum()):,} with q <= {rl['fdr_q']} "
               f"({rl['fdr_family']} family); {len(cands)} candidates after the committed rules.\n"
               f"Ledger: {total:,} tests logged in total (disclose this variant count). Rules commit {rl.get('_commit')}.\n")
    (out / "summary.txt").write_text(summary)
    return {"run_id": run_id, "out": out, "tests": len(atlas), "candidates": len(cands), "summary": summary}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="rebuild the measurement table from the cache")
    ap.add_argument("--allow-fetch", action="store_true", help="allow API calls on cache misses (default: offline)")
    args = ap.parse_args()
    rl = rules_mod.load()                                    # refuses unless complete and committed
    import os
    os.chdir(ROOT)
    import nb
    from playground import measure
    NB = nb.load()
    measure.go_offline(NB) if not args.allow_fetch else None
    taxonomy = pd.DataFrame(NB["api_get_all"]("/stocks/taxonomies/vX/disclosures", {"limit": 1000}))
    taxonomy.to_csv(OUT / "taxonomy.csv", index=False)       # the explorer reads it
    mfile = OUT / "measurements.pkl"
    if args.rebuild or not mfile.exists():
        events = pd.read_csv(OUT / "events_discovery.csv", parse_dates=["filing_date", "accepted_at", "t_0", "t_pre"])
        nulls = pd.read_csv(OUT / "nulls_discovery.csv", parse_dates=["filing_date", "t_0", "t_pre"])
        print(f"building measurements for {len(events):,} events and {len(nulls):,} null days (offline)", flush=True)
        table, failures = measure.build(NB, events, nulls, rl["earnings_tags"], allow_fetch=args.allow_fetch)
        table.to_pickle(mfile)
        failures.to_csv(OUT / "measurement_failures.csv", index=False)
        print(f"{len(table):,} measurement rows; {len(failures):,} rows could not be priced (see measurement_failures.csv)")
    table = pd.read_pickle(mfile)
    res = run_scan(table, taxonomy, rl, OUT / "scan", OUT / "ledger.csv", rules_mod.git_head())
    print(res["summary"])
    print(f"Atlas and candidates in {res['out']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
