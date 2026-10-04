"""Synthetic tests for src/oldnews/classify.py and src/oldnews/tests.py (no real data, no network).
Run from the repo root:  .venv/Scripts/python.exe tests/test_oldnews_stats.py"""
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402

from oldnews import classify as cl        # noqa: E402
from oldnews import tests as T            # noqa: E402
from playground import ledger, stats      # noqa: E402

FAILS = []
NP, NB = 2000, 2000            # fewer permutations / resamples than the real run, for speed


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{': ' + detail if detail else ''}")
    if not ok:
        FAILS.append(name)


def raises(f, exc=PermissionError):
    try:
        f()
        return False
    except exc:
        return True


def with_date(df, col, date, n=1):
    d = df.head(n).copy()
    d[col] = pd.Timestamp(date)
    return d



SESS = pd.bdate_range("2022-01-03", "2023-12-29")
PEOPLE = ["ceo_departure", "cfo_departure", "executive_officer_departure", "director_departure",
          "ceo_appointment", "cfo_appointment", "executive_officer_appointment", "director_appointment"]
PLACEBO = ["annual_meeting_results", "shareholder_proposal_outcome", "dividend_declaration"]


def synth(n_people=300, n_placebo=150, effect=-0.3, placebo_effect=0.0, seed=0, ref=None):
    """Events, nulls, gap and outcome tables with the exact interface columns. Old-news people events get
    `effect` added to y (placebo old events get `placebo_effect`); every row shares its ticker's level."""
    rng = np.random.default_rng(seed)
    ev, nl, gp, used = [], [], [], set()

    def gap_row(rid, scale):
        ok = rng.random() > 0.05
        div = np.nan if rng.random() < 0.45 else rng.normal(0.005 * scale, 0.02)
        vol = np.nan if rng.random() < 0.3 else (0.0 if rng.random() < 0.02 else rng.lognormal(0, 0.8) * scale)
        why = [] if ok else ["no_price"]
        why += ["d_iv_gap: no 1m mark at both ends"] if np.isnan(div) else []
        bk = lambda: str(rng.choice(["1m", "2m", "3-6m"], p=[0.6, 0.3, 0.1]))               # noqa: E731
        return {"row_id": rid, "r_gap": rng.normal(0, 0.02), "r_mkt": rng.normal(0, 0.01), "sigma_d": 0.015,
                "gap_move": abs(rng.normal(0, scale)) if ok else np.nan, "iv_gap_start": 0.25, "iv_tpre": 0.26,
                "d_iv_gap": div, "vol_gap_ratio": vol, "usable": ok, "reason": "; ".join(why),
                "spot_bucket": f"{bk()}|{bk()}", "n_eff": float(rng.integers(1, 6)),
                "stale_sessions": float(rng.choice([0, 0, 0, 0, 1, 2])), "n_baseline": int(rng.integers(0, 6))}

    for i in range(n_people + n_placebo):
        grp = "people" if i < n_people else "placebo"
        tk = f"T{rng.integers(25):02d}"
        k = int(rng.integers(80, len(SESS) - 70))
        n_gap = int(rng.integers(1, 6))
        if grp == "people":
            tags = list(rng.choice(PEOPLE, size=int(rng.integers(1, 3)), replace=False))
            if rng.random() < 0.1:
                tags.append("business_update")
        else:
            tags = [str(rng.choice(PLACEBO))] + (["share_repurchase_program"] if rng.random() < 0.1 else [])
        cues = (rng.random(3) < [0.3, 0.3, 0.15]).astype(int)
        acc = f"0000-{i:05d}"
        rid = f"event|{acc}"
        late = bool(rng.random() < 0.85)
        ev.append({"row_id": rid, "kind": "event", "accession_number": acc, "ticker": tk,
                   "filing_date": SESS[k - 1], "accepted_at": SESS[k - 1] + pd.Timedelta(hours=17),
                   "event_date": SESS[k - 1 - n_gap] + pd.Timedelta(days=1), "gap_start": SESS[k - 1 - n_gap],
                   "t_pre": SESS[k - 1], "t_0": SESS[k], "n_gap": n_gap, "lag_bd": n_gap if late else 0, "late": late,
                   "group": grp, "role": "CEO" if grp == "people" else "placebo", "tags": "|".join(tags),
                   "earnings_excluded": bool(rng.random() < 0.1), "cue_dated_prior": cues[0],
                   "cue_prior_wording": cues[1], "cue_related_filing": cues[2], "T": int(cues.sum()),
                   "text_source": "supporting_text"})
        gp.append(gap_row(rid, 1.3))
        for rnd in (1, 2):
            k2 = int(np.clip(k + rng.choice([-1, 1]) * rng.integers(6, 60), n_gap + 2, len(SESS) - 70))
            while (nid := f"null|{tk}|{SESS[k2 - 1].date()}|{SESS[k2].date()}|{rnd}") in used:
                k2 += 1
            used.add(nid)
            nl.append({"row_id": nid, "kind": "null", "event_row_id": rid, "ticker": tk, "t_pre": SESS[k2 - 1],
                       "t_0": SESS[k2], "gap_start": SESS[k2 - 1 - n_gap], "n_gap": n_gap, "round": rnd})
            gp.append(gap_row(nid, 1.0))
    events, nulls, gap = pd.DataFrame(ev), pd.DataFrame(nl), pd.DataFrame(gp)
    c = cl.classify(events, gap, cl.reference(nulls, gap) if ref is None else ref)
    shift = {r: (effect if g == "people" else placebo_effect) for r, g in zip(c.row_id[c.old], c.group[c.old])}
    rows = pd.concat([events[["row_id", "ticker", "t_0"]], nulls[["row_id", "ticker", "t_0"]]], ignore_index=True)
    level = {f"T{j:02d}": rng.normal(-0.1, 0.2) for j in range(25)}
    grid = pd.MultiIndex.from_product([rows.index, T.BUCKETS, T.HORIZONS, T.OTMS],
                                      names=["i", "bucket", "horizon", "otm"]).to_frame(index=False)
    grid = grid.join(rows, on="i").drop(columns="i")
    hz = grid["horizon"].map(lambda h: 21 if h == "expiry" else int(h))
    grid["entry_date"] = grid["t_0"]
    grid["exit_date"] = grid["t_0"] + pd.to_timedelta(hz * 7 // 5, unit="D")
    grid["iv0"] = 0.25
    grid["y"] = grid["ticker"].map(level) + rng.normal(0, 0.25, len(grid)) + grid["row_id"].map(shift).fillna(0.0)
    grid["rv"] = grid["iv0"] * np.exp(grid["y"])
    grid["put_strike"], grid["put_premium"], grid["put_volume_entry"] = 97.0, 1.2, 50
    grid["csp_gross"] = rng.normal(0.004, 0.01, len(grid))
    grid["csp_net"] = grid["csp_gross"] - 0.001
    grid["csp_net2x"] = grid["csp_gross"] - 0.002
    grid["usable"] = rng.random(len(grid)) > 0.03
    return events, nulls, gap, grid.drop(columns=["ticker", "t_0"])


def tmp_log(d: Path) -> T.Log:
    return T.Log(d / "ledger.csv", "test-run", "synthetic-head")


TMP = Path(tempfile.mkdtemp(prefix="oldnews_stats_"))
events, nulls, gap, outcome = synth()
REAL_FROZEN = ROOT / "src" / "oldnews" / "zref_frozen.json"       # the committed file: tests must never touch it
REAL_BEFORE = REAL_FROZEN.read_bytes() if REAL_FROZEN.exists() else None
FDIR = TMP / "frozen"
FDIR.mkdir()
FROZEN_FILE = FDIR / "zref_frozen.json"
for name, df in zip(("events", "nulls", "gap"), (events, nulls, gap)):      # no outcome file: freeze may not need one
    df.to_csv(FDIR / f"{name}_discovery.csv", index=False)
DOC = cl.freeze("discovery", FDIR, FROZEN_FILE)
cl.FROZEN_PATH = FROZEN_FILE                                      # classify and run read this file
FROZEN_BYTES = FROZEN_FILE.read_bytes()
FROZEN = cl.load_frozen()

# ---- constants fixed by the test plan ----------------------------------------------------------------------
check("seed, permutations, resamples, horizons fixed by the plan",
      T.SEED == 20261003 and T.N_PERM == 10_000 and T.N_BOOT == 10_000
      and T.HORIZONS == ("1", "2", "3", "5", "10", "21", "42", "63", "expiry"))
check("five weight sets and three cutoffs as committed",
      cl.WEIGHT_SETS == {"equal": (1, 1), "math_only": (1, 0), "words_only": (0, 1), "math_heavy": (1, 0.5),
                         "words_heavy": (0.5, 1)} and cl.CUTOFFS == (0.5, 1.0, 1.5)
      and T.PRIMARY == {"weights": "equal", "cutoff": 1.0, "bucket": "1m", "horizon": "10", "otm": 3,
                        "category": "all8"})

# ---- classification ----------------------------------------------------------------------------------------
c = cl.classify(events, gap)
ref = c.attrs["reference"]
G = cl.gap_table(gap)
null_ok = G.loc[nulls.row_id.unique()]
null_ok = null_ok[null_ok.gap_usable]
doc = json.loads(FROZEN_BYTES)
sha = lambda ids: hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()               # noqa: E731
check("frozen file: first entries say the constants come from discovery only, inputs only",
      list(doc)[:3] == ["_note", "source", "uses_outcomes"] and doc["source"] == "discovery" and doc["uses_outcomes"] is False
      and "DISCOVERY window only" in doc["_note"] and "never outcomes" in doc["_note"])
check("the three inputs are gap_move, d_iv_gap and log(volume ratio)",
      cl.INPUTS == ("gap_move", "d_iv_gap", "log_vol_gap_ratio") and list(doc["constants"]) == list(cl.INPUTS)
      and "log(" in doc["inputs"]["log_vol_gap_ratio"])
check("frozen file: six constants = mean and sd (ddof=1) of each input over the usable ordinary days",
      all(np.isclose(doc["constants"][k]["mean"], null_ok[k].dropna().mean())
          and np.isclose(doc["constants"][k]["sd"], null_ok[k].dropna().std(ddof=1)) for k in cl.INPUTS)
      and all(set(v) == {"mean", "sd", "n", "row_ids_sha256"} for v in doc["constants"].values()),
      str({k: (round(v["mean"], 3), round(v["sd"], 3), v["n"]) for k, v in doc["constants"].items()}))
check("frozen file: n and sha256 per input (each over the days where that input exists), sample n, dates, sha256",
      all(doc["constants"][k]["n"] == null_ok[k].notna().sum() and doc["constants"][k]["row_ids_sha256"] == sha(null_ok.index[null_ok[k].notna()])
          for k in cl.INPUTS)
      and doc["constants"]["d_iv_gap"]["n"] < doc["constants"]["gap_move"]["n"] == len(null_ok)
      and doc["n_null_rows"] == len(null_ok)
      and doc["window_t_0"] == {"first": str(nulls.t_0.min().date()), "last": str(nulls.t_0.max().date())}
      and doc["null_row_ids_sha256"] == sha(null_ok.index))
raw_nulls = gap.set_index("row_id").loc[nulls.row_id.unique()]
check("ordinary days without a gap move are left out of every constant, even if they carry another input",
      (~raw_nulls.usable & raw_nulls.d_iv_gap.notna()).any()
      and not np.isclose(doc["constants"]["d_iv_gap"]["mean"], raw_nulls.d_iv_gap.dropna().mean()))
check("classify applies exactly the frozen constants",
      np.allclose(ref[["mean", "sd"]], FROZEN[["mean", "sd"]]) and np.allclose(ref.loc[list(doc["constants"]), "mean"],
                                                                                 [v["mean"] for v in doc["constants"].values()]))
z_null = (null_ok[list(cl.INPUTS)] - ref["mean"]) / ref["sd"]
check("on the freeze sample itself each z-score has mean 0, sd 1 (over the rows where it exists)",
      np.allclose(z_null.mean(), 0) and np.allclose(z_null.std(ddof=1), 1))
check("freeze refuses any label but discovery, and writes nothing",
      raises(lambda: cl.freeze("insample", FDIR, TMP / "no.json"), ValueError) and not (TMP / "no.json").exists())
check("constants are frozen once: a second freeze refuses and leaves the file as it was",
      raises(lambda: cl.freeze("discovery", FDIR, FROZEN_FILE), FileExistsError) and FROZEN_FILE.read_bytes() == FROZEN_BYTES)
alt = TMP / "alt.json"
cl.freeze("discovery", FDIR, alt)
check("freeze is deterministic (same inputs, same bytes)", alt.read_bytes() == FROZEN_BYTES)
check("freeze reads no outcome file", not list(FDIR.glob("outcome*")))
s = c[c.scored]
zc = [f"z_{k}" for k in cl.INPUTS]
Z = np.column_stack([(s[k] - ref.at[k, "mean"]) / ref.at[k, "sd"] for k in cl.INPUTS])
check("M = mean of the AVAILABLE standardised inputs (gap move required, the others may be missing)",
      np.allclose(s.M, np.nanmean(Z, axis=1)) and np.isfinite(Z[:, 0]).all() and np.isnan(Z[:, 1:]).any()
      and (s.n_inputs == np.isfinite(Z).sum(axis=1)).all() and set(s.n_inputs) == {1, 2, 3},
      str(s.n_inputs.value_counts().sort_index().to_dict()))
ok = all(np.allclose(s[f"S_{w}"], a * s.M + b * s["T"]) for w, (a, b) in cl.WEIGHT_SETS.items())
ok &= all((s[cl.old_col(w, k)] == (s[f"S_{w}"] >= k)).all() for w in cl.WEIGHT_SETS for k in cl.CUTOFFS)
check("S = w_m M + w_t T and old = S >= cutoff for all 15 rules", ok)
check("cutoff is inclusive (words only: S = T = 1 is old)",
      (s.loc[s["T"] == 1, cl.old_col("words_only", 1.0)]).all() and not s.loc[s["T"] == 0, cl.old_col("words_only", 1.0)].any())
check("primary columns: old = old_equal_1, S = S_equal",
      (c.old == c["old_equal_1"]).all() and np.allclose(c.S.fillna(-9), c.S_equal.fillna(-9)))
u = c[~c.gap_usable]
check("no gap move -> unscored, never old, S missing, reason recorded",
      len(u) > 0 and (~u.scored).all() and (~u.old).all() and u.S.isna().all() and (u.news == "unscored").all()
      and u.unscored_reason.str.startswith("no gap_move: no_price").all() and (c[c.scored].unscored_reason == "").all(),
      f"{len(u)} rows")
check("old and surprise both present", 0 < c.old.sum() < c.scored.sum(), f"old {c.old.sum()} of {c.scored.sum()} scored")
# the volume input: log of the ratio; ratio 0 or missing means the input is missing, not that the event is dropped
ids3 = list(events.row_id[:6])
gx = gap.set_index("row_id").copy()
gx.loc[ids3, ["gap_move", "d_iv_gap", "vol_gap_ratio", "usable"]] = [
    [0.5, 0.01, 4.0, True], [0.5, 0.01, 0.0, True], [0.5, 0.01, np.nan, True],
    [0.5, np.nan, np.nan, True], [np.nan, 0.01, 4.0, True], [0.5, 0.01, 4.0, False]]
gx = gx.reset_index()
cx = cl.classify(events, gx).set_index("row_id").loc[ids3]
zg, zd = (0.5 - ref.at["gap_move", "mean"]) / ref.at["gap_move", "sd"], (0.01 - ref.at["d_iv_gap", "mean"]) / ref.at["d_iv_gap", "sd"]
zv = (np.log(4.0) - ref.at["log_vol_gap_ratio", "mean"]) / ref.at["log_vol_gap_ratio", "sd"]
check("volume enters as log(ratio): z = (log 4 - mean) / sd", np.isclose(cx.z_log_vol_gap_ratio.iloc[0], zv)
      and np.isclose(cx.log_vol_gap_ratio.iloc[0], np.log(4.0)))
check("a ratio of 0 (no log) or a missing input does not drop the event: M is the mean of the others",
      np.isclose(cx.M.iloc[0], (zg + zd + zv) / 3) and cx.n_inputs.iloc[0] == 3
      and np.isclose(cx.M.iloc[1], (zg + zd) / 2) and cx.n_inputs.iloc[1] == 2 and cx.vol_zero.iloc[1] and not cx.vol_zero.iloc[0]
      and np.isclose(cx.M.iloc[2], (zg + zd) / 2) and np.isclose(cx.M.iloc[3], zg) and cx.n_inputs.iloc[3] == 1
      and cx.scored.iloc[:4].all())
check("no gap move (or the file says unusable) -> not scored, even with the other inputs present",
      not cx.scored.iloc[4:].any() and cx.M.iloc[4:].isna().all() and cx.old.iloc[4:].eq(False).all()
      and cx.unscored_reason.iloc[4].startswith("no gap_move") and cx.unscored_reason.iloc[5].startswith("no gap_move"))
ev_t = events.copy()
ev_t.loc[7, "T"] = np.nan
ev_g = events.copy()
cg = cl.classify(ev_t, gap[gap.row_id != events.row_id[8]]).set_index("row_id")
check("unscored reasons: T missing, no gap row", cg.at[events.row_id[7], "unscored_reason"] == "T missing"
      and cg.at[events.row_id[8], "unscored_reason"] == "no gap row" and not cg.scored[[events.row_id[7], events.row_id[8]]].any())
c_sub = cl.classify(events.iloc[::3], gap)
cs_, c_ = c_sub.set_index("row_id"), c.set_index("row_id").loc[c_sub.row_id]
check("an event gets the same z-scores, M and labels whichever other events it is classified with",
      np.allclose(cs_[[*zc, "M", "S"]].fillna(-9), c_[[*zc, "M", "S"]].fillna(-9)) and (cs_.old == c_.old).all())
check("classification never reads outcomes or ordinary days (signature: events, gap, ref)",
      list(cl.classify.__code__.co_varnames[:3]) == ["events", "gap", "ref"])

# ---- date guards, label by label (same rules as pipeline.WINDOWS and trade.check_dates) -----------------------
for lab in ("discovery", "dryrun"):
    check(f"{lab}: refuses t_pre, t_0 or gap_start on or after 2024-01-01",
          all(raises(lambda c=c: cl.guard_dates(lab, {"e": with_date(events, c, "2024-01-01")}))
              for c in ("t_pre", "t_0", "gap_start")) and
          raises(lambda: cl.guard_dates(lab, {"n": with_date(nulls, "t_0", "2025-03-03")})))
    check(f"{lab}: allows 2023-12-29 entries and exits in 2024 and 2025",
          not raises(lambda: cl.guard_dates(lab, {"e": with_date(events, "t_0", "2023-12-29"),
                                                  "o": with_date(outcome, "exit_date", "2025-12-30")})))
    check(f"{lab}: refuses a usable exit on or after 2026-01-01; an unusable one is ignored",
          raises(lambda: cl.guard_dates(lab, {"o": with_date(outcome, "exit_date", "2026-01-02").assign(usable=True)}))
          and not raises(lambda: cl.guard_dates(lab, {"o": with_date(outcome, "exit_date", "2026-01-02").assign(usable=False)})))
check("insample: allows 2024-2025 entries, refuses entries on or after 2026-01-01",
      not raises(lambda: cl.guard_dates("insample", {"e": with_date(events, "t_0", "2025-12-31")}))
      and raises(lambda: cl.guard_dates("insample", {"e": with_date(events, "t_0", "2026-01-02")}))
      and raises(lambda: cl.guard_dates("insample", {"o": with_date(outcome, "entry_date", "2026-02-02")})))
check("insample: refuses a usable exit in 2026",
      raises(lambda: cl.guard_dates("insample", {"o": with_date(outcome, "exit_date", "2026-01-05").assign(usable=True)})))
check("holdout: allowed with 2026+ dates (the pipeline gates it behind RUN_HOLDOUT)",
      cl.guard_dates("holdout", {"e": with_date(events, "t_0", "2026-03-02"),
                                 "o": with_date(outcome, "exit_date", "2026-09-01")}) is None)
check("oos: refused unless RUN_OOS is True in the namespace given",
      raises(lambda: cl.guard_dates("oos", {}, {})) and raises(lambda: cl.guard_dates("oos", {}, {"RUN_OOS": False}))
      and raises(lambda: cl.guard_dates("oos", {}, {"RUN_OOS": 1})) and raises(lambda: cl.guard_dates("oos", {})))
check("oos: allowed once a human has set RUN_OOS = True",
      cl.guard_dates("oos", {"e": with_date(events, "t_0", "2026-03-02")}, {"RUN_OOS": True}) is None)
check("any other label gets discovery dates only",
      raises(lambda: cl.guard_dates("sealed", {"e": with_date(events, "t_0", "2024-01-02")}))
      and not raises(lambda: cl.guard_dates("sealed", {"e": events})))
check("window caps as in the pipeline", cl.entry_cap("discovery") == cl.entry_cap("dryrun") == pd.Timestamp("2024-01-01")
      and cl.entry_cap("insample") == pd.Timestamp("2026-01-01") and cl.entry_cap("holdout") is None
      and cl.entry_cap("oos") is None)

# ---- the window edge: drop and count, never fail ------------------------------------------------------------
edge_ev = events.copy()
edge_ev.loc[:2, "t_0"] = pd.Timestamp("2024-01-02")             # filed late on the last day: enters next session
gone_ids = set(edge_ev.row_id[:3])
edge_nl = nulls.copy()
own_2024 = edge_nl.index[~edge_nl.event_row_id.isin(gone_ids)][0]
edge_nl.loc[own_2024, "t_0"] = pd.Timestamp("2024-01-03")        # an ordinary day of a kept event reaches 2024
gone_nl_ids = set(edge_nl.row_id[edge_nl.event_row_id.isin(gone_ids)]) | {edge_nl.row_id[own_2024]}
kept, drops = cl.restrict("discovery", {"events": edge_ev, "nulls": edge_nl, "gap": gap, "outcome": outcome})
dc = dict(zip(drops.what, drops.n))
check("window edge: events with t_0 on or after the cap are dropped, with all their rows",
      len(kept["events"]) == len(events) - 3 and not set(kept["events"].row_id) & gone_ids
      and not set(kept["nulls"].row_id) & gone_nl_ids
      and not set(kept["gap"].row_id) & (gone_ids | gone_nl_ids) and not set(kept["outcome"].row_id) & (gone_ids | gone_nl_ids)
      and len(kept["gap"]) == len(gap) - len(gone_ids | gone_nl_ids), str(dc))
check("window edge: every drop is counted",
      list(dc.values())[0] == 3 and list(dc.values())[1] == len(gone_nl_ids) and list(dc.values())[2] == 3 + len(gone_nl_ids))
check("window edge: restricted tables pass the guard", not raises(lambda: cl.guard_dates("discovery", kept)))
kept_h, drops_h = cl.restrict("holdout", {"events": edge_ev, "nulls": edge_nl, "gap": gap})
check("gated labels drop nothing", len(kept_h["events"]) == len(edge_ev) and drops_h.empty)
edge_dir = TMP / "edgefreeze"
edge_dir.mkdir()
for name, df in zip(("events", "nulls", "gap"), (edge_ev, edge_nl, gap)):
    df.to_csv(edge_dir / f"{name}_discovery.csv", index=False)
edge_doc = cl.freeze("discovery", edge_dir, edge_dir / "z.json")
keep_ids = sorted(set(nulls.row_id) - gone_nl_ids)
keep_ok = cl.gap_table(gap).loc[keep_ids]
keep_ok = keep_ok[keep_ok.gap_usable]
check("freeze uses no ordinary day that falls on or after 2024-01-01 or belongs to a dropped event",
      edge_doc["n_null_rows"] == len(keep_ok) and edge_doc["null_row_ids_sha256"] ==
      hashlib.sha256("\n".join(sorted(keep_ok.index)).encode()).hexdigest()
      and edge_doc["window_t_0"]["last"] < "2024-01-01")

# ---- reading CSVs: "null" and "NA" are text, not missing -------------------------------------------------------
tricky = TMP / "tricky.csv"
tricky.write_text("row_id,kind,ticker,x,reason\nnull|NA|1,null,NA,1.5,\nevent|2,event,NA,,no_price\n", encoding="utf-8")
rt = cl.read_table(tricky)
check("read_table keeps kind='null' and ticker='NA'; empty cells are missing",
      list(rt.kind) == ["null", "event"] and list(rt.ticker) == ["NA", "NA"] and np.isnan(rt.x[1]) and rt.reason.isna()[0]
      and rt.reason[1] == "no_price" and list(rt.row_id) == ["null|NA|1", "event|2"])
check("no read_csv in classify.py or tests.py uses pandas' default NA parsing",
      all("keep_default_na=False" in l for f in ("classify.py", "tests.py")
          for l in (ROOT / "src" / "oldnews" / f).read_text(encoding="utf-8").splitlines() if "pd.read_csv(" in l))

# ---- permutation and bootstrap internals -------------------------------------------------------------------
d = np.array([-3.0, -2.0, 1.0, 2.0])
o = np.array([True, True, False, False])
obs, perm = T._label_perm(d, o, 20000, np.random.default_rng(1))
p1, p2 = T._p_values(obs, perm, "less")
check("label permutation matches exact enumeration (1/6 one-sided, 1/3 two-sided)",
      obs == -4 and abs(p1 - 1 / 6) < 0.01 and abs(p2 - 1 / 3) < 0.01, f"p1={p1:.4f}, p2={p2:.4f}")
obs, perm = T._set_perm(np.array([0.0]), np.array([[1.0, 2.0]]), 20000, np.random.default_rng(1))
p1, _ = T._p_values(obs, perm, "less")
check("set permutation matches enumeration (event 0 vs {1, 2}: obs -1.5, p 1/3)",
      np.isclose(obs, -1.5) and set(np.round(perm, 6)) == {-1.5, 0.0, 1.5} and abs(p1 - 1 / 3) < 0.01, f"p={p1:.4f}")
rng = np.random.default_rng(3)
pl = [T._p_values(*T._label_perm(rng.normal(0, 1, 40), rng.permutation(np.r_[np.ones(15), np.zeros(25)]).astype(bool),
                                 400, rng), "less")[0] for _ in range(400)]
check("label permutation calibrated under the null", 0.02 <= np.mean(np.array(pl) < 0.05) <= 0.09,
      f"{np.mean(np.array(pl) < 0.05):.3f} below 0.05")
ps = []
for _ in range(400):
    N = rng.normal(0, 1, (30, 6))
    N[rng.random((30, 6)) < 0.3] = np.nan
    N[:, 0] = rng.normal(0, 1, 30)
    ps.append(T._p_values(*T._set_perm(rng.normal(0, 1, 30), N, 400, rng), "less")[0])
check("set permutation calibrated under the null", 0.02 <= np.mean(np.array(ps) < 0.05) <= 0.09,
      f"{np.mean(np.array(ps) < 0.05):.3f} below 0.05")

# ---- matched outcome and H1b pool on hand-built tables ---------------------------------------------------
hc = pd.DataFrame({"row_id": ["e1", "e2", "e3"], "ticker": ["A", "B", "C"], "group": "people", "tags": "ceo_departure",
                   "late": True, "earnings_excluded": False, "scored": True, "gap_usable": True,
                   "t_0": pd.Timestamp("2022-03-01"), "gap_move": [2.0, 5.0, 0.1], "old": [True, True, False]})
for w in cl.WEIGHT_SETS:
    for k in cl.CUTOFFS:
        hc[cl.old_col(w, k)] = hc.old
hn = pd.DataFrame({"event_row_id": ["e1", "e1", "e2", "e2", "e3", "e3"], "row_id": ["n1", "n2", "n3", "n4", "n5", "n6"],
                   "round": [1, 2, 1, 2, 1, 2]})
hg = pd.DataFrame({"row_id": ["n1", "n2", "n3", "n4", "n5", "n6"], "gap_move": [1.0, 2.0, 3.0, np.nan, 0.5, 0.2],
                   "d_iv_gap": 0.0, "vol_gap_ratio": 1.0, "usable": [True, True, True, True, True, False], "reason": ""})
ho = pd.DataFrame({"row_id": ["e1", "e2", "e3", "n1", "n2", "n3", "n4", "n5", "n6"],
                   "y": [0.0, 1.0, 3.0, 10.0, 20.0, 30.0, np.nan, 7.0, 99.0],
                   "usable": True}).assign(bucket="1m", horizon=10, otm=3)
hi = T.Inputs(hc, hn, cl.gap_table(hg), T._norm_outcome(ho))
y = T.cell(hi, "1m", "10", 3)
Nm = T.matched_nulls(hi, hc.row_id, y)
dd, keep = stats.matched_diffs(hc.row_id.map(y).to_numpy(float), Nm)
check("matched outcome: event y minus mean of its usable matched nulls; NaN null ignored",
      np.allclose(dd, [0 - 15, 1 - 30, 3 - (7 + 99) / 2]) and keep.all(), str(dd))
r = T._h1b_row(hi, "all8", "equal", 1.0, "1m", "10", 3, 500, 500)
check("H1b pool: ordinary days with gap_move >= event's (inclusive), unusable gaps excluded; too-large dropped",
      r["n"] == 1 and np.isclose(r["mean_comp"], 25.0) and np.isclose(r["effect"], -25.0) and "1 old events dropped" in r["note"],
      f"n={r['n']}, comp={r['mean_comp']}, note={r['note']!r}")

# ---- the tests on synthetic data ---------------------------------------------------------------------------
inp = T.prepare(events, nulls, gap, outcome)
log = tmp_log(TMP)
h = T.h1(inp, log, NP, NB).iloc[0]
check("H1 detects an injected -0.3 effect", h.effect < 0 and h.p_one_sided < 0.01 and h.ci_lo < -0.3 < h.ci_hi,
      f"effect {h.effect:+.3f} [{h.ci_lo:+.3f}, {h.ci_hi:+.3f}], p={h.p_one_sided:.4f}, n={h.n} ({h.n_old}/{h.n_comp})")
check("H1 reproducible (fixed seed)", T.h1(inp, log, NP, NB).iloc[0].equals(h))
c2 = inp.classified
excl = c2.row_id[~(c2.late & ~c2.earnings_excluded & c2.scored & (c2.group == "people"))]
out2 = outcome.copy()
out2.loc[out2.row_id.isin(excl), "y"] = -50.0
h_ex = T.h1(T.prepare(events, nulls, gap, out2), log, NP, NB).iloc[0]
check("non-late, earnings and unscored filings never enter H1", np.isclose(h_ex.effect, h.effect) and h_ex.n == h.n)
cnt = T.counts(inp).set_index("step")
check("counts: final step equals H1 n, old + surprise add up",
      cnt.at["at least one usable matched ordinary day", "people"] == h.n
      and cnt.at["  of which old news", "people"] == h.n_old and cnt.at["  of which surprise news", "people"] == h.n_comp,
      str(cnt["people"].to_dict()))
check("counts list why events are not scored", any(s.startswith("  not scored: no gap_move: no_price") for s in cnt.index))
cov = T.coverage(inp).set_index("what")
cv = cov["people"]
cc = inp.classified
b_ = cc.late & ~cc.earnings_excluded & (cc.group == "people")
s_ = cc[b_ & cc.scored]
check("coverage: filings, scored and not-scored reasons add up",
      cv["late, earnings-excluded filings"] == b_.sum() and cv["scored: gap move exists"] == len(s_)
      and sum(v for k, v in cv.items() if k.startswith("not scored")) == (b_ & ~cc.scored).sum())
check("coverage: inputs used, missing inputs, zero volume",
      sum(cv[f"scored with {k} of 3 inputs"] for k in (1, 2, 3)) == len(s_)
      and cv["  implied-vol change missing (no 1m mark at both ends)"] == s_.d_iv_gap.isna().sum()
      and cv["  volume input missing"] == s_.log_vol_gap_ratio.isna().sum()
      and cv["    of which volume ratio = 0 (log undefined)"] == s_.vol_zero.sum() > 0)
check("coverage: spot buckets, stale marks and baseline sessions each add up to the scored events",
      all(sum(cv[f"spot at {w} from the {k} pair"] for k in T.BUCKETS) == len(s_) for w in ("gap start", "t_pre"))
      and cv["gap-start mark fresh (0 sessions old)"] + cv["gap-start mark stale (1 to 3 sessions old)"] == len(s_)
      and sum(v for k, v in cv.items() if k.startswith("volume baseline")) == len(s_)
      and cv["median sessions in the gap-move formula (n_eff)"] >= 1, str(cv.to_dict())[:120])
p = T.placebo(inp, log, NP, NB).iloc[0]
check("placebo with no injected effect is not significant (two-sided)", p.p_two_sided > 0.05 and p.group == "placebo",
      f"effect {p.effect:+.3f}, p2={p.p_two_sided:.3f}, n={p.n}")
check("placebo's own p is two-sided, H1's is one-sided", p.p == p.p_two_sided and h.p == h.p_one_sided)
b = T.h1b(inp, log, NP, NB).iloc[0]
check("H1b detects the injected effect against the gap-move pool", b.effect < 0 and b.p_one_sided < 0.01,
      f"effect {b.effect:+.3f}, p={b.p_one_sided:.4f}, n={b.n}, median pool {b.n_comp}")
e0 = synth(effect=0.0, seed=1, ref=FROZEN)
h0 = T.h1(T.prepare(*e0), log, NP, NB).iloc[0]
check("no injected effect: H1 not significant", h0.p_one_sided > 0.05, f"effect {h0.effect:+.3f}, p={h0.p_one_sided:.3f}")
ep = synth(effect=+0.3, seed=2, ref=FROZEN)
hp = T.h1(T.prepare(*ep), log, NP, NB).iloc[0]
check("wrong-sign effect: one-sided p is large", hp.effect > 0 and hp.p_one_sided > 0.5, f"p={hp.p_one_sided:.3f}")

prof = T.horizon_profile(inp, log, NP, NB)
ok = len(prof) == 27 and all(list(prof[prof.test == t].horizon) == list(T.HORIZONS) for t in ("H1", "H1b", "P"))
ok &= all(np.allclose(prof[prof.test == t].q_bh, stats.benjamini_hochberg(prof[prof.test == t].p.to_numpy()), equal_nan=True)
          for t in ("H1", "H1b", "P"))
check("horizon profile: 9 horizons x 3 tests, BH within each test", ok)
check("profile h=10 matches the primary H1", np.isclose(prof[(prof.test == "H1") & (prof.horizon == "10")].effect.iloc[0], h.effect))
sens = T.sensitivity(inp, log, NP, NB)
check("sensitivity: 13 one-at-a-time rows",
      sens.dimension.value_counts().to_dict() == {"primary": 1, "weights": 4, "cutoff": 2, "bucket": 2, "otm": 2, "category": 2},
      str(sens.dimension.value_counts().to_dict()))
check("sensitivity primary row equals H1", np.isclose(sens[sens.dimension == "primary"].effect.iloc[0], h.effect))
cat = sens.set_index("category")
check("departures-only is a subset, plus_extra adds filings",
      cat.at["departures", "n"] < h.n < cat.at["plus_extra", "n"], f"{cat.at['departures', 'n']} < {h.n} < {cat.at['plus_extra', 'n']}")
check("words-only rule gives the same sample, different labels",
      sens.set_index("weights").at["words_only", "n"] == h.n)

led = pd.read_csv(log.path)
check("every test is in the ledger (seed, permutations, head, run id)",
      len(led) == 7 + 27 + 13 and (led.seed == T.SEED).all() and (led.n_perm == NP).all()
      and (led.git_head == "synthetic-head").all() and set(led.kind) == {"oldnews_H1", "oldnews_H1b", "oldnews_P"},
      f"{len(led)} rows")

# ---- run(label) end to end ----------------------------------------------------------------------------------
dd_ = TMP / "run"
dd_.mkdir()
for name, df in zip(("events", "nulls", "gap", "outcome"), (events, nulls, gap, outcome)):
    df.to_csv(dd_ / f"{name}_discovery.csv", index=False)
t0 = time.time()
res = T.run("discovery", dd_, n_perm=NP, n_boot=NB, log=T.Log(dd_ / "ledger.csv", "run-1", "synthetic-head"))
el = time.time() - t0
files = {"counts.csv", "coverage.csv", "h1.csv", "h1b.csv", "placebo.csv", "profile.csv", "sensitivity.csv", "summary.md"}
check("run writes every table and the summary", files <= {f.name for f in (dd_ / "results_discovery").iterdir()}
      and (dd_ / "classified_discovery.csv").exists() and (dd_ / "zref_discovery.csv").exists(), f"{el:.1f}s")
summ = (dd_ / "results_discovery" / "summary.md").read_text(encoding="utf-8")
check("summary states counts, coverage, H1 effect, CI, p and n",
      all(k in summ for k in ("## Counts", "## Coverage of the gap inputs", "95% CI", "one-sided p", "n = ", "log of the ratio")))
check("run reproduces the direct H1", np.isclose(res["h1"].effect.iloc[0], h.effect))
res2 = T.run("discovery", dd_, n_perm=NP, n_boot=NB, log=T.Log(dd_ / "ledger.csv", "run-2", "synthetic-head"))
check("two runs give identical tables", all(res[k].equals(res2[k]) for k in res))
cls = pd.read_csv(dd_ / "classified_discovery.csv")
check("classified table has the trade agent's `old` column", "old" in cls.columns and cls.old.dtype == bool)
small = synth(n_people=25, n_placebo=10, seed=4, ref=FROZEN)
sd_ = TMP / "small"
sd_.mkdir()
for name, df in zip(("events", "nulls", "gap", "outcome"), small):
    df.to_csv(sd_ / f"{name}_discovery.csv", index=False)
T.run("discovery", sd_, n_perm=300, n_boot=300, log=T.Log(sd_ / "ledger.csv", "small", "synthetic-head"))
check("fewer than 30 events -> labelled descriptive",
      "DESCRIPTIVE" in (sd_ / "results_discovery" / "summary.md").read_text(encoding="utf-8"))
ev_bad = events.copy()
ev_bad.loc[3, "t_pre"] = "2024-01-03"                           # t_0 still inside the window: a corrupt row
ev_bad.to_csv(dd_ / "events_discovery.csv", index=False)
check("run hard-fails on a corrupt discovery row (t_pre in 2024, t_0 not)",
      raises(lambda: T.run("discovery", dd_, n_perm=10, n_boot=10, log=T.Log(dd_ / "l2.csv", "x", "y"))))
events.to_csv(dd_ / "events_discovery.csv", index=False)
out_bad = outcome.copy()
out_bad.loc[5, "entry_date"] = pd.Timestamp("2024-02-01")
out_bad.to_csv(dd_ / "outcome_discovery.csv", index=False)
check("run hard-fails on a kept row whose outcome entry is in 2024",
      raises(lambda: T.run("discovery", dd_, n_perm=10, n_boot=10, log=T.Log(dd_ / "l2.csv", "x", "y"))))
outcome.to_csv(dd_ / "outcome_discovery.csv", index=False)

# the window edge through run(): extra 2024 events are dropped and counted, H1 is unchanged
ed = TMP / "edge"
ed.mkdir()
extra = events.head(3).copy()
extra["row_id"] = [f"event|edge{i}" for i in range(3)]
extra["t_0"] = pd.Timestamp("2024-01-02")
extra_nl = nulls[nulls.event_row_id.isin(events.row_id[:3])].copy()
extra_nl["event_row_id"] = extra_nl.event_row_id.map(dict(zip(events.row_id[:3], extra.row_id)))
extra_nl["row_id"] = extra_nl.row_id + "|edge"
pd.concat([events, extra]).to_csv(ed / "events_discovery.csv", index=False)
pd.concat([nulls, extra_nl]).to_csv(ed / "nulls_discovery.csv", index=False)
gap.to_csv(ed / "gap_discovery.csv", index=False)
outcome.to_csv(ed / "outcome_discovery.csv", index=False)
res_e = T.run("discovery", ed, n_perm=NP, n_boot=NB, log=T.Log(ed / "ledger.csv", "edge", "synthetic-head"))
dr = pd.read_csv(ed / "results_discovery" / "dropped.csv")
check("run drops the window-edge events and says how many (dropped.csv, summary.md)",
      dr.n.iloc[0] == 3 and dr.n.iloc[1] == len(extra_nl)
      and "Dropped at the window edge" in (ed / "results_discovery" / "summary.md").read_text(encoding="utf-8"), str(dr.n.tolist()))
check("dropping them leaves H1 exactly as without them", res_e["h1"].iloc[0].equals(res["h1"].iloc[0]))

# a ticker called "NA" and kind "null" survive the files
na_ev = events.copy()
na_ev.loc[na_ev.ticker == "T05", "ticker"] = "NA"
nd = TMP / "na"
nd.mkdir()
na_ev.to_csv(nd / "events_discovery.csv", index=False)
nulls.assign(ticker=nulls.ticker.replace("T05", "NA")).to_csv(nd / "nulls_discovery.csv", index=False)
gap.to_csv(nd / "gap_discovery.csv", index=False)
outcome.to_csv(nd / "outcome_discovery.csv", index=False)
T.run("discovery", nd, n_perm=NP, n_boot=NB, log=T.Log(nd / "ledger.csv", "na", "synthetic-head"))
cna = cl.read_table(nd / "classified_discovery.csv")
check("ticker 'NA' is kept through run(): same labels and H1 as with 'T05'",
      (cna.ticker == "NA").sum() == (events.ticker == "T05").sum() and cna.ticker.notna().all()
      and (cna.set_index("row_id").old == cl.read_table(dd_ / "classified_discovery.csv").set_index("row_id").old).all()
      and np.isclose(pd.read_csv(nd / "results_discovery" / "h1.csv").effect[0], res["h1"].effect.iloc[0], rtol=1e-12))


# holdout and oos through run(), with every date shifted into 2026 and later
def shifted(label, years=4, ctx=None):
    d = TMP / label
    d.mkdir()
    sh = lambda df, cols: df.assign(**{c: pd.to_datetime(df[c]) + pd.DateOffset(years=years) for c in cols})  # noqa: E731
    sh(events, ["filing_date", "accepted_at", "event_date", "gap_start", "t_pre", "t_0"]).to_csv(d / f"events_{label}.csv", index=False)
    sh(nulls, ["gap_start", "t_pre", "t_0"]).to_csv(d / f"nulls_{label}.csv", index=False)
    (gap if ctx is None else ctx).to_csv(d / f"gap_{label}.csv", index=False)
    sh(outcome, ["entry_date", "exit_date"]).to_csv(d / f"outcome_{label}.csv", index=False)
    return d


hd = shifted("holdout")
res_h = T.run("holdout", hd, n_perm=NP, n_boot=NB, log=T.Log(hd / "ledger.csv", "h", "synthetic-head"))
check("holdout runs on 2026+ dates, nothing dropped; summary says the label is gated upstream",
      np.isclose(res_h["h1"].effect.iloc[0], res["h1"].effect.iloc[0]) and "Dropped at the window edge" not in
      (hd / "results_holdout" / "summary.md").read_text(encoding="utf-8")
      and "gated upstream" in (hd / "results_holdout" / "summary.md").read_text(encoding="utf-8"))
od = shifted("oos")
check("oos refuses to run (and writes nothing) without RUN_OOS",
      raises(lambda: T.run("oos", od, n_perm=10, n_boot=10)) and raises(lambda: cl.run("oos", od, {"RUN_OOS": False}))
      and not (od / "results_oos").exists() and not (od / "classified_oos.csv").exists())
res_o = T.run("oos", od, {"RUN_OOS": True}, n_perm=NP, n_boot=NB, log=T.Log(od / "ledger.csv", "o", "synthetic-head"))
check("oos runs once RUN_OOS is True in the namespace passed", np.isclose(res_o["h1"].effect.iloc[0], res["h1"].effect.iloc[0]))
for lab in ("discovery", "insample"):
    dd2 = TMP / f"deny_{lab}"
    dd2.mkdir()
    for f in ("events", "nulls", "gap", "outcome"):
        (dd2 / f"{f}_{lab}.csv").write_bytes((od / f"{f}_oos.csv").read_bytes())
    check(f"{lab} cannot be pointed at 2026+ data (everything is dropped or refused, nothing is computed)",
          raises(lambda: T.run(lab, dd2, n_perm=10, n_boot=10), (PermissionError, ValueError))
          and not (dd2 / f"results_{lab}").exists())

# ---- the frozen constants across windows -----------------------------------------------------------------------
c_disc = pd.read_csv(dd_ / "classified_discovery.csv").set_index("row_id")
zr = {lab: (TMP / lab / f"zref_{lab}.csv").read_text() for lab in ("holdout", "oos")}
zr["discovery"] = (dd_ / "zref_discovery.csv").read_text()
check("the constants applied are the same in discovery, holdout and oos", len(set(zr.values())) == 1)
gap_other = gap.copy()
is_null = gap_other.row_id.str.startswith("null|")
gap_other.loc[is_null, ["gap_move", "d_iv_gap", "vol_gap_ratio"]] *= 5       # ordinary days of the new window look different
ind = shifted("insample", 2, gap_other)
T.run("insample", ind, n_perm=NP, n_boot=NB, log=T.Log(ind / "ledger.csv", "i", "synthetic-head"))
c_ins = pd.read_csv(ind / "classified_insample.csv").set_index("row_id")
cols = [*zc, "M", "S"]
check("an event is standardised identically in discovery and insample, even when that window's ordinary days differ",
      np.allclose(c_disc[cols].fillna(-9), c_ins.loc[c_disc.index, cols].fillna(-9)) and (c_disc.old == c_ins.loc[c_disc.index].old).all()
      and (ind / "zref_insample.csv").read_text() == zr["discovery"])
cl_h = pd.read_csv(hd / "classified_holdout.csv").set_index("row_id")
check("an event is standardised identically in the holdout window",
      np.allclose(c_disc[cols].fillna(-9), cl_h.loc[c_disc.index, cols].fillna(-9)))
check("running other labels did not change the frozen constants", FROZEN_FILE.read_bytes() == FROZEN_BYTES)
missing = TMP / "nowhere" / "zref_frozen.json"
cl.FROZEN_PATH = missing
fresh = TMP / "fresh"
fresh.mkdir()
for f in ("events", "nulls", "gap", "outcome"):
    (fresh / f"{f}_discovery.csv").write_bytes((dd_ / f"{f}_discovery.csv").read_bytes())
check("classify.run refuses when the frozen file is missing, and does not create it",
      raises(lambda: cl.run("discovery", fresh), FileNotFoundError) and not missing.exists()
      and not (fresh / "classified_discovery.csv").exists())
check("tests.run refuses too, before reading or writing anything",
      raises(lambda: T.run("discovery", fresh, n_perm=10, n_boot=10), FileNotFoundError) and not missing.exists()
      and not (fresh / "results_discovery").exists() and not (fresh / "ledger.csv").exists())
check("classify() with no constants refuses too", raises(lambda: cl.classify(events, gap), FileNotFoundError))
bad = TMP / "bad.json"
bad.write_text(FROZEN_BYTES.decode().replace('"discovery"', '"insample"', 1), encoding="utf-8")
check("a constants file that does not say 'discovery' is refused", raises(lambda: cl.load_frozen(bad), ValueError))
bad.write_text(FROZEN_BYTES.decode().replace('"uses_outcomes": false', '"uses_outcomes": true'), encoding="utf-8")
check("a constants file that does not say it uses no outcomes is refused", raises(lambda: cl.load_frozen(bad), ValueError))
cl.FROZEN_PATH = FROZEN_FILE
check("no run wrote or changed the committed src/oldnews/zref_frozen.json",
      (REAL_FROZEN.read_bytes() if REAL_FROZEN.exists() else None) == REAL_BEFORE)

print(f"\nlast full run on {len(events)} synthetic events: {el:.1f}s at {NP} permutations")
print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
