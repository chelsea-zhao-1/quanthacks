"""Synthetic tests for src/oldnews/classify.py and src/oldnews/tests.py (no real data, no network).
Run from the repo root:  .venv/Scripts/python.exe tests/test_oldnews_stats.py

Every synthetic table is dated inside the allowed window (2024-2025). The retired 2022-23 constants file is read
only as an explicit archive, and the committed constants files are never written."""
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
RETIRED_MSG = ("2022-2023 is outside the allowed 2024-2025 window and overlaps the sealed placeholder "
               "(2023-06-01..2023-08-31)")


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


def message(f, exc=PermissionError):
    """The text of the exception f raises (None if it does not raise exc)."""
    try:
        f()
    except exc as e:
        return str(e)
    return None


def with_date(df, col, date, n=1):
    d = df.head(n).copy()
    d[col] = pd.Timestamp(date)
    return d


SESS = pd.bdate_range("2024-01-02", "2025-12-31")
PEOPLE = ["ceo_departure", "cfo_departure", "executive_officer_departure", "director_departure",
          "ceo_appointment", "cfo_appointment", "executive_officer_appointment", "director_appointment"]
PLACEBO = ["annual_meeting_results", "shareholder_proposal_outcome", "dividend_declaration"]


def synth(n_people=300, n_placebo=150, effect=-0.3, placebo_effect=0.0, seed=0, ref=None):
    """Events, nulls, gap and outcome tables with the exact interface columns, dated 2024-25. Old-news people
    events get `effect` added to y (placebo old events get `placebo_effect`); every row shares its ticker's level."""
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


def write_tables(d: Path, label: str, tables, with_outcome=True):
    d.mkdir(parents=True, exist_ok=True)
    for name, df in zip(("events", "nulls", "gap", "outcome"), tables):
        if name != "outcome" or with_outcome:
            df.to_csv(d / f"{name}_{label}.csv", index=False)
    return d


TMP = Path(tempfile.mkdtemp(prefix="oldnews_stats_"))
events, nulls, gap, outcome = synth()
REAL_ARCHIVE = ROOT / "src" / "oldnews" / "zref_frozen.json"            # the retired 2022-23 file: archive only
REAL_ARCHIVE_BEFORE = REAL_ARCHIVE.read_bytes() if REAL_ARCHIVE.exists() else None
REAL_FROZEN_IN = ROOT / "src" / "oldnews" / "zref_frozen_insample.json"  # the committed 2024-25 file: never touched
REAL_IN_BEFORE = REAL_FROZEN_IN.read_bytes() if REAL_FROZEN_IN.exists() else None
FDIR = TMP / "frozen"
FROZEN_FILE = FDIR / "zref_frozen_insample.json"
cl.FROZEN_PATH_INSAMPLE = FROZEN_FILE                                    # everything below reads and writes this one
cl.FROZEN_PATH = FROZEN_FILE
assert cl.frozen_path() == FROZEN_FILE and FROZEN_FILE != REAL_FROZEN_IN
write_tables(FDIR, "insample", (events, nulls, gap), with_outcome=False)  # no outcome file: freeze may not need one
DOC = cl.freeze("insample", FDIR)                                         # default path: the (patched) insample file
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

# ---- the frozen 2024-25 constants and the classification ------------------------------------------------------
c = cl.classify(events, gap, FROZEN)
ref = c.attrs["reference"]
G = cl.gap_table(gap)
null_ok = G.loc[nulls.row_id.unique()]
null_ok = null_ok[null_ok.gap_usable]
doc = json.loads(FROZEN_BYTES)
sha = lambda ids: hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()               # noqa: E731
check("frozen file: first entries say the constants come from the 2024-25 window only, inputs only",
      list(doc)[:3] == ["_note", "source", "uses_outcomes"] and doc["source"] == "insample" and doc["uses_outcomes"] is False
      and "2024-25 window only" in doc["_note"] and "never outcomes" in doc["_note"] and DOC == doc)
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
      and "2024-01-01" <= doc["window_t_0"]["first"] and doc["window_t_0"]["last"] < "2026-01-01"
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
check("constants are frozen once: a second freeze refuses and leaves the file as it was",
      raises(lambda: cl.freeze("insample", FDIR), FileExistsError) and FROZEN_FILE.read_bytes() == FROZEN_BYTES)
alt = TMP / "alt.json"
cl.freeze("insample", FDIR, alt)
check("freeze is deterministic (same inputs, same bytes)", alt.read_bytes() == FROZEN_BYTES)
check("freeze reads no outcome file", not list(FDIR.glob("outcome*")))
check("freeze writes only the insample file: no label but insample can be frozen",
      message(lambda: cl.freeze("discovery", FDIR, TMP / "x.json")) == RETIRED_MSG
      and message(lambda: cl.freeze("dryrun", FDIR, TMP / "x.json")) == RETIRED_MSG
      and all(raises(lambda l=l: cl.freeze(l, FDIR, TMP / "x.json"), ValueError) for l in ("holdout", "oos", "sealed"))
      and not (TMP / "x.json").exists())
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
cx = cl.classify(events, gx, FROZEN).set_index("row_id").loc[ids3]
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
cg = cl.classify(ev_t, gap[gap.row_id != events.row_id[8]], FROZEN).set_index("row_id")
check("unscored reasons: T missing, no gap row", cg.at[events.row_id[7], "unscored_reason"] == "T missing"
      and cg.at[events.row_id[8], "unscored_reason"] == "no gap row" and not cg.scored[[events.row_id[7], events.row_id[8]]].any())
c_sub = cl.classify(events.iloc[::3], gap, FROZEN)
cs_, c_ = c_sub.set_index("row_id"), c.set_index("row_id").loc[c_sub.row_id]
check("an event gets the same z-scores, M and labels whichever other events it is classified with",
      np.allclose(cs_[[*zc, "M", "S"]].fillna(-9), c_[[*zc, "M", "S"]].fillna(-9)) and (cs_.old == c_.old).all())
check("classification never reads outcomes or ordinary days (signature: events, gap, ref)",
      list(cl.classify.__code__.co_varnames[:3]) == ["events", "gap", "ref"])
check("classify() needs its constants passed in: no default file", raises(lambda: cl.classify(events, gap), ValueError))

# ---- the window guard, label by label (.claude/ctx/06_window_guard.md; same rules as trade.check_dates) -------
NOWHERE = TMP / "nowhere"                       # a folder that does not exist: a refusal must come before any read
check("window constants: 2024-01-01 to 2026-01-01, sealed placeholder 2023-06-01..2023-08-31",
      cl.window("insample") == (pd.Timestamp("2024-01-01"), pd.Timestamp("2026-01-01"))
      and cl.window("holdout") is None and cl.window("oos") is None
      and cl.holdout_window({}) == (pd.Timestamp("2023-06-01"), pd.Timestamp("2023-08-31")) == cl.HOLDOUT_PLACEHOLDER
      and cl.holdout_window({"HOLDOUT_START": "2024-09-02", "HOLDOUT_END": "2024-09-30"})
      == (pd.Timestamp("2024-09-02"), pd.Timestamp("2024-09-30")))
for lab in ("discovery", "dryrun"):
    fs = {"guard_dates": lambda: cl.guard_dates(lab, {"e": events}), "check_label": lambda: cl.check_label(lab),
          "load_tables": lambda: cl.load_tables(lab, NOWHERE), "classify.run": lambda: cl.run(lab, NOWHERE),
          "tests.run": lambda: T.run(lab, NOWHERE, n_perm=10, n_boot=10), "default_zref_source": lambda: cl.default_zref_source(lab),
          "freeze": lambda: cl.freeze(lab, NOWHERE, TMP / "r.json")}
    check(f"{lab} is retired: every entry point refuses with the stated message, before reading anything",
          all(message(f) == RETIRED_MSG for f in fs.values()), str({k: message(f) == RETIRED_MSG for k, f in fs.items()}))
retired_dir = write_tables(TMP / "retired", "discovery", (events, nulls, gap, outcome))     # files exist: still refused
check("a retired label is refused even when its files exist, and nothing is written",
      message(lambda: cl.run("discovery", retired_dir)) == RETIRED_MSG
      and message(lambda: T.run("dryrun", retired_dir, n_perm=10, n_boot=10)) == RETIRED_MSG
      and not (retired_dir / "classified_discovery.csv").exists() and not (retired_dir / "results_discovery").exists()
      and not (retired_dir / "ledger.csv").exists())
check("unknown labels are refused (and name the allowed ones)",
      all("allowed: insample, holdout" in (message(lambda l=l: cl.check_label(l)) or "") for l in ("sealed", "confirmation", "", "Insample")))
dates_ok = {"e": with_date(events, "t_0", "2024-01-02"), "n": with_date(nulls, "t_0", "2025-12-31"),
            "o": with_date(outcome, "exit_date", "2025-12-31").assign(usable=True)}
check("insample: allows dates from 2024-01-01 up to 2025-12-31, entries and usable exits",
      cl.guard_dates("insample", dates_ok) == (pd.Timestamp("2024-01-01"), pd.Timestamp("2026-01-01")))
check("insample: refuses any entry-side date before 2024-01-01",
      all(raises(lambda c_=c_: cl.guard_dates("insample", {"e": with_date(events, c_, "2023-12-29")}))
          for c_ in ("filing_date", "gap_start", "t_pre", "t_0"))
      and raises(lambda: cl.guard_dates("insample", {"n": with_date(nulls, "t_0", "2023-12-29")}))
      and raises(lambda: cl.guard_dates("insample", {"o": with_date(outcome, "entry_date", "2023-12-29")})))
check("insample: refuses entries on or after 2026-01-01, and a usable exit on or after it; an unusable exit is ignored",
      all(raises(lambda c_=c_: cl.guard_dates("insample", {"e": with_date(events, c_, "2026-01-01")}))
          for c_ in ("t_pre", "t_0", "gap_start"))
      and raises(lambda: cl.guard_dates("insample", {"o": with_date(outcome, "exit_date", "2026-01-02").assign(usable=True)}))
      and not raises(lambda: cl.guard_dates("insample", {"o": with_date(outcome, "exit_date", "2026-01-02").assign(usable=False)})))
sealed = {"HOLDOUT_START": "2024-09-02", "HOLDOUT_END": "2024-09-30"}
check("insample: never overlaps the notebook's HOLDOUT_START..HOLDOUT_END (boundaries included)",
      all(raises(lambda d=d: cl.guard_dates("insample", {"e": with_date(events, "t_0", d)}, sealed))
          for d in ("2024-09-02", "2024-09-10", "2024-09-30"))
      and not any(raises(lambda d=d: cl.guard_dates("insample", {"e": with_date(events, "t_0", d)}, sealed))
                  for d in ("2024-08-30", "2024-10-01"))
      and raises(lambda: cl.guard_dates("insample", {"o": with_date(outcome, "exit_date", "2024-09-15").assign(usable=True)}, sealed))
      and not raises(lambda: cl.guard_dates("insample", {"o": with_date(outcome, "exit_date", "2024-09-15").assign(usable=False)}, sealed)))
check("holdout: only when RUN_HOLDOUT is True in the namespace; then any dates (2026+ included)",
      all(raises(lambda ns=ns: cl.guard_dates("holdout", {}, ns)) for ns in ({}, {"RUN_HOLDOUT": False}, {"RUN_HOLDOUT": 1},
                                                                             {"RUN_OOS": True}))
      and raises(lambda: cl.guard_dates("holdout", {})) and raises(lambda: cl.load_tables("holdout", NOWHERE))
      and raises(lambda: cl.run("holdout", NOWHERE)) and raises(lambda: T.run("holdout", NOWHERE, n_perm=10, n_boot=10))
      and cl.guard_dates("holdout", {"e": with_date(events, "t_0", "2026-03-02"),
                                     "o": with_date(outcome, "exit_date", "2027-09-01")}, {"RUN_HOLDOUT": True}) is None)
check("oos: only when RUN_OOS is True in the namespace; RUN_HOLDOUT does not open it",
      all(raises(lambda ns=ns: cl.guard_dates("oos", {}, ns)) for ns in ({}, {"RUN_OOS": False}, {"RUN_OOS": 1}, {"RUN_HOLDOUT": True}))
      and raises(lambda: cl.guard_dates("oos", {})) and raises(lambda: cl.run("oos", NOWHERE))
      and raises(lambda: T.run("oos", NOWHERE, n_perm=10, n_boot=10))
      and cl.guard_dates("oos", {"e": with_date(events, "t_0", "2026-03-02")}, {"RUN_OOS": True}) is None)
check("insample needs no switch; every allowed label uses the 2024-25 constants",
      [cl.default_zref_source("insample", {}), cl.default_zref_source("holdout", {"RUN_HOLDOUT": True}),
       cl.default_zref_source("oos", {"RUN_OOS": True})] == ["insample"] * 3
      and raises(lambda: cl.default_zref_source("holdout", {})) and raises(lambda: cl.default_zref_source("oos", {})))

# ---- the end of the window: drop and count; before the start or inside the sealed window: refuse -----------------
edge_ev = events.copy()
edge_ev.loc[:2, "t_0"] = pd.Timestamp("2026-01-02")             # filed late on the last day: enters next session
gone_ids = set(edge_ev.row_id[:3])
edge_nl = nulls.copy()
own_2026 = edge_nl.index[~edge_nl.event_row_id.isin(gone_ids)][0]
edge_nl.loc[own_2026, "t_0"] = pd.Timestamp("2026-01-05")        # an ordinary day of a kept event reaches 2026
gone_nl_ids = set(edge_nl.row_id[edge_nl.event_row_id.isin(gone_ids)]) | {edge_nl.row_id[own_2026]}
kept, drops = cl.restrict("insample", {"events": edge_ev, "nulls": edge_nl, "gap": gap, "outcome": outcome})
dc = dict(zip(drops.what, drops.n))
check("window end: events with t_0 on or after 2026-01-01 are dropped, with all their rows",
      len(kept["events"]) == len(events) - 3 and not set(kept["events"].row_id) & gone_ids
      and not set(kept["nulls"].row_id) & gone_nl_ids
      and not set(kept["gap"].row_id) & (gone_ids | gone_nl_ids) and not set(kept["outcome"].row_id) & (gone_ids | gone_nl_ids)
      and len(kept["gap"]) == len(gap) - len(gone_ids | gone_nl_ids), str(dc))
check("window end: every drop is counted",
      list(dc.values())[0] == 3 and list(dc.values())[1] == len(gone_nl_ids) and list(dc.values())[2] == 3 + len(gone_nl_ids))
check("window end: restricted tables pass the guard", not raises(lambda: cl.guard_dates("insample", kept)))
pre = events.copy()
pre.loc[4, "gap_start"] = pd.Timestamp("2023-12-29")           # a gap that starts before the window: refused, not dropped
kept_pre, drops_pre = cl.restrict("insample", {"events": pre, "nulls": nulls, "gap": gap})
check("window start: a row dated before 2024-01-01 is not silently dropped; the guard refuses it",
      len(kept_pre["events"]) == len(pre) and raises(lambda: cl.guard_dates("insample", kept_pre)))
kept_h, drops_h = cl.restrict("holdout", {"events": edge_ev, "nulls": edge_nl, "gap": gap})
check("holdout and oos drop nothing", len(kept_h["events"]) == len(edge_ev) and drops_h.empty)
edge_dir = write_tables(TMP / "edgefreeze", "insample", (edge_ev, edge_nl, gap), with_outcome=False)
edge_doc = cl.freeze("insample", edge_dir, edge_dir / "z.json")
keep_ids = sorted(set(nulls.row_id) - gone_nl_ids)
keep_ok = cl.gap_table(gap).loc[keep_ids]
keep_ok = keep_ok[keep_ok.gap_usable]
check("freeze uses no ordinary day that falls on or after 2026-01-01 or belongs to a dropped event",
      edge_doc["n_null_rows"] == len(keep_ok) and edge_doc["null_row_ids_sha256"] ==
      hashlib.sha256("\n".join(sorted(keep_ok.index)).encode()).hexdigest()
      and edge_doc["window_t_0"]["last"] < "2026-01-01")
pre_dir = write_tables(TMP / "prefreeze", "insample", (pre, nulls, gap), with_outcome=False)
check("freeze refuses a table with a date before 2024-01-01 and writes nothing",
      raises(lambda: cl.freeze("insample", pre_dir, pre_dir / "z.json")) and not (pre_dir / "z.json").exists())

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
src_lines = [l for f in ("classify.py", "tests.py") for l in (ROOT / "src" / "oldnews" / f).read_text(encoding="utf-8").splitlines()]
check("nothing in classify.py or tests.py builds a path to data/oldnews/_unused_2022_23",
      not any("_unused" in l and any(k in l for k in ("Path(", "read_", "glob", "listdir", "DATA_DIR /", "open(")) for l in src_lines))

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
                   "t_0": pd.Timestamp("2024-03-01"), "gap_move": [2.0, 5.0, 0.1], "old": [True, True, False]})
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
inp = T.prepare(events, nulls, gap, outcome, classified=c)
log = tmp_log(TMP)
h = T.h1(inp, log, NP, NB).iloc[0]
check("H1 detects an injected -0.3 effect", h.effect < 0 and h.p_one_sided < 0.01 and h.ci_lo < -0.3 < h.ci_hi,
      f"effect {h.effect:+.3f} [{h.ci_lo:+.3f}, {h.ci_hi:+.3f}], p={h.p_one_sided:.4f}, n={h.n} ({h.n_old}/{h.n_comp})")
check("H1 reproducible (fixed seed)", T.h1(inp, log, NP, NB).iloc[0].equals(h))
c2 = inp.classified
excl = c2.row_id[~(c2.late & ~c2.earnings_excluded & c2.scored & (c2.group == "people"))]
out2 = outcome.copy()
out2.loc[out2.row_id.isin(excl), "y"] = -50.0
h_ex = T.h1(T.prepare(events, nulls, gap, out2, classified=c), log, NP, NB).iloc[0]
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
h0 = T.h1(T.prepare(*e0, classified=cl.classify(e0[0], e0[2], FROZEN)), log, NP, NB).iloc[0]
check("no injected effect: H1 not significant", h0.p_one_sided > 0.05, f"effect {h0.effect:+.3f}, p={h0.p_one_sided:.3f}")
ep = synth(effect=+0.3, seed=2, ref=FROZEN)
hp = T.h1(T.prepare(*ep, classified=cl.classify(ep[0], ep[2], FROZEN)), log, NP, NB).iloc[0]
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

# ---- run('insample') end to end: pooled primary result, then each t_0 year ---------------------------------------
dd_ = write_tables(TMP / "run", "insample", (events, nulls, gap, outcome))
t0 = time.time()
res = T.run("insample", dd_, n_perm=NP, n_boot=NB, log=T.Log(dd_ / "ledger.csv", "run-1", "synthetic-head"))
el = time.time() - t0
files = {"counts.csv", "coverage.csv", "h1.csv", "h1b.csv", "placebo.csv", "profile.csv", "sensitivity.csv", "summary.md",
         "by_year.csv", "profile_by_year.csv", "counts_by_year.csv"}
check("run writes every table and the summary", files <= {f.name for f in (dd_ / "results_insample").iterdir()}
      and (dd_ / "classified_insample.csv").exists() and (dd_ / "zref_insample.csv").exists(), f"{el:.1f}s")
summ = (dd_ / "results_insample" / "summary.md").read_text(encoding="utf-8")
check("summary states counts, coverage, H1 effect, CI, p and n",
      all(k in summ for k in ("## Counts", "## Coverage of the gap inputs", "95% CI", "one-sided p", "n = ", "log of the ratio")))
check("summary names the 2024-25 constants file, the date window and the sealed window",
      "frozen insample constants" in summ and "zref_frozen_insample.json" in summ
      and "every date inside [2024-01-01, 2026-01-01)" in summ and "sealed window 2023-06-01..2023-08-31" in summ)
check("run reproduces the direct H1", np.isclose(res["h1"].effect.iloc[0], h.effect))
res2 = T.run("insample", dd_, n_perm=NP, n_boot=NB, log=T.Log(dd_ / "ledger.csv", "run-2", "synthetic-head"))
check("two runs give identical tables", all(res[k].equals(res2[k]) for k in res))
cls = pd.read_csv(dd_ / "classified_insample.csv")
check("classified table has the trade agent's `old` column", "old" in cls.columns and cls.old.dtype == bool)
check("the zref csv records the source of the constants applied",
      (pd.read_csv(dd_ / "zref_insample.csv").source == "insample").all())

# pooled primary, by-year robustness
check("primary tables are pooled and the summary says so",
      all((res[k].period == "pooled").all() for k in ("h1", "h1b", "placebo", "profile", "sensitivity"))
      and "Primary result: pooled over t_0 years 2024, 2025" in summ)
by = res["by_year"]
check("by-year table: H1, H1b and P for 2024 and 2025, with n per group",
      sorted(by.period.unique()) == ["2024", "2025"] and all(sorted(by[by.period == y_].test) == ["H1", "H1b", "P"] for y_ in ("2024", "2025"))
      and by[["n", "n_old", "n_comp"]].notna().all().all() and (by.n > 0).all(), str(by[["period", "test", "n", "n_old", "n_comp"]].values.tolist()))
pool = res["h1"].iloc[0]
check("the two years partition the pooled H1 and P samples exactly",
      by[(by.test == "H1")].n.sum() == pool.n and by[(by.test == "H1")].n_old.sum() == pool.n_old
      and by[(by.test == "H1")].n_comp.sum() == pool.n_comp and by[(by.test == "P")].n.sum() == res["placebo"].n.iloc[0])
inp_i = T.prepare(events, nulls, gap, outcome, classified=cl.classify(events, gap, FROZEN))
y24 = T.restrict_year(inp_i, 2024)
check("restrict_year keeps only that t_0 year's events and their ordinary days",
      (y24.classified.t_0.dt.year == 2024).all() and 0 < len(y24.classified) < len(inp_i.classified)
      and set(y24.nulls.event_row_id) <= set(y24.classified.row_id)
      and len(T.restrict_year(inp_i, 2024).classified) + len(T.restrict_year(inp_i, 2025).classified) == len(inp_i.classified))
h24 = T.h1(y24, T.Log(TMP / "scratch_led.csv", "d", "d"), NP, NB, period="2024").iloc[0]
check("the by-year H1 row is H1 on that year's events", np.isclose(by[(by.period == "2024") & (by.test == "H1")].effect.iloc[0], h24.effect)
      and by[(by.period == "2024") & (by.test == "H1")].n.iloc[0] == h24.n)
pby = res["profile_by_year"]
check("horizon profile by year: 9 horizons x 3 tests x 2 years, BH within each test and year",
      len(pby) == 54 and all(list(pby[(pby.period == y_) & (pby.test == t)].horizon) == list(T.HORIZONS) for y_ in ("2024", "2025") for t in ("H1", "H1b", "P"))
      and all(np.allclose(g_.q_bh, stats.benjamini_hochberg(g_.p.to_numpy()), equal_nan=True) for _, g_ in pby.groupby(["period", "test"])))
cby = res["counts_by_year"].set_index("step")
check("counts by year: n per step and group for each year",
      {"people_2024", "people_2025", "placebo_2024", "placebo_2025"} <= set(cby.columns)
      and cby.at["at least one usable matched ordinary day", "people_2024"] + cby.at["at least one usable matched ordinary day", "people_2025"] == pool.n)
led_i = pd.read_csv(dd_ / "ledger.csv")
led_i = led_i[led_i.run_id == "run-1"]
yr = led_i["filter"].str.extract(r"year=(\d{4})")[0]
check("every pooled and by-year test is in the ledger, the year in the filter column",
      len(led_i) == 43 + 2 * (3 + 27) and yr.notna().sum() == 60 and yr.value_counts().to_dict() == {"2024": 30, "2025": 30}
      and set(led_i.kind) == {"oldnews_H1", "oldnews_H1b", "oldnews_P"} and (led_i.seed == T.SEED).all(), f"{len(led_i)} rows")
check("summary: by-year section with each year's H1, H1b and P, the profile and the counts",
      all(k in summ for k in ("## By year", "### 2024", "### 2025", "Horizon profile by year", "Counts by year", "**H1** (negative)",
                              "**H1b** (negative)", "**P** (no difference)")))
check("years= turns the split on or off for any allowed label",
      "by_year" not in T.run("insample", dd_, n_perm=100, n_boot=100, log=T.Log(TMP / "scratch_led2.csv", "x", "y"), years=())
      and set(T.run("insample", dd_, n_perm=100, n_boot=100, log=T.Log(TMP / "scratch_led2.csv", "x", "y"),
                    years=(2025,))["by_year"].period) == {"2025"})
small = synth(n_people=25, n_placebo=10, seed=4, ref=FROZEN)
sd_ = write_tables(TMP / "small", "insample", small)
T.run("insample", sd_, n_perm=300, n_boot=300, log=T.Log(sd_ / "ledger.csv", "small", "synthetic-head"))
small_txt = (sd_ / "results_insample" / "summary.md").read_text(encoding="utf-8")
check("fewer than 30 events -> labelled descriptive, pooled and in each year",
      "DESCRIPTIVE" in small_txt and small_txt.count("DESCRIPTIVE") >= 3 and pd.read_csv(sd_ / "results_insample" / "by_year.csv").descriptive.all())

# corrupt rows and the window edge through run()
ev_bad = events.copy()
ev_bad.loc[3, "t_pre"] = "2026-01-03"                           # t_0 still inside the window: a corrupt row
ev_bad.to_csv(dd_ / "events_insample.csv", index=False)
check("run hard-fails on a corrupt row (t_pre on or after 2026-01-01, t_0 not)",
      raises(lambda: T.run("insample", dd_, n_perm=10, n_boot=10, log=T.Log(dd_ / "l2.csv", "x", "y"))))
ev_old = events.copy()
ev_old.loc[3, "gap_start"] = "2023-12-29"                       # a gap that starts before the window
ev_old.to_csv(dd_ / "events_insample.csv", index=False)
check("run hard-fails on an event whose gap starts before 2024-01-01",
      raises(lambda: T.run("insample", dd_, n_perm=10, n_boot=10, log=T.Log(dd_ / "l2.csv", "x", "y")))
      and raises(lambda: cl.run("insample", dd_)))
events.to_csv(dd_ / "events_insample.csv", index=False)
out_bad = outcome.copy()
out_bad.loc[5, "entry_date"] = pd.Timestamp("2026-02-01")
out_bad.to_csv(dd_ / "outcome_insample.csv", index=False)
check("run hard-fails on a kept row whose outcome entry is in 2026",
      raises(lambda: T.run("insample", dd_, n_perm=10, n_boot=10, log=T.Log(dd_ / "l2.csv", "x", "y"))))
out_h = outcome.copy()
out_h.loc[out_h.index[7], ["entry_date", "exit_date"]] = pd.Timestamp("2024-09-10")
out_h.to_csv(dd_ / "outcome_insample.csv", index=False)
check("run hard-fails on a row inside the notebook's sealed window (HOLDOUT_START..END in NB)",
      raises(lambda: T.run("insample", dd_, sealed, n_perm=10, n_boot=10, log=T.Log(dd_ / "l2.csv", "x", "y"))))
outcome.to_csv(dd_ / "outcome_insample.csv", index=False)

ed = TMP / "edge"
extra = events.head(3).copy()
extra["row_id"] = [f"event|edge{i}" for i in range(3)]
extra["t_0"] = pd.Timestamp("2026-01-02")
extra_nl = nulls[nulls.event_row_id.isin(events.row_id[:3])].copy()
extra_nl["event_row_id"] = extra_nl.event_row_id.map(dict(zip(events.row_id[:3], extra.row_id)))
extra_nl["row_id"] = extra_nl.row_id + "|edge"
write_tables(ed, "insample", (pd.concat([events, extra]), pd.concat([nulls, extra_nl]), gap, outcome))
res_e = T.run("insample", ed, n_perm=NP, n_boot=NB, log=T.Log(ed / "ledger.csv", "edge", "synthetic-head"))
dr = pd.read_csv(ed / "results_insample" / "dropped.csv")
check("run drops the events entering on or after 2026-01-01 and says how many (dropped.csv, summary.md)",
      dr.n.iloc[0] == 3 and dr.n.iloc[1] == len(extra_nl)
      and "Dropped at the window edge" in (ed / "results_insample" / "summary.md").read_text(encoding="utf-8"), str(dr.n.tolist()))
check("dropping them leaves H1 exactly as without them", res_e["h1"].iloc[0].equals(res["h1"].iloc[0]))

# a ticker called "NA" and kind "null" survive the files
na_ev = events.copy()
na_ev.loc[na_ev.ticker == "T05", "ticker"] = "NA"
nd = write_tables(TMP / "na", "insample", (na_ev, nulls.assign(ticker=nulls.ticker.replace("T05", "NA")), gap, outcome))
T.run("insample", nd, n_perm=NP, n_boot=NB, log=T.Log(nd / "ledger.csv", "na", "synthetic-head"))
cna = cl.read_table(nd / "classified_insample.csv")
check("ticker 'NA' is kept through run(): same labels and H1 as with 'T05'",
      (cna.ticker == "NA").sum() == (events.ticker == "T05").sum() and cna.ticker.notna().all()
      and (cna.set_index("row_id").old == cl.read_table(dd_ / "classified_insample.csv").set_index("row_id").old).all()
      and np.isclose(pd.read_csv(nd / "results_insample" / "h1.csv").effect[0], res["h1"].effect.iloc[0], rtol=1e-12))


# ---- holdout and oos through run(): behind their switches, same 2024-25 constants ------------------------------
def shifted(label, years=2, ctx=None, name=None):
    """The tables with every date moved `years` later (2024-25 becomes 2026-27, as a judges' window might be)."""
    d = TMP / (name or label)
    sh = lambda df, cols: df.assign(**{c: pd.to_datetime(df[c]) + pd.DateOffset(years=years) for c in cols})  # noqa: E731
    return write_tables(d, label, (sh(events, ["filing_date", "accepted_at", "event_date", "gap_start", "t_pre", "t_0"]),
                                   sh(nulls, ["gap_start", "t_pre", "t_0"]), gap if ctx is None else ctx,
                                   sh(outcome, ["entry_date", "exit_date"])))


JUDGES = {"RUN_HOLDOUT": True}
HUMAN = {"RUN_OOS": True}
hd = shifted("holdout")
check("holdout refuses to run without RUN_HOLDOUT, and writes nothing",
      raises(lambda: T.run("holdout", hd, n_perm=10, n_boot=10)) and raises(lambda: cl.run("holdout", hd, {"RUN_HOLDOUT": False}))
      and not (hd / "results_holdout").exists() and not (hd / "classified_holdout.csv").exists())
res_h = T.run("holdout", hd, JUDGES, n_perm=NP, n_boot=NB, log=T.Log(hd / "ledger.csv", "h", "synthetic-head"))
h_txt = (hd / "results_holdout" / "summary.md").read_text(encoding="utf-8")
check("holdout runs on 2026+ dates behind RUN_HOLDOUT: nothing dropped, no year split, insample constants",
      np.isclose(res_h["h1"].effect.iloc[0], res["h1"].effect.iloc[0]) and "Dropped at the window edge" not in h_txt
      and "switched by RUN_HOLDOUT" in h_txt and "by_year" not in res_h and "## By year" not in h_txt
      and (pd.read_csv(hd / "zref_holdout.csv").source == "insample").all() and "zref_frozen_insample.json" in h_txt)
od = shifted("oos")
check("oos refuses to run (and writes nothing) without RUN_OOS",
      raises(lambda: T.run("oos", od, n_perm=10, n_boot=10)) and raises(lambda: cl.run("oos", od, {"RUN_OOS": False}))
      and raises(lambda: T.run("oos", od, JUDGES, n_perm=10, n_boot=10))
      and not (od / "results_oos").exists() and not (od / "classified_oos.csv").exists())
res_o = T.run("oos", od, HUMAN, n_perm=NP, n_boot=NB, log=T.Log(od / "ledger.csv", "o", "synthetic-head"))
check("oos runs once RUN_OOS is True in the namespace passed", np.isclose(res_o["h1"].effect.iloc[0], res["h1"].effect.iloc[0]))
dd2 = TMP / "deny_insample"
dd2.mkdir()
for f in ("events", "nulls", "gap", "outcome"):
    (dd2 / f"{f}_insample.csv").write_bytes((od / f"{f}_oos.csv").read_bytes())
check("insample cannot be pointed at 2026+ data (everything is dropped, nothing is computed)",
      raises(lambda: T.run("insample", dd2, n_perm=10, n_boot=10), (PermissionError, ValueError))
      and not (dd2 / "results_insample").exists())

# ---- the frozen constants across windows -----------------------------------------------------------------------
c_main = pd.read_csv(dd_ / "classified_insample.csv").set_index("row_id")
zr = {"insample": (dd_ / "zref_insample.csv").read_text(), "holdout": (hd / "zref_holdout.csv").read_text(),
      "oos": (od / "zref_oos.csv").read_text()}
check("the constants applied are the same in insample, holdout and oos", len(set(zr.values())) == 1)
gap_other = gap.copy()
gap_other.loc[gap_other.row_id.str.startswith("null|"), ["gap_move", "d_iv_gap", "vol_gap_ratio"]] *= 5   # new window's ordinary days differ
hd2 = shifted("holdout", 3, gap_other, name="holdout2")
cols = [*zc, "M", "S"]
cl.run("holdout", hd2, JUDGES)
c_h2 = pd.read_csv(hd2 / "classified_holdout.csv").set_index("row_id")
check("an event is standardised identically in insample and in a window whose ordinary days look different",
      np.allclose(c_main[cols].fillna(-9), c_h2.loc[c_main.index, cols].fillna(-9))
      and (c_main.old == c_h2.loc[c_main.index].old).all() and (hd2 / "zref_holdout.csv").read_text() == zr["insample"])
check("running other labels did not change the frozen constants", FROZEN_FILE.read_bytes() == FROZEN_BYTES)

# ---- missing, malformed and retired constants -----------------------------------------------------------------------
missing = TMP / "nowhere" / "zref_frozen_insample.json"
cl.FROZEN_PATH_INSAMPLE = missing
fresh = write_tables(TMP / "fresh", "insample", (events, nulls, gap, outcome))
check("classify.run refuses when the insample constants are missing, and does not create them",
      raises(lambda: cl.run("insample", fresh), FileNotFoundError) and not missing.exists()
      and not (fresh / "classified_insample.csv").exists())
check("tests.run refuses too, before reading or writing anything",
      raises(lambda: T.run("insample", fresh, n_perm=10, n_boot=10), FileNotFoundError) and not missing.exists()
      and not (fresh / "results_insample").exists() and not (fresh / "ledger.csv").exists())
check("holdout and oos refuse too (no fallback to any other file)",
      raises(lambda: cl.run("holdout", hd, JUDGES), FileNotFoundError) and raises(lambda: cl.run("oos", od, HUMAN), FileNotFoundError)
      and raises(lambda: cl.load_frozen(), FileNotFoundError))
cl.FROZEN_PATH_INSAMPLE = FROZEN_FILE
bad = TMP / "bad.json"
bad.write_text(FROZEN_BYTES.decode().replace('"source": "insample"', '"source": "elsewhere"', 1), encoding="utf-8")
check("a constants file whose source is neither insample nor the archive is refused", raises(lambda: cl.load_frozen(bad), ValueError))
bad.write_text(FROZEN_BYTES.decode().replace('"uses_outcomes": false', '"uses_outcomes": true'), encoding="utf-8")
check("a constants file that does not say it uses no outcomes is refused", raises(lambda: cl.load_frozen(bad), ValueError))
old_copy = TMP / "old_discovery_constants.json"
old_copy.write_text(FROZEN_BYTES.decode().replace('"source": "insample"', '"source": "discovery"', 1), encoding="utf-8")
check("retired constants (source 'discovery') cannot be loaded by any label",
      message(lambda: cl.load_frozen(old_copy)) == cl.RETIRED_ZREF_MSG
      and raises(lambda: cl.run("insample", dd_, zref=old_copy)) and raises(lambda: T.run("insample", dd_, n_perm=10, n_boot=10, zref=old_copy))
      and raises(lambda: cl.run("holdout", hd, JUDGES, zref=old_copy)) and raises(lambda: cl.run("oos", od, HUMAN, zref=old_copy))
      and raises(lambda: cl.run("insample", dd_, zref_source="discovery")) and raises(lambda: T.run("insample", dd_, n_perm=10, n_boot=10, zref_source="discovery"))
      and raises(lambda: cl.frozen_path("discovery")) and raises(lambda: cl.load_frozen(None, "discovery")))
arch = cl.load_frozen(old_copy, archive=True)
check("... except as an explicit read-only archive read (archive=True with a path): constants load, flagged as archive",
      arch.attrs["archive"] is True and arch.attrs["source"] == "discovery" and np.allclose(arch[["mean", "sd"]], FROZEN[["mean", "sd"]])
      and raises(lambda: cl.load_frozen(archive=True), ValueError)
      and old_copy.read_text() != FROZEN_BYTES.decode() and cl.load_frozen(FROZEN_FILE, archive=True).attrs["source"] == "insample")
check("the archive read does not make the retired file usable by classify.run or tests.run",
      raises(lambda: cl.run("insample", dd_, zref=old_copy)) and "archive" not in cl.run.__code__.co_varnames
      and "archive" not in T.run.__code__.co_varnames)
if REAL_ARCHIVE.exists():
    check("the real 2022-23 file src/oldnews/zref_frozen.json: archive read only; every label refuses it",
          cl.load_frozen(REAL_ARCHIVE, archive=True).attrs["source"] == "discovery"
          and raises(lambda: cl.load_frozen(REAL_ARCHIVE)) and raises(lambda: cl.run("insample", dd_, zref=REAL_ARCHIVE))
          and raises(lambda: T.run("insample", dd_, n_perm=10, n_boot=10, zref=REAL_ARCHIVE))
          and raises(lambda: cl.run("holdout", hd, JUDGES, zref=REAL_ARCHIVE)) and raises(lambda: cl.run("oos", od, HUMAN, zref=REAL_ARCHIVE)))
check("no code path loads the archive: classify.py and tests.py only mention ARCHIVE_PATH where it is defined",
      sum("ARCHIVE_PATH" in l for l in src_lines) == 1 and cl.ARCHIVE_PATH == REAL_ARCHIVE
      and cl.frozen_path() == FROZEN_FILE and cl.FROZEN_PATH == FROZEN_FILE)
check("no run wrote or changed the committed constants files (2022-23 archive, 2024-25 insample)",
      (REAL_ARCHIVE.read_bytes() if REAL_ARCHIVE.exists() else None) == REAL_ARCHIVE_BEFORE
      and (REAL_FROZEN_IN.read_bytes() if REAL_FROZEN_IN.exists() else None) == REAL_IN_BEFORE)

shutil.rmtree(TMP, ignore_errors=True)
print(f"\nlast full run on {len(events)} synthetic events: {el:.1f}s at {NP} permutations")
print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
