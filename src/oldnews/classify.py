"""Old news or surprise news (docs/test_plan.md, "Classification").

Every late 8-K gets a math score M and a word score T, both built only from data known at `t_pre` (the last
close before EDGAR acceptance), so the label never looks ahead:

- M is the mean of the AVAILABLE standardised inputs among three: gap move (required), implied-volatility
  change over the gap, and log(option volume in the gap / its recent average). Each input is standardised with
  FROZEN constants (mean and standard deviation of that input over the usable discovery-window ordinary days;
  gap inputs only, never outcomes), computed once by `freeze` into src/oldnews/zref_frozen.json and applied
  unchanged to every window, so no event is standardised with data from after its own entry outside discovery.
  `run` and `classify` only read that file and refuse without it.
- T is the number of word cues present (0 to 3), computed by the events agent.
- S = w_m * M + w_t * T for each of the five fixed weight sets; old news if S >= cutoff.

An event is "scored" only when its gap move exists and T is finite; the others are counted with the reason. The same scored sample is used
for every weight set, so the sensitivity rows differ only in the rule, never in the sample.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

INPUTS = ("gap_move", "d_iv_gap", "log_vol_gap_ratio")      # the volume input is log(volume ratio)
GAP_COVERAGE_COLS = ("spot_bucket", "n_eff", "stale_sessions", "n_baseline")
WEIGHT_SETS: dict[str, tuple[float, float]] = {     # name: (w_m, w_t), fixed by the test plan
    "equal": (1.0, 1.0),
    "math_only": (1.0, 0.0),
    "words_only": (0.0, 1.0),
    "math_heavy": (1.0, 0.5),
    "words_heavy": (0.5, 1.0),
}
CUTOFFS = (0.5, 1.0, 1.5)
PRIMARY_WEIGHTS, PRIMARY_CUTOFF = "equal", 1.0

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "oldnews"
FROZEN_PATH = ROOT / "src" / "oldnews" / "zref_frozen.json"     # committed; written only by freeze()
DISCOVERY_END = pd.Timestamp("2024-01-01")       # discovery and the dry run: no entry-side date on or after this
HARD_STOP = pd.Timestamp("2026-01-01")           # nothing is ever computed on or after this day (rule 8)
ENTRY_CAPS = {"discovery": DISCOVERY_END, "dryrun": DISCOVERY_END, "insample": HARD_STOP}    # as pipeline.WINDOWS
DEFAULT_CAP = DISCOVERY_END                      # any other label gets discovery dates only (pipeline.DEFAULT_RULES)
GATED = ("holdout", "oos")                       # no date cap here; the pipeline runs them only behind RUN_HOLDOUT / RUN_OOS
ENTRY_COLS = ("filing_date", "gap_start", "t_pre", "t_0", "entry_date")
EXIT_COL = "exit_date"
ID_COLS = {"row_id": str, "event_row_id": str, "ticker": str, "accession_number": str}


def read_table(path: Path) -> pd.DataFrame:
    """read_csv where only empty cells (and 'nan') are missing. Pandas' defaults would turn kind="null" and a
    ticker such as "NA" into NaN."""
    return pd.read_csv(path, dtype=ID_COLS, keep_default_na=False, na_values=["", "nan", "NaN"])


def as_bool(s: pd.Series) -> pd.Series:
    """CSV-safe boolean: True/False, 1/0 and 'true'/'false' all work; missing values are False."""
    if s.dtype == bool:
        return s
    return s.map(lambda v: str(v).strip().lower() in {"1", "1.0", "true", "yes"}).astype(bool)


def old_col(weights: str, cutoff: float) -> str:
    """Column name of the old-news flag for one weight set and cutoff, e.g. old_equal_1."""
    return f"old_{weights}_{cutoff:g}"


def entry_cap(label: str) -> pd.Timestamp | None:
    """The first day no entry-side date may reach; None for the gated labels (holdout, oos)."""
    return None if label in GATED else ENTRY_CAPS.get(label, DEFAULT_CAP)


def guard_dates(label: str, frames: dict[str, pd.DataFrame], NB: dict | None = None) -> pd.Timestamp | None:
    """Hard rules on dates, the same as pipeline.WINDOWS and trade.check_dates.

    discovery and dryrun: refuse any filing, gap-start, t_pre, t_0 or entry date on or after 2024-01-01.
    insample: the same for 2026-01-01. Any other label: as discovery. For all of these, a usable exit on or after
    2026-01-01 is refused as well. holdout: allowed (the judges' sealed window; the pipeline runs it only after
    they set RUN_HOLDOUT). oos: refused unless the notebook namespace `NB` (default: the running notebook's
    namespace, `__main__`) has RUN_OOS is True. Returns the entry cap used (None for holdout and oos)."""
    if label == "oos":
        ns = NB if NB is not None else vars(sys.modules["__main__"])
        if ns.get("RUN_OOS") is not True:
            raise PermissionError("the out-of-sample window runs only after a human sets RUN_OOS = True (section 2)")
    cap = entry_cap(label)
    if cap is None:
        return None
    for name, df in frames.items():
        if df is None:
            continue
        for col in ENTRY_COLS:
            if col in df.columns and (pd.to_datetime(df[col], errors="coerce") >= cap).any():
                d = pd.to_datetime(df[col], errors="coerce")
                raise PermissionError(f"{name}.{col} has dates on or after {cap.date()} "
                                      f"(latest {d.max().date()}); refusing to compute for label {label!r}")
        if EXIT_COL in df.columns:
            late = pd.to_datetime(df[EXIT_COL], errors="coerce") >= HARD_STOP
            if "usable" in df.columns:
                late &= as_bool(df["usable"])
            if late.any():
                raise PermissionError(f"{name} has usable exits on or after {HARD_STOP.date()}; "
                                      f"refusing to compute for label {label!r}")
    return cap


def restrict(label: str, tables: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Drop, and count, everything that enters on or after the label's entry cap, so the guard never has to fail
    on a legitimate window edge: events whose t_0 is on or after the cap (a filing accepted late on the last
    day enters the next session), ordinary days whose own dates reach it, and every row of the dropped ids.
    Returns the remaining tables and a count table (`what`, `n`). Gated labels keep everything."""
    cap = entry_cap(label)
    if cap is None:
        return dict(tables), pd.DataFrame({"what": [], "n": []})
    ev, nl = tables["events"], tables["nulls"]
    gone_ev = pd.to_datetime(ev["t_0"], errors="coerce") >= cap
    gone_nl = nl["event_row_id"].isin(set(ev.loc[gone_ev, "row_id"]))
    for col in ("t_pre", "t_0", "gap_start"):
        if col in nl.columns:
            gone_nl |= pd.to_datetime(nl[col], errors="coerce") >= cap
    ids = set(ev.loc[gone_ev, "row_id"]) | set(nl.loc[gone_nl, "row_id"])
    out = {k: (v[~v["row_id"].isin(ids)] if k in ("gap", "outcome") and v is not None else v)
           for k, v in tables.items()}
    out["events"], out["nulls"] = ev[~gone_ev], nl[~gone_nl]
    counts = [("events with t_0 on or after " + str(cap.date()), int(gone_ev.sum())),
              ("ordinary days dropped (their event, or their own dates, on or after the cap)", int(gone_nl.sum()))]
    counts += [(f"{k} rows of dropped ids", int(len(tables[k]) - len(out[k]))) for k in ("gap", "outcome")
               if tables.get(k) is not None]
    return out, pd.DataFrame(counts, columns=["what", "n"])


def load_tables(label: str, data_dir: Path = DATA_DIR, NB: dict | None = None,
                names: tuple[str, ...] = ("events", "nulls", "gap")) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Read data_dir/<name>_<label>.csv safely, drop what the window edge forces out, then check dates."""
    data_dir = Path(data_dir)
    if label == "oos":
        guard_dates(label, {}, NB)                      # refuse before reading anything
    tables = {k: read_table(data_dir / f"{k}_{label}.csv") for k in names}
    n_before = len(tables["events"])
    tables, dropped = restrict(label, tables)
    if len(tables["events"]) == 0:
        raise ValueError(f"no events left for label {label!r}: {n_before} read, "
                         f"{n_before - len(tables['events'])} on or after the date cap; nothing to classify")
    guard_dates(label, tables, NB)
    return tables, dropped


def short_reason(r) -> str:
    """The first clause of a gap-table reason (up to the first ';' or '('), so reasons can be counted."""
    return "no reason given" if pd.isna(r) else (re.split(r"[;(]", str(r))[0].strip() or "no reason given")


def gap_table(gap: pd.DataFrame) -> pd.DataFrame:
    """The gap file indexed by row_id: the three inputs as numbers (volume as log of the ratio; a ratio of 0 has
    no log, so that input is missing), gap_usable (the file says usable and gap_move is finite), gap_reason and
    the coverage columns spot_bucket, n_eff, stale_sessions, n_baseline when present."""
    missing = {"row_id", "usable", "gap_move", "d_iv_gap", "vol_gap_ratio"} - set(gap.columns)
    if missing:
        raise ValueError(f"gap table is missing columns {sorted(missing)}")
    if gap["row_id"].duplicated().any():
        raise ValueError("gap table has duplicate row_id values")
    g = gap.set_index("row_id")
    out = pd.DataFrame(index=g.index)
    out["gap_move"] = pd.to_numeric(g["gap_move"], errors="coerce")
    out["d_iv_gap"] = pd.to_numeric(g["d_iv_gap"], errors="coerce")
    ratio = pd.to_numeric(g["vol_gap_ratio"], errors="coerce")
    out["vol_gap_ratio"] = ratio
    out["log_vol_gap_ratio"] = np.log(ratio.where(ratio > 0))
    out["vol_zero"] = (ratio == 0).fillna(False).astype(bool)
    out["gap_usable"] = as_bool(g["usable"]) & np.isfinite(out["gap_move"])
    out["gap_reason"] = g["reason"] if "reason" in g.columns else np.nan
    for c in GAP_COVERAGE_COLS:
        out[c] = g[c] if c in g.columns else np.nan
    return out


def reference(nulls: pd.DataFrame, gap: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation (ddof=1) of each input over the usable null rows (gap_move available), using
    each input's own finite values, so n can differ by input. Gap inputs only, never outcomes. Used by `freeze`;
    classification never calls it. `.attrs` lists the row ids behind each input (`row_ids_by_input`) and in all
    (`row_ids`)."""
    g = gap_table(gap)
    ids = pd.Index(nulls["row_id"].unique())
    ref_rows = g.loc[g.index.intersection(ids)]
    ref_rows = ref_rows[ref_rows["gap_usable"]]
    rows, by_input = [], {}
    for col in INPUTS:
        x = ref_rows[col]
        x = x[np.isfinite(x)]
        sd = float(x.to_numpy(float).std(ddof=1)) if len(x) > 1 else np.nan
        if not (len(x) > 1 and sd > 0):
            raise ValueError(f"cannot standardise {col}: {len(x)} usable null values, sd={sd}")
        rows.append({"input": col, "mean": float(x.mean()), "sd": sd, "n_null": len(x)})
        by_input[col] = sorted(x.index)
    out = pd.DataFrame(rows).set_index("input")
    out.attrs["row_ids"] = sorted(ref_rows.index)
    out.attrs["row_ids_by_input"] = by_input
    return out


def _sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


# ---- the frozen constants ----------------------------------------------------------------------------------
def freeze(label: str = "discovery", data_dir: Path = DATA_DIR, path: Path | None = None,
           overwrite: bool = False) -> dict:
    """Compute the six standardisation constants ONCE and write them to src/oldnews/zref_frozen.json.

    The only function that writes that file; `run` never does. The constants are the mean and standard deviation
    of gap_move, d_iv_gap and log(vol_gap_ratio) over the usable discovery-window ordinary days (people and
    placebo nulls together; gap inputs only, no outcomes; nothing on or after 2024-01-01), each over the days
    where that input exists. The file also records, per input, the count and a sha256 of the row ids, and for
    the sample as a whole the count, the date range and a sha256 of the row ids. It then ships committed and is
    applied unchanged to every window. Refuses any label but "discovery", and refuses to replace an existing
    file unless overwrite=True (a change is a plan amendment, made in its own commit)."""
    if label != "discovery":
        raise ValueError(f"the constants come from the discovery window only, not {label!r}")
    path = Path(FROZEN_PATH if path is None else path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists: the constants are frozen once. Pass overwrite=True only as a "
                              "documented plan amendment, in its own commit.")
    tables, _ = load_tables(label, data_dir, None, ("events", "nulls", "gap"))
    ref = reference(tables["nulls"], tables["gap"])
    ids, by_input = ref.attrs["row_ids"], ref.attrs["row_ids_by_input"]
    t0 = pd.to_datetime(tables["nulls"].set_index("row_id").loc[ids, "t_0"])
    doc = {
        "_note": ("FROZEN CONSTANTS. Computed once from the DISCOVERY window only (filings 2022-01-01 to "
                  "2023-12-31): usable ordinary days, people and placebo together, gap inputs only, never outcomes. "
                  "Applied unchanged to discovery, confirmation, the dry run and the sealed window "
                  "(docs/test_plan.md, Classification)."),
        "source": "discovery",
        "uses_outcomes": False,
        "inputs": {"gap_move": "|r_gap - r_mkt| / (sigma * sqrt(n))", "d_iv_gap": "1m ATM IV at t_pre minus at gap start",
                   "log_vol_gap_ratio": "log(ATM-pair volume in the gap / mean over the 5 sessions before); a ratio "
                                        "of 0 has no log and counts as missing"},
        "window_t_0": {"first": str(t0.min().date()), "last": str(t0.max().date())},
        "n_null_rows": len(ids),
        "null_row_ids_sha256": _sha(ids),
        "ddof": 1,
        "constants": {c: {"mean": float(ref.at[c, "mean"]), "sd": float(ref.at[c, "sd"]), "n": int(ref.at[c, "n_null"]),
                          "row_ids_sha256": _sha(by_input[c])} for c in INPUTS},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc


def load_frozen(path: Path | None = None) -> pd.DataFrame:
    """The frozen constants as a table (index: input; columns mean, sd, n_null). Refuses if the file is missing
    or malformed. Never writes: only `freeze` does."""
    path = Path(FROZEN_PATH if path is None else path)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Classification needs the frozen constants: run "
                                "oldnews.classify.freeze() once on the discovery data and commit the file.")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("source") != "discovery" or doc.get("uses_outcomes") is not False:
        raise ValueError(f"{path} does not say the constants come from discovery inputs only; refusing")
    try:
        ref = pd.DataFrame({c: doc["constants"][c] for c in INPUTS}).T[["mean", "sd", "n"]].astype(float)
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{path} is missing a constant ({e!r}); refusing") from e
    if not (np.isfinite(ref.to_numpy()).all() and (ref["sd"] > 0).all()):
        raise ValueError(f"{path} holds a non-finite constant or a non-positive sd; refusing")
    ref = ref.rename(columns={"n": "n_null"})
    ref["n_null"] = ref["n_null"].astype(int)
    ref.index.name = "input"
    ref.attrs.update(source=doc["source"], sha256=doc["null_row_ids_sha256"], window_t_0=doc["window_t_0"],
                     n_null_rows=int(doc["n_null_rows"]))
    return ref


def classify(events: pd.DataFrame, gap: pd.DataFrame, ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """The labelled event table: every events row plus its inputs, z-scores, M, T, S for each weight set and
    old_<weights>_<cutoff> flags. `old`, `S` and `news` (old | surprise | unscored) are the primary rule.

    M is the mean of the AVAILABLE standardised inputs; the gap move is required, the other two may be missing
    (`n_inputs` says how many M used). An event without a gap move, or without T, is not scored: `scored` is
    False and `unscored_reason` says why, so it can be counted and reported.

    The z-scores always use the frozen discovery constants (`ref` only lets tests pass others), so an event gets
    the same M whichever window, or whichever other events, it is classified with."""
    missing = {"row_id", "T"} - set(events.columns)
    if missing:
        raise ValueError(f"events table is missing columns {sorted(missing)}")
    if events["row_id"].duplicated().any():
        raise ValueError("events table has duplicate row_id values")
    ref = load_frozen() if ref is None else ref
    g = gap_table(gap)
    in_gap = events["row_id"].isin(g.index).to_numpy()
    ev = events.join(g, on="row_id")
    ev["gap_usable"] = ev["gap_usable"].fillna(False).astype(bool)
    ev["vol_zero"] = ev["vol_zero"].fillna(False).astype(bool)
    for col in INPUTS:
        ev[f"z_{col}"] = (ev[col] - ref.at[col, "mean"]) / ref.at[col, "sd"]
    z = ev[[f"z_{c}" for c in INPUTS]].to_numpy(float)
    avail = np.isfinite(z)
    ev["n_inputs"] = avail.sum(axis=1)
    has_gap = avail[:, 0] & ev["gap_usable"].to_numpy(bool)             # the gap move is required
    ev["M"] = np.where(has_gap, np.where(avail, z, 0.0).sum(axis=1) / np.maximum(avail.sum(axis=1), 1), np.nan)
    ev["T"] = pd.to_numeric(ev["T"], errors="coerce")
    ev["scored"] = ev["gap_usable"] & np.isfinite(ev["M"]) & np.isfinite(ev["T"])
    why = pd.Series("", index=ev.index)
    why = why.mask(~ev["scored"], "T missing")
    why = why.mask(~ev["scored"] & ~(ev["gap_usable"] & np.isfinite(ev["M"])),
                   "no gap_move: " + ev["gap_reason"].map(short_reason))
    ev["unscored_reason"] = why.mask(~ev["scored"] & ~in_gap, "no gap row")
    for name, (w_m, w_t) in WEIGHT_SETS.items():
        s = (w_m * ev["M"] + w_t * ev["T"]).where(ev["scored"])
        ev[f"S_{name}"] = s
        for c in CUTOFFS:
            ev[old_col(name, c)] = ev["scored"] & (s >= c)
    ev["S"] = ev[f"S_{PRIMARY_WEIGHTS}"]
    ev["old"] = ev[old_col(PRIMARY_WEIGHTS, PRIMARY_CUTOFF)]
    ev["news"] = np.where(ev["scored"], np.where(ev["old"], "old", "surprise"), "unscored")
    ev.attrs["reference"] = ref
    return ev


def run(label: str, data_dir: Path = DATA_DIR, NB: dict | None = None) -> pd.DataFrame:
    """Read events, nulls and gap for `label`, drop and count the window-edge rows, check dates, classify with the
    frozen constants (refuses if src/oldnews/zref_frozen.json is missing; never writes it), write
    classified_<label>.csv and zref_<label>.csv (a copy of the constants applied), and return the labelled table
    (`.attrs` holds `reference` and `dropped`). Pass NB (the notebook namespace) so oos can see RUN_OOS."""
    ref = load_frozen()
    tables, dropped = load_tables(label, data_dir, NB)
    return save(label, Path(data_dir), classify(tables["events"], tables["gap"], ref), dropped)


def save(label: str, data_dir: Path, labelled: pd.DataFrame, dropped: pd.DataFrame) -> pd.DataFrame:
    """Write the labelled table and the standardisation reference next to the inputs."""
    labelled.attrs["dropped"] = dropped
    labelled.to_csv(data_dir / f"classified_{label}.csv", index=False)
    labelled.attrs["reference"].to_csv(data_dir / f"zref_{label}.csv")
    return labelled
