"""Old news or surprise news (docs/test_plan.md, "Classification").

Every late 8-K gets a math score M and a word score T, both built only from data known at `t_pre` (the last
close before EDGAR acceptance), so the label never looks ahead:

- M is the mean of the AVAILABLE standardised inputs among three: gap move (required), implied-volatility
  change over the gap, and log(option volume in the gap / its recent average). Each input is standardised with
  FROZEN constants (mean and standard deviation of that input over the usable 2024-25 ordinary days; gap inputs
  only, never outcomes), computed once by `freeze` into src/oldnews/zref_frozen_insample.json and applied
  unchanged to every event of every label (insample, holdout, oos), so no event is standardised with data from
  after its own entry. `run` and `classify` only read that file and refuse without it. The older 2022-23
  constants (src/oldnews/zref_frozen.json) are retired: no label can load them, only a test can read them, as an
  explicit read-only archive.
- T is the number of word cues present (0 to 3), computed by the events agent.
- S = w_m * M + w_t * T for each of the five fixed weight sets; old news if S >= cutoff.

An event is "scored" only when its gap move exists and T is finite; the others are counted with the reason. The same scored sample is used
for every weight set, so the sensitivity rows differ only in the rule, never in the sample.

Window guard (starter notebook cells 10 and 48; .claude/ctx/06_window_guard.md): label "insample" only uses dates in
[2024-01-01, 2026-01-01) outside the notebook's sealed HOLDOUT_START..HOLDOUT_END; "discovery" and "dryrun" are
retired and refuse; "holdout" runs only when RUN_HOLDOUT is True; "oos" only when RUN_OOS is True. Nothing
reads data/oldnews/_unused_2022_23/.
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
FROZEN_PATH_INSAMPLE = ROOT / "src" / "oldnews" / "zref_frozen_insample.json"   # the constants every label uses; written only by freeze()
FROZEN_PATH = FROZEN_PATH_INSAMPLE                                                # name other modules use for "the constants file"
ARCHIVE_PATH = ROOT / "src" / "oldnews" / "zref_frozen.json"                      # RETIRED 2022-23 constants: archive, read-only
ZREF_SOURCES = ("insample",)
WINDOW_START = pd.Timestamp("2024-01-01")        # first day of the in-sample window (options history starts here)
HARD_STOP = pd.Timestamp("2026-01-01")           # nothing is ever computed on or after this day (rule 8)
HOLDOUT_PLACEHOLDER = (pd.Timestamp("2023-06-01"), pd.Timestamp("2023-08-31"))    # the notebook's sealed placeholder
RETIRED = ("discovery", "dryrun")
RETIRED_MSG = ("2022-2023 is outside the allowed 2024-2025 window and overlaps the sealed placeholder "
               "(2023-06-01..2023-08-31)")
RETIRED_ZREF_MSG = ("the constants frozen on 2022-23 (src/oldnews/zref_frozen.json) are retired: 2022-2023 is outside the "
                    "allowed 2024-2025 window. Use zref_frozen_insample.json")
SWITCHES = {"holdout": "RUN_HOLDOUT", "oos": "RUN_OOS"}     # labels that run only when the notebook switch is on
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


# ---- the window guard --------------------------------------------------------------------------------------
def _namespace(NB: dict | None) -> dict:
    """The notebook namespace (RUN_HOLDOUT, RUN_OOS, HOLDOUT_START, ...): NB if given, else the running `__main__`."""
    return NB if NB is not None else vars(sys.modules["__main__"])


def check_label(label: str, NB: dict | None = None) -> None:
    """Refuse a label before anything is read. discovery and dryrun are retired; holdout needs RUN_HOLDOUT True
    and oos needs RUN_OOS True in the notebook namespace; insample is allowed (its dates are checked by
    `guard_dates`); any other label is refused."""
    if label in RETIRED:
        raise PermissionError(RETIRED_MSG)
    if label == "insample":
        return
    if label in SWITCHES:
        if _namespace(NB).get(SWITCHES[label]) is not True:
            who = "the judges set" if label == "holdout" else "a human sets"
            raise PermissionError(f"label {label!r} runs only after {who} {SWITCHES[label]} = True (section 2)")
        return
    raise PermissionError(f"unknown label {label!r}; allowed: insample, holdout (RUN_HOLDOUT), oos (RUN_OOS)")


def holdout_window(NB: dict | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """HOLDOUT_START..HOLDOUT_END from the notebook namespace, else the placeholder 2023-06-01..2023-08-31."""
    ns = _namespace(NB)
    start, end = ns.get("HOLDOUT_START"), ns.get("HOLDOUT_END")
    if start is None or end is None:
        return HOLDOUT_PLACEHOLDER
    return pd.Timestamp(start), pd.Timestamp(end)


def window(label: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """The date range [start, stop) every date of the label must lie in; None for holdout and oos (no date cap
    here, the label itself is switched)."""
    return (WINDOW_START, HARD_STOP) if label == "insample" else None


def guard_dates(label: str, frames: dict[str, pd.DataFrame | None], NB: dict | None = None) -> tuple | None:
    """Hard stop before any computation, the same rules as trade.check_dates.

    The label is checked first (`check_label`). For "insample", every date used (filing_date, gap_start, t_pre,
    t_0, entry_date, and the exit_date of every usable row) must lie in [2024-01-01, 2026-01-01) and outside
    HOLDOUT_START..HOLDOUT_END (the notebook's, else the placeholder). holdout and oos have no date cap.
    Returns the window used, or None."""
    check_label(label, NB)
    if label != "insample":
        return None
    h0, h1 = holdout_window(NB)
    for name, df in frames.items():
        if df is None:
            continue
        for col in (*ENTRY_COLS, EXIT_COL):
            if col not in df.columns:
                continue
            d = pd.to_datetime(df[col], errors="coerce")
            if col == EXIT_COL and "usable" in df.columns:
                d = d.where(as_bool(df["usable"]))                  # an unusable row's exit is never used
            for bad, why in ((d < WINDOW_START, f"before {WINDOW_START.date()}"),
                             (d >= HARD_STOP, f"on or after {HARD_STOP.date()}"),
                             ((d >= h0) & (d <= h1), f"inside the sealed window {h0.date()}..{h1.date()}")):
                if bad.any():
                    raise PermissionError(f"{name}.{col} has {int(bad.sum())} dates {why}; refusing to run "
                                          f"label {label!r} (allowed: {WINDOW_START.date()}..{HARD_STOP.date()}, exclusive)")
    return WINDOW_START, HARD_STOP


def restrict(label: str, tables: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Drop, and count, what enters on or after the end of the window (2026-01-01), so the guard never has to
    fail on a legitimate edge: events whose t_0 is on or after it (a filing accepted late on the last day enters
    the next session), ordinary days whose own dates reach it, and every row of the dropped ids. Dates before
    2024-01-01 or inside the sealed window are NOT dropped: the guard refuses them. Returns the remaining tables
    and a count table (`what`, `n`). holdout and oos keep everything."""
    win = window(label)
    if win is None:
        return dict(tables), pd.DataFrame({"what": [], "n": []})
    cap = win[1]
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
    """Read data_dir/<name>_<label>.csv safely, drop what the end of the window forces out, then check dates.
    The label is checked before any file is touched (retired labels and unswitched ones refuse)."""
    data_dir = Path(data_dir)
    check_label(label, NB)
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
def frozen_path(source: str = "insample") -> Path:
    """Where the constants live: src/oldnews/zref_frozen_insample.json, the only file any label uses. Looked up
    at call time. The retired 2022-23 file is never returned."""
    if source == "discovery":
        raise PermissionError(RETIRED_ZREF_MSG)
    if source not in ZREF_SOURCES:
        raise ValueError(f"constants source must be one of {ZREF_SOURCES}, not {source!r}")
    return Path(FROZEN_PATH_INSAMPLE)


def default_zref_source(label: str, NB: dict | None = None) -> str:
    """Every allowed label (insample, holdout, oos) uses the 2024-25 constants. Retired and unknown labels
    refuse (and holdout / oos refuse unless their switch is on)."""
    check_label(label, NB)
    return "insample"


NOTE = ("FROZEN CONSTANTS. Computed once from the 2024-25 window only (filings 2024-01-01 to "
        "2025-12-31): usable ordinary days, people and placebo together, gap inputs only, never outcomes. "
        "Applied unchanged to the 2024-25 test window and to the sealed window "
        "(docs/test_plan.md, Windows amendment and Classification).")


def freeze(label: str = "insample", data_dir: Path = DATA_DIR, path: Path | None = None,
           overwrite: bool = False) -> dict:
    """Compute the six standardisation constants ONCE and write them to src/oldnews/zref_frozen_insample.json.

    The only function that writes that file; `run` never does. The constants are the mean and standard deviation
    of gap_move, d_iv_gap and log(vol_gap_ratio) over the usable ordinary days of the insample null rows (people
    and placebo together; gap inputs only, no outcomes; the window guard applies), each over the days where that
    input exists. The file also records, per input, the count and a sha256 of the row ids, and for the sample as
    a whole the count, the date range and a sha256 of the row ids; `source` is "insample". It then ships
    committed. Only the label "insample" can be frozen (discovery and dryrun are retired), and an existing file
    is never replaced unless overwrite=True (a change is a plan amendment, made in its own commit)."""
    if label in RETIRED:
        raise PermissionError(RETIRED_MSG)
    if label != "insample":
        raise ValueError(f"constants can be frozen on the label 'insample' only, not {label!r}")
    path = frozen_path("insample") if path is None else Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists: the constants are frozen once. Pass overwrite=True only as a "
                              "documented plan amendment, in its own commit.")
    tables, _ = load_tables(label, data_dir, None, ("events", "nulls", "gap"))
    ref = reference(tables["nulls"], tables["gap"])
    ids, by_input = ref.attrs["row_ids"], ref.attrs["row_ids_by_input"]
    t0 = pd.to_datetime(tables["nulls"].set_index("row_id").loc[ids, "t_0"])
    doc = {
        "_note": NOTE,
        "source": label,
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


def load_frozen(path: Path | None = None, source: str | None = None, archive: bool = False) -> pd.DataFrame:
    """The frozen constants as a table (index: input; columns mean, sd, n_null). With no path: the insample file.
    Refuses if the file is missing or malformed, or if its recorded `source` is not "insample". A file recorded
    as "discovery" (the retired 2022-23 constants) is refused for every caller, except as an explicit read-only
    archive read (`archive=True` with an explicit path, meant for tests). Never writes: only `freeze` does."""
    if path is None:
        if archive:
            raise ValueError("an archive read needs an explicit path")
        path = frozen_path(source or "insample")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Classification needs the frozen 2024-25 constants: run "
                                "oldnews.classify.freeze('insample') once on the insample data, and commit the file. "
                                "Nothing falls back to another file.")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("uses_outcomes") is not False:
        raise ValueError(f"{path} does not say the constants use no outcomes; refusing")
    if doc.get("source") == "discovery":
        if not archive:
            raise PermissionError(RETIRED_ZREF_MSG)
    elif doc.get("source") not in ZREF_SOURCES:
        raise ValueError(f"{path} does not say the constants come from the insample window; refusing")
    elif source is not None and doc["source"] != source:
        raise ValueError(f"{path} holds constants frozen on {doc['source']!r}, but {source!r} was asked for; refusing")
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
                     n_null_rows=int(doc["n_null_rows"]), path=str(path), archive=bool(archive))
    return ref


def classify(events: pd.DataFrame, gap: pd.DataFrame, ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """The labelled event table: every events row plus its inputs, z-scores, M, T, S for each weight set and
    old_<weights>_<cutoff> flags. `old`, `S` and `news` (old | surprise | unscored) are the primary rule.

    M is the mean of the AVAILABLE standardised inputs; the gap move is required, the other two may be missing
    (`n_inputs` says how many M used). An event without a gap move, or without T, is not scored: `scored` is
    False and `unscored_reason` says why, so it can be counted and reported.

    The z-scores always use frozen constants (passed in as `ref`), so an event gets
    the same M whichever window, or whichever other events, it is classified with. The constants must be passed
    (`load_frozen()`); nothing is loaded here, so no window can pick up another file by accident."""
    missing = {"row_id", "T"} - set(events.columns)
    if missing:
        raise ValueError(f"events table is missing columns {sorted(missing)}")
    if events["row_id"].duplicated().any():
        raise ValueError("events table has duplicate row_id values")
    if ref is None:
        raise ValueError("classify needs the frozen constants: pass ref=load_frozen()")
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


def run(label: str, data_dir: Path = DATA_DIR, NB: dict | None = None, zref_source: str | None = None,
        zref: Path | None = None) -> pd.DataFrame:
    """Read events, nulls and gap for `label`, drop and count the rows past the end of the window, check dates,
    classify with the frozen constants, write classified_<label>.csv and zref_<label>.csv (a copy of the
    constants applied, with their source), and return the labelled table (`.attrs` holds `reference` and
    `dropped`).

    Labels: insample, holdout (only when RUN_HOLDOUT is True in NB) and oos (only when RUN_OOS is True); discovery
    and dryrun are retired and refuse, before anything is read. Constants: always zref_frozen_insample.json
    (`zref` may name that file explicitly; a file recorded as "discovery" is refused). A missing file is an
    error, never a fallback; this function never writes it. Pass NB (the notebook namespace) so the switches
    and HOLDOUT_START / HOLDOUT_END can be seen."""
    check_label(label, NB)
    ref = load_frozen(zref, zref_source)
    tables, dropped = load_tables(label, data_dir, NB)
    return save(label, Path(data_dir), classify(tables["events"], tables["gap"], ref), dropped)


def save(label: str, data_dir: Path, labelled: pd.DataFrame, dropped: pd.DataFrame) -> pd.DataFrame:
    """Write the labelled table and the standardisation reference next to the inputs."""
    labelled.attrs["dropped"] = dropped
    labelled.to_csv(data_dir / f"classified_{label}.csv", index=False)
    ref = labelled.attrs["reference"]
    ref.assign(source=ref.attrs.get("source", "")).to_csv(data_dir / f"zref_{label}.csv")
    return labelled
