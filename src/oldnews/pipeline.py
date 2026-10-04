"""Old news, new news: the committed hypothesis test (docs/hypothesis.md, docs/test_plan.md) as one call.

    run_oldnews(start, end, label, allow_fetch=False)

runs every step in order, each in its own module, and writes everything to data/oldnews/ (git-ignored):

    0  inputs    every TOP_100 8-K in the window and its matched ordinary days   raw/events_<label>.csv, raw/nulls_<label>.csv
    1  events    late people-news and placebo filings, gaps, word cues            events_<label>.csv, nulls_<label>.csv
    2  measure   gap inputs, realised vs implied volatility, the put's P&L        gap_<label>.csv, outcome_<label>.csv
    3  classify  old news or surprise news (weights fixed in the test plan)       classified_<label>.csv
    4  tests     H1, H1b, placebo P, every horizon, the sensitivity grid          results_<label>/
    5  trade     the cash-secured put on old news, net of costs, with capacity   trade_<label>/
    6  figures   fade curve, one-event walkthrough, placebo                       figures/<label>/

Where the data come from. Steps 1-6 read only the API cache. Step 0 reuses the tables the 2024-2025 download wrote
(data/playground/events_insample.csv and nulls_insample.csv) when they exist. Otherwise (the judges' sealed window,
or a clean machine) it builds them through the notebook's own functions: every tag's disclosures, EDGAR acceptance
times inside the notebook's build_events, and two matched ordinary days per filing (seed 20261003). Only with
allow_fetch=True may those functions download what is missing, and then the option data for every row are
downloaded too, one request at a time through the notebook's api_get, before any step computes anything.

Labels and date guards (.claude/ctx/06_window_guard.md; the starter notebook's cells 9, 10 and 48; Massive: only
2024-2025 may be looked at):

    insample   the test. Every t_pre, t_0, gap, ordinary-day, panel and exit date lies in [2024-01-01, 2026-01-01)
               and outside the notebook's sealed placeholder HOLDOUT_START..HOLDOUT_END.
    holdout    the judges' sealed window: only when RUN_HOLDOUT is True; any dates; downloads allowed.
    oos        2026 (OOS_START..OOS_END), once, by a human: only when RUN_OOS is True; downloads allowed with
               allow_fetch, after a combined request estimate that must stay under a cap (default 40,000).
    discovery, dryrun   retired: any call refuses. (A dry run of the judges' path is `dry_run()`, label insample.)

Each module also checks its own dates.

Usage
    in the notebook:  pipeline.run_oldnews(start, end, label, allow_fetch=..., NB=globals())
    in a terminal:    .venv/Scripts/python.exe src/oldnews/pipeline.py --label insample
                      .venv/Scripts/python.exe src/oldnews/pipeline.py --dry-run     (quiet; scratch folder; deleted)
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import inspect
import io
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent.parent
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

OUT = ROOT / "data" / "oldnews"
RAW = OUT / "raw"
PLAYGROUND = ROOT / "data" / "playground"

SEED = 20261003
OOS_START = "2026-01-01"
N_NULLS, NULL_WINDOW, NULL_GAP = 2, 60, 5      # ordinary days: per filing, +/- sessions, min sessions from any 8-K
REQUESTS_PER_ROW = 28                          # upper bound per (ticker, pre-event session), all three buckets
MEASURED_RPS = 6.0                             # requests per second, sequential, on our key (a conservative figure)
FILINGS_PER_MONTH = 75                         # rough count of TOP_100 8-Ks (all tags) per month, for the step-0 estimate only
OOS_REQUEST_CAP = 40_000                       # the one-time 2026 run refuses a larger request estimate unless the caller raises it
PEOPLE_PLACEBO_SHARE = 0.42                    # rough share of TOP_100 8-Ks in the people and placebo sets (2024-25: 751 of ~1,800)
PANEL_EXTRA = 400                              # at most this many extra ticker-dates priced only for the market panel (r_mkt)
FAILURES_TO_STOP = 5                           # consecutive API failures before a download stops (CLAUDE.md rule 17)
ZREF_FROZEN = SRC / "oldnews" / "zref_frozen_insample.json"   # the z-score constants (classify.frozen_path() wins); read here, never written. The old zref_frozen.json is a retired archive and is never read.
FREEZING_LABEL = "insample"                    # the test label: it may run steps 0-2 before the constants exist (they are frozen from its gap tables)
SEALED_PLACEHOLDER = ("2023-06-01", "2023-08-31")    # the notebook's HOLDOUT_START..HOLDOUT_END when NB does not say
RETIRED_LABELS = ("discovery", "dryrun")
RETIRED_MESSAGE = ("2022-2023 is outside the allowed 2024-2025 window and overlaps the sealed placeholder "
                   "(2023-06-01..2023-08-31)")
STAGES = ("inputs", "events", "measure", "classify", "tests", "trade", "figures")


@dataclass(frozen=True)
class Window:
    start: str
    end: str
    hard_stop: str | None             # no date on or after this day (None: the judges' or OOS window)
    clip_before: str | None = None    # no option bar, so no exit, on or after this day
    first_date: str | None = None     # no date before this day


WINDOWS = {"insample": Window("2024-01-01", "2025-12-31", hard_stop="2026-01-01", clip_before="2026-01-01",
                              first_date="2024-01-01")}


def check_label(label: str) -> None:
    """Refuse the retired labels, everywhere (pipeline, figures, report): no code path reads 2022-2023."""
    if label in RETIRED_LABELS:
        raise PermissionError(f"label {label!r} is retired: {RETIRED_MESSAGE}.")


_QUIET = False


def log(msg: str) -> None:
    if not _QUIET:
        print(f"[oldnews {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class FrozenConstantsMissing(RuntimeError):
    """src/oldnews/zref_frozen_insample.json is not there or is unusable. Not a FileNotFoundError, so a notebook cell never
    mistakes it for "this window is not in the cache"."""


def frozen_path() -> Path:
    """Where classify reads the frozen constants (one source of truth), else the default location."""
    try:
        from oldnews import classify
        return Path(classify.frozen_path("insample"))
    except (ImportError, AttributeError):
        return ZREF_FROZEN


def check_frozen_constants(path: Path | None = None) -> str:
    """Read-only preflight: the z-score constants (frozen once, committed) must exist and be valid. Returns the
    first 12 characters of the file's SHA-256 for the run log. This pipeline never writes the file: the only
    writer is classify.freeze, run once by a person."""
    path = Path(path or frozen_path())
    if not path.is_file():
        raise FrozenConstantsMissing(
            f"{path} is missing. It holds the z-score constants, frozen once so that every window (the 2024-2025 "
            "test and the judges' sealed window) is standardised the same way; it is committed with the code. "
            "Restore it with `git checkout -- src/oldnews/zref_frozen_insample.json`. If it was never created, a person runs "
            "oldnews.classify.freeze('insample') once and commits it (the pipeline does not create it).")
    try:
        from oldnews import classify
        classify.load_frozen(path)                 # the stats module's own validation, read only
    except ImportError:
        pass
    except (ValueError, KeyError, OSError) as e:
        raise FrozenConstantsMissing(f"{path} is not usable ({e}). Restore it with "
                                     "`git checkout -- src/oldnews/zref_frozen_insample.json`.") from e
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def frozen_window_warning(path: Path | None = None) -> str | None:
    """A warning (not a refusal) when the frozen file says it was computed from dates before 2024-01-01."""
    path = Path(path or frozen_path())
    try:
        first = json.loads(path.read_text(encoding="utf-8"))["window_t_0"]["first"]
        if pd.Timestamp(first) < pd.Timestamp("2024-01-01"):
            return (f"WARNING: the frozen z-score constants were computed from dates as early as {first}, before the "
                    "allowed 2024-2025 window; re-freeze them (classify.freeze) before this run is used in a deliverable")
    except (OSError, KeyError, ValueError, TypeError):
        pass
    return None


def estimate(n_requests: int, rps: float = MEASURED_RPS) -> str:
    """"~N requests, about M minutes" at the sequential rate measured on our key."""
    minutes = n_requests / rps / 60
    span = f"{minutes:.0f} minutes" if minutes < 90 else f"{minutes / 60:.1f} hours"
    return f"~{n_requests:,} requests, about {span} at {rps:g} requests per second"


def market_panel_rows(rows: pd.DataFrame, raw_ev: pd.DataFrame, raw_nu: pd.DataFrame,
                      max_extra: int = PANEL_EXTRA, seed: int = SEED, w: Window | None = None,
                      NB: dict | None = None) -> pd.DataFrame:
    """Extra (ticker, t_pre, t_0) rows priced only to widen r_mkt's ticker panel: a seeded sample, at most
    `max_extra`, of the window's filings and ordinary days that the run does not already price. The rows the run
    prices are in the panel anyway (measure adds every one it prices). With a guarded window `w` (insample) only
    rows with every date in [first_date, hard_stop) and outside the sealed placeholder qualify: the panel never
    uses 2022-2023 rows, whatever the prepared tables hold."""
    cols = ["ticker", "t_pre", "t_0"]
    have = set(zip(rows["ticker"], pd.to_datetime(rows["t_pre"])))
    pool = pd.concat([raw_ev[cols], raw_nu[cols]], ignore_index=True)
    pool["t_pre"], pool["t_0"] = pd.to_datetime(pool["t_pre"]), pd.to_datetime(pool["t_0"])
    if w is not None:
        pool = pool[in_bounds(pool, w, NB)]
    pool = pool.drop_duplicates(["ticker", "t_pre"]).sort_values(["t_pre", "ticker"])
    pool = pool[[(t, d) not in have for t, d in zip(pool["ticker"], pool["t_pre"])]].reset_index(drop=True)
    if len(pool) > max_extra:
        pool = pool.iloc[np.sort(np.random.default_rng(seed).choice(len(pool), size=max_extra, replace=False))]
    return pool.reset_index(drop=True)


def flag(s: pd.Series) -> pd.Series:
    """A 0/1, True/False or "True"/"False" column as booleans (CSV round trips change the type)."""
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes"])


# ---------------------------------------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------------------------------------
def sealed_range(NB: dict | None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The notebook's sealed placeholder (HOLDOUT_START..HOLDOUT_END), else the starter's default."""
    NB = NB or {}
    return (pd.Timestamp(NB.get("HOLDOUT_START") or SEALED_PLACEHOLDER[0]),
            pd.Timestamp(NB.get("HOLDOUT_END") or SEALED_PLACEHOLDER[1]))


def window_rules(NB: dict, start: str, end: str, label: str) -> Window:
    """The date rules for this run. Refuses anything the hard rules forbid; never switches a guard on."""
    for name, d in (("start", start), ("end", end)):
        try:
            pd.Timestamp(d)
        except ValueError:
            raise ValueError(f"{name} = {d!r} is not a real date") from None
    if not start < end:
        raise ValueError(f"start {start} must be before end {end}")
    check_label(label)
    if label == "oos":
        if NB.get("RUN_OOS") is not True:
            raise PermissionError("the out-of-sample window runs only after a human sets RUN_OOS = True (section 2)")
        o0, o1 = NB.get("OOS_START", OOS_START), NB.get("OOS_END")
        if start < o0 or (o1 and end > o1):
            raise PermissionError(f"label 'oos' runs on {o0}..{o1} only; got {start}..{end}")
        print(NB.get("OOS_WARNING", "WARNING: OUT-OF-SAMPLE RUN"), file=sys.stderr)
        return Window(start, end, hard_stop=None)
    if label == "holdout":
        if NB.get("RUN_HOLDOUT") is not True:
            raise PermissionError("the sealed window runs only after the judges set RUN_HOLDOUT = True (section 2)")
        return Window(start, end, hard_stop=None)
    if label != "insample":
        raise PermissionError(f"unknown label {label!r}: the labels are insample (the test), holdout (the judges' "
                              "sealed window) and oos (one human run); discovery and dryrun are retired")
    rules = WINDOWS["insample"]
    if start < rules.first_date or end >= rules.hard_stop:
        raise PermissionError(f"label 'insample' allows filings from {rules.first_date} to before {rules.hard_stop} only; "
                              f"got {start}..{end}")
    s0, s1 = sealed_range(NB)
    if pd.Timestamp(start) <= s1 and pd.Timestamp(end) >= s0:
        raise PermissionError(f"{start}..{end} overlaps the sealed placeholder {s0.date()}..{s1.date()}; refusing")
    return replace(rules, start=start, end=end)


DATE_COLUMNS = ("t_pre", "t_0", "gap_start")


def in_bounds(df: pd.DataFrame, w: Window, NB: dict | None = None) -> pd.Series:
    """True where every date column present (t_pre, t_0, gap_start) lies in [first_date, hard_stop) and outside the
    sealed placeholder; a missing date (NaT) does not exclude a row. All True for the unguarded windows."""
    ok = pd.Series(True, index=df.index)
    if w.hard_stop is None:
        return ok
    s0, s1 = sealed_range(NB)
    for col in DATE_COLUMNS:
        if col in df:
            d = pd.to_datetime(df[col])
            bad = (d >= pd.Timestamp(w.hard_stop)) | ((d >= s0) & (d <= s1))
            if w.first_date:
                bad |= d < pd.Timestamp(w.first_date)
            ok &= ~bad.fillna(False)
    return ok


def check_dates(df: pd.DataFrame, w: Window, what: str, NB: dict | None = None) -> None:
    """Hard fail: no pre-event, entry or gap date outside the window or inside the sealed placeholder."""
    if w.hard_stop is None or df is None or df.empty:
        return
    bad = ~in_bounds(df, w, NB)
    if bad.any():
        raise PermissionError(f"{int(bad.sum())} {what} rows have a t_pre, t_0 or gap_start outside "
                              f"[{w.first_date}, {w.hard_stop}) or inside the sealed placeholder; refusing. The module "
                              "that built this table must drop its window-edge rows.")


@contextlib.contextmanager
def sandbox(NB: dict, allow_fetch: bool, clip_before: str | None):
    """Offline (unless allow_fetch) and clipped for the duration of the block, then restored exactly.

    In the notebook NB is the notebook's own globals(); without the restore, a run (or a module that calls
    go_offline itself) would leave the notebook unable to fetch, or clipped, in every later cell."""
    from playground.measure import CacheMiss, clip_to, go_offline

    session = NB["SESSION"]
    had_get, old_get = "get" in vars(session), vars(session).get("get")
    missing = object()
    saved = {k: NB.get(k, missing) for k in ("option_bars", "LAST_SESSION", "_clipped_to", "fetch_acceptance_time")}
    try:
        if clip_before:
            clip_to(NB, clip_before)
        if not allow_fetch:
            go_offline(NB)
            sec = saved["fetch_acceptance_time"]
            if sec is not missing:
                def cached_acceptance_time(url: str):
                    key = Path(NB["CACHE_DIR"]) / ("sec_" + hashlib.sha1(url.encode()).hexdigest() + ".txt")
                    if not key.exists():
                        raise CacheMiss(f"EDGAR header not in cache: {url}")
                    return sec(url)

                NB["fetch_acceptance_time"] = cached_acceptance_time
        yield
    finally:
        if had_get:
            session.get = old_get
        else:
            vars(session).pop("get", None)
        for k, v in saved.items():
            if v is missing:
                NB.pop(k, None)
            else:
                NB[k] = v


def restoring(NB: dict):
    """Run a block that may switch the notebook offline (events.build, measure.measure) and undo it after."""
    return sandbox(NB, allow_fetch=True, clip_before=None)


# ---------------------------------------------------------------------------------------------------------
# Step 0 · inputs: every TOP_100 8-K in the window and its matched ordinary days
# ---------------------------------------------------------------------------------------------------------
EVENT_DATES = ["filing_date", "accepted_at", "t_0", "t_pre"]
NULL_DATES = ["filing_date", "t_0", "t_pre"]


ID_COLS = {"row_id": str, "event_row_id": str, "ticker": str, "accession_number": str}


def _read(path: Path, dates: list[str]) -> pd.DataFrame:
    """read_csv where only empty cells are missing (pandas' defaults would turn a ticker such as "NA" into NaN)."""
    df = pd.read_csv(path, dtype=ID_COLS, keep_default_na=False,
                     na_values=[""])
    for c in dates:
        if c in df:
            df[c] = pd.to_datetime(df[c])
    return df


def has_prepared(source: str) -> bool:
    return (PLAYGROUND / f"events_{source}.csv").exists() and (PLAYGROUND / f"nulls_{source}.csv").exists()


def slice_prepared(source: str, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The download's tables for `source`, cut to filings dated start..end (and those filings' ordinary days)."""
    ev = _read(PLAYGROUND / f"events_{source}.csv", EVENT_DATES)
    nu = _read(PLAYGROUND / f"nulls_{source}.csv", NULL_DATES)
    ev = ev[(ev.filing_date >= pd.Timestamp(start)) & (ev.filing_date <= pd.Timestamp(end))].reset_index(drop=True)
    keep = pd.MultiIndex.from_frame(ev[["ticker", "filing_date"]])
    nu = nu[pd.MultiIndex.from_frame(nu[["ticker", "filing_date"]]).isin(keep)].reset_index(drop=True)
    return ev, nu


def build_from_api(NB: dict, start: str, end: str, hard_stop: str | None, allow_fetch: bool = False
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The download's event and ordinary-day tables (src/discovery_fetch.py), for any window.

    Every tag in the taxonomy goes through the notebook's build_events, so acceptance times and the sessions
    they imply (t_0, t_pre) come from the same code in every window, and the disclosures and EDGAR headers that
    step 1 reads land in the cache. One row per (filer, filing date) with all its tags; then two ordinary days
    per filing, drawn with the download's rule and seed."""
    taxonomy = pd.DataFrame(NB["api_get_all"]("/stocks/taxonomies/vX/disclosures", {"limit": 1000}))
    tags = sorted(taxonomy["tertiary_category"])
    months = max((pd.Timestamp(end) - pd.Timestamp(start)).days / 30.4, 1.0)
    headers = int(round(months * FILINGS_PER_MONTH))
    log(f"0 inputs: {len(tags)} disclosure queries (about 1 to 3 requests each) plus one SEC EDGAR header per TOP_100 "
        f"filing, about {headers:,} (rough); " + estimate(len(tags) * 2 + headers, 5.0)
        + ". Each response is cached after the first run." if allow_fetch else
        f"0 inputs: {len(tags)} disclosure queries and one cached EDGAR header per TOP_100 filing (cache only)")
    parts, quiet = [], io.StringIO()
    for tag in tags:
        with contextlib.redirect_stdout(quiet):        # build_events prints two lines per tag
            try:
                ev = NB["build_events"](tag, start, end, NB["TOP_100"])
            except KeyError:                           # the starter's build_events fails when no filer has tickers
                continue
        if len(ev):
            parts.append(ev.assign(tag=tag))
    if not parts:
        return (pd.DataFrame(columns=["cik", "ticker", "tags", *EVENT_DATES]),
                pd.DataFrame(columns=["ticker", "round", *NULL_DATES]))
    allev = pd.concat(parts, ignore_index=True)
    joined = allev.groupby(["cik", "filing_date"])["tag"].apply(lambda s: "|".join(sorted(set(s)))).rename("tags")
    events = (allev.sort_values("accepted_at").drop_duplicates(["cik", "filing_date"])
                   .drop(columns=["tag", "supporting_text"]).merge(joined, on=["cik", "filing_date"])
                   .sort_values(["filing_date", "ticker"]).reset_index(drop=True))
    if hard_stop:
        events = events[events.t_0 < pd.Timestamp(hard_stop)].reset_index(drop=True)
    log(f"0 inputs: {len(events)} TOP_100 filings in {start}..{end}; {events.accepted_at.notna().sum()} with an "
        "EDGAR acceptance time")
    return events, draw_nulls(NB, events, start, end, hard_stop)


def draw_nulls(NB: dict, events: pd.DataFrame, start: str, end: str, hard_stop: str | None) -> pd.DataFrame:
    """Ordinary days exactly as the download drew them: same ticker, within +/-60 sessions of the
    filing-date session, more than 5 sessions from both sessions any 8-K of that ticker can enter at, two per
    filing, seed 20261003, never outside the window nor after the last completed session."""
    CAL = NB["CAL"]
    stop = pd.Timestamp(hard_stop) if hard_stop else pd.Timestamp(end) + pd.Timedelta(days=1)
    pos = {d: i for i, d in enumerate(CAL)}
    lo_i = CAL.searchsorted(pd.Timestamp(start))
    hi_i = min(CAL.searchsorted(stop) - 1, pos[NB["LAST_SESSION"]])
    fd_i = lambda d: pos[NB["session_on_or_after"](d)]            # noqa: E731
    inside = events[(events.filing_date >= pd.Timestamp(start)) & (events.filing_date <= pd.Timestamp(end))]
    busy = {t: {fd_i(d) + k for d in g["filing_date"] for k in (0, 1)} for t, g in inside.groupby("ticker")}
    rng = np.random.default_rng(SEED)
    rows = []
    for e in events.itertuples(index=False):
        i0 = fd_i(e.filing_date)
        cand = [i for i in range(max(i0 - NULL_WINDOW, lo_i + 1), min(i0 + NULL_WINDOW, hi_i) + 1)
                if all(abs(i - j) > NULL_GAP for j in busy.get(e.ticker, ()))]
        picks = rng.choice(cand, size=min(N_NULLS, len(cand)), replace=False) if cand else []
        for r, i in enumerate(sorted(picks), 1):
            rows.append({"ticker": e.ticker, "filing_date": e.filing_date, "round": r, "t_0": CAL[i], "t_pre": CAL[i - 1]})
    return pd.DataFrame(rows, columns=["ticker", "filing_date", "round", "t_0", "t_pre"])


def prepare_inputs(NB: dict, w: Window, label: str, source: str | None, allow_fetch: bool = False
                   ) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Step 0. Returns (filings, ordinary days, where they came from) and writes them to data/oldnews/raw/.
    Rows with a date outside the window (a filing in the first days of 2024 whose last close before acceptance was in
    2023) are dropped here and counted."""
    if source != "api" and has_prepared(source or label):
        src = source or label
        ev, nu = slice_prepared(src, w.start, w.end)
        origin = f"the {src} download's tables" + (f", cut to {w.start}..{w.end}" if src != label else "")
    else:
        ev, nu = build_from_api(NB, w.start, w.end, w.hard_stop, allow_fetch)
        origin = "the API, through the notebook's own cached functions"
    ok_ev, ok_nu = in_bounds(ev, w, NB), in_bounds(nu, w, NB)
    if not (ok_ev.all() and ok_nu.all()):
        log(f"0 inputs: dropped {int((~ok_ev).sum())} filings and {int((~ok_nu).sum())} ordinary days with a date outside "
            f"[{w.first_date}, {w.hard_stop}) or inside the sealed placeholder (window edge)")
        ev, nu = ev[ok_ev].reset_index(drop=True), nu[ok_nu].reset_index(drop=True)
    check_dates(ev, w, "filing", NB)
    check_dates(nu, w, "ordinary-day", NB)
    RAW.mkdir(parents=True, exist_ok=True)
    ev.to_csv(RAW / f"events_{label}.csv", index=False)
    nu.to_csv(RAW / f"nulls_{label}.csv", index=False)
    (RAW / f"inputs_{label}.json").write_text(json.dumps(
        {"start": w.start, "end": w.end, "origin": origin, "built": time.strftime("%Y-%m-%d %H:%M:%S"),
         "filings": len(ev), "ordinary_days": len(nu)}, indent=1))
    return ev, nu, origin


def events_estimate(NB: dict, w: Window, label: str) -> tuple[int, str]:
    """events.estimate_requests: uncached Massive pages + EDGAR headers + full texts. When some disclosures are not
    cached, their filings are unknown, so a rough upper bound for their SEC requests is added and said so."""
    months = max((pd.Timestamp(w.end) - pd.Timestamp(w.start)).days / 30.4, 1.0)
    filings = int(round(months * FILINGS_PER_MONTH))
    try:
        from oldnews import events as E
        est = _call(E.estimate_requests, window=(w.start, w.end), label=label, NB=NB)
    except (ImportError, AttributeError):
        return 2 * 120 + filings + int(round(filings * PEOPLE_PLACEBO_SHARE)), "rough upper bound (no events.estimate_requests)"
    n = int(est.get("massive_pages", 0)) + int(est.get("sec_headers", 0)) + int(est.get("sec_full_text", 0))
    note = (f"events.estimate_requests: {est.get('massive_pages', 0)} Massive pages, {est.get('sec_headers', 0)} EDGAR "
            f"headers, {est.get('sec_full_text', 0)} full texts")
    if est.get("unknown_filings"):
        extra = filings + int(round(filings * PEOPLE_PLACEBO_SHARE)) + 2 * len(est.get("massive_more_pages_possible", []))
        n += extra
        note += f", plus ~{extra:,} for filings and later pages not yet known (rough)"
    return n, note


def measure_estimate(NB: dict, label: str, rows: pd.DataFrame | None, panel: pd.DataFrame | None,
                     n_rows: int, panel_max_extra: int) -> tuple[int, str]:
    """measure.estimate_requests on the run's rows and panel when they exist (exact count of uncached keys), else a
    rough upper bound from the projected number of rows."""
    if rows is not None:
        try:
            from oldnews import measure as M
            est = _call(M.estimate_requests, rows, label, NB, panel_rows=panel)
            return int(est["requests_estimate"]), (f"measure.estimate_requests: {est['uncached']:,} of {est['keys']:,} "
                                                   "(ticker, pre-event session) keys not cached")
        except (ImportError, AttributeError, KeyError, TypeError):
            pass
        keys = rows[["ticker", "t_pre"]].drop_duplicates().shape[0] + (0 if panel is None else len(panel))
        return keys * REQUESTS_PER_ROW, "upper bound: every key uncached"
    return (n_rows + panel_max_extra) * REQUESTS_PER_ROW, "rough upper bound from the projected rows"


def check_estimate(stage: str, w: Window, label: str, parts: list[tuple[str, int, str]], max_requests: int | None) -> int:
    """Print the combined estimate and refuse above the cap (CLAUDE.md rule 18), before the fetch it covers."""
    total = sum(n for _, n, _ in parts)
    detail = " + ".join(f"{name} ~{n:,} ({note})" for name, n, note in parts)
    print(f"[oldnews] request estimate {stage}, {label} {w.start}..{w.end}: {detail} = ~{total:,}; " + estimate(total)
          + (f"; cap {max_requests:,}" if max_requests is not None else ""), flush=True)
    if max_requests is not None and total > max_requests:
        raise RuntimeError(f"the request estimate (~{total:,}) exceeds the cap of {max_requests:,}; nothing more was "
                           "fetched. Raise --max-requests (after asking, CLAUDE.md rule 18) or narrow the window.")
    return total


def preflight_estimate(NB: dict, w: Window, label: str, max_requests: int | None, panel_max_extra: int) -> dict:
    """Before anything is fetched: the events step's estimate plus a projected measure estimate (its rows are known
    only after the events step; the exact count is checked again before the option download)."""
    months = max((pd.Timestamp(w.end) - pd.Timestamp(w.start)).days / 30.4, 1.0)
    rows = int(round(months * FILINGS_PER_MONTH * PEOPLE_PLACEBO_SHARE)) * (1 + N_NULLS)
    ev, ev_note = events_estimate(NB, w, label)
    me, me_note = measure_estimate(NB, label, None, None, rows, panel_max_extra)
    total = check_estimate("before any fetch", w, label, [("events", ev, ev_note), ("measure", me, me_note)], max_requests)
    return {"events": ev, "measure_projected": me, "total": total}


def warm_cache(NB: dict, keys: pd.DataFrame, max_requests: int | None = None) -> None:
    """With allow_fetch only: price every (ticker, t_pre) once through the notebook's price_event, one request
    at a time, so its chain and option bars are cached before measure (which never fetches) runs. Prints the
    request and runtime estimate first and refuses if it exceeds max_requests (CLAUDE.md rule 18). Stops after
    FAILURES_TO_STOP API failures in a row (rule 17); a data quirk in one row is skipped."""
    keys = keys.drop_duplicates(["ticker", "t_pre"]).reset_index(drop=True)
    n_req = len(keys) * REQUESTS_PER_ROW
    log(f"2 download: option data for {len(keys):,} (ticker, pre-event session) pairs; at most "
        + estimate(n_req) + " if none is cached (cached ones cost nothing)")
    if max_requests is not None and n_req > max_requests:
        raise RuntimeError(f"the estimate of {n_req:,} requests exceeds max_requests = {max_requests:,}; "
                           "nothing was fetched. Raise max_requests (after asking, CLAUDE.md rule 18) or narrow the window.")
    req = NB["requests"]
    fails, t0 = 0, time.time()
    for i, k in enumerate(keys.itertuples(index=False), 1):
        try:
            NB["price_event"](k.ticker, pd.Timestamp(k.t_pre), pd.Timestamp(k.t_0), pd.Timestamp(k.t_0),
                              NB["EXPIRY_BUCKETS"], NB["OTM_GRID"])
            fails = 0
        except req.RequestException as e:
            fails += 1
            log(f"  {k.ticker} {pd.Timestamp(k.t_pre).date()}: {type(e).__name__} ({fails} in a row)")
            if fails >= FAILURES_TO_STOP:
                raise RuntimeError("repeated API failures; stopping (never hammer the API). Rerun later: "
                                   "everything downloaded so far is cached.") from e
            time.sleep(30 * fails)
        except Exception as e:  # noqa: BLE001 (one row's data quirk must not stop the download)
            log(f"  {k.ticker} {pd.Timestamp(k.t_pre).date()}: skipped ({type(e).__name__}: {str(e)[:80]})")
        if i % 100 == 0:
            log(f"  {i:,}/{len(keys):,} pairs, {(time.time() - t0) / 60:.1f} min")


# ---------------------------------------------------------------------------------------------------------
# Steps 1-5 · the other modules
# ---------------------------------------------------------------------------------------------------------
def _call(fn: Callable, *args, **kwargs):
    """Call fn with only the keyword arguments it accepts (a stand-in or a newer module may take fewer)."""
    params = inspect.signature(fn).parameters
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        kwargs = {k: v for k, v in kwargs.items() if k in params}
    return fn(*args, **kwargs)


@dataclass
class Steps:
    """The five steps. Defaults are the oldnews modules; tests pass synthetic stand-ins."""
    events: Callable[..., tuple[pd.DataFrame, pd.DataFrame]] | None = None
    measure: Callable[..., tuple[pd.DataFrame, pd.DataFrame]] | None = None
    classify: Callable[..., pd.DataFrame] | None = None
    tests: Callable[..., Any] | None = None
    trade: Callable[..., Any] | None = None
    missing: list[str] = field(default_factory=list)


STEP_FUNCTIONS = {            # step -> (module in src/oldnews, function names to try in order)
    "events": ("events", ["build"]),
    "measure": ("measure", ["measure"]),
    "classify": ("classify", ["run"]),
    "tests": ("tests", ["run"]),
    "trade": ("trade", ["run"]),
}


def _missing(name: str, err: Exception) -> Callable:
    def stub(*a, **k):
        raise NotImplementedError(f"step '{name}' is not available yet ({err})")
    return stub


def default_steps() -> Steps:
    """Resolve each step from its module; a module that is not there yet becomes a stub that says so."""
    steps = Steps()
    for step, (module, names) in STEP_FUNCTIONS.items():
        try:
            m = importlib.import_module(f"oldnews.{module}")
            fn = next(getattr(m, n) for n in names if hasattr(m, n))
        except Exception as e:  # noqa: BLE001 (any import failure means the step is not ready)
            fn = _missing(step, e)
            steps.missing.append(step)
        setattr(steps, step, fn)
    return steps


@contextlib.contextmanager
def superset_disclosures(NB: dict, superset: Window | None):
    """Cache only: answer a disclosure query for a sub-window from the cached response for the superset window,
    filtered by filing date. The API cache is keyed by the full URL, so a sub-window's own query (the dry run's
    2024-07 to 2024-12) is not in it, but the full 2024-2025 window's responses are, and a narrower query returns
    exactly the rows of the wider one that fall inside its dates. A query the cache cannot answer still raises
    CacheMiss."""
    from playground.measure import CacheMiss

    if superset is None or "fetch_disclosures" not in NB:
        yield
        return
    orig = NB["fetch_disclosures"]

    def fetch(tag: str, start: str, end: str) -> pd.DataFrame:
        try:
            return orig(tag, start, end)
        except CacheMiss:
            df = orig(tag, superset.start, superset.end)
            if df.empty:
                return df
            fd = pd.to_datetime(df["filing_date"])
            return df[(fd >= pd.Timestamp(start)) & (fd <= pd.Timestamp(end))].reset_index(drop=True)

    NB["fetch_disclosures"] = fetch
    try:
        yield
    finally:
        NB["fetch_disclosures"] = orig


def full_text_mode(label: str, allow_fetch: bool) -> str | None:
    """The word score's text source, as the committed test plan says (commit 34a11a4): the full EDGAR filing text.
    insample reads the cached full texts; holdout and oos (each behind its notebook switch) fetch the missing ones
    through events.SecFetcher, the polite SEC client, when allow_fetch, else read the cache."""
    if label == "insample":
        return "cache"
    if label in ("holdout", "oos"):
        return "fetch" if allow_fetch else "cache"
    return None


def events_step(fn: Callable, NB: dict, label: str, w: Window, superset: Window | None = None,
                allow_fetch: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """events.build for this run's own dates: `window=(first filing date, last filing date)`. The module owns the
    date guard and the switch each label needs (RUN_HOLDOUT for the sealed window, RUN_OOS for 2026), the
    acceptance-time sessions and the look-backs, and reads the ordinary days from data/oldnews/raw/. A sub-window
    gets exactly what the judges' window gets: its own dates, no history from before them. `superset` (the full
    2024-2025 window) lets a cache-only run answer its disclosure queries from the cached wider window."""
    with restoring(NB), superset_disclosures(NB, None if label == "holdout" else superset):
        return _call(fn, label, NB=NB, window=(w.start, w.end), out_dir=OUT, playground=RAW,
                     full_text=full_text_mode(label, allow_fetch))


def measurement_rows(events: pd.DataFrame, nulls: pd.DataFrame) -> pd.DataFrame:
    """The events and their ordinary days as one table of rows to price (00_shared.md columns)."""
    cols = ["row_id", "kind", "ticker", "t_pre", "t_0", "gap_start", "n_gap"]
    rows = pd.concat([events.reindex(columns=cols), nulls.reindex(columns=cols)], ignore_index=True)
    for c in ("t_pre", "t_0", "gap_start"):
        rows[c] = pd.to_datetime(rows[c])
    return rows


def _summary(folder: Path) -> str | None:
    f = folder / "summary.md"
    return f.read_text(encoding="utf-8") if f.exists() else None


# ---------------------------------------------------------------------------------------------------------
# The whole test
# ---------------------------------------------------------------------------------------------------------
@contextlib.contextmanager
def _output_root(out_dir: Path | None):
    """Point every output at `out_dir` for the block (a scratch folder for the dry run), then restore."""
    global OUT, RAW
    if out_dir is None:
        yield
        return
    saved = OUT, RAW
    OUT = Path(out_dir)
    RAW = OUT / "raw"
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        yield
    finally:
        OUT, RAW = saved


@contextlib.contextmanager
def _quiet(on: bool):
    """Silence this module's log and every step's prints (the stats module prints its headline result)."""
    global _QUIET
    if not on:
        yield
        return
    saved, _QUIET = _QUIET, True
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        _QUIET = saved


def run_oldnews(start: str, end: str, label: str, allow_fetch: bool = False, NB: dict | None = None,
                source: str | None = None, market_panel: bool = True, panel_max_extra: int = PANEL_EXTRA,
                max_requests: int | None = None, steps: Steps | None = None, figures: bool = True,
                out_dir: Path | None = None, quiet: bool = False, stages: dict | None = None) -> dict:
    """Run the hypothesis test on filings dated start..end and return every table it produced.

    label         "insample" (the test, 2024-2025), "holdout" (the judges' sealed window, needs RUN_HOLDOUT) or
                  "oos" (one human run, needs RUN_OOS; may fetch, capped). "discovery" and "dryrun" are retired.
    allow_fetch   False: cache only; a missing response stops the run. True: the notebook's own cached API
                  functions download what is missing first (the judges' path). Never set it for the test window.
    NB            the notebook namespace (pass globals() in the notebook); loaded from the notebook if None.
    source        "api" to build the inputs through the API functions even when prepared tables exist.
    market_panel  also price a seeded sample of at most `panel_max_extra` (default 400) ticker-dates beyond the
                  rows the run prices, so r_mkt (the market's return over the gap) is the median across more
                  TOP_100 tickers (test plan). For insample only rows inside 2024-2025 qualify.
    max_requests  with allow_fetch: refuse before fetching if the printed estimate exceeds this many requests
                  (oos: 40,000 unless given).
    out_dir       write everything here instead of data/oldnews/ (a scratch folder).
    quiet         print nothing (no log, no step output, no results).
    stages        a dict to fill with each stage's status ("ran") as it finishes, even if a later one fails.

    Needs src/oldnews/zref_frozen_insample.json (read, never written); without it the run stops before fetching anything.
    """
    check_label(label)
    with _output_root(out_dir), _quiet(quiet):
        return _run_oldnews(start, end, label, allow_fetch, NB, source, market_panel, panel_max_extra, max_requests,
                            steps, figures, {} if stages is None else stages)


def _run_oldnews(start, end, label, allow_fetch, NB, source, market_panel, panel_max_extra, max_requests, steps,
                 figures, stages: dict) -> dict:
    t_start = time.time()
    if NB is None:
        import nb
        NB = nb.load()
    w = window_rules(NB, start, end, label)
    if label == "oos" and max_requests is None:
        max_requests = OOS_REQUEST_CAP
    try:
        zref_sha: str | None = check_frozen_constants()
    except FrozenConstantsMissing as e:
        if label != FREEZING_LABEL:
            raise FrozenConstantsMissing(f"{e} Nothing was fetched or computed.") from e
        zref_sha = None
        log("the frozen z-score constants do not exist yet: steps 0 to 2 will run, then the run stops before classify")
    warning = frozen_window_warning() if zref_sha else None
    if warning:
        log(warning)
    if source and source != "api" and not has_prepared(source):
        raise FileNotFoundError(f"no prepared tables for {source!r} in {PLAYGROUND}")
    full = WINDOWS["insample"]
    superset = full if label == "insample" and (w.start, w.end) != (full.start, full.end) else None
    steps = steps or default_steps()
    if steps.missing:
        log(f"not available yet: {', '.join(steps.missing)} (the run stops at the first of them)")
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"{label}: filings {w.start}..{w.end}; "
        f"{'every date in [' + w.first_date + ', ' + w.hard_stop + '), outside the sealed placeholder' if w.hard_stop else 'the judges/OOS window'}; "
        f"{'no exit on or after ' + w.clip_before if w.clip_before else 'exits up to the last session'}; "
        f"{'fetch allowed' if allow_fetch else 'cache only'}; frozen z-score constants {zref_sha or 'not yet'}")

    result: dict[str, Any] = {"label": label, "start": w.start, "end": w.end, "window": w, "stages": stages}
    stop = w.hard_stop              # None for the sealed and out-of-sample windows: measure decides by label and switch
    if allow_fetch:
        result["request_estimate"] = preflight_estimate(NB, w, label, max_requests, panel_max_extra)
    with sandbox(NB, allow_fetch, w.clip_before):
        raw_ev, raw_nu, origin = prepare_inputs(NB, w, label, source, allow_fetch)
        stages["inputs"] = "ran"
        log(f"0 inputs: {len(raw_ev)} TOP_100 filings and {len(raw_nu)} ordinary days, from {origin}")

        ev, nu = events_step(steps.events, NB, label, w, superset, allow_fetch)
        check_dates(ev, w, "event", NB)
        check_dates(nu, w, "ordinary-day", NB)
        result.update(events=ev, nulls=nu)
        stages["events"] = "ran"
        log(f"1 events: {len(ev)} filings in the people and placebo sets, {len(nu)} matched ordinary days")

        rows = measurement_rows(ev, nu)
        panel = market_panel_rows(rows, raw_ev, raw_nu, panel_max_extra, w=w, NB=NB) if market_panel else None
        if panel is not None:
            log(f"2 measure: the market panel adds {len(panel)} ticker-dates to the {len(rows)} rows priced "
                f"(at most {panel_max_extra}, seed {SEED})")
        if allow_fetch:
            me, me_note = measure_estimate(NB, label, rows, panel, len(rows), panel_max_extra)
            result["request_estimate"]["measure"] = me
            check_estimate("before the option download", w, label,
                           [("events (already fetched)", result["request_estimate"]["events"], "estimate"),
                            ("measure", me, me_note)], max_requests)
            warm_cache(NB, pd.concat([rows[["ticker", "t_pre", "t_0"]].dropna(), panel], ignore_index=True)
                       if panel is not None else rows[["ticker", "t_pre", "t_0"]].dropna(), None)
        with restoring(NB):
            gap, outcome = _call(steps.measure, rows, label, NB=NB, panel_rows=panel,
                                 **({"hard_stop": stop} if stop else {}), out_dir=OUT)
        result.update(gap=gap, outcome=outcome)
        stages["measure"] = "ran"
        n_ok = int(flag(gap["usable"]).sum()) if "usable" in gap else 0
        log(f"2 measure: {len(gap)} gap rows ({n_ok} usable), {len(outcome)} outcome rows")

        if zref_sha is None:                       # the first run of the test: the constants come from its own gap tables
            try:
                zref_sha = check_frozen_constants()
            except FrozenConstantsMissing as e:
                raise FrozenConstantsMissing(
                    f"{e}\n\nSteps 0 to 2 are finished and saved in {OUT} (events, nulls, gap, outcome for "
                    f"{label!r}; measure is resumable, so a rerun skips them). Next: a person freezes the constants "
                    "once (oldnews.classify.freeze('insample')), commits the file, and reruns this step.") from e
        classified = _call(steps.classify, label, data_dir=OUT, NB=NB)
        result["classified"] = classified
        stages["classify"] = "ran"
        log(f"3 classify: {len(classified)} filings scored, "
            f"{int(flag(classified['old']).sum()) if 'old' in classified else '?'} labelled old news")

        result["tests"] = _call(steps.tests, label, data_dir=OUT, NB=NB)
        result["tests_summary"] = _summary(OUT / f"results_{label}")
        stages["tests"] = "ran"
        log("4 tests: written to " + str(OUT / f"results_{label}"))
        result["trade"] = _call(steps.trade, label, labels=classified, data_dir=OUT, NB=NB)
        result["trade_summary"] = _summary(OUT / f"trade_{label}")
        stages["trade"] = "ran"
        log("5 trade: written to " + str(OUT / f"trade_{label}"))

        if figures:
            result["figures"] = make_figures(result, NB)
            stages["figures"] = "ran"
            log(f"6 figures: {', '.join(result['figures'])} in {OUT / 'figures' / label}")

    (OUT / f"run_{label}.json").write_text(json.dumps(
        {"label": label, "start": w.start, "end": w.end, "hard_stop": w.hard_stop, "clip_before": w.clip_before,
         "allow_fetch": allow_fetch, "inputs": origin, "zref_frozen_sha256_12": zref_sha,
         "market_panel_extra": 0 if panel is None else len(panel), "filings": len(raw_ev), "events": len(ev),
         "ordinary_days": len(nu), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
         "minutes": round((time.time() - t_start) / 60, 1)}, indent=1))
    return result


DRY_RUN_WINDOW = ("2024-07-01", "2024-12-31")


def dry_run(start: str = DRY_RUN_WINDOW[0], end: str = DRY_RUN_WINDOW[1], NB: dict | None = None) -> dict:
    """A quiet end-to-end check that the judges' path runs, on a sub-window of 2024, from the cache only.

    It runs label insample with every output in a temporary scratch folder, which it deletes afterwards, and prints
    only: the number of events, the wall time, and whether each stage ran. It never prints a result or a table.
    Run it only once the 2024-2025 download has finished."""
    t0 = time.time()
    scratch = Path(tempfile.mkdtemp(prefix="oldnews_dryrun_"))
    stages: dict[str, str] = {}
    n_events, error = None, None
    try:
        res = run_oldnews(start, end, "insample", allow_fetch=False, NB=NB, out_dir=scratch, quiet=True, stages=stages)
        n_events = len(res["events"])
    except Exception as e:  # noqa: BLE001 (report the type and the first words of the message, never a table)
        error = f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    wall = time.time() - t0
    out = {"events": n_events, "wall_seconds": round(wall, 1), "stages": stages, "error": error,
           "scratch_deleted": not scratch.exists()}
    print(f"dry run of the judges' path: label insample, filings {start}..{end}, cache only")
    print(f"events in the people and placebo sets: {n_events if n_events is not None else 'n/a'}")
    first_missing = next((s for s in STAGES if s not in stages), None)
    for st in STAGES:
        print(f"  {st}: {stages.get(st) or ('FAILED' if st == first_missing and error else 'not reached')}")
    if error:
        print(f"stopped: {error}")
    print(f"wall time: {wall / 60:.1f} minutes ({wall:.0f} s); scratch folder deleted: {out['scratch_deleted']}")
    return out


def report(result: dict) -> None:
    """What every run prints (test plan, "Sealed window"): counts, the H1 effect with its interval and
    p-value, and "descriptive" when fewer than 30 events qualify. The numbers are the modules' own."""
    print(f"Old news, new news · {result['label']} · filings {result['start']}..{result['end']}\n")
    for name in ("tests_summary", "trade_summary"):
        text = result.get(name)
        print(text if text else f"({name.replace('_', ' ')}: no summary.md written)")


def notebook_run(start: str, end: str, label: str, allow_fetch: bool = False, NB: dict | None = None,
                 show: bool = True, **kwargs) -> dict | None:
    """run_oldnews for a notebook cell: prints the report and shows the figures. A window that is not in the
    cache (a clean machine, cache only) is explained instead of stopping the notebook; every other error,
    including a refused date, stops it."""
    from playground.measure import CacheMiss

    try:
        result = run_oldnews(start, end, label, allow_fetch=allow_fetch, NB=NB, **kwargs)
    except (CacheMiss, FileNotFoundError) as e:
        if "zref_frozen" in str(e):          # the frozen constants are missing: never report that as a cache miss
            raise
        print(f"Old news, new news · {label}: not run. This window is not in the API cache ({str(e)[:160]}).\n"
              "Set OLDNEWS_ALLOW_FETCH = True to download it through the notebook's own cached functions "
              "(the request estimate is printed before any option data are fetched).")
        return None
    report(result)
    if show and result.get("figures"):
        try:
            from IPython.display import Image, display
            for path in result["figures"].values():
                display(Image(filename=str(path)))
        except ImportError:
            print("figures: " + ", ".join(str(p) for p in result["figures"].values()))
    return result


# ---------------------------------------------------------------------------------------------------------
# Step 6 · figures
# ---------------------------------------------------------------------------------------------------------
def spot_path(NB: dict, row: pd.Series, before: int = 5, after: int = 21, bucket: str = "1m") -> pd.Series:
    """Daily parity spot for one event, from `before` sessions ahead of its gap to `after` sessions past entry,
    from the same pricing object, and the same fresh-mark rule, the measurement uses (call inside sandbox():
    cache only by default)."""
    from oldnews import measure

    CAL = NB["CAL"]
    t_pre, t_0 = pd.Timestamp(row["t_pre"]), pd.Timestamp(row["t_0"])
    priced, _ = NB["price_event"](row["ticker"], t_pre, t_0, t_0, NB["EXPIRY_BUCKETS"], NB["OTM_GRID"])
    pe = next((p for p in priced if p.bucket == bucket), None)       # all buckets above: the cached chain query
    if pe is None:
        return pd.Series(dtype=float)
    first = CAL[max(CAL.searchsorted(pd.Timestamp(row["gap_start"])) - before, 0)]
    last = min(CAL[min(CAL.searchsorted(t_0) + after, len(CAL) - 1)], pe.expiry_session, NB["LAST_SESSION"])
    days = CAL[(CAL >= first) & (CAL <= last)]
    return measure.spot_path(pe, days).rename("spot")


WALK_NOTE = ("One illustrative filing, not evidence: chosen without looking at outcomes (old news, all three math "
             "inputs, median gap length).")


def make_figures(result: dict, NB: dict | None, show: bool = False) -> dict[str, Path]:
    """The three figures for one run, saved to data/oldnews/figures/<label>/. The first two come straight from the
    stats module's horizon profile."""
    from oldnews import figures as F

    tests = result.get("tests")
    if not (isinstance(tests, dict) and "profile" in tests):
        raise ValueError("figures need the stats module's horizon profile (results_<label>/profile.csv)")
    data = F.Inputs(profile=tests["profile"], classified=result["classified"], outcome=F.primary_rows(result["outcome"]),
                    nulls=result["nulls"])
    F.attach_files(data, OUT, result["label"])      # profile_by_year.csv and the trade tables, if the run wrote them
    walk = None
    if NB is not None:
        for _, row in F.walkthrough_candidates(data.classified, data.outcome, data.nulls).head(5).iterrows():
            try:                              # the next candidate in the same rule order if the cache cannot price this one
                spot = spot_path(NB, row)
            except Exception as e:  # noqa: BLE001 (a missing cache entry only costs this candidate)
                log(f"walkthrough candidate {row['ticker']} {pd.Timestamp(row['t_0']).date()} skipped: "
                    f"{type(e).__name__}: {str(e)[:100]}")
                continue
            if len(spot.dropna()) >= 10:
                walk = (row, spot)
                break
    return F.make_all(data, OUT / "figures" / result["label"], result["label"],
                      title_suffix=f"filings {result['start']}..{result['end']}", walk=walk,
                      walk_note=WALK_NOTE, show=show)


def figures_only(label: str, NB: dict | None = None, show: bool = False) -> dict[str, Path]:
    """Draw the three figures from the tables a finished run wrote (data/oldnews/), computing nothing else.
    The walkthrough reads the API cache offline; nothing is fetched."""
    from oldnews import figures as F

    check_label(label)
    record = OUT / f"run_{label}.json"
    w = WINDOWS.get(label)
    if record.exists():
        r = json.loads(record.read_text(encoding="utf-8"))
        start, end = r["start"], r["end"]
    elif w:
        start, end = w.start, w.end
    else:
        raise FileNotFoundError(f"{record} not found: run the pipeline for {label!r} first")
    if NB is None:
        import nb
        NB = nb.load()
    w = window_rules(NB, start, end, label)       # the same date guards and switches as a full run
    data = F.read_tables(label, OUT)
    result = {"label": label, "start": start, "end": end, "classified": data.classified, "outcome": data.outcome,
              "nulls": data.nulls, "tests": {"profile": data.profile}}
    with sandbox(NB, False, w.clip_before):
        return make_figures(result, NB, show)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the old-news test on one window (cache only unless --allow-fetch).")
    ap.add_argument("--label", default=None, help="insample (the test) | holdout | oos; discovery and dryrun are retired")
    ap.add_argument("--start", default=None, help="first filing date (default: the label's window)")
    ap.add_argument("--end", default=None, help="last filing date (default: the label's window)")
    ap.add_argument("--source", default=None, help="'api' to build the inputs through the API functions")
    ap.add_argument("--allow-fetch", action="store_true", help="let the notebook's cached API functions download")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--figures-only", action="store_true",
                    help="draw the figures from the tables a finished run of this label wrote; compute nothing else")
    ap.add_argument("--max-requests", type=int, default=None,
                    help=f"refuse before fetching if the request estimate exceeds this (oos default {OOS_REQUEST_CAP:,})")
    ap.add_argument("--dry-run", action="store_true",
                    help=f"quiet end-to-end check on {DRY_RUN_WINDOW[0]}..{DRY_RUN_WINDOW[1]} in a scratch folder (deleted)")
    args = ap.parse_args()
    os.chdir(ROOT)                      # the notebook's cache and .env paths are relative to the repo root
    if args.dry_run:
        out = dry_run(args.start or DRY_RUN_WINDOW[0], args.end or DRY_RUN_WINDOW[1])
        return 0 if out["error"] is None else 1
    if not args.label:
        ap.error("--label is required (insample, holdout or oos)")
    check_label(args.label)
    if args.figures_only:
        for name, path in figures_only(args.label).items():
            print(f"{name}: {path}")
        return 0
    NB = None
    w = WINDOWS.get(args.label)
    start, end = args.start or (w and w.start), args.end or (w and w.end)
    if args.label == "oos":
        import nb                       # the notebook as the human edited it: RUN_OOS comes from its section 2
        NB = nb.load()
        if NB.get("RUN_OOS") is not True:
            print("refused: RUN_OOS is not True in the notebook (section 2). Only a human switches it on, once.",
                  file=sys.stderr)
            return 2
        start, end = args.start or NB["OOS_START"], args.end or NB["OOS_END"]
    if not (start and end):
        ap.error(f"--start and --end are required for label {args.label!r}")
    report(run_oldnews(start, end, args.label, allow_fetch=args.allow_fetch, NB=NB, source=args.source,
                       max_requests=args.max_requests, figures=not args.no_figures))
    return 0


if __name__ == "__main__":
    sys.exit(main())
