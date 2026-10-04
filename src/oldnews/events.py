"""Event and ordinary-day tables for the old-news test (docs/test_plan.md: "Events", "Ordinary days", "Word score T").

One row per 8-K (accession number) in the people-news set or the placebo set, built offline from the API cache:
  tags, excerpts    Massive's disclosure responses for every taxonomy tag, TOP_100 tickers only
  acceptance time   the cached EDGAR submission header (.massive_cache/sec_<sha1(filing_url)>.txt):
  and event date    <ACCEPTANCE-DATETIME> and CONFORMED PERIOD OF REPORT
  t_0, t_pre        the notebook's own entry_session and pre_session (15:30 cutoff, early closes) and calendar
Ordinary days are the matched draws already in data/playground/nulls_<label>.csv; each gets its event's gap length.

Windows and guards: build(label, NB, window=(start, end)) takes the filing-date window from the caller (default
WINDOWS[label], or the notebook's HOLDOUT_ and OOS_ dates). The label alone sets the date guard, checked before any
data are read (resolve_window):
  discovery, dryrun, other labels  refuse any t_pre, t_0 or gap date on or after 2024-01-01
  insample                         refuse any on or after 2026-01-01
  holdout                          no guard, only when NB has RUN_HOLDOUT is True (the judges' switch); the one label that
                                   may fetch a disclosure response or EDGAR header missing from the cache, through the
                                   notebook's own cached fetch_disclosures and fetch_acceptance_time
  oos                              no guard, only when NB has RUN_OOS is True (a human's switch); cache only
Every other label is cache only: the notebook is put offline and a miss raises.

Interface (extends 00_shared.md):
  nulls row_id        "null|ticker|t_pre|t_0|round|accession" (the matched event's accession), so every row_id across
                      events and nulls is unique even when two events of one ticker and filing date share an ordinary day
  dropped_<label>.csv row_id, ticker, filing_date, accepted_at, reason: people and placebo filings whose own t_0 or t_pre
                      differs from data/playground/events_<label>.csv. That file keeps one row per (cik, filing_date),
                      the earliest-accepted accession, and the cache was built with its sessions. A later accession filed
                      the same day (after the close) would enter at a close before it was accepted, which is lookahead,
                      so it is dropped, with its ordinary days. Counted as an exclusion. Discovery: 5 events (BK
                      2022-04-12, NEE 2022-05-23, TMUS 2022-09-07, CVX 2023-01-27, HD 2023-06-20). The earlier same-day
                      filing stays in the events table; dropped filings still count in the related-filing and earnings pools.

Definitions (business days are NYSE sessions; every input is public by the t_pre close or is the filing itself):
  lag_bd              sessions from the event date (inclusive) to the acceptance date (exclusive); late = lag_bd >= 1
  gap_start           last session strictly before the event date; n_gap = sessions from gap_start to t_pre.
                      Left empty when the event date is after the acceptance date (a bad header; never late).
  earnings_excluded   the filing has an earnings tag, or the same ticker has an earnings-tagged filing whose t_0 is
                      within 5 sessions of this row's t_0
  cue_dated_prior     the text writes a date earlier than the filing date within 80 characters of "announc" or
                      "press release"
  cue_prior_wording   the text matches "previously (announced|disclosed|reported)"
  cue_related_filing  the same ticker has another people-news filing whose event date or filing date is in the 30
                      calendar days before this event date, and which was accepted before this row's t_pre close
  T                   the number of cues present (0 to 3)
Text: every distinct excerpt Massive attached to the accession (one per tag row), lower-cased, whitespace collapsed.
Groups: "people" has a people-news tag (role = most senior: CEO, CFO, officer, director); "placebo" has a placebo tag
and no people-news tag. Both keep non-late rows; the test sets are the rows with late = 1.

Usage (repo root):  .venv/Scripts/python.exe src/oldnews/events.py [--label discovery]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

PEOPLE_TAGS = {
    "ceo_departure": "CEO", "ceo_appointment": "CEO",
    "cfo_departure": "CFO", "cfo_appointment": "CFO",
    "executive_officer_departure": "officer", "executive_officer_appointment": "officer",
    "director_departure": "director", "director_appointment": "director",
}
ROLE_RANK = ["CEO", "CFO", "officer", "director"]          # most senior first
PLACEBO_TAGS = ["annual_meeting_results", "shareholder_proposal_outcome", "executive_compensation_change",
                "bylaw_amendment", "charter_amendment", "dividend_declaration", "equity_compensation_grant"]
EARNINGS_TAGS = ["quarterly_earnings", "annual_earnings", "preliminary_results"]
REQUIRED_TAGS = sorted(set(PEOPLE_TAGS) | set(PLACEBO_TAGS) | set(EARNINGS_TAGS))

# Default windows: label -> (first filing date, last filing date, hard stop: no t_pre, t_0 or gap date on or after it).
# holdout and oos have none: the caller passes window= (or the notebook's HOLDOUT_/OOS_ dates are used).
WINDOWS = {"discovery": ("2022-01-01", "2023-12-31", "2024-01-01"),
           "dryrun": ("2023-07-01", "2023-12-31", "2024-01-01"),
           "insample": ("2024-01-01", "2025-12-31", "2026-01-01")}
PRIOR = {"insample": "discovery"}       # an earlier cached window, read only to look back (related filings, earnings)
NEVER = "2026-01-01"                     # out-of-sample start: no guarded label may reach it
DISCOVERY_STOP = "2024-01-01"            # hard stop of discovery, dryrun and any unnamed label
LIVE_LABELS = {"holdout": "RUN_HOLDOUT", "oos": "RUN_OOS"}   # labels with no date guard, each behind its notebook switch

EARNINGS_SESSIONS = 5
CUE_CHARS = 80
RELATED_DAYS = 30
YEARLESS_ROLLBACK_DAYS = 330

OUT_DIR = ROOT / "data" / "oldnews"
PLAYGROUND = ROOT / "data" / "playground"
_cache = Path(os.environ.get("MASSIVE_CACHE_DIR", ".massive_cache"))
CACHE_DIR = _cache if _cache.is_absolute() else ROOT / _cache

EVENT_COLUMNS = ["row_id", "kind", "accession_number", "ticker", "filing_date", "accepted_at", "event_date",
                 "gap_start", "t_pre", "t_0", "n_gap", "lag_bd", "late", "group", "role", "tags", "earnings_excluded",
                 "cue_dated_prior", "cue_prior_wording", "cue_related_filing", "T", "text_source"]
DROPPED_COLUMNS = ["row_id", "ticker", "filing_date", "accepted_at", "reason"]
NULL_COLUMNS = ["row_id", "kind", "event_row_id", "ticker", "t_pre", "t_0", "gap_start", "n_gap", "round"]
# Every read_csv uses these: "null" and the ticker "NA" must not become missing values; only empty cells are.
CSV_KW = {"keep_default_na": False, "na_values": [""]}


# ---- Calendar ----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Clock:
    """The trading calendar and the notebook's session rules, injected so tests can use a synthetic calendar."""
    cal: pd.DatetimeIndex
    entry_session: Callable[[pd.Timestamp, pd.Timestamp], pd.Timestamp]   # (accepted_at, filing_date) -> t_0
    pre_session: Callable[[pd.Timestamp, pd.Timestamp], pd.Timestamp]     # (accepted_at, filing_date) -> t_pre
    close_time: Callable[[pd.Timestamp], pd.Timedelta]                    # day -> close after midnight ET

    @classmethod
    def from_nb(cls, NB: dict) -> "Clock":
        return cls(NB["CAL"], NB["entry_session"], NB["pre_session"], NB["close_time"])

    def pos(self, day) -> int:
        """Index of the first session on or after `day` (a session's own index)."""
        return int(self.cal.searchsorted(pd.Timestamp(day), side="left"))

    def session_before(self, day) -> pd.Timestamp:
        """The last session strictly before `day`; NaT if the calendar does not reach back that far."""
        i = self.pos(day) - 1
        return self.cal[i] if i >= 0 else pd.NaT


# ---- Word cues ---------------------------------------------------------------------------------------------------------
_MONTHS = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8,
           "september": 9, "october": 10, "november": 11, "december": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4,
           "jun": 6, "jul": 7, "aug": 8, "sept": 9, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
DATE_RE = re.compile(rf"\b({_MONTH_ALT})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+((?:19|20)\d{{2}})\b)?")
KEYWORD_RE = re.compile(r"announc|press release")
PRIOR_WORDING_RE = re.compile(r"previously (announced|disclosed|reported)")


def normalize(text) -> str:
    """Lower-case and collapse whitespace, so phrases split across lines still match."""
    return re.sub(r"\s+", " ", text.lower()).strip() if isinstance(text, str) else ""


def written_dates(text: str, filing_date) -> list[tuple[int, int, pd.Timestamp]]:
    """(start, end, date) of every 'Month D[, YYYY]' date in normalized text (full or abbreviated month name).
    A date written without a year takes the filing's year, or the year before when that would put it more than
    330 days after the filing (a late-December date in a January filing). Impossible dates are skipped."""
    fd = pd.Timestamp(filing_date).normalize()
    out = []
    for m in DATE_RE.finditer(text):
        month, day = _MONTHS[m.group(1)], int(m.group(2))
        try:
            d = pd.Timestamp(year=int(m.group(3)) if m.group(3) else fd.year, month=month, day=day)
        except ValueError:
            continue
        if not m.group(3) and (d - fd).days > YEARLESS_ROLLBACK_DAYS:
            d = d - pd.DateOffset(years=1)
        out.append((m.start(), m.end(), d))
    return out


def cue_dated_prior(text: str, filing_date) -> int:
    """1 if a date earlier than the filing date is written within CUE_CHARS characters of 'announc' or
    'press release' (gap between the two spans; overlapping or adjacent counts as 0)."""
    keys = [(k.start(), k.end()) for k in KEYWORD_RE.finditer(text)]
    if not keys:
        return 0
    fd = pd.Timestamp(filing_date).normalize()
    return int(any(d < fd and any(max(ks - e, s - ke, 0) <= CUE_CHARS for ks, ke in keys)
                   for s, e, d in written_dates(text, fd)))


def cue_prior_wording(text: str) -> int:
    return int(bool(PRIOR_WORDING_RE.search(text)))


# ---- EDGAR header ------------------------------------------------------------------------------------------------------
ACCEPT_RE = re.compile(r"<ACCEPTANCE-DATETIME>(\d{14})")
PERIOD_RE = re.compile(r"CONFORMED PERIOD OF REPORT:\s*(\d{8})")


def header_path(filing_url: str, cache_dir: Path = CACHE_DIR) -> Path:
    """Where the notebook's fetch_acceptance_time caches the first 4 KB of the EDGAR submission."""
    return Path(cache_dir) / ("sec_" + hashlib.sha1(filing_url.encode()).hexdigest() + ".txt")


def read_header(path: Path) -> tuple[bool, pd.Timestamp, pd.Timestamp]:
    """(header cached, acceptance time, event date) from a cached EDGAR header. Never fetches."""
    if not path.exists():
        return False, pd.NaT, pd.NaT
    txt = path.read_text(encoding="latin-1", errors="replace")
    a, p = ACCEPT_RE.search(txt), PERIOD_RE.search(txt)
    accepted = pd.to_datetime(a.group(1), format="%Y%m%d%H%M%S") if a else pd.NaT
    event = pd.to_datetime(p.group(1), format="%Y%m%d", errors="coerce") if p else pd.NaT
    return True, accepted, event


# ---- Filings -----------------------------------------------------------------------------------------------------------
def load_disclosures(NB: dict, tags: list[str], start: str, end: str) -> tuple[pd.DataFrame, list[str]]:
    """Cached disclosure rows for these tags with filing dates in [start, end], one row per TOP_100 ticker,
    plus the tags whose response is not in the cache. Requires go_offline(NB): a cache miss never fetches."""
    from playground.measure import CacheMiss
    parts, missing = [], []
    for tag in tags:
        try:
            df = NB["fetch_disclosures"](tag, start, end)
        except CacheMiss:
            missing.append(tag)
            continue
        if len(df) and "tickers" in df:              # some tags have no filer with a ticker field: no TOP_100 rows
            parts.append(df)
    if not parts:
        return pd.DataFrame(columns=["cik", "accession_number", "filing_date", "tertiary_category",
                                     "supporting_text", "filing_url", "ticker"]), missing
    ex = pd.concat(parts, ignore_index=True).explode("tickers").rename(columns={"tickers": "ticker"})
    ex["ticker"] = ex["ticker"].map(NB["normalize_ticker"])
    return ex[ex["ticker"].isin(NB["TOP_100"])].reset_index(drop=True), missing


def accession_table(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per accession: cik, ticker, filing_date, filing_url, tags (sorted, pipe-joined) and text (its distinct
    excerpts in a fixed order, newline-joined)."""
    rows = rows.sort_values(["accession_number", "ticker", "tertiary_category"], kind="stable")
    g = rows.groupby("accession_number", sort=True)
    out = g.agg(cik=("cik", "first"), ticker=("ticker", "first"), filing_date=("filing_date", "first"),
                filing_url=("filing_url", "first"))
    out["tags"] = g["tertiary_category"].agg(lambda s: "|".join(sorted(set(s))))
    out["text"] = g["supporting_text"].agg(
        lambda s: "\n".join(dict.fromkeys(t.strip() for t in s if isinstance(t, str) and t.strip())))
    out["filing_date"] = pd.to_datetime(out["filing_date"])
    return out.reset_index()


def date_filings(acc: pd.DataFrame, clock: Clock, cache_dir: Path = CACHE_DIR,
                 fetch: Callable[[str], object] | None = None) -> pd.DataFrame:
    """Add has_header, accepted_at, event_date, t_0 and t_pre (the notebook's rules; with no acceptance time they
    fall back to the notebook's conservative sessions). With `fetch` (the notebook's fetch_acceptance_time, which
    caches the EDGAR header under cache_dir), a header missing from the cache is fetched first; a CacheMiss from it
    (the notebook offline) or a failed fetch leaves the filing without a header, and it is counted as dropped."""
    from playground.measure import CacheMiss

    def header(url: str):
        path = header_path(url, cache_dir)
        if fetch is not None and not path.exists():
            try:
                fetch(url)
            except CacheMiss:
                pass
        return read_header(path)

    hdr = [header(u) for u in acc["filing_url"]]
    out = acc.assign(has_header=[h[0] for h in hdr], accepted_at=pd.to_datetime([h[1] for h in hdr]),
                     event_date=pd.to_datetime([h[2] for h in hdr]))
    out["t_0"] = pd.to_datetime([clock.entry_session(a, f) for a, f in zip(out["accepted_at"], out["filing_date"])])
    out["t_pre"] = pd.to_datetime([clock.pre_session(a, f) for a, f in zip(out["accepted_at"], out["filing_date"])])
    return out


DROP_REASON = "entry would precede acceptance; same-day earlier filing in the playground table"


def playground_conflicts(acc: pd.DataFrame, playground_events: pd.DataFrame) -> pd.Series:
    """True for a dated filing whose own t_0 or t_pre differs from the playground file's row for the same ticker and
    filing date. That file keeps one row per (cik, filing_date), the earliest-accepted accession, and the cache was built
    with its sessions. A later accession filed the same day (after the close) differs, and the cache's entry close would
    come before that filing was accepted (lookahead), so such filings cannot be tested. Filings the file lacks are False."""
    pg = playground_events.drop_duplicates(["ticker", "filing_date"]).set_index(["ticker", "filing_date"])
    idx = pd.MultiIndex.from_arrays([acc["ticker"], acc["filing_date"]])
    pg_t0, pg_pre = pg["t_0"].reindex(idx).to_numpy(), pg["t_pre"].reindex(idx).to_numpy()
    have = ~(pd.isna(pg_t0) | pd.isna(pg_pre))
    differs = (pg_t0 != acc["t_0"].to_numpy()) | (pg_pre != acc["t_pre"].to_numpy())
    return pd.Series(have & differs & acc["has_header"].to_numpy(), index=acc.index)


def dropped_table(acc: pd.DataFrame, conflict: pd.Series) -> pd.DataFrame:
    """The people-news and placebo filings in conflict (see playground_conflicts), for the record."""
    d = select_groups(acc[conflict]).sort_values(["filing_date", "ticker", "accession_number"])
    return pd.DataFrame({"row_id": "event|" + d["accession_number"], "ticker": d["ticker"], "filing_date": d["filing_date"],
                         "accepted_at": d["accepted_at"], "reason": DROP_REASON}).reset_index(drop=True)


def tag_sets(tags: pd.Series) -> pd.Series:
    return tags.fillna("").str.split("|").map(lambda t: set(x for x in t if x))


# ---- Table pieces ------------------------------------------------------------------------------------------------------
def gap_fields(ev: pd.DataFrame, clock: Clock) -> pd.DataFrame:
    """lag_bd, late, gap_start and n_gap for dated filings."""
    acc_day = ev["accepted_at"].dt.normalize().fillna(ev["filing_date"])
    lag = [clock.pos(a) - clock.pos(e) for a, e in zip(acc_day, ev["event_date"])]
    out = ev.assign(lag_bd=pd.array(lag, dtype="Int64"))
    out["late"] = (out["lag_bd"] >= 1).astype(int)
    gs = [clock.session_before(e) if l >= 0 else pd.NaT for e, l in zip(out["event_date"], lag)]
    out["gap_start"] = pd.to_datetime(gs)
    out["n_gap"] = pd.array([clock.pos(t) - clock.pos(g) if pd.notna(g) else pd.NA
                             for t, g in zip(out["t_pre"], out["gap_start"])], dtype="Int64")
    return out


def earnings_flag(ev: pd.DataFrame, earnings: pd.DataFrame, clock: Clock, k: int = EARNINGS_SESSIONS) -> pd.Series:
    """1 if the filing has an earnings tag, or the same ticker has an earnings filing whose t_0 is within k sessions."""
    own = tag_sets(ev["tags"]).map(lambda t: bool(t & set(EARNINGS_TAGS)))
    by_ticker = {t: np.array([clock.pos(d) for d in g["t_0"]]) for t, g in earnings.groupby("ticker")}
    near = [bool(len(by_ticker.get(t, ())) and np.abs(by_ticker[t] - clock.pos(t0)).min() <= k)
            for t, t0 in zip(ev["ticker"], ev["t_0"])]
    return (own.to_numpy() | np.array(near, dtype=bool)).astype(int)


def related_filing_cue(ev: pd.DataFrame, people: pd.DataFrame, clock: Clock, days: int = RELATED_DAYS) -> np.ndarray:
    """1 if the same ticker has another people-news filing whose event date or filing date falls in the `days`
    calendar days before this row's event date, accepted before this row's t_pre close (public by then).
    A pool filing with no acceptance time counts as accepted at the end of its filing date."""
    accepted = people["accepted_at"].fillna(people["filing_date"] + pd.Timedelta(hours=23, minutes=59))
    pool = {t: g for t, g in people.assign(_acc=accepted).groupby("ticker")}
    out = []
    for r in ev.itertuples(index=False):
        g = pool.get(r.ticker)
        if g is None or pd.isna(r.event_date):
            out.append(0)
            continue
        known_by = r.t_pre + clock.close_time(r.t_pre)
        lo = r.event_date - pd.Timedelta(days=days)
        win = ((g["event_date"] >= lo) & (g["event_date"] < r.event_date)) | \
              ((g["filing_date"] >= lo) & (g["filing_date"] < r.event_date))
        out.append(int(((g["accession_number"] != r.accession_number) & (g["_acc"] < known_by) & win).any()))
    return np.array(out, dtype=int)


def select_groups(acc: pd.DataFrame) -> pd.DataFrame:
    """Keep people-news filings (role = most senior people tag) and placebo-tag filings with no people tag."""
    tags = tag_sets(acc["tags"])
    roles = tags.map(lambda t: [PEOPLE_TAGS[x] for x in t if x in PEOPLE_TAGS])
    people = roles.map(bool)
    placebo = ~people & tags.map(lambda t: bool(t & set(PLACEBO_TAGS)))
    keep = people | placebo
    out = acc[keep].copy()
    out["group"] = np.where(people[keep], "people", "placebo")
    out["role"] = [min(r, key=ROLE_RANK.index) if r else "placebo" for r in roles[keep]]
    return out


def event_table(acc: pd.DataFrame, people_pool: pd.DataFrame, earnings_pool: pd.DataFrame, clock: Clock,
                hard_stop: str | None) -> tuple[pd.DataFrame, dict]:
    """The events table from dated accessions (see date_filings), plus counts of every row dropped and why."""
    ev = select_groups(acc)
    counts: dict = {}
    for grp, g in ev.groupby("group"):
        counts[f"{grp}_filings"] = len(g)
        counts[f"{grp}_no_header"] = int((~g["has_header"]).sum())
        counts[f"{grp}_no_event_date"] = int((g["has_header"] & g["event_date"].isna()).sum())
        counts[f"{grp}_no_acceptance_time_kept"] = int((g["has_header"] & g["accepted_at"].isna()
                                                        & g["event_date"].notna()).sum())
    ev = ev[ev["has_header"] & ev["event_date"].notna()]
    entry_late = ev["t_0"] >= pd.Timestamp(hard_stop) if hard_stop else pd.Series(False, index=ev.index)
    for grp, g in ev[entry_late].groupby("group"):
        counts[f"{grp}_entry_on_or_after_{hard_stop}"] = len(g)
    ev = gap_fields(ev[~entry_late], clock)
    for grp, g in ev.groupby("group"):
        counts[f"{grp}_event_after_acceptance_kept"] = int(g["gap_start"].isna().sum())

    ev["earnings_excluded"] = earnings_flag(ev, earnings_pool, clock)
    text = ev["text"].map(normalize)
    ev["text_source"] = np.where(text.str.len() > 0, "excerpt", "none")
    ev["cue_dated_prior"] = [cue_dated_prior(t, f) for t, f in zip(text, ev["filing_date"])]
    ev["cue_prior_wording"] = text.map(cue_prior_wording)
    ev["cue_related_filing"] = related_filing_cue(ev, people_pool, clock)
    ev["T"] = ev[["cue_dated_prior", "cue_prior_wording", "cue_related_filing"]].sum(axis=1)
    ev["row_id"] = "event|" + ev["accession_number"]
    ev["kind"] = "event"
    ev = ev.sort_values(["filing_date", "ticker", "accession_number"]).reset_index(drop=True)
    return ev[EVENT_COLUMNS], counts


def null_table(ev: pd.DataFrame, drawn: pd.DataFrame, clock: Clock) -> pd.DataFrame:
    """Matched ordinary days for each event (drawn per ticker and filing date by the playground download), each with
    a pseudo-gap of its event's n_gap sessions ending at its t_pre (gap_start empty when the event has no gap)."""
    keys = ev[["row_id", "accession_number", "ticker", "filing_date", "n_gap"]].rename(columns={"row_id": "event_row_id"})
    m = keys.merge(drawn[["ticker", "filing_date", "round", "t_0", "t_pre"]], on=["ticker", "filing_date"])
    starts = []
    for t_pre, n in zip(m["t_pre"], m["n_gap"]):
        i = clock.pos(t_pre) - int(n) if pd.notna(n) and n >= 1 else -1
        starts.append(clock.cal[i] if i >= 0 else pd.NaT)
    m["gap_start"] = pd.to_datetime(starts)
    m["round"] = m["round"].astype(int)
    m["row_id"] = ("null|" + m["ticker"] + "|" + m["t_pre"].dt.strftime("%Y-%m-%d") + "|"
                   + m["t_0"].dt.strftime("%Y-%m-%d") + "|" + m["round"].astype(str) + "|" + m["accession_number"])
    m["kind"] = "null"
    return m.sort_values(["event_row_id", "round"]).reset_index(drop=True)[NULL_COLUMNS]


def check_dates(df: pd.DataFrame, cols: list[str], hard_stop: str | None, name: str) -> None:
    """Hard fail on any listed date on or after the hard stop (None: the label has no date guard)."""
    if hard_stop is None:
        return
    for c in cols:
        bad = df[c].notna() & (df[c] >= pd.Timestamp(hard_stop))
        if bad.any():
            raise ValueError(f"{name}: {int(bad.sum())} rows have {c} on or after {hard_stop}; refusing")


def resolve_window(label: str, NB: dict, window: tuple | None = None) -> tuple[str, str, str | None]:
    """(start, end, hard_stop) for this run, or a refusal. The label alone sets the date guard; a third item in
    `window` (the pipeline passes one) is ignored, so a caller cannot loosen or tighten it.
      discovery, dryrun, any other label   no t_pre, t_0 or gap date on or after 2024-01-01
      insample                              none on or after 2026-01-01 (the out-of-sample start)
      holdout                               no guard; only when the namespace has RUN_HOLDOUT is True (the judges' switch)
      oos                                   no guard; only when the namespace has RUN_OOS is True (a human's switch)
    The window defaults to WINDOWS[label], or for holdout and oos to the notebook's HOLDOUT_ and OOS_ dates."""
    if label in LIVE_LABELS:
        switch = LIVE_LABELS[label]
        if NB.get(switch) is not True:
            raise PermissionError(f"label {label!r} runs only when {switch} is True in the notebook (section 2); refusing")
        stop = None
        default = (NB.get("HOLDOUT_START"), NB.get("HOLDOUT_END")) if label == "holdout" else (NB.get("OOS_START"), NB.get("OOS_END"))
    else:
        stop = WINDOWS[label][2] if label in WINDOWS else DISCOVERY_STOP
        if stop > NB.get("OOS_START", NEVER):
            raise PermissionError(f"label {label!r} would reach the out-of-sample period; refusing")
        default = WINDOWS[label][:2] if label in WINDOWS else (None, None)
    start, end = (window[0], window[1]) if window is not None else default
    if start is None or end is None:
        raise ValueError(f"label {label!r} has no default window; pass window=(start, end)")
    try:
        start, end = str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date())
    except ValueError:
        raise ValueError(f"window {start!r}..{end!r} holds a date that is not real") from None
    if start > end:
        raise ValueError(f"window start {start} is after its end {end}")
    if stop is not None and end >= stop:
        raise PermissionError(f"label {label!r} allows filing dates before {stop} only; window ends {end}")
    return start, end, stop


# ---- Build -------------------------------------------------------------------------------------------------------------
def build(label: str = "discovery", NB: dict | None = None, out_dir: Path = OUT_DIR,
          playground: Path = PLAYGROUND, write: bool = True, window: tuple | None = None,
          cache_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build events_<label>.csv, nulls_<label>.csv and dropped_<label>.csv from the API cache. Counts of dropped rows
    are in events.attrs["counts"].

    window    (start, end) of filing dates, optionally with a third item that is ignored (see resolve_window); default
              WINDOWS[label], or the notebook's HOLDOUT_/OOS_ dates. The date guard and the switch each label needs are
              in resolve_window and checked before anything is read.
    cache_dir the API cache (default: the notebook's CACHE_DIR). Read-only for every label but holdout.
    Cache only (the notebook is put offline) for discovery, dryrun, insample, oos and any other label. Holdout leaves
    the notebook as the caller set it and, where a disclosure response or EDGAR header is not cached, fetches it
    through the notebook's own cached fetch_disclosures and fetch_acceptance_time (offline callers get a CacheMiss)."""
    if NB is None:
        import nb
        NB = nb.load()
    start, end, hard_stop = resolve_window(label, NB, window)
    from playground.measure import go_offline
    if cache_dir is not None:
        NB["CACHE_DIR"] = Path(cache_dir)
    cache_dir = Path(NB.get("CACHE_DIR", CACHE_DIR))
    fetching = label == "holdout"
    if not fetching:
        go_offline(NB)
    clock = Clock.from_nb(NB)

    taxonomy = sorted(pd.DataFrame(NB["api_get_all"]("/stocks/taxonomies/vX/disclosures", {"limit": 1000}))
                      ["tertiary_category"])
    rows, missing = load_disclosures(NB, taxonomy, start, end)
    if set(missing) & set(REQUIRED_TAGS):
        raise RuntimeError(f"disclosures not cached for {sorted(set(missing) & set(REQUIRED_TAGS))}")
    fetch = NB["fetch_acceptance_time"] if fetching else None
    acc = date_filings(accession_table(rows), clock, cache_dir, fetch)
    conflict = playground_conflicts(acc, pd.read_csv(playground / f"events_{label}.csv",
                                                     parse_dates=["filing_date", "t_0", "t_pre"], **CSV_KW))
    dropped = dropped_table(acc, conflict)              # people and placebo filings only; the pools below keep every filing
    pool = acc
    # Window edges (discovery, late people events kept): the related-filing look-back cannot see December 2021 filings,
    # so 12 events with an event date before 2022-01-31 may miss a cue; the earnings check cannot see January 2024
    # filings, so 2 events with t_0 on or after 2023-12-21 may miss an earnings exclusion. Both only understate.
    if label in PRIOR:                        # look back into the earlier window for related filings and earnings
        p0, p1, _ = WINDOWS[PRIOR[label]]
        prior_rows, _ = load_disclosures(NB, sorted(set(PEOPLE_TAGS) | set(EARNINGS_TAGS)), p0, p1)
        pool = pd.concat([date_filings(accession_table(prior_rows), clock, cache_dir), acc], ignore_index=True)
    pool_tags = tag_sets(pool["tags"])
    people_pool = pool[pool_tags.map(lambda t: bool(t & set(PEOPLE_TAGS)))]
    earnings_pool = pool[pool_tags.map(lambda t: bool(t & set(EARNINGS_TAGS)))]

    events, counts = event_table(acc[~acc["accession_number"].isin(dropped["row_id"].str.removeprefix("event|"))],
                                 people_pool, earnings_pool, clock, hard_stop)
    counts["tags_not_cached"] = missing
    counts["entry_precedes_acceptance_dropped"] = len(dropped)
    drawn = pd.read_csv(playground / f"nulls_{label}.csv", parse_dates=["filing_date", "t_0", "t_pre"], **CSV_KW)
    nulls = null_table(events, drawn, clock)
    check_dates(events, ["t_pre", "t_0", "gap_start"], hard_stop, "events")
    check_dates(nulls, ["t_pre", "t_0", "gap_start"], hard_stop, "nulls")
    ids = pd.concat([events["row_id"], nulls["row_id"]])
    assert ids.is_unique, f"{int(ids.duplicated().sum())} duplicate row_ids across events and nulls"
    assert not set(dropped["row_id"]) & set(events["row_id"]), "a dropped filing is still in the events table"
    events.attrs["counts"], events.attrs["dropped"] = counts, dropped
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        events.to_csv(out_dir / f"events_{label}.csv", index=False)
        nulls.to_csv(out_dir / f"nulls_{label}.csv", index=False)
        dropped.to_csv(out_dir / f"dropped_{label}.csv", index=False)
    return events, nulls


def summary(events: pd.DataFrame, nulls: pd.DataFrame) -> dict:
    """The counts the team reports: events, late events, exclusions, cue frequencies, ordinary days."""
    late = events[events["late"] == 1]
    lp, lpl = late[late["group"] == "people"], late[late["group"] == "placebo"]
    lp_in, lpl_in = lp[lp["earnings_excluded"] == 0], lpl[lpl["earnings_excluded"] == 0]
    cues = ["cue_dated_prior", "cue_prior_wording", "cue_related_filing"]
    return {
        "people_events": int((events["group"] == "people").sum()),
        "people_late": len(lp), "people_late_earnings_excluded": int(lp["earnings_excluded"].sum()),
        "people_late_kept": len(lp_in),
        "people_late_kept_by_role": lp_in["role"].value_counts().to_dict(),
        "placebo_events": int((events["group"] == "placebo").sum()),
        "placebo_late": len(lpl), "placebo_late_earnings_excluded": int(lpl["earnings_excluded"].sum()),
        "placebo_late_kept": len(lpl_in),
        "cue_counts_people_late_kept": {c: int(lp_in[c].sum()) for c in cues},
        "T_distribution_people_late_kept": lp_in["T"].value_counts().sort_index().to_dict(),
        "cue_counts_placebo_late_kept": {c: int(lpl_in[c].sum()) for c in cues},
        "lag_bd_people_late_kept_quantiles": lp_in["lag_bd"].astype(float).quantile([.5, .9, 1]).to_dict(),
        "null_rows": len(nulls), "null_rows_with_gap": int(nulls["gap_start"].notna().sum()),
        "events_without_nulls": int((~events["row_id"].isin(nulls["event_row_id"])).sum()),
        "late_people_kept_without_nulls": int((~lp_in["row_id"].isin(nulls["event_row_id"])).sum()),
        "duplicate_row_ids_events_and_nulls": int(pd.concat([events["row_id"], nulls["row_id"]]).duplicated().sum()),
        "dropped_entry_precedes_acceptance": events.attrs.get("dropped", pd.DataFrame(columns=DROPPED_COLUMNS))["row_id"].tolist(),
        "dropped": events.attrs.get("counts", {}),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", default="discovery", help="discovery | dryrun | insample (holdout and oos need their notebook switch)")
    ap.add_argument("--start", default=None, help="first filing date (default: the label's window)")
    ap.add_argument("--end", default=None, help="last filing date (default: the label's window)")
    args = ap.parse_args()
    os.chdir(ROOT)                       # the notebook's cache path is relative to the repo root
    ev, nu = build(args.label, window=(args.start, args.end) if args.start or args.end else None)
    for k, v in summary(ev, nu).items():
        print(f"{k}: {v}")
    print(f"wrote {OUT_DIR / f'events_{args.label}.csv'} and {OUT_DIR / f'nulls_{args.label}.csv'}")
