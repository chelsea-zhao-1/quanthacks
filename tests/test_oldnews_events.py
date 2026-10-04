"""Synthetic tests for src/oldnews/events.py (no real data, no network, no API key).
Run:  .venv/Scripts/python.exe tests/test_oldnews_events.py"""
import hashlib
import inspect
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402

from oldnews import events as E                 # noqa: E402
from playground.measure import CacheMiss        # noqa: E402

FAILS = []
T = pd.Timestamp


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{': ' + detail if detail else ''}")
    if not ok:
        FAILS.append(name)


# ---- A synthetic calendar with the notebook's session rules (15:30 cutoff, 16:00 close) ------------------------------
CAL = pd.bdate_range("2021-12-01", "2027-12-31").drop(pd.DatetimeIndex(["2022-01-17", "2022-04-15"]))


def entry_session(a, f):
    if pd.isna(a):
        return CAL[CAL.searchsorted(T(f), side="right")]
    day = a.normalize()
    t0 = CAL[CAL.searchsorted(day, side="right")] if a - day >= pd.Timedelta(hours=15, minutes=30) else CAL[CAL.searchsorted(day)]
    return max(t0, CAL[CAL.searchsorted(T(f))])


def pre_session(a, f):
    if pd.isna(a):
        return CAL[CAL.searchsorted(T(f)) - 1]
    day = a.normalize()
    return day if day in CAL and a - day >= pd.Timedelta(hours=16) else CAL[CAL.searchsorted(day) - 1]


CLOCK = E.Clock(CAL, entry_session, pre_session, lambda d: pd.Timedelta(hours=16))


def dated(rows):
    """A dated accession table (as date_filings returns) from (accession, ticker, filing_date, accepted_at, event_date, tags, text)."""
    df = pd.DataFrame(rows, columns=["accession_number", "ticker", "filing_date", "accepted_at", "event_date", "tags", "text"])
    for c in ("filing_date", "accepted_at", "event_date"):
        df[c] = pd.to_datetime(df[c])
    df["has_header"] = True
    df["cik"], df["filing_url"] = "0000000001", "u"
    df["t_0"] = [entry_session(a, f) for a, f in zip(df.accepted_at, df.filing_date)]
    df["t_pre"] = [pre_session(a, f) for a, f in zip(df.accepted_at, df.filing_date)]
    return df


# ---- Text cues -------------------------------------------------------------------------------------------------------
check("normalize lower-cases and collapses whitespace", E.normalize("As  Previously\n\tDisclosed ") == "as previously disclosed")
check("normalize turns missing text into ''", E.normalize(np.nan) == "")
d = E.written_dates(E.normalize("On March 17, the director announced ... effective December 31, 2022. Sept. 30, 2022"), "2022-03-22")
check("written dates: year-less, full and abbreviated", [x[2] for x in d] == [T("2022-03-17"), T("2022-12-31"), T("2022-09-30")], str(d))
check("year-less late-December date in a January filing is last year",
      [x[2] for x in E.written_dates("on december 28 the ceo resigned", "2022-01-03")] == [T("2021-12-28")])
check("year-less future date in a February filing stays this year",
      [x[2] for x in E.written_dates("retire effective december 31", "2022-02-10")] == [T("2022-12-31")])
check("impossible dates and month-year phrases are skipped",
      E.written_dates("june 31, 2022 and may 2022 and fiscal 2022", "2022-08-01") == [])
check("1st/2nd ordinals parse", [x[2] for x in E.written_dates("on june 1st, 2022 it", "2022-06-03")] == [T("2022-06-01")])

fd = "2022-03-10"
check("dated prior: earlier date next to 'announced'", E.cue_dated_prior("on march 7, 2022, the company announced that", fd) == 1)
check("dated prior: 'press release' counts", E.cue_dated_prior("a press release dated march 8, 2022 is attached", fd) == 1)
check("dated prior: a date on the filing date does not count",
      E.cue_dated_prior("on march 10, 2022, the company announced that", fd) == 0)
check("dated prior: a later date does not count", E.cue_dated_prior("announced that effective april 1, 2022", fd) == 0)
near, far = "march 7, 2022" + "-" * 80 + "announced", "march 7, 2022" + "-" * 81 + "announced"
check("dated prior: 80-character window (gap between spans)", E.cue_dated_prior(near, fd) == 1 and E.cue_dated_prior(far, fd) == 0)
check("dated prior: keyword before the date also counts", E.cue_dated_prior("announced on march 7, 2022 that", fd) == 1)
check("dated prior: no keyword, no cue", E.cue_dated_prior("on march 7, 2022, mr. x resigned", fd) == 0)
check("prior wording: the three phrases", all(E.cue_prior_wording(E.normalize(f"As previously {w}, Ms. X")) == 1
                                             for w in ("announced", "Disclosed", "REPORTED")))
check("prior wording: other phrasing does not count", E.cue_prior_wording("as previously filed and announced previously") == 0)

# ---- EDGAR header ------------------------------------------------------------------------------------------------------
url = "https://www.sec.gov/Archives/edgar/data/1/0000000001-22-000001.txt"
check("header path uses the notebook's sha1 cache name",
      E.header_path(url, Path("c")) == Path("c") / f"sec_{hashlib.sha1(url.encode()).hexdigest()}.txt")
with tempfile.TemporaryDirectory() as tmp:
    p = E.header_path(url, Path(tmp))
    p.write_text("<SEC-HEADER>x\n<ACCEPTANCE-DATETIME>20220307171500\nCONFORMED PERIOD OF REPORT:\t20220304\nFILED AS OF DATE: 20220308\n")
    got = E.read_header(p)
    check("header: acceptance time and event date", got == (True, T("2022-03-07 17:15:00"), T("2022-03-04")), str(got))
    p.write_text("<SEC-HEADER>no fields here")
    has, a, ev = E.read_header(p)
    check("header without fields gives NaT", has and pd.isna(a) and pd.isna(ev))
    has, a, ev = E.read_header(Path(tmp) / "missing.txt")
    check("missing header is reported, never fetched", not has and pd.isna(a) and pd.isna(ev))

    acc = pd.DataFrame({"accession_number": ["A"], "ticker": ["X"], "filing_date": [T("2022-03-08")], "filing_url": [url]})
    p.write_text("<ACCEPTANCE-DATETIME>20220307171500\nCONFORMED PERIOD OF REPORT:\t20220304\n")
    df = E.date_filings(acc, CLOCK, Path(tmp))
    r = df.iloc[0]
    check("date_filings: t_0 next session after a 17:15 acceptance, t_pre that day's close",
          r.t_0 == T("2022-03-08") and r.t_pre == T("2022-03-07") and r.event_date == T("2022-03-04"))

# ---- Disclosures and accessions ---------------------------------------------------------------------------------------
def fake_fetch(tag, start, end):
    if tag == "uncached":
        raise CacheMiss("not in cache")
    if tag == "notickers":
        return pd.DataFrame({"accession_number": ["Z"]})
    return pd.DataFrame({"cik": ["1", "2"], "accession_number": ["A1", "B1"], "filing_date": pd.to_datetime(["2022-03-08"] * 2),
                         "tertiary_category": [tag, tag], "supporting_text": [f"text {tag}", "shared"],
                         "filing_url": ["u1", "u2"], "tickers": [["brk/b"], ["NOTTOP"]]})


NBF = {"fetch_disclosures": fake_fetch, "TOP_100": ["BRK.B"], "normalize_ticker": lambda t: t.strip().upper().replace("/", ".")}
rows, missing = E.load_disclosures(NBF, ["ceo_departure", "director_appointment", "uncached", "notickers"], "2022-01-01", "2023-12-31")
check("load_disclosures: TOP_100 only, tickers normalized, misses listed",
      set(rows.ticker) == {"BRK.B"} and len(rows) == 2 and missing == ["uncached"], f"{len(rows)} rows, missing {missing}")
dup = pd.concat([rows, rows.assign(tertiary_category="executive_compensation_change", supporting_text="text ceo_departure")])
at = E.accession_table(dup)
check("accession_table: one row per accession, tags sorted, excerpts de-duplicated",
      len(at) == 1 and at.tags[0] == "ceo_departure|director_appointment|executive_compensation_change"
      and at.text[0] == "text ceo_departure\ntext director_appointment", repr(at.text[0]))

g = E.select_groups(pd.DataFrame({"tags": ["director_appointment|cfo_departure|executive_compensation_change",
                                           "executive_officer_appointment|director_departure",
                                           "dividend_declaration|bylaw_amendment", "share_repurchase_program",
                                           "ceo_appointment|annual_meeting_results"]}))
check("groups and most-senior role", list(g.group) == ["people", "people", "placebo", "people"]
      and list(g.role) == ["CFO", "officer", "placebo", "CEO"], f"{list(g.group)} {list(g.role)}")

# ---- Gap fields: late, gap start, n_gap --------------------------------------------------------------------------------
gf = E.gap_fields(dated([
    ("fri_mon_pre", "X", "2022-03-07", "2022-03-07 08:00", "2022-03-04", "ceo_departure", ""),   # event Fri, filed Mon pre-open
    ("sat_mon", "X", "2022-03-07", "2022-03-07 08:00", "2022-03-05", "ceo_departure", ""),       # event Sat, filed Mon pre-open
    ("same_day_pm", "X", "2022-03-07", "2022-03-07 17:00", "2022-03-07", "ceo_departure", ""),   # event Mon, filed Mon evening
    ("tue_fri", "X", "2022-03-11", "2022-03-11 10:00", "2022-03-08", "ceo_departure", ""),       # event Tue, filed Fri intraday
    ("future", "X", "2022-03-07", "2022-03-07 10:00", "2022-03-09", "ceo_departure", ""),        # bad header: event after filing
    ("holiday", "X", "2022-04-18", "2022-04-18 09:00", "2022-04-14", "ceo_departure", ""),       # event Thu, Good Friday closed
]), CLOCK).set_index("accession_number")
exp = {"fri_mon_pre": (1, 1, "2022-03-03", 1), "sat_mon": (0, 0, "2022-03-04", 0), "same_day_pm": (0, 0, "2022-03-04", 1),
       "tue_fri": (3, 1, "2022-03-07", 3), "holiday": (1, 1, "2022-04-13", 1)}
for k, (lag, late, gs, n) in exp.items():
    r = gf.loc[k]
    check(f"gap fields: {k}", (r.lag_bd, r.late, r.gap_start, r.n_gap) == (lag, late, T(gs), n),
          f"lag {r.lag_bd}, late {r.late}, gap_start {r.gap_start}, n_gap {r.n_gap}")
r = gf.loc["future"]
check("gap fields: event after acceptance has no gap and is not late", r.lag_bd == -2 and r.late == 0
      and pd.isna(r.gap_start) and pd.isna(r.n_gap))
check("late filings always have a gap of at least one session", (gf[gf.late == 1].n_gap >= 1).all())

# ---- Earnings exclusion -------------------------------------------------------------------------------------------------
ev = pd.DataFrame({"ticker": ["X", "X", "X", "Y"], "t_0": pd.to_datetime(["2022-06-01", "2022-06-16", "2022-06-17", "2022-06-01"]),
                   "tags": ["ceo_departure|quarterly_earnings", "ceo_departure", "ceo_departure", "ceo_departure"]})
earn = pd.DataFrame({"ticker": ["X"], "t_0": pd.to_datetime(["2022-06-09"])})
check("earnings: own tag, +-5 sessions, not 6, not another ticker", list(E.earnings_flag(ev, earn, CLOCK)) == [1, 1, 0, 0],
      str(list(E.earnings_flag(ev, earn, CLOCK))))

# ---- Related earlier filing ---------------------------------------------------------------------------------------------
pool = dated([
    ("P1", "X", "2022-05-25", "2022-05-25 10:00", "2022-05-24", "director_departure", ""),   # 10 days before event: counts
    ("P2", "Y", "2022-04-01", "2022-04-01 10:00", "2022-03-31", "director_departure", ""),   # 35 days before: too early
    ("P3", "Z", "2022-06-06", "2022-06-06 18:00", "2022-05-30", "director_departure", ""),   # event in window, accepted after t_pre close
    ("P4", "W", "2022-06-06", "2022-06-06 12:00", "2022-06-01", "director_departure", ""),   # filed after the event date, public by t_pre
    ("P5", "W", "2022-06-03", "2022-06-03 12:00", "2022-06-03", "director_departure", ""),   # same event date: not "before"
])
rel = dated([
    ("E1", "X", "2022-06-06", "2022-06-06 08:00", "2022-06-03", "ceo_departure", ""),
    ("E2", "Y", "2022-06-06", "2022-06-06 08:00", "2022-05-06", "ceo_departure", ""),
    ("E3", "Z", "2022-06-06", "2022-06-06 08:00", "2022-06-03", "ceo_departure", ""),
    ("E4", "W", "2022-06-06", "2022-06-06 18:00", "2022-06-03", "ceo_departure", ""),
    ("P1", "X", "2022-05-25", "2022-05-25 10:00", "2022-05-24", "director_departure", ""),  # itself: never counts
])
got = list(E.related_filing_cue(rel, pd.concat([pool, rel.iloc[[0]]]), CLOCK))
check("related filing: window, other accession, public by the t_pre close", got == [1, 0, 0, 1, 0], str(got))
got = list(E.related_filing_cue(rel.iloc[[3]], pool[pool.accession_number == "P5"], CLOCK))
check("related filing: same event date is not 'before'", got == [0], str(got))

# ---- Events table end to end -------------------------------------------------------------------------------------------
acc = dated([
    ("A1", "X", "2022-03-07", "2022-03-07 08:00", "2022-03-04", "ceo_departure", "On March 4, 2022, the Company announced"),
    ("A2", "X", "2022-03-07", "2022-03-07 17:00", "2022-03-07", "annual_meeting_results", "As previously disclosed"),
    ("A3", "Y", "2023-12-29", "2023-12-29 17:00", "2023-12-27", "cfo_appointment", ""),           # enters 2024: dropped
    ("A4", "Y", "2022-05-02", "2022-05-02 08:00", "2022-04-28", "bylaw_amendment|director_appointment", ""),
    ("A5", "Y", "2022-05-02", "2022-05-02 08:00", "2022-04-28", "share_repurchase_program", "x"),  # neither set
])
acc.loc[acc.accession_number == "A4", "has_header"] = False
evt, counts = E.event_table(acc, acc[acc.tags.str.contains("ceo|cfo|director")], acc.iloc[:0], CLOCK, "2024-01-01")
SHARED = ["row_id", "kind", "accession_number", "ticker", "filing_date", "accepted_at", "event_date", "gap_start", "t_pre",
          "t_0", "n_gap", "lag_bd", "late", "group", "role", "tags", "earnings_excluded", "cue_dated_prior",
          "cue_prior_wording", "cue_related_filing", "T", "text_source"]
check("events: columns exactly as the shared interface", list(evt.columns) == SHARED and list(evt.columns) == E.EVENT_COLUMNS)
check("events: drops entry in 2024 and missing headers, keeps the rest", list(evt.accession_number) == ["A1", "A2"]
      and counts["people_entry_on_or_after_2024-01-01"] == 1 and counts["people_no_header"] == 1, str(counts))
r1, r2 = evt.iloc[0], evt.iloc[1]
check("events: row ids, groups, roles, cues and T",
      (r1.row_id, r1.group, r1.role, r1.cue_dated_prior, r1["T"], r1.text_source, r1.late) == ("event|A1", "people", "CEO", 1, 1, "excerpt", 1)
      and (r2.group, r2.role, r2.cue_prior_wording, r2.late) == ("placebo", "placebo", 1, 0))
check("events: a placebo row gets the related cue from an earlier people filing", (r2.cue_related_filing, r2["T"]) == (1, 2))

# ---- Ordinary days -----------------------------------------------------------------------------------------------------
evn = pd.DataFrame({"row_id": ["event|A1", "event|A1b", "event|A2"], "accession_number": ["A1", "A1b", "A2"],
                    "ticker": ["X", "X", "X"], "filing_date": pd.to_datetime(["2022-03-07", "2022-03-07", "2022-03-08"]),
                    "n_gap": pd.array([3, 2, 0], dtype="Int64")})       # A1 and A1b: one ticker, one filing date
drawn = pd.DataFrame({"ticker": ["X", "X", "X", "Q"], "filing_date": pd.to_datetime(["2022-03-07", "2022-03-07", "2022-03-08", "2022-03-07"]),
                      "round": [1, 2, 1, 1], "t_0": pd.to_datetime(["2022-02-01", "2022-04-20", "2022-02-02", "2022-02-01"]),
                      "t_pre": pd.to_datetime(["2022-01-31", "2022-04-19", "2022-02-01", "2022-01-31"])})
nl = E.null_table(evn, drawn, CLOCK)
check("nulls: columns exactly as the shared interface", list(nl.columns) == E.NULL_COLUMNS)
check("nulls: joined to their event by ticker and filing date",
      list(nl.event_row_id) == ["event|A1", "event|A1", "event|A1b", "event|A1b", "event|A2"], str(list(nl.event_row_id)))
check("nulls: pseudo-gap of the event's own n_gap sessions (calendar index, skips the holiday)",
      list(nl.gap_start[:4]) == [T("2022-01-26"), T("2022-04-13"), T("2022-01-27"), T("2022-04-14")] and pd.isna(nl.gap_start[4]),
      str(list(nl.gap_start)))
check("nulls: row id ends with the matched event's accession",
      nl.row_id[0] == "null|X|2022-01-31|2022-02-01|1|A1" and nl.row_id[2] == "null|X|2022-01-31|2022-02-01|1|A1b"
      and (nl.kind == "null").all())
every = pd.concat([pd.Series(["event|A1", "event|A1b", "event|A2"]), nl.row_id])
check("every row_id across events and nulls is unique (two events share a ticker, filing date and ordinary days)",
      every.is_unique and nl.row_id.is_unique, f"{len(every)} ids")

# ---- Playground times -------------------------------------------------------------------------------------------------
base = dated([
    ("M1", "X", "2022-03-07", "2022-03-07 08:00", "2022-03-04", "ceo_departure", ""),            # same times as the playground
    ("M2", "X", "2022-03-07", "2022-03-07 17:00", "2022-03-07", "bylaw_amendment", ""),          # accepted later the same day
    ("M3", "Y", "2022-03-08", "2022-03-08 08:00", "2022-03-07", "ceo_departure", ""),            # ticker missing in the playground file
    ("M4", "X", "2022-03-07", "2022-03-07 17:30", "2022-03-07", "share_repurchase_program", ""),  # conflicts, but in neither set
])
pgt = pd.DataFrame({"ticker": ["X"], "filing_date": [T("2022-03-07")], "t_0": [T("2022-03-07")], "t_pre": [T("2022-03-04")]})
confl = E.playground_conflicts(base, pgt)
check("conflicts: only filings whose own times differ from the playground's (after-close sibling, not equal or absent)",
      dict(zip(base.accession_number, confl)) == {"M1": False, "M2": True, "M3": False, "M4": True}, str(dict(zip(base.accession_number, confl))))
check("conflicts: a filing without a cached header is never flagged",
      not E.playground_conflicts(base.assign(has_header=False), pgt).any())
dr = E.dropped_table(base, confl)
check("dropped table: people and placebo filings only, the interface columns and the reason",
      list(dr.columns) == E.DROPPED_COLUMNS and list(dr.row_id) == ["event|M2"] and dr.accepted_at[0] == T("2022-03-07 17:00")
      and dr.reason[0] == "entry would precede acceptance; same-day earlier filing in the playground table", str(dr.to_dict("list")))
kept = base[~base.accession_number.isin(["M2"])]
fe, _ = E.event_table(kept, base, base.iloc[:0], CLOCK, "2024-01-01")
dn = E.null_table(fe, pd.DataFrame({"ticker": ["X", "Y"], "filing_date": pd.to_datetime(["2022-03-07", "2022-03-08"]), "round": [1, 1],
                                    "t_0": pd.to_datetime(["2022-02-01", "2022-02-02"]), "t_pre": pd.to_datetime(["2022-01-31", "2022-02-01"])}), CLOCK)
check("a dropped filing leaves neither an event row nor ordinary days; the others keep theirs",
      "event|M2" not in set(fe.row_id) and set(fe.row_id) == {"event|M1", "event|M3"}
      and set(dn.event_row_id) == {"event|M1", "event|M3"} and pd.concat([fe.row_id, dn.row_id]).is_unique)

# ---- CSV reading: "null" and the ticker "NA" are values, only empty cells are missing ------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    f = Path(tmp) / "x.csv"
    f.write_text("ticker,filing_date,tags,gap_start\nNA,2022-03-07,null,\nT,2022-03-08,none|n/a,2022-03-01\n")
    got = pd.read_csv(f, parse_dates=["filing_date", "gap_start"], **E.CSV_KW)
    check("read_csv settings keep 'NA', 'null' and 'n/a' as text and empty cells missing",
          list(got.ticker) == ["NA", "T"] and list(got.tags) == ["null", "none|n/a"] and got.gap_start.isna().tolist() == [True, False])

# ---- Date guards ---------------------------------------------------------------------------------------------------------
def raises(f):
    try:
        f()
    except (ValueError, AssertionError):
        return True
    return False


check("check_dates refuses a discovery date in 2024",
      raises(lambda: E.check_dates(pd.DataFrame({"t_0": [T("2024-01-02")]}), ["t_0"], "2024-01-01", "x")))
check("check_dates refuses a 2026 date at the insample hard stop",
      raises(lambda: E.check_dates(pd.DataFrame({"t_0": [T("2026-01-02")]}), ["t_0"], "2026-01-01", "x")))
check("check_dates accepts discovery dates and blanks",
      not raises(lambda: E.check_dates(pd.DataFrame({"t_0": [T("2023-12-29"), pd.NaT]}), ["t_0"], "2024-01-01", "x")))
check("check_dates with no hard stop (holdout, oos) accepts any date",
      not raises(lambda: E.check_dates(pd.DataFrame({"t_0": [T("2026-09-15")]}), ["t_0"], None, "x")))


# ---- Windows and guards by label (.claude/ctx/06_window_guard.md) --------------------------------------------------------
import requests as _requests            # noqa: E402


def _no_network(*a, **k):
    raise AssertionError("a test reached the network")


_requests.get = _no_network             # every test must stay offline; tests that fetch pass their own fake


def refused(f, kind=PermissionError, text=None):
    try:
        f()
    except kind as e:
        return text is None or text in str(e)
    except Exception as e:  # noqa: BLE001
        print("   raised", type(e).__name__, str(e)[:100])
    return False


RETIRED = ("2022-2023 is outside the allowed 2024-2025 window and overlaps the sealed placeholder (2023-06-01..2023-08-31)")
NB0 = {"OOS_START": "2026-01-01", "OOS_END": "2026-08-31", "HOLDOUT_START": "2023-06-01", "HOLDOUT_END": "2023-08-31"}
rw = lambda label, window=None, **flags: E.resolve_window(label, {**NB0, **flags}, window)      # noqa: E731
check("retired labels refuse with the stated message, whatever the window and switches",
      all(refused(lambda l=l, w=w: rw(l, w, RUN_OOS=True, RUN_HOLDOUT=True), text=RETIRED)
          for l in ("discovery", "dryrun") for w in (None, ("2022-01-01", "2023-12-31"), ("2024-03-01", "2024-03-31")))
      and E.RETIRED_MESSAGE == RETIRED and refused(lambda: E.resolve_window("discovery", {}), text=RETIRED))
check("build refuses discovery and dryrun before loading the notebook or reading any file",
      refused(lambda: E.build("discovery", None, playground=Path("/nonexistent")), text=RETIRED)
      and refused(lambda: E.build("dryrun", {}, playground=Path("/nonexistent"), window=("2024-01-01", "2024-06-30")), text=RETIRED)
      and "discovery" not in E.WINDOWS and "dryrun" not in E.WINDOWS and not hasattr(E, "PRIOR"))
check("insample default window and a window argument inside [2024-01-01, 2026-01-01)",
      rw("insample") == ("2024-01-01", "2025-12-31", "2026-01-01")
      and rw("insample", (T("2024-03-01"), "2024-03-31")) == ("2024-03-01", "2024-03-31", "2026-01-01")
      and rw("insample", ("2025-12-31", "2025-12-31")) == ("2025-12-31", "2025-12-31", "2026-01-01"))
check("insample refuses a window that starts before 2024, reaches 2026 or sits in 2022-23",
      all(refused(lambda w=w: rw("insample", w)) for w in (("2023-12-31", "2024-06-30"), ("2025-06-01", "2026-01-01"),
                                                          ("2026-03-01", "2026-03-31"), ("2022-01-01", "2023-12-31"),
                                                          ("2023-06-01", "2023-08-31"), ("2024-01-01", "2026-08-31"))))
check("insample refuses a window that overlaps the sealed window the judges set, even inside 2024-25",
      refused(lambda: rw("insample", ("2024-06-01", "2025-12-31"), HOLDOUT_START="2025-03-01", HOLDOUT_END="2025-03-31"))
      and refused(lambda: rw("insample", ("2024-06-01", "2025-03-01"), HOLDOUT_START="2025-03-01", HOLDOUT_END="2025-03-31"))
      and refused(lambda: rw("insample", ("2025-03-31", "2025-12-31"), HOLDOUT_START="2025-03-01", HOLDOUT_END="2025-03-31"))
      and rw("insample", ("2024-06-01", "2025-02-28"), HOLDOUT_START="2025-03-01", HOLDOUT_END="2025-03-31")[:2] == ("2024-06-01", "2025-02-28")
      and rw("insample", ("2025-04-01", "2025-12-31"), HOLDOUT_START="2025-03-01", HOLDOUT_END="2025-03-31")[:2] == ("2025-04-01", "2025-12-31"))
check("without HOLDOUT_START and END in the namespace the sealed placeholder 2023-06-01..2023-08-31 applies",
      E.sealed_window({}) == ("2023-06-01", "2023-08-31") and E.sealed_window({"HOLDOUT_START": "2025-03-01", "HOLDOUT_END": "2025-03-31"}) == ("2025-03-01", "2025-03-31")
      and E.resolve_window("insample", {"OOS_START": "2026-01-01"}, ("2024-01-01", "2024-06-30"))[:2] == ("2024-01-01", "2024-06-30"))
check("an unnamed label is refused", refused(lambda: rw("myrun", ("2024-05-01", "2024-05-31"))) and refused(lambda: rw("myrun")))
check("a third window item is ignored: it cannot loosen the insample guard or the holdout switch",
      refused(lambda: rw("insample", ("2025-06-01", "2026-02-01", "2030-01-01")))
      and rw("insample", ("2024-05-01", "2024-05-31", "2020-01-01")) == ("2024-05-01", "2024-05-31", "2026-01-01")
      and rw("holdout", ("2026-09-01", "2026-09-30", "2026-01-01"), RUN_HOLDOUT=True) == ("2026-09-01", "2026-09-30", None))
check("holdout needs RUN_HOLDOUT to be exactly True (False, missing, 1 and 'yes' refuse)",
      all(refused(lambda f=f: rw("holdout", **f)) for f in ({}, {"RUN_HOLDOUT": False}, {"RUN_HOLDOUT": 1}, {"RUN_HOLDOUT": "yes"}))
      and rw("holdout", RUN_HOLDOUT=True) == ("2023-06-01", "2023-08-31", None))
check("holdout with its switch has no date guard, even in 2026 and beyond",
      rw("holdout", ("2026-09-01", "2027-02-28"), RUN_HOLDOUT=True) == ("2026-09-01", "2027-02-28", None))
check("oos needs RUN_OOS to be exactly True, then takes the notebook's OOS dates",
      all(refused(lambda f=f: rw("oos", **f)) for f in ({}, {"RUN_OOS": False}, {"RUN_OOS": 1}))
      and rw("oos", RUN_OOS=True) == ("2026-01-01", "2026-08-31", "2026-09-01"))
check("each switch unlocks only its own label; neither loosens insample",
      refused(lambda: rw("holdout", RUN_OOS=True)) and refused(lambda: rw("oos", RUN_HOLDOUT=True))
      and refused(lambda: rw("insample", ("2025-06-01", "2026-03-01"), RUN_OOS=True, RUN_HOLDOUT=True))
      and refused(lambda: rw("insample", ("2026-01-01", "2026-03-01"), RUN_OOS=True, RUN_HOLDOUT=True)))
check("insample refuses when its hard stop passes the notebook's out-of-sample start",
      refused(lambda: E.resolve_window("insample", {"OOS_START": "2025-07-01"}, ("2025-01-01", "2025-03-01"))))
check("window validation: not a real date, start after end, no default",
      refused(lambda: rw("insample", ("2024-02-30", "2024-03-31")), ValueError) and refused(lambda: rw("insample", ("2024-05-01", "2024-04-01")), ValueError)
      and refused(lambda: E.resolve_window("holdout", {"RUN_HOLDOUT": True}), ValueError))
check("build keeps its old signature (positional label, NB, out_dir, playground, write; new arguments last; insample default)",
      list(inspect.signature(E.build).parameters)[:5] == ["label", "NB", "out_dir", "playground", "write"]
      and list(inspect.signature(E.build).parameters)[5:] == ["window", "cache_dir", "full_text"]
      and inspect.signature(E.build).parameters["label"].default == "insample")

# outside_allowed and the table-level bounds
SEALED = ("2023-06-01", "2023-08-31")
d = pd.DataFrame({"a": pd.to_datetime(["2023-12-29", "2024-01-02", "2025-12-31", "2026-01-01", "2023-07-04", None, "2025-03-15"])})
check("outside_allowed: before the first date, on or after the stop, inside the sealed window; blanks are fine",
      list(E.outside_allowed(d, ["a"], "2024-01-01", "2026-01-01", SEALED)) == [True, False, False, True, True, False, False]
      and list(E.outside_allowed(d, ["a"], "2024-01-01", "2026-01-01", ("2025-03-01", "2025-03-31")))[-1] is True)
acc_b = dated([
    ("B1", "X", "2024-01-03", "2024-01-03 08:00", "2023-12-28", "ceo_departure", "x"),        # the gap starts 2023-12-27
    ("B2", "X", "2024-03-12", "2024-03-12 08:00", "2024-03-07", "ceo_departure", "x"),        # fine
])
bounds = ("2024-01-01", "2026-01-01", SEALED)
e_in, c_in = E.event_table(acc_b, acc_b, acc_b.iloc[:0], CLOCK, "2026-01-01", bounds)
e_all, c_all = E.event_table(acc_b, acc_b, acc_b.iloc[:0], CLOCK, "2026-01-01")
check("event_table with bounds drops an event whose gap starts before 2024, counts it and records why; without bounds it stays",
      list(e_in.accession_number) == ["B2"] and c_in["people_gap_reaches_before_2024-01-01"] == 1
      and list(e_all.accession_number) == ["B1", "B2"] and "people_gap_reaches_before_2024-01-01" not in c_all
      and e_in.attrs["outside"].to_dict("list")["row_id"] == ["event|B1"]
      and e_in.attrs["outside"].reason.tolist() == ["gap reaches before 2024-01-01"] and list(e_in.attrs["outside"].columns) == E.DROPPED_COLUMNS, str(c_in))
rs = E.outside_reason(pd.DataFrame({"a": pd.to_datetime(["2023-12-29", "2026-01-02", "2024-05-01", "2023-07-04", None]),
                                    "b": pd.to_datetime(["2024-02-01", "2024-02-01", "2024-02-01", "2026-01-02", None])}), ["a", "b"],
                      "2024-01-01", "2026-01-01", SEALED)
check("outside_reason: a date before 2024 gets the gap reason (it wins), a later or sealed date the other reason, blanks none",
      rs.tolist() == ["gap reaches before 2024-01-01", "a date is on or after 2026-01-01 or inside the sealed window", "",
                      "gap reaches before 2024-01-01", ""], str(rs.tolist()))
dn_b = E.null_table(e_all, pd.DataFrame({"ticker": ["X", "X", "X", "X"], "filing_date": pd.to_datetime(["2024-01-03"] * 2 + ["2024-03-12"] * 2),
                                         "round": [1, 2, 1, 2], "t_0": pd.to_datetime(["2024-06-04", "2024-06-05", "2024-01-03", "2024-06-06"]),
                                         "t_pre": pd.to_datetime(["2024-06-03", "2024-06-04", "2024-01-02", "2024-06-05"])}), CLOCK, bounds)
check("null_table with bounds drops an ordinary day whose pseudo-gap starts before 2024 and counts it",
      dn_b.attrs["outside_allowed"] == 1 and len(dn_b) == 3 and dn_b.gap_start.dropna().min() >= T("2024-01-01"),
      f"{dn_b.attrs['outside_allowed']} dropped, {len(dn_b)} kept")


# ---- build() for every label, on a fake notebook namespace (no real cache, no network) ---------------------------------------
class Session:                      # stands in for requests.Session; go_offline sets .get on it
    pass


def fil(acc, date, accepted, period, tags=("ceo_departure",), cached=True):
    return {"acc": acc, "ticker": "AAA", "date": date, "accepted": accepted, "period": period, "tags": list(tags), "cached": cached}


def furl(f):
    return f"https://www.sec.gov/Archives/edgar/data/1/{f['acc']}.txt"


def times(f):
    a, d = pd.to_datetime(f["accepted"], format="%Y%m%d%H%M%S"), T(f["date"])
    return entry_session(a, d), pre_session(a, d)


def sub(*docs):
    """An EDGAR full submission text from (type, body) documents."""
    return "<SEC-HEADER>x</SEC-HEADER>\n" + "".join(
        f"<DOCUMENT>\n<TYPE>{t}\n<SEQUENCE>{i}\n<FILENAME>f{i}.htm\n<TEXT>\n{b}\n</TEXT>\n</DOCUMENT>\n" for i, (t, b) in enumerate(docs, 1))


def run_build(label, filings, window=None, fetchable=True, positional=False, full_text=None, full_docs=None, **flags):
    """build() against a fake namespace: returns (events, nulls, namespace, calls, output file names, dropped table)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cache, pg, out = tmp / "cache", tmp / "pg", tmp / "out"
        cache.mkdir(), pg.mkdir()
        by_url, calls = {furl(f): f for f in filings}, {"header": [], "disclosures": []}
        hdr = lambda f: f"<ACCEPTANCE-DATETIME>{f['accepted']}\nCONFORMED PERIOD OF REPORT:\t{f['period']}\n"      # noqa: E731
        for f in filings:
            if f["cached"]:
                E.header_path(furl(f), cache).write_text(hdr(f))
        for acc_no, raw in (full_docs or {}).items():                  # cached full submission texts, as SecFetcher writes them
            E.full_text_path(furl(next(f for f in filings if f["acc"] == acc_no)), cache).write_text(raw)

        def fetch_disclosures(tag, start, end):
            calls["disclosures"].append((tag, start, end))
            rows = [{"cik": "1", "accession_number": f["acc"], "filing_date": T(f["date"]), "tertiary_category": tag,
                     "supporting_text": "text", "filing_url": furl(f), "tickers": ["aaa"]}
                    for f in filings if tag in f["tags"] and start <= f["date"] <= end]
            return pd.DataFrame(rows)

        def fetch_acceptance_time(url):
            calls["header"].append(url)
            if not fetchable:
                raise CacheMiss("EDGAR header not in cache")
            E.header_path(url, NB["CACHE_DIR"]).write_text(hdr(by_url[url]))

        NB = {"CAL": CAL, "entry_session": entry_session, "pre_session": pre_session, "close_time": lambda d: pd.Timedelta(hours=16),
              "fetch_disclosures": fetch_disclosures, "fetch_acceptance_time": fetch_acceptance_time,
              "normalize_ticker": lambda t: t.strip().upper().replace("/", "."), "TOP_100": ["AAA"],
              "api_get_all": lambda path, params=None: [{"tertiary_category": t} for t in E.REQUIRED_TAGS],
              "SESSION": Session(), "CACHE_DIR": cache, "SEC_USER_AGENT": "Test Agent test@example.org",
              "RUN_OOS": False, "RUN_HOLDOUT": False, **NB0, **flags}
        stop = {"insample": "2026-01-01"}.get(label, "2100-01-01")
        evs, nus = [], []
        for f in filings:                       # the playground's tables, as its download wrote them (no entry at or after the stop)
            t0, tp = times(f)
            if t0 < T(stop):
                evs.append({"ticker": f["ticker"], "filing_date": T(f["date"]), "t_0": t0, "t_pre": tp})
                i = CAL.searchsorted(T(f["date"]))
                nus += [{"ticker": f["ticker"], "filing_date": T(f["date"]), "round": r, "t_0": CAL[i - 20 - r], "t_pre": CAL[i - 21 - r]}
                        for r in (1, 2)]
        pd.DataFrame(evs).to_csv(pg / f"events_{label}.csv", index=False)
        pd.DataFrame(nus).to_csv(pg / f"nulls_{label}.csv", index=False)
        if positional:
            ev, nu = E.build(label, NB, out, pg, True, window=window, full_text=full_text)
        else:
            ev, nu = E.build(label, NB, out_dir=out, playground=pg, window=window, full_text=full_text)
        calls["cached_after"] = [furl(f) for f in filings if E.header_path(furl(f), cache).exists()]
        return ev, nu, NB, calls, sorted(p.name for p in out.iterdir()), ev.attrs["dropped"]


def offline(NB):
    try:
        NB["SESSION"].get("https://api.massive.com/x")
    except CacheMiss:
        return True
    except AttributeError:
        return False
    return False


INS = [fil("I1", "2025-03-11", "20250311080000", "20250306"),                      # kept: late, 3 sessions
       fil("I4", "2024-02-06", "20240206080000", "20240131"),                      # kept; one of its ordinary days has a gap before 2024
       fil("I3", "2024-01-03", "20240103080000", "20231228"),                      # its gap starts 2023-12-27: dropped, counted
       fil("I2", "2025-12-31", "20251231170000", "20251229"),                      # enters on or after 2026-01-01: dropped, counted
       fil("I5", "2025-05-13", "20250513080000", "20250508", cached=False)]        # no cached header: counted
ev, nu, NBr, calls, files, dr = run_build("insample", INS, positional=True)
check("insample: kept events are I1 and I4; the 2023 gap, the 2026 entry and the missing header are each dropped and counted",
      sorted(ev.accession_number) == ["I1", "I4"] and ev.attrs["counts"]["people_gap_reaches_before_2024-01-01"] == 1
      and ev.attrs["counts"]["people_entry_on_or_after_2026-01-01"] == 1 and ev.attrs["counts"]["people_no_header"] == 1, str(ev.attrs["counts"]))
check("insample: an ordinary day whose pseudo-gap starts before 2024 is dropped and counted; the rest keep their ids",
      ev.attrs["counts"]["null_rows_gap_reaches_before_2024-01-01"] == 1 and ev.attrs["counts"]["null_rows_outside_allowed_other"] == 0
      and len(nu) == 3 and nu.row_id.is_unique, f"{len(nu)} nulls, {ev.attrs['counts']}")
check("insample: dropped_insample.csv lists the gap-before-2024 event with its reason",
      dr.to_dict("list")["row_id"] == ["event|I3"] and dr.reason.tolist() == ["gap reaches before 2024-01-01"]
      and list(dr.columns) == E.DROPPED_COLUMNS, str(dr.to_dict("list")))
INS3 = [fil("I6", "2024-01-10", "20240110080000", "20231229"), fil("I7", "2024-01-25", "20240125080000", "20240122")]
ev3_, nu3_, *_r, dr3_ = run_build("insample", INS3)
check("insample pools: a filing dropped for its December gap still counts as a related earlier filing by its own (2024) filing date",
      list(ev3_.accession_number) == ["I7"] and dr3_.reason.tolist() == ["gap reaches before 2024-01-01"]
      and ev3_.cue_related_filing.tolist() == [1], f"{list(ev3_.accession_number)} {ev3_.cue_related_filing.tolist()}")
check("insample: every t_pre, t_0 and gap start in events and nulls lies in [2024-01-01, 2026-01-01) and outside the sealed window",
      not E.outside_allowed(ev, ["t_pre", "t_0", "gap_start"], "2024-01-01", "2026-01-01", SEALED).any()
      and not E.outside_allowed(nu, ["t_pre", "t_0", "gap_start"], "2024-01-01", "2026-01-01", SEALED).any()
      and ev.t_0.min() >= T("2024-01-01"))
check("insample: no disclosure query reaches outside 2024-2025 (no 2022-23 look-back), cache only, three files",
      calls["disclosures"] and all(s >= "2024-01-01" and e < "2026-01-01" for _, s, e in calls["disclosures"])
      and offline(NBr) and calls["header"] == [] and files == ["dropped_insample.csv", "events_insample.csv", "nulls_insample.csv"], str(files))
check("insample refuses a window that reaches 2026, starts before 2024 or overlaps the sealed window, before reading anything",
      all(refused(lambda w=w, f=f: E.build("insample", {**NBr, "SESSION": Session(), **f}, playground=Path("/nonexistent"), window=w))
          for w, f in ((("2025-06-01", "2026-01-01"), {}), (("2023-12-01", "2024-03-31"), {}), (("2023-06-01", "2023-08-31"), {}),
                       (("2024-01-01", "2025-12-31"), {"HOLDOUT_START": "2025-03-01", "HOLDOUT_END": "2025-03-31"}))))
ev2, nu2, *_ = run_build("insample", INS, window=("2024-07-01", "2025-12-31"), HOLDOUT_START="2024-02-01", HOLDOUT_END="2024-02-28")
check("insample with a moved sealed window and a window that clears it: builds, and drops what touches the sealed window",
      sorted(ev2.accession_number) == ["I1"] and not E.outside_allowed(nu2, ["t_pre", "t_0", "gap_start"], "2024-07-01", "2026-01-01", ("2024-02-01", "2024-02-28")).any())

HOLD = [fil("H1", "2026-09-15", "20260915080000", "20260910", cached=False), fil("H2", "2026-09-22", "20260922080000", "20260917")]
check("holdout without RUN_HOLDOUT is refused before anything is read or fetched",
      refused(lambda: run_build("holdout", HOLD, window=("2026-09-01", "2026-09-30"))))
check("holdout with only RUN_OOS on is still refused", refused(lambda: run_build("holdout", HOLD, window=("2026-09-01", "2026-09-30"), RUN_OOS=True)))
ev, nu, NBr, calls, files, dr = run_build("holdout", HOLD, window=("2026-09-01", "2026-09-30"), RUN_HOLDOUT=True)
check("holdout: a 2026 window builds, with no date guard on t_pre, t_0 or the gap",
      list(ev.row_id) == ["event|H1", "event|H2"] and ev.t_0.min() >= T("2026-09-01") and ev.late.tolist() == [1, 1]
      and nu.event_row_id.nunique() == 2 and ev.gap_start.notna().all(), f"{list(ev.row_id)}")
check("holdout: the header missing from the cache is fetched through the notebook's fetch_acceptance_time (once); the cached one is not",
      calls["header"] == [furl(HOLD[0])] and furl(HOLD[0]) in calls["cached_after"] and not offline(NBr)
      and not hasattr(NBr["SESSION"], "get"))
ev, nu, NBr, calls, files, dr = run_build("holdout", HOLD, window=("2026-09-01", "2026-09-30"), RUN_HOLDOUT=True, fetchable=False)
check("holdout: a header that cannot be fetched leaves the filing out and counted, not a crash",
      list(ev.row_id) == ["event|H2"] and ev.attrs["counts"]["people_no_header"] == 1)
ev, nu, NBr, calls, files, dr = run_build("holdout", [fil("H3", "2023-07-18", "20230718080000", "20230713")], RUN_HOLDOUT=True)
check("holdout without window= uses the notebook's HOLDOUT_START and HOLDOUT_END (the judges' dates, whatever they are)",
      list(ev.row_id) == ["event|H3"])
check("holdout still refuses an unknown or missing switch value",
      refused(lambda: run_build("holdout", HOLD, window=("2026-09-01", "2026-09-30"), RUN_HOLDOUT="yes")))

OOS = [fil("O1", "2026-03-10", "20260310080000", "20260305")]
check("oos without RUN_OOS is refused; RUN_HOLDOUT alone does not unlock it",
      refused(lambda: run_build("oos", OOS)) and refused(lambda: run_build("oos", OOS, RUN_HOLDOUT=True)))
ev, nu, NBr, calls, files, dr = run_build("oos", OOS, RUN_OOS=True)
check("oos with RUN_OOS: the notebook's OOS window, fetching allowed like holdout (the notebook is not put offline)",
      list(ev.row_id) == ["event|O1"] and not offline(NBr) and ev.t_0.min() >= T("2026-01-01"))


# ---- Full-text variant: parsing -----------------------------------------------------------------------------------------------
FD = "2025-03-11"
raw = sub(("8-K", "x8k"), ("EX-99.1", "x99"), ("EX-101.INS", "<xbrl/>"), ("GRAPHIC", "begin 644 logo.jpg"), ("EX-10.1", "agreement"), ("8-K/A", "amend"))
check("split_documents reads every document type; keep_documents keeps the 8-K, 8-K/A and EX-99 exhibits only",
      [t for t, _ in E.split_documents(raw)] == ["8-K", "EX-99.1", "EX-101.INS", "GRAPHIC", "EX-10.1", "8-K/A"]
      and [t for t, _ in E.split_documents(E.keep_documents(raw))] == ["8-K", "EX-99.1", "8-K/A"]
      and "begin 644" not in E.keep_documents(raw) and "agreement" not in E.keep_documents(raw))
check("a truncated last document is still read", [t for t, _ in E.split_documents(raw[:raw.index("</DOCUMENT>") + 11] + "<DOCUMENT>\n<TYPE>EX-99.1\n<TEXT>\npart")] == ["8-K", "EX-99.1"])
check("html_to_text drops head, style, script and the inline-XBRL header, decodes entities, turns tags into spaces",
      E.normalize(E.html_to_text("<html><head><title>March 7, 2025 announced</title></head><body><style>p{}</style>"
                                 "<ix:header><ix:hidden>March 6, 2025 announced</ix:hidden></ix:header>"
                                 "<p>On&#160;March&#160;7,<b> 2025</b>, we&#8217;re <i>previously</i>announced</p><script>var a=1</script></body></html>"))
      == "on march 7, 2025 , we’re previously announced")
B_HIDDEN = ("<html><head><title>Mar 7, 2025 announced</title></head><body><ix:header><ix:hidden>March 7, 2025 announced</ix:hidden>"
            "</ix:header><p>On March&#160;12, 2025, the Company announced that Ms. Y will retire.</p></body></html>")
B_DATED = "<p>On March&#160;7, 2025, the Company announced that Ms. Y will retire.</p>"
B_WORDING = "<p>As <b>previously</b> announced, Mr. X will step down.</p>"
c = E.full_text_cues(sub(("8-K", B_HIDDEN)), FD)
check("full text: a date only in the head or hidden XBRL, or a later date, does not fire the dated-prior cue",
      c == {"full_text_status": "ok", "cue_dated_prior_full": 0, "cue_prior_wording_full": 0, "cue_exhibit_dated_prior_full": 0}, str(c))
c = E.full_text_cues(sub(("8-K", B_DATED)), FD)
check("full text: an earlier date next to 'announced' fires the dated-prior cue only",
      (c["cue_dated_prior_full"], c["cue_prior_wording_full"], c["cue_exhibit_dated_prior_full"]) == (1, 0, 0), str(c))
c = E.full_text_cues(sub(("8-K", B_WORDING)), FD)
check("full text: 'previously' split by a tag still fires the prior-wording cue", (c["cue_dated_prior_full"], c["cue_prior_wording_full"]) == (0, 1), str(c))
EX_PRIOR = "<html><body><p>Exhibit 99.1</p><p>CUPERTINO, Calif., March 7, 2025 &#8212; the company announced a new officer.</p></body></html>"
c = E.full_text_cues(sub(("8-K", "<p>Item 5.02 departure.</p>"), ("EX-99.1", EX_PRIOR)), FD)
check("full text: an EX-99 press release dated before the filing fires the exhibit flag (and the dated-prior cue)",
      (c["cue_exhibit_dated_prior_full"], c["cue_dated_prior_full"], c["full_text_status"]) == (1, 1, "ok"), str(c))
c = E.full_text_cues(sub(("8-K", "x"), ("EX-99.1", EX_PRIOR.replace("March 7, 2025", "March 11, 2025"))), FD)
check("full text: a press release dated the filing date is not dated before it", (c["cue_exhibit_dated_prior_full"], c["cue_dated_prior_full"]) == (0, 0), str(c))
c = E.full_text_cues(sub(("8-K", "x"), ("EX-99.1", "<p>Effective April 1, 2025 the board announced a change. Dated March 7, 2025.</p>")), FD)
check("full text: the exhibit flag reads the FIRST date in its head (a later effective date first means no flag)", c["cue_exhibit_dated_prior_full"] == 0, str(c))
c = E.full_text_cues(sub(("8-K", "x"), ("EX-99.1", "<p>" + "x " * 1000 + " March 7, 2025</p>")), FD)
check("full text: a date beyond the first 1,500 characters of the exhibit is not its dateline", c["cue_exhibit_dated_prior_full"] == 0)
c = E.full_text_cues(sub(("8-K", "<p>zz effective March 7, 2025</p>"), ("EX-99.1", "<p>announced the appointment</p>")), FD)
check("full text: the date at the end of one document and a keyword at the start of the next are not 'near' each other",
      (c["cue_dated_prior_full"], c["cue_exhibit_dated_prior_full"]) == (0, 0), str(c))
c = E.full_text_cues(sub(("EX-99.1", EX_PRIOR), ("EX-10.1", B_DATED)), FD)
check("full text: no 8-K document gives status 'empty'; other exhibits are never read", c["full_text_status"] == "empty" and c["cue_dated_prior_full"] == 1, str(c))
check("full text of an empty or header-only submission is 'empty'", E.full_text_cues("<SEC-HEADER>x</SEC-HEADER>", FD)["full_text_status"] == "empty")


# ---- Full-text variant: the polite fetcher (a scripted fake network, no real requests) --------------------------------------
class FakeResp:
    def __init__(self, status=200, body=b"", headers=None):
        self.status_code, self.body, self.headers = status, body, headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_content(self, n):
        for i in range(0, len(self.body), n):
            yield self.body[i:i + n]


class Net:
    """Scripted responses per URL (the last repeats); records each request with its fake time, and every sleep."""
    def __init__(self, script):
        self.script, self.log, self.sleeps, self.t = {k: list(v) for k, v in script.items()}, [], [], 0.0

    def get(self, url, headers=None, timeout=None, stream=None):
        self.log.append((url, dict(headers or {}), self.t))
        q = self.script[url]
        r = q.pop(0) if len(q) > 1 else q[0]
        if isinstance(r, Exception):
            raise r
        return r

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s

    def clock(self):
        return self.t


UA = "Test Agent test@example.org"
OK = lambda body="<p>x</p>": FakeResp(200, sub(("8-K", body)).encode("latin-1"))       # noqa: E731
ur = lambda k: f"https://www.sec.gov/Archives/edgar/data/1/{k}.txt"                      # noqa: E731


def make_fetcher(net, tmp, **kw):
    return E.SecFetcher(UA, tmp, get=net.get, sleep=net.sleep, clock=net.clock, **kw)


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    net = Net({ur(i): [OK()] for i in range(9)})
    fx = make_fetcher(net, tmp)
    got = [fx.fetch(ur(i)) for i in range(9)]
    times_ = [t for _, _, t in net.log]
    gaps = np.diff(times_)
    check("fetcher: nine filings take nine requests, never faster than 8 per second, with the real User-Agent",
          fx.requests == 9 and all(g is not None for g in got) and gaps.min() >= 0.125 - 1e-9 and 8 / (times_[-1] - times_[0]) <= 8 + 1e-9
          and all(h["User-Agent"] == UA for _, h, _ in net.log), f"min gap {gaps.min():.3f}s")
    check("fetcher: each document is cached as sec_full_<sha1(filing_url)>.txt, trimmed, and never fetched again",
          E.full_text_path(ur(0), tmp).name == "sec_full_" + hashlib.sha1(ur(0).encode()).hexdigest() + ".txt"
          and E.full_text_path(ur(0), tmp).exists() and fx.fetch(ur(0)) == got[0] and len(net.log) == 9)
check("fetcher: the rate is capped at 8 per second however high max_rps is set",
      abs(E.SecFetcher(UA, ".", max_rps=100).interval - 0.125) < 1e-12 and abs(E.SecFetcher(UA, ".", max_rps=2).interval - 0.5) < 1e-12)
check("fetcher: an empty or placeholder SEC_USER_AGENT is refused",
      all(refused(lambda u=u: E.SecFetcher(u, "."), ValueError) for u in ("", "  ", "your@email.edu", "team-name x")))
with tempfile.TemporaryDirectory() as tmp:
    net = Net({ur("a"): [FakeResp(503), OK()], ur("b"): [FakeResp(429, headers={"Retry-After": "3"}), OK()],
               ur("c"): [_requests.ConnectionError("x"), _requests.Timeout("y"), _requests.ConnectionError("z")],
               ur("d"): [FakeResp(503)], ur("e"): [FakeResp(404)]})
    fx = make_fetcher(net, Path(tmp))
    ra, rb = fx.fetch(ur("a")), fx.fetch(ur("b"))
    check("fetcher: a 503 is retried after a backoff of 1 s, and a 429 waits for its Retry-After",
          ra is not None and rb is not None and 1 in net.sleeps and 3.0 in net.sleeps and [u for u, _, _ in net.log].count(ur("a")) == 2)
    n0 = len(net.sleeps)
    rc, rd, re_ = fx.fetch(ur("c")), fx.fetch(ur("d")), fx.fetch(ur("e"))
    check("fetcher: connection errors and 5xx are tried 3 times with backoff 1, 2, 4 and then given up; a 404 is not retried",
          rc is None and rd is None and re_ is None and [u for u, _, _ in net.log].count(ur("c")) == 3
          and [u for u, _, _ in net.log].count(ur("d")) == 3 and [u for u, _, _ in net.log].count(ur("e")) == 1
          and {1, 2, 4} <= set(net.sleeps[n0:]) and fx.failures == 3)
with tempfile.TemporaryDirectory() as tmp:
    net = Net({ur(i): [FakeResp(404)] for i in range(5)} | {ur("ok"): [OK()]} | {ur(i): [FakeResp(404)] for i in range(10, 15)})
    fx = make_fetcher(net, Path(tmp))
    stopped = None
    try:
        for i in range(5):
            fx.fetch(ur(i))
    except RuntimeError as e:
        stopped = (i, str(e))
    check("fetcher: it stops with a RuntimeError at the fifth failed filing in a row (never hammer the SEC)",
          stopped is not None and stopped[0] == 4 and "5 filings in a row" in stopped[1], str(stopped))
    fx2 = make_fetcher(Net({ur(i): [FakeResp(404)] for i in range(10, 15)} | {ur("ok"): [OK()]}), Path(tmp) / "b")
    for i in (10, 11, 12, 13):
        fx2.fetch(ur(i))
    fx2.fetch(ur("ok"))
    fx2.fetch(ur(14))
    check("fetcher: a success resets the streak (four failures, a success, one more failure does not stop)", fx2.failures == 1)
with tempfile.TemporaryDirectory() as tmp:
    big = sub(("8-K", "<p>" + "a" * 300_000 + "</p>"))
    net = Net({ur("big"): [FakeResp(200, big.encode("latin-1"))]})
    fx = make_fetcher(net, Path(tmp), max_bytes=70_000)
    txt = fx.fetch(ur("big"))
    check("fetcher: a response is read only up to max_bytes", txt is not None and len(txt) < 200_000 and "</p>" not in txt[-300:] or len(txt) < len(big), str(len(txt)))


# ---- Full-text variant: add_full_text, the guards and build(full_text=) ----------------------------------------------------------
def events_frame():
    rows = [("a", "2025-03-11", 1, 0, 1, "people"), ("b", "2025-03-12", 0, 0, 0, "people"), ("c", "2025-03-13", 1, 1, 0, "people"),
            ("d", "2025-03-14", 1, 0, 0, "people"), ("e", "2025-03-17", 1, 0, 0, "placebo"), ("f", "2025-03-18", 1, 0, 0, "placebo")]
    ev = pd.DataFrame(rows, columns=["accession_number", "filing_date", "late", "earnings_excluded", "cue_related_filing", "group"])
    ev["filing_date"], ev["T"], ev["row_id"] = pd.to_datetime(ev["filing_date"]), 0, "event|" + ev["accession_number"]
    return ev


URLS = {k: ur(k) for k in "abcdef"}
NBF = {"RUN_HOLDOUT": False, "SEC_USER_AGENT": UA}
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    cached = {"a": sub(("8-K", B_DATED)), "e": sub(("EX-99.1", EX_PRIOR)), "f": sub(("8-K", "<p>x</p>"), ("EX-99.1", EX_PRIOR))}
    for k, raw_ in cached.items():
        E.full_text_path(URLS[k], tmp).write_text(E.keep_documents(raw_))
    ev0 = events_frame()
    out = E.add_full_text(ev0, URLS, NBF, "insample", cache_dir=tmp)
    check("add_full_text: existing columns untouched, FULL_COLUMNS appended, the input table unchanged",
          out[list(ev0.columns)].equals(ev0) and list(out.columns) == list(ev0.columns) + E.FULL_COLUMNS and list(ev0.columns) == list(events_frame().columns))
    o = out.set_index("accession_number")
    check("add_full_text: only late filings without an earnings exclusion are read; the rest are not_run with empty cues",
          o.full_text_status.to_dict() == {"a": "ok", "b": "not_run", "c": "not_run", "d": "missing", "e": "empty", "f": "ok"}
          and o.loc[["b", "c", "d", "e"], E.FULL_COLUMNS[1:]].isna().all().all())
    check("add_full_text: cues, the related cue copied, T_full (three plan cues) and T_full4 (plus the exhibit flag)",
          o.loc["a", E.FULL_COLUMNS[1:]].tolist() == [1, 0, 1, 0, 2, 2] and o.loc["f", E.FULL_COLUMNS[1:]].tolist() == [1, 0, 0, 1, 1, 2],
          f"{o.loc['a', E.FULL_COLUMNS[1:]].tolist()} {o.loc['f', E.FULL_COLUMNS[1:]].tolist()}")
    fs = E.full_text_summary(out)
    check("full_text_summary counts the kept late filings by status and cue",
          fs["status"] == {"ok": 2, "missing": 1, "empty": 1} and fs["people"]["filings"] == 1 and fs["people"]["cue_dated_prior_full"] == 1
          and fs["placebo"]["filings"] == 1 and fs["placebo"]["cue_exhibit_dated_prior_full"] == 1, str(fs))
    net = Net({URLS["d"]: [OK(B_WORDING)]})
    fx = make_fetcher(net, tmp)
    out2 = E.add_full_text(ev0, URLS, NBF, "insample", allow_fetch=True, cache_dir=tmp, fetcher=fx)
    o2 = out2.set_index("accession_number")
    check("add_full_text with fetching: only the filing missing from the cache is requested, then cached",
          len(net.log) == 1 and net.log[0][0] == URLS["d"] and o2.loc["d", "full_text_status"] == "ok" and o2.loc["d", "cue_prior_wording_full"] == 1
          and E.full_text_path(URLS["d"], tmp).exists() and o2.loc["a", "T_full"] == 2)
    out3 = E.add_full_text(ev0, URLS, NBF, "insample", cache_dir=tmp)
    check("add_full_text offline then reads the cached document too", out3.set_index("accession_number").loc["d", "full_text_status"] == "ok")

    # the strict data rule: nothing is fetched or read for 2022-23
    net = Net({URLS[k]: [OK()] for k in "abcdef"})
    old = events_frame().assign(filing_date=lambda d: d.filing_date - pd.DateOffset(years=2))
    check("full text refuses discovery, dryrun, oos and unnamed labels, and holdout without its switch, fetching nothing",
          all(refused(lambda l=l: E.add_full_text(old, URLS, NBF, l, allow_fetch=True, cache_dir=tmp, fetcher=make_fetcher(net, tmp)))
              for l in ("discovery", "dryrun", "oos", "myrun", "holdout")) and net.log == [])
    check("insample full text refuses a filing dated before 2024-01-01 and fetches nothing",
          refused(lambda: E.add_full_text(old, URLS, NBF, "insample", allow_fetch=True, cache_dir=tmp, fetcher=make_fetcher(net, tmp)), text="2024-01-01")
          and net.log == [])
    check("holdout full text runs with RUN_HOLDOUT True, on any dates",
          E.add_full_text(old, URLS, {**NBF, "RUN_HOLDOUT": True}, "holdout", cache_dir=tmp).full_text_status.eq("not_run").sum() == 2)

INS2 = [fil("I1", "2025-03-11", "20250311080000", "20250306"), fil("I4", "2024-02-06", "20240206080000", "20240131")]
ev0, *_ = run_build("insample", INS2)
ev1, nu1, NBr, calls, files, dr = run_build("insample", INS2, full_text="cache", full_docs={"I1": sub(("8-K", B_DATED), ("EX-99.1", EX_PRIOR))})
o = ev1.set_index("accession_number")
check("build(full_text='cache') on insample: the full-text columns are added and every other column is identical to the plain build",
      ev1[E.EVENT_COLUMNS].reset_index(drop=True).equals(ev0[E.EVENT_COLUMNS].reset_index(drop=True)) and list(ev1.columns) == E.EVENT_COLUMNS + E.FULL_COLUMNS
      and o.loc["I1", "full_text_status"] == "ok" and o.loc["I1", "cue_dated_prior_full"] == 1 and o.loc["I1", "cue_exhibit_dated_prior_full"] == 1
      and o.loc["I4", "full_text_status"] == "missing", str(o[E.FULL_COLUMNS].to_dict("index")))
check("build(full_text=) refuses before reading anything for oos, retired labels and a bad mode; holdout needs its switch",
      refused(lambda: E.build("oos", {**NB0, "RUN_OOS": False}, playground=Path("/nonexistent"), window=("2026-03-01", "2026-03-31"), full_text="cache"))
      and refused(lambda: E.build("dryrun", None, full_text="cache"), text=RETIRED)
      and refused(lambda: E.build("insample", {**NB0}, playground=Path("/nonexistent"), full_text="sometimes"), ValueError)
      and refused(lambda: E.build("holdout", {"RUN_HOLDOUT": False}, playground=Path("/nonexistent"), window=("2026-09-01", "2026-09-30"), full_text="cache")))
ev3, *_ = run_build("holdout", [fil("H3", "2026-09-15", "20260915080000", "20260910")], window=("2026-09-01", "2026-09-30"), RUN_HOLDOUT=True,
                    full_text="cache", full_docs={"H3": sub(("8-K", B_WORDING))})
check("build(full_text='cache') on holdout (RUN_HOLDOUT True): the same columns", ev3.full_text_status.tolist() == ["ok"] and ev3.cue_prior_wording_full.tolist() == [1])
# build(full_text='fetch') reaches sec.gov only through SecFetcher (here: the scripted fake), once per uncached late filing
_requests.get = Net({furl(INS2[0]): [OK(B_DATED)], furl(INS2[1]): [OK(B_WORDING)]}).get
ev4, *_ = run_build("insample", INS2, full_text="fetch")
_requests.get = _no_network
check("build(full_text='fetch'): uncached late filings are fetched through requests.get (the scripted fake) and read",
      sorted(ev4.full_text_status) == ["ok", "ok"] and ev4.set_index("accession_number").loc["I1", "cue_dated_prior_full"] == 1)

# ---- oos: the one-time 2026 run (only with RUN_OOS True; this module never sets it) -------------------------------------
check("oos window: inside OOS_START..OOS_END only, hard stop the day after OOS_END",
      rw("oos", ("2026-02-01", "2026-03-31"), RUN_OOS=True) == ("2026-02-01", "2026-03-31", "2026-09-01")
      and refused(lambda: rw("oos", ("2026-08-01", "2026-09-30"), RUN_OOS=True))
      and refused(lambda: rw("oos", ("2025-12-01", "2026-02-01"), RUN_OOS=True))
      and refused(lambda: E.resolve_window("oos", {"RUN_OOS": True}, ("2026-02-01", "2026-03-31"))))
check("oos with RUN_OOS False, missing, 1 or 'yes' refuses build before reading or fetching anything",
      all(refused(lambda f=f: E.build("oos", {**NB0, **f}, playground=Path("/nonexistent"))) for f in ({}, {"RUN_OOS": False}, {"RUN_OOS": 1}, {"RUN_OOS": "yes"})))
check("the module never switches RUN_OOS on", "RUN_OOS\"] = True" not in Path(E.__file__).read_text(encoding="utf-8")
      and "RUN_OOS'] = True" not in Path(E.__file__).read_text(encoding="utf-8"))
OOS2 = [fil("O2", "2026-03-10", "20260310080000", "20260305", cached=False),                # header fetched like holdout
        fil("O3", "2026-08-31", "20260831170000", "20260826"),                             # enters 2026-09-01: dropped, counted
        fil("O4", "2026-01-08", "20260108080000", "20260105"),                             # related to P1 (2025-12)
        fil("P1", "2025-12-15", "20251215080000", "20251212", tags=("director_departure",)),
        fil("P0", "2023-12-15", "20231215080000", "20231212", tags=("director_departure",))]
ev, nu, NBr, calls, files, dr = run_build("oos", OOS2, RUN_OOS=True, full_text="cache")
o = ev.set_index("accession_number")
check("oos: the uncached header is fetched; an entry after OOS_END is dropped and counted; no entry, pre-event or gap date after OOS_END",
      sorted(ev.accession_number) == ["O2", "O4"] and calls["header"] == [furl(OOS2[0])]
      and ev.attrs["counts"]["people_entry_on_or_after_2026-09-01"] == 1
      and ev[["t_pre", "t_0", "gap_start"]].max().max() <= T("2026-08-31") and nu[["t_pre", "t_0", "gap_start"]].max().max() <= T("2026-08-31"),
      str(ev.attrs["counts"]))
check("oos pools: look back into 2024-25 only (a 2025 people filing is a related earlier filing; nothing before 2024 is queried)",
      o.loc["O4", "cue_related_filing"] == 1 and all(s >= "2024-01-01" for _, s, _e in calls["disclosures"])
      and any(s == "2024-01-01" and e == "2025-12-31" for _, s, e in calls["disclosures"]))
check("oos full text runs with RUN_OOS True (cache mode: missing documents are reported, not fetched)",
      set(ev.full_text_status) <= {"missing", "not_run"} and (ev.full_text_status == "missing").sum() == 2)
ern = lambda acc, date, accepted: fil(acc, date, accepted, date.replace("-", ""), tags=("quarterly_earnings",))   # noqa: E731
EV_O = fil("O5", "2026-03-10", "20260310080000", "20260305")
ev_after, *_ = run_build("oos", [EV_O, ern("E1", "2026-03-12", "20260312070000")], RUN_OOS=True)
ev_before, *_ = run_build("oos", [EV_O, ern("E2", "2026-03-05", "20260305070000")], RUN_OOS=True)
check("oos earnings exclusion is the committed insample rule: a later earnings filing within 5 sessions excludes, as does an earlier one",
      ev_after.earnings_excluded.tolist() == [1] and ev_before.earnings_excluded.tolist() == [1])
ev_ins, *_ = run_build("insample", [fil("Q1", "2025-03-11", "20250311080000", "20250306"), ern("E3", "2025-03-13", "20250313070000")])
check("insample keeps the committed +-5 sessions rule (a later earnings filing excludes)",
      ev_ins.set_index("accession_number").loc["Q1", "earnings_excluded"] == 1)
seen_calls = []
_orig_flag = E.earnings_flag


def _spy(ev, earnings, clock, *a, **k):
    seen_calls.append((a, tuple(sorted(k.items()))))
    return _orig_flag(ev, earnings, clock, *a, **k)


E.earnings_flag = _spy
try:
    run_build("insample", [fil("Q1", "2025-03-11", "20250311080000", "20250306")])
    run_build("oos", [EV_O], RUN_OOS=True)
finally:
    E.earnings_flag = _orig_flag
check("insample and oos call the same exclusion function with the same parameters",
      len(seen_calls) == 2 and seen_calls[0] == seen_calls[1] and "known_only" not in inspect.signature(E.earnings_flag).parameters,
      str(seen_calls))


# ---- estimate_requests: counts uncached requests, fetches nothing --------------------------------------------------------
import json  # noqa: E402
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    NBE = {**NB0, "RUN_OOS": True, "BASE_URL": "https://api.massive.com", "TOP_100": ["AAA"], "CACHE_DIR": tmp,
           "normalize_ticker": lambda t: t.strip().upper().replace("/", ".")}
    put = lambda path, params, payload: E._api_cache_file(NBE, path, params, tmp).write_text(json.dumps(payload))   # noqa: E731
    put("/stocks/taxonomies/vX/disclosures", {"limit": 1000},
        {"results": [{"tertiary_category": "ceo_departure"}, {"tertiary_category": "quarterly_earnings"}]})
    q = lambda t: {"tertiary_category": t, "filing_date.gte": "2026-01-01", "filing_date.lte": "2026-08-31", "limit": 1000, "sort": "filing_date.asc"}   # noqa: E731
    u1, u2 = ur("est1"), ur("est2")
    put("/stocks/filings/8-K/vX/disclosures", q("ceo_departure"),
        {"results": [{"tickers": ["aaa"], "filing_url": u1}, {"tickers": ["ZZZ"], "filing_url": ur("other")}],
         "next_url": "https://api.massive.com/stocks/filings/8-K/vX/disclosures?cursor=abc"})
    put("/stocks/filings/8-K/vX/disclosures", q("quarterly_earnings"), {"results": [{"tickers": ["aaa"], "filing_url": u2}]})
    E.header_path(u2, tmp).write_text("<ACCEPTANCE-DATETIME>20260305070000\n")
    _requests.get = _no_network
    est = E.estimate_requests(None, "oos", NBE)
    check("estimate_requests: uncached Massive pages (a next page, the 11 look-back queries), SEC headers and full texts, nothing fetched",
          est["massive_pages"] == 12 and est["sec_headers"] == 1 and est["sec_full_text"] == 1 and est["unknown_filings"]
          and len(est["massive_more_pages_possible"]) == 12 and est["window"] == ("2026-01-01", "2026-08-31"), str(est))
    check("estimate_requests applies the same guards (oos without RUN_OOS, retired labels)",
          refused(lambda: E.estimate_requests(None, "oos", {**NBE, "RUN_OOS": False}))
          and refused(lambda: E.estimate_requests(None, "discovery", NBE), text=RETIRED))
    est_i = E.estimate_requests(("2024-01-01", "2025-12-31"), "insample", {**NBE, "RUN_OOS": False})
    check("estimate_requests for insample asks no look-back queries", est_i["massive_pages"] == 2 and est_i["window"] == ("2024-01-01", "2025-12-31"), str(est_i))

print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
