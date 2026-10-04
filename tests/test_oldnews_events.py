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


# ---- Windows and guards by label ------------------------------------------------------------------------------------------
def refused(f, kind=PermissionError):
    try:
        f()
    except kind:
        return True
    except Exception as e:  # noqa: BLE001
        print("   raised", type(e).__name__, str(e)[:100])
    return False


NB0 = {"OOS_START": "2026-01-01", "OOS_END": "2026-08-31", "HOLDOUT_START": "2023-06-01", "HOLDOUT_END": "2023-08-31"}
rw = lambda label, window=None, **flags: E.resolve_window(label, {**NB0, **flags}, window)      # noqa: E731
check("resolve_window defaults: discovery, dryrun, insample",
      [rw(l) for l in ("discovery", "dryrun", "insample")] ==
      [("2022-01-01", "2023-12-31", "2024-01-01"), ("2023-07-01", "2023-12-31", "2024-01-01"), ("2024-01-01", "2025-12-31", "2026-01-01")])
check("resolve_window: a window argument replaces the default dates", rw("discovery", ("2023-07-01", "2023-09-30")) ==
      ("2023-07-01", "2023-09-30", "2024-01-01") and rw("insample", (T("2024-03-01"), "2024-03-31")) == ("2024-03-01", "2024-03-31", "2026-01-01"))
check("discovery and dryrun refuse a window that reaches 2024-01-01",
      refused(lambda: rw("discovery", ("2023-12-01", "2024-01-01"))) and refused(lambda: rw("dryrun", ("2023-12-01", "2024-03-31")))
      and refused(lambda: rw("discovery", ("2024-06-01", "2024-06-30"))))
check("insample refuses a window that reaches 2026-01-01, and one that starts there",
      refused(lambda: rw("insample", ("2025-06-01", "2026-01-01"))) and refused(lambda: rw("insample", ("2026-03-01", "2026-03-31"))))
check("an unnamed label gets the discovery rules and needs a window",
      rw("myrun", ("2022-05-01", "2022-05-31")) == ("2022-05-01", "2022-05-31", "2024-01-01")
      and refused(lambda: rw("myrun", ("2023-12-01", "2024-02-01"))) and refused(lambda: rw("myrun"), ValueError))
check("a third window item is ignored: it cannot loosen or tighten the guard",
      refused(lambda: rw("discovery", ("2023-12-01", "2024-02-01", "2030-01-01")))
      and rw("discovery", ("2022-05-01", "2022-05-31", "2020-01-01")) == ("2022-05-01", "2022-05-31", "2024-01-01")
      and rw("holdout", ("2026-09-01", "2026-09-30", "2026-01-01"), RUN_HOLDOUT=True) == ("2026-09-01", "2026-09-30", None))
check("holdout needs RUN_HOLDOUT to be exactly True (False, missing, 1 and 'yes' refuse)",
      all(refused(lambda f=f: rw("holdout", **f)) for f in ({}, {"RUN_HOLDOUT": False}, {"RUN_HOLDOUT": 1}, {"RUN_HOLDOUT": "yes"}))
      and rw("holdout", RUN_HOLDOUT=True) == ("2023-06-01", "2023-08-31", None))
check("holdout with its switch has no date guard, even in 2026 and beyond",
      rw("holdout", ("2026-09-01", "2027-02-28"), RUN_HOLDOUT=True) == ("2026-09-01", "2027-02-28", None))
check("oos needs RUN_OOS to be exactly True, then takes the notebook's OOS dates",
      all(refused(lambda f=f: rw("oos", **f)) for f in ({}, {"RUN_OOS": False}, {"RUN_OOS": 1}))
      and rw("oos", RUN_OOS=True) == ("2026-01-01", "2026-08-31", None))
check("each switch unlocks only its own label; RUN_OOS never loosens a guarded label",
      refused(lambda: rw("holdout", RUN_OOS=True)) and refused(lambda: rw("oos", RUN_HOLDOUT=True))
      and refused(lambda: rw("discovery", ("2026-01-01", "2026-03-01"), RUN_OOS=True, RUN_HOLDOUT=True))
      and refused(lambda: rw("insample", ("2025-06-01", "2026-03-01"), RUN_OOS=True, RUN_HOLDOUT=True)))
check("a guarded label whose stop passes the out-of-sample start refuses",
      refused(lambda: E.resolve_window("insample", {"OOS_START": "2025-07-01"}, ("2025-01-01", "2025-03-01"))))
check("window validation: not a real date, start after end, no default",
      refused(lambda: rw("discovery", ("2022-02-30", "2022-03-31")), ValueError) and refused(lambda: rw("discovery", ("2022-05-01", "2022-04-01")), ValueError)
      and refused(lambda: E.resolve_window("holdout", {"RUN_HOLDOUT": True}), ValueError))
check("build keeps its old signature (positional label, NB, out_dir, playground, write; new arguments last)",
      list(inspect.signature(E.build).parameters)[:5] == ["label", "NB", "out_dir", "playground", "write"]
      and list(inspect.signature(E.build).parameters)[5:] == ["window", "cache_dir"])


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


def run_build(label, filings, window=None, fetchable=True, positional=False, **flags):
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

        def fetch_disclosures(tag, start, end):
            calls["disclosures"].append(tag)
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
              "SESSION": Session(), "CACHE_DIR": cache, "RUN_OOS": False, "RUN_HOLDOUT": False, **NB0, **flags}
        stop = {"discovery": "2024-01-01", "dryrun": "2024-01-01", "insample": "2026-01-01"}.get(label, "2100-01-01")
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
            ev, nu = E.build(label, NB, out, pg, True, window=window)
        else:
            ev, nu = E.build(label, NB, out_dir=out, playground=pg, window=window)
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


GOOD = fil("D1", "2022-06-07", "20220607080000", "20220602")
NOHDR = fil("D2", "2022-09-13", "20220913080000", "20220908", cached=False)
ev, nu, NBr, calls, files, dr = run_build("discovery", [GOOD, NOHDR], positional=True)
check("discovery: built from its default window; one late people event, its two ordinary days, three files",
      list(ev.row_id) == ["event|D1"] and ev.late.tolist() == [1] and len(nu) == 2 and files == ["dropped_discovery.csv", "events_discovery.csv", "nulls_discovery.csv"],
      f"{list(ev.row_id)} {files}")
check("discovery: cache only (the notebook is offline, no header was fetched) and the filing without a header is counted",
      offline(NBr) and calls["header"] == [] and ev.attrs["counts"]["people_no_header"] == 1, str(ev.attrs["counts"]))
check("discovery refuses a window into 2024 before reading anything (no playground files, no disclosure calls)",
      refused(lambda: E.build("discovery", {**NBr, "SESSION": Session()}, playground=Path("/nonexistent"), window=("2023-12-01", "2024-01-31"))))
check("discovery refuses 2026 even with both switches on",
      refused(lambda: E.build("discovery", {**NBr, "RUN_OOS": True, "RUN_HOLDOUT": True}, playground=Path("/nonexistent"), window=("2026-03-01", "2026-03-31"))))

ev, nu, NBr, calls, files, dr = run_build("dryrun", [fil("R1", "2023-09-12", "20230912080000", "20230907")])
check("dryrun: default window 2023-07-01..2023-12-31, offline", list(ev.row_id) == ["event|R1"] and offline(NBr) and calls["header"] == [])
check("dryrun refuses a window into 2024",
      refused(lambda: E.build("dryrun", NBr, playground=Path("/nonexistent"), window=("2023-07-01", "2024-01-31"))))
ev, nu, NBr, calls, files, dr = run_build("myrun", [fil("U1", "2022-06-07", "20220607080000", "20220602")], window=("2022-05-01", "2022-07-31"))
check("an unnamed label with a window builds under the discovery rules",
      list(ev.row_id) == ["event|U1"] and "events_myrun.csv" in files and refused(lambda: E.build("myrun", NBr, window=("2023-12-01", "2024-01-31"))))

INS = [fil("I1", "2025-03-11", "20250311080000", "20250306"), fil("I2", "2025-12-31", "20251231170000", "20251229"),
       fil("PR1", "2023-12-20", "20231220080000", "20231215")]
ev, nu, NBr, calls, files, dr = run_build("insample", INS)
check("insample: default window, offline, an entry on or after 2026-01-01 is dropped and counted (and the prior window is read for look-back only)",
      list(ev.row_id) == ["event|I1"] and offline(NBr) and ev.attrs["counts"]["people_entry_on_or_after_2026-01-01"] == 1
      and ev.t_0.max() < T("2026-01-01") and nu.t_0.max() < T("2026-01-01"), f"{list(ev.row_id)} {ev.attrs['counts']}")
check("insample refuses a window that reaches 2026-01-01",
      refused(lambda: E.build("insample", NBr, playground=Path("/nonexistent"), window=("2025-06-01", "2026-01-01"))))

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
check("holdout without window= uses the notebook's HOLDOUT_START and HOLDOUT_END", list(ev.row_id) == ["event|H3"])
check("holdout still refuses an unknown or missing switch value",
      refused(lambda: run_build("holdout", HOLD, window=("2026-09-01", "2026-09-30"), RUN_HOLDOUT="yes")))

OOS = [fil("O1", "2026-03-10", "20260310080000", "20260305")]
check("oos without RUN_OOS is refused; RUN_HOLDOUT alone does not unlock it",
      refused(lambda: run_build("oos", OOS)) and refused(lambda: run_build("oos", OOS, RUN_HOLDOUT=True)))
ev, nu, NBr, calls, files, dr = run_build("oos", OOS, RUN_OOS=True)
check("oos with RUN_OOS: the notebook's OOS window, cache only (offline, nothing fetched)",
      list(ev.row_id) == ["event|O1"] and offline(NBr) and calls["header"] == [] and ev.t_0.min() >= T("2026-01-01"))

print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
