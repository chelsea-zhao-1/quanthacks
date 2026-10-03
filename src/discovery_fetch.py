"""Overnight download for the 2022-2023 discovery window: price every TOP_100 8-K event and its matched
ordinary days with the notebook's own code, so every chain and option bar lands in the API cache.
Nothing is analysed here and no returns are computed; the scan runs later on the cache.

Order: all events, then null round 1 for every event, then round 2, so a partial run stays balanced.
Resumable: progress is logged per row and every response is cached by URL, so a rerun skips finished work.

Safety (CLAUDE.md): entry or ordinary days at or after 2024-01-01 are refused (hard fail); the out-of-sample
switch must be off; requests are sequential and throttled; HTTP refusals stop the run after a few in a row;
a STOP file in the output folder ends the run cleanly after the current row.

Environment:
  PLAYGROUND_DIR     where the event lists, progress log and heartbeat go (default data/playground)
  MASSIVE_CACHE_DIR  API cache folder (default .massive_cache, as the notebook); on HiPerGator: $SLURM_TMPDIR
  CACHE_ARCHIVE      if set, the cache is packed to this .tar.gz every 2 hours and at exit (HiPerGator: on Blue)
  MAX_RPS            request ceiling per second (default 8; the API is latency-bound at ~6 sequentially)

Usage:  python src/discovery_fetch.py [--no-acceptance] [--limit N]
The same job serves the in-sample confirmation of candidates (src/playground/confirm.py prints the command):
        python src/discovery_fetch.py --label insample --start 2024-01-01 --end 2025-12-31 --hard-stop 2026-01-01
            --clip-before 2026-01-01 --tags a,b --busy-csv data/census/top100_events.csv --max-requests N
"""
import argparse
import json
import os
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)                      # the notebook reads .env relative to the repo root
sys.path.insert(0, str(ROOT / "src"))
import nb                           # noqa: E402

START, END = "2022-01-01", "2023-12-31"   # defaults: the discovery window
HARD_STOP = "2024-01-01"            # no entry or ordinary day at or after this date (CLAUDE.md, discovery window)
REQUESTS_PER_ROW = 28               # measured on the discovery run
N_NULLS, NULL_WINDOW, NULL_GAP = 2, 60, 5   # per event: draws, +/- sessions around t_0, min sessions from any 8-K
SEED = 20261003
FAILURES_TO_STOP = 5
CHECKPOINT_SECONDS = 2 * 3600

OUT = Path(os.environ.get("PLAYGROUND_DIR", ROOT / "data" / "playground"))
CACHE = Path(os.environ.get("MASSIVE_CACHE_DIR", ".massive_cache"))
ARCHIVE = os.environ.get("CACHE_ARCHIVE")
MAX_RPS = float(os.environ.get("MAX_RPS", "8"))


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}", flush=True)


def pack_cache() -> None:
    """Pack the cache to CACHE_ARCHIVE atomically (only between rows, so no file is half-written)."""
    if not ARCHIVE:
        return
    t = time.time()
    tmp = ARCHIVE + ".tmp"
    with tarfile.open(tmp, "w:gz") as tar:
        tar.add(CACHE, arcname="massive_cache")
    os.replace(tmp, ARCHIVE)
    log(f"cache packed to {ARCHIVE} in {time.time() - t:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-acceptance", action="store_true",
                    help="skip EDGAR acceptance times (conservative sessions); affected events re-download later")
    ap.add_argument("--limit", type=int, default=None, help="only the first N events (dress rehearsal)")
    ap.add_argument("--label", default="discovery", help="names the event and null files (events_<label>.csv)")
    ap.add_argument("--start", default=globals()["START"])
    ap.add_argument("--end", default=globals()["END"])
    ap.add_argument("--hard-stop", default=globals()["HARD_STOP"], help="refuse entry or ordinary days on or after this date")
    ap.add_argument("--out", default=None, help="output folder (default PLAYGROUND_DIR or data/playground)")
    ap.add_argument("--tags", default=None, help="comma-separated tertiary tags (default: every tag)")
    ap.add_argument("--busy-csv", default=None,
                    help="CSV of every tagged 8-K (ticker, filing_date) used to keep null days clear; default: the event file")
    ap.add_argument("--clip-before", default=None, help="never fetch option bars dated on or after this day")
    ap.add_argument("--max-requests", type=int, default=None, help="refuse if the estimate exceeds this (CLAUDE.md rule 18)")
    args = ap.parse_args()
    START, END, HARD_STOP = args.start, args.end, args.hard_stop        # noqa: N806 (shadow the defaults)
    OUT = Path(args.out) if args.out else globals()["OUT"]              # noqa: N806

    NB = nb.load()
    pd, np, CAL = NB["pd"], NB["np"], NB["CAL"]
    requests = NB["requests"]
    assert NB["RUN_OOS"] is False, "the out-of-sample switch is on; refusing to run"
    assert HARD_STOP <= NB["OOS_START"] and END < NB["OOS_START"], "this window reaches the out-of-sample period; refusing"
    if args.clip_before:
        from playground.measure import clip_to
        clip_to(NB, args.clip_before)
        log(f"option bars clipped to before {args.clip_before}")
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    NB["CACHE_DIR"] = CACHE         # the notebook's api_get and fetch_acceptance_time read this global

    if args.no_acceptance:
        NB["fetch_acceptance_time"] = lambda url: None
        log("acceptance times OFF: every event uses the conservative sessions (next-day entry)")
    elif "your@email.edu" in NB["SEC_USER_AGENT"] or "team-name" in NB["SEC_USER_AGENT"]:
        log("SEC_USER_AGENT is still the placeholder. Set a real contact in the notebook, or pass --no-acceptance.")
        return 2

    # ---- Throttle and count every request the notebook makes ---------------------------------------
    stats = {"requests": 0, "last": 0.0}
    session_get = NB["SESSION"].get

    def throttled_get(*a, **k):
        wait = stats["last"] + 1 / MAX_RPS - time.time()
        if wait > 0:
            time.sleep(wait)
        stats["last"] = time.time()
        stats["requests"] += 1
        return session_get(*a, **k)

    NB["SESSION"].get = throttled_get

    # ---- Events: every tag, one row per (filer, filing date), sessions from acceptance times ---------
    events_file = OUT / f"events_{args.label}.csv"
    events = None
    if events_file.exists():
        events = pd.read_csv(events_file, parse_dates=["filing_date", "accepted_at", "t_0", "t_pre"])
        if not args.no_acceptance and events["accepted_at"].isna().all():
            log("events were built without acceptance times; rebuilding them with acceptance times")
            events = None
        else:
            log(f"{len(events)} events loaded from {events_file}")
    if events is None:
        taxonomy = pd.DataFrame(NB["api_get_all"]("/stocks/taxonomies/vX/disclosures", {"limit": 1000}))
        parts = []
        wanted = args.tags.split(",") if args.tags else sorted(taxonomy["tertiary_category"])
        unknown = set(wanted) - set(taxonomy["tertiary_category"])
        assert not unknown, f"unknown tags: {sorted(unknown)}"
        for tag in wanted:
            try:
                ev = NB["build_events"](tag, START, END, NB["TOP_100"])
            except KeyError as e:                 # the starter's build_events fails when no filer has tickers
                log(f"{tag}: no ticker field in its disclosures ({e}); no TOP_100 events")
                continue
            if len(ev):
                parts.append(ev.assign(tag=tag))
        allev = pd.concat(parts, ignore_index=True)
        tags = allev.groupby(["cik", "filing_date"])["tag"].apply(lambda s: "|".join(sorted(set(s)))).rename("tags")
        events = (allev.sort_values("accepted_at").drop_duplicates(["cik", "filing_date"])
                       .drop(columns=["tag", "supporting_text"]).merge(tags, on=["cik", "filing_date"])
                       .sort_values(["filing_date", "ticker"]).reset_index(drop=True))
        late = events["t_0"] >= pd.Timestamp(HARD_STOP)
        if late.any():
            log(f"dropping {late.sum()} events whose entry session falls on or after {HARD_STOP}")
            events = events[~late].reset_index(drop=True)
        events.to_csv(events_file, index=False)
        log(f"{len(events)} distinct events written to {events_file}")
    if args.limit:
        events = events.head(args.limit)

    # ---- Matched nulls: same ticker, +/- NULL_WINDOW sessions, > NULL_GAP sessions from any 8-K -------
    # Centred on the filing-date session, and kept clear of both sessions an 8-K can enter at (filing-date
    # session or the next), so the draws do not change when acceptance times arrive later.
    nulls_file = OUT / f"nulls_{args.label}.csv"
    pos = {d: i for i, d in enumerate(CAL)}
    lo_i, hi_i = CAL.searchsorted(pd.Timestamp(START)), CAL.searchsorted(pd.Timestamp(HARD_STOP)) - 1
    fd_i = lambda d: pos[NB["session_on_or_after"](d)]           # noqa: E731
    if nulls_file.exists() and not args.limit:
        nulls = pd.read_csv(nulls_file, parse_dates=["filing_date", "t_0", "t_pre"])
        log(f"{len(nulls)} null days loaded from {nulls_file}")
    else:
        all_ev = pd.read_csv(args.busy_csv or events_file, parse_dates=["filing_date"])
        all_ev = all_ev[(all_ev.filing_date >= pd.Timestamp(START)) & (all_ev.filing_date <= pd.Timestamp(END))]
        busy = {t: {fd_i(d) + k for d in g["filing_date"] for k in (0, 1)} for t, g in all_ev.groupby("ticker")}
        rng = np.random.default_rng(SEED)
        rows = []
        for e in events.itertuples(index=False):
            i0 = fd_i(e.filing_date)
            cand = [i for i in range(max(i0 - NULL_WINDOW, lo_i + 1), min(i0 + NULL_WINDOW, hi_i) + 1)
                    if all(abs(i - j) > NULL_GAP for j in busy.get(e.ticker, ()))]
            picks = rng.choice(cand, size=min(N_NULLS, len(cand)), replace=False) if cand else []
            for r, i in enumerate(sorted(picks), 1):
                rows.append({"ticker": e.ticker, "filing_date": e.filing_date, "round": r, "t_0": CAL[i], "t_pre": CAL[i - 1]})
        nulls = pd.DataFrame(rows, columns=["ticker", "filing_date", "round", "t_0", "t_pre"])
        if not args.limit:
            nulls.to_csv(nulls_file, index=False)
        log(f"{len(nulls)} null days drawn (seed {SEED}); {len(events) * N_NULLS - len(nulls)} short of {N_NULLS} per event")

    # ---- Hard fail on anything at or after the discovery cutoff -----------------------------------------
    for name, df in (("events", events), ("nulls", nulls)):
        for col in ("t_0", "t_pre"):
            assert (df[col] < pd.Timestamp(HARD_STOP)).all(), f"{name}.{col} reaches {HARD_STOP}; refusing"

    # ---- Work list in priority order, minus rows already done ---------------------------------------
    work = [("event", e.ticker, e.t_pre, e.t_0, e.filing_date) for e in events.itertuples(index=False)]
    for r in range(1, N_NULLS + 1):
        work += [("null", n.ticker, n.t_pre, n.t_0, n.t_0) for n in nulls[nulls["round"] == r].itertuples(index=False)]
    progress_file = OUT / "progress.csv"
    done = set()
    if progress_file.exists():
        prog = pd.read_csv(progress_file)
        done = set(prog.loc[prog.status.isin(["ok", "dropped"]), "row_id"])
    todo = [w for w in work if f"{w[0]}|{w[1]}|{w[2].date()}|{w[3].date()}" not in done]
    log(f"{len(work)} rows ({len(events)} events, {len(nulls)} null days); {len(work) - len(todo)} already done; "
        f"~{len(todo) * REQUESTS_PER_ROW:,} requests to go at <= {MAX_RPS:g}/s (cached rows cost nothing)")
    if args.max_requests is not None and len(todo) * REQUESTS_PER_ROW > args.max_requests:
        log(f"estimate {len(todo) * REQUESTS_PER_ROW:,} exceeds --max-requests {args.max_requests:,}; ask before fetching")
        return 4

    if not progress_file.exists():
        progress_file.write_text("row_id,status,buckets,requests,seconds,note\n")
    t_start, last_pack, failures, n_done = time.time(), time.time(), 0, 0
    state = "crashed"               # overwritten on every normal way out
    try:
        for kind, ticker, t_pre, t_0, event_date in todo:
            if (OUT / "STOP").exists():
                log("STOP file found; stopping cleanly")
                state = "stopped (STOP file)"
                break
            row_id = f"{kind}|{ticker}|{t_pre.date()}|{t_0.date()}"
            r0, s0 = stats["requests"], time.time()
            try:
                got, notes = NB["price_event"](ticker, t_pre, t_0, event_date, NB["EXPIRY_BUCKETS"], NB["OTM_GRID"])
                status, note, failures = ("ok" if got else "dropped"), "; ".join(notes), 0
            except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
                failures += 1
                code = getattr(getattr(e, "response", None), "status_code", None)
                status, note = "error", f"{type(e).__name__} {code or ''}".strip()
                log(f"{row_id}: {note} ({failures} in a row)")
                if failures >= FAILURES_TO_STOP:
                    log("repeated failures; stopping entirely (never hammer the API)")
                    state = "stopped (repeated API failures)"
                    return 3
                time.sleep(30 * failures)
            except Exception as e:                       # a data quirk in one row: log it and move on
                status, note = "error", f"{type(e).__name__}: {e}"[:200]
            with open(progress_file, "a", encoding="utf-8") as f:
                f.write(f"{row_id},{status},{len(got) if status == 'ok' else 0},{stats['requests'] - r0},"
                        f"{time.time() - s0:.1f},\"{note.replace(chr(34), chr(39))}\"\n")
            n_done += 1
            el = time.time() - t_start
            (OUT / "heartbeat.json").write_text(json.dumps({
                "state": "running", "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "last_row": row_id, "done_this_run": n_done,
                "remaining": len(todo) - n_done, "requests_this_run": stats["requests"],
                "req_per_s": round(stats["requests"] / el, 2), "eta_hours": round((len(todo) - n_done) * el / n_done / 3600, 2),
                "failures_in_a_row": failures}, indent=1))
            if n_done % 50 == 0:
                log(f"{n_done}/{len(todo)} rows, {stats['requests']:,} requests, {stats['requests'] / el:.1f}/s")
            if time.time() - last_pack > CHECKPOINT_SECONDS:
                pack_cache()
                last_pack = time.time()
        else:
            state = "finished"
    finally:
        hb_file = OUT / "heartbeat.json"
        hb = json.loads(hb_file.read_text()) if hb_file.exists() else {}
        hb.update(state=state, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
        hb_file.write_text(json.dumps(hb, indent=1))
        pack_cache()
    log(f"finished this run: {n_done} rows, {stats['requests']:,} requests in {(time.time() - t_start) / 3600:.2f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
