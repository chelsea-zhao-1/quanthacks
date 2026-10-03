"""One-screen status of a download run, safe to share: counts and progress only, no prices, no key.

Usage:  python src/status.py [label]      (label: discovery (default) or insample; folder from PLAYGROUND_DIR,
                                            default data/playground, or data/playground/insample for insample)
"""
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def row_ids(df: pd.DataFrame, kind: str) -> set[str]:
    """The download job's row ids, so stale rows from earlier runs (other sessions) are not counted."""
    return {f"{kind}|{t}|{pd.Timestamp(p).date()}|{pd.Timestamp(z).date()}" for t, p, z in zip(df.ticker, df.t_pre, df.t_0)}


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "discovery"
    default = ROOT / "data" / "playground" / ("insample" if label == "insample" else "")
    out = Path(os.environ.get("PLAYGROUND_DIR", default))
    hb_file, prog_file = out / "heartbeat.json", out / "progress.csv"
    if not hb_file.exists():
        print(f"No heartbeat yet in {out}: the download has not started.")
        return
    hb = json.loads(hb_file.read_text())
    age_min = (time.time() - hb_file.stat().st_mtime) / 60
    ev_file, nu_file = out / f"events_{label}.csv", out / f"nulls_{label}.csv"
    events = pd.read_csv(ev_file, usecols=["ticker", "t_pre", "t_0"]) if ev_file.exists() else pd.DataFrame(columns=["ticker", "t_pre", "t_0"])
    nulls = pd.read_csv(nu_file, usecols=["ticker", "t_pre", "t_0"]) if nu_file.exists() else pd.DataFrame(columns=["ticker", "t_pre", "t_0"])
    want_ev, want_nu = row_ids(events, "event"), row_ids(nulls, "null")
    prog = pd.read_csv(prog_file) if prog_file.exists() else pd.DataFrame(columns=["row_id", "status", "requests", "note"])
    last = prog.drop_duplicates("row_id", keep="last")
    current = last[last.row_id.isin(want_ev | want_nu)]
    done = current[current.status.isin(["ok", "dropped"])]
    errors = current[current.status == "error"]
    state = hb.get("state", "unknown")
    if state == "running" and age_min > 5:
        state = "NOT RESPONDING (process may have died)"

    print(f"Download ({label}) as of {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  state: {state.upper()} (last heartbeat {age_min:.0f} min ago"
          f"{'; STOP file present' if (out / 'STOP').exists() else ''})")
    print(f"  unique rows done: {len(done):,} of {len(want_ev) + len(want_nu):,} "
          f"({len(done) / max(len(want_ev) + len(want_nu), 1):.0%}) - events {done.row_id.isin(want_ev).sum():,} of "
          f"{len(want_ev):,}, null days {done.row_id.isin(want_nu).sum():,} of {len(want_nu):,}")
    print(f"  no usable bucket: {int((current.status == 'dropped').sum()):,}; rows still failing: {len(errors):,} "
          f"({hb.get('failures_in_a_row', 0)} in a row now)")
    print(f"  requests: {int(prog['requests'].sum()):,} total over all runs; {hb.get('req_per_s')}/s last run; "
          f"ETA {hb.get('eta_hours')} h")
    if len(errors):
        print("  failing: " + "; ".join(f"{r.row_id}: {r.note}" for r in errors.tail(3).itertuples()))


if __name__ == "__main__":
    main()
