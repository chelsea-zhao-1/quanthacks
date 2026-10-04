"""Spoken research brief: turn one pipeline run's own results into a short MP3 (ElevenLabs).

Optional add-on, not part of the judged notebook. Reads only the run's result files:
    data/oldnews/results_<label>/note_numbers.md
    data/oldnews/results_<label>/summary.md
    data/oldnews/trade_<label>/summary.md
Any number it cannot find is spoken as missing, never invented.

Usage (from the repo root):
    .venv/Scripts/python.exe extras/audio_brief.py --label insample --dry-run
    .venv/Scripts/python.exe extras/audio_brief.py --label insample [--voice-id ID]

The ElevenLabs key is read by this code from the environment or from .env (ELEVENLABS_API_KEY).
It is never printed, logged or passed on a command line.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "oldnews"
MAX_CHARS = 1900
DEFAULT_VOICE = "JBFqnCBsd6RMkjVDRZzb"
FALLBACK_VOICE = "21m00Tcm4TlvDq8ikWAM"
MISSING = "a number we could not find in this run's files"

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
        "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


# ---------- numbers for speech ----------
def int_words(n: int) -> str:
    """Spell a non-negative integer below one million."""
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else "-" + ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return ONES[n // 100] + " hundred" + ("" if rest == 0 else " and " + int_words(rest))
    rest = n % 1000
    return int_words(n // 1000) + " thousand" + ("" if rest == 0 else " " + int_words(rest))


def say_num(x: float | None, dp: int, signed: bool = False) -> str:
    """0.0966, dp=3, signed -> 'plus zero point zero nine seven'. None -> missing phrase."""
    if x is None:
        return MISSING
    s = f"{abs(x):.{dp}f}"
    whole, _, frac = s.partition(".")
    words = int_words(int(whole))
    if frac:
        words += " point " + " ".join(ONES[int(d)] for d in frac)
    if float(s) == 0:
        return words
    if x < 0:
        return "minus " + words
    return ("plus " + words) if signed else words


def say_int(n: int | None) -> str:
    return MISSING if n is None else int_words(n)


# ---------- parsing ----------
def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        print(f"note: {path.relative_to(ROOT)} not found; its numbers will be spoken as missing")
        return ""


def num(s: str | None) -> float | None:
    if s is None:
        return None
    m = re.search(r"[+-]?\d+(?:\.\d+)?", s.replace(",", ""))
    return float(m.group()) if m else None


def md_tables(text: str) -> list[list[dict[str, str]]]:
    """Every markdown table in text, as a list of row dicts keyed by header."""
    tables, lines, i = [], text.splitlines(), 0
    while i < len(lines) - 1:
        if lines[i].startswith("|") and re.match(r"^\|[\s\-|:]+\|$", lines[i + 1].strip()):
            head = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            rows, i = [], i + 2
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(dict(zip(head, cells)))
                i += 1
            tables.append(rows)
        else:
            i += 1
    return tables


def section(text: str, heading_start: str) -> str:
    """Body of the '## <heading_start>...' section up to the next '## ' heading."""
    m = re.search(r"^## " + re.escape(heading_start) + r".*?$(.*?)(?=^## |\Z)", text, re.M | re.S)
    return m.group(1) if m else ""


TEST_RE = re.compile(
    r"effect ([+-]?\d+\.\d+) \(95% CI ([+-]?\d+\.\d+) to ([+-]?\d+\.\d+)\), "
    r"one-sided p = (\d+\.\d+), two-sided p = (\d+\.\d+); n = (\d+) \(old (\d+), comparison (\d+)")


def parse_test(sec: str) -> dict:
    m = TEST_RE.search(sec)
    if not m:
        return {}
    g = m.groups()
    return {"eff": float(g[0]), "lo": float(g[1]), "hi": float(g[2]), "p1": float(g[3]),
            "p2": float(g[4]), "n": int(g[5]), "old": int(g[6]), "comp": int(g[7])}


def find_row(tables, need_cols, avoid_cols, **match) -> dict:
    for rows in tables:
        if not rows:
            continue
        cols = set(rows[0])
        if not set(need_cols) <= cols or cols & set(avoid_cols):
            continue
        for r in rows:
            if all(r.get(k) == v for k, v in match.items()):
                return r
    return {}


def gather(label: str) -> dict:
    res = read(DATA / f"results_{label}" / "summary.md")
    trade = read(DATA / f"trade_{label}" / "summary.md")
    notes = read(DATA / f"results_{label}" / "note_numbers.md")
    note = {r.get("item", ""): r for t in md_tables(notes) for r in t}

    h1 = parse_test(section(res, "H1 (primary)"))
    pl = parse_test(section(res, "P (placebo"))
    tt = md_tables(trade)
    b1 = find_row(tt, ["book", "cost", "total_return_pct", "n_trades"], ["year", "quarter"], book="old", cost="1x")
    b2 = find_row(tt, ["book", "cost", "total_return_pct", "n_trades"], ["year", "quarter"], book="old", cost="2x")
    rk = find_row(tt, ["book", "cost", "sharpe"], ["year"], book="old", cost="1x")
    power = note.get("diagnostics: events per group to detect 0.10", {}).get("value")
    mean1 = note.get("trade: mean per taken trade, h = 10, 1x (% of collateral)", {}).get("value")
    return {
        "h1": h1, "pl": pl,
        "n_trades": int(num(b1.get("n_trades"))) if num(b1.get("n_trades")) is not None else None,
        "ret1": num(b1.get("total_return_pct")), "ret2": num(b2.get("total_return_pct")),
        "dd1": num(b1.get("max_dd_mtm_pct")), "sharpe": num(rk.get("sharpe")),
        "mean1": num(mean1), "power": int(num(power)) if num(power) is not None else None,
    }


# ---------- script ----------
def build_script(label: str, d: dict) -> str:
    h1, pl = d["h1"], d["pl"]
    name = "in-sample" if label == "insample" else label.replace("_", " ")
    parts = [f"This is the spoken research brief for the {name} run of our Trade the eight K project."]
    if label != "insample":
        parts.append("If you are hearing this for the judges' sealed window, "
                     "these numbers come from that window.")
    parts += [
        "What we asked. When a company files a late eight K about an executive or a director, "
        "the news is often already out. Does that old news make options overprice the coming move, "
        "more than surprise news does?",
        "Every rule, from the old news label to the costs, was fixed and committed before we saw the data.",
        "What we measured. For each filing, the volatility the stock actually realised over the next ten "
        "sessions, against the implied volatility priced into its options, compared with the same stock "
        "on matched ordinary days. We repeated the test on a placebo: late, scheduled filings that carry no people news.",
    ]
    if h1:
        parts.append(
            f"The result. {say_int(h1['n']).capitalize()} events: {say_int(h1['old'])} old news and "
            f"{say_int(h1['comp'])} surprise. Old minus surprise was {say_num(h1['eff'], 3, True)}, "
            f"with a ninety-five percent interval from {say_num(h1['lo'], 3, True)} to "
            f"{say_num(h1['hi'], 3, True)}, and one-sided p equals {say_num(h1['p1'], 2)}. "
            "We had predicted a negative number.")
    else:
        parts.append(f"The result. The main test result is {MISSING}.")
    if pl:
        parts.append(f"The placebo filings gave {say_num(pl['eff'], 3, True)}, two-sided p equals "
                     f"{say_num(pl['p2'], 2)}: a similar picture.")
    else:
        parts.append(f"The placebo result is {MISSING}.")
    parts.append(
        "The trade. We sold a one-month put three percent below the stock price on each old news "
        f"event and held it ten sessions: {say_int(d['n_trades'])} trades. Net of costs, the book "
        f"returned {say_num(d['ret1'], 2, True)} percent in total. At double costs, "
        f"{say_num(d['ret2'], 2, True)} percent. The maximum drawdown was "
        f"{say_num(abs(d['dd1']) if d['dd1'] is not None else None, 2)} percent"
        + (f", and the Sharpe ratio was {say_num(d['sharpe'], 2)}." if d["sharpe"] is not None else "."))
    power = (f" To detect a difference of zero point one, we would need about "
             f"{say_int(d['power'])} events per group." if d["power"] is not None else "")
    parts.append(
        "The honest conclusion. We found no edge. The options market may already price old news "
        "efficiently, our old news label may be too noisy, or the sample may be too small." + power)
    parts.append("Thank you for listening.")
    return "\n\n".join(parts)


# ---------- ElevenLabs ----------
def load_key() -> str | None:
    key = os.environ.get("ELEVENLABS_API_KEY")
    if key:
        return key.strip()
    env = ROOT / ".env"
    if not env.exists():
        return None
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].strip()
        if line.startswith("ELEVENLABS_API_KEY") and "=" in line:
            val = line.split("=", 1)[1].strip().strip('"').strip("'")
            return val or None
    return None


def error_detail(resp: requests.Response, key: str) -> str:
    try:
        det = resp.json().get("detail", "")
    except ValueError:
        det = ""
    if isinstance(det, dict):
        det = det.get("message") or det.get("status") or ""
    det = str(det)[:300]
    return det.replace(key, "[redacted]") if key else det


def tts(text: str, voice_id: str, key: str) -> requests.Response:
    return requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers={"xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
        json={"text": text, "model_id": "eleven_multilingual_v2",
              "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}},
        timeout=120,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", required=True, help="run label, e.g. insample or holdout")
    ap.add_argument("--dry-run", action="store_true", help="print the script; no API call")
    ap.add_argument("--voice-id", default=None, help=f"ElevenLabs voice (default {DEFAULT_VOICE})")
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", args.label):
        print("label must be letters, digits, _ or -")
        return 2

    text = build_script(args.label, gather(args.label))
    print(text)
    print(f"\n[{len(text)} characters]")
    if len(text) > MAX_CHARS:
        print(f"script is over {MAX_CHARS} characters; not sending")
        return 1
    if args.dry_run:
        return 0

    key = load_key()
    if not key:
        print("ELEVENLABS_API_KEY not set (environment or .env)")
        return 1
    voices = [args.voice_id] if args.voice_id else [DEFAULT_VOICE, FALLBACK_VOICE]
    resp = None
    for voice in voices:  # at most two calls; the second only if the first voice is 404
        try:
            resp = tts(text, voice, key)
        except requests.RequestException as exc:
            print(f"request failed: {type(exc).__name__}")
            return 1
        if resp.status_code != 404:
            break
    if resp is None or resp.status_code != 200:
        print(f"HTTP {resp.status_code}: {error_detail(resp, key)}")
        return 1

    out = DATA / "audio" / f"brief_{args.label}.mp3"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(resp.content)
    print(f"saved {out.relative_to(ROOT)} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
