"""Event census: how many 8-K events each disclosure category has in each research window.

Counts only: no option prices, no returns. The out-of-sample window (2026-01-01 onward) is never
requested. Acceptance times are not needed for counts, so nothing is fetched from sec.gov.

Requests: 1 taxonomy page + one paginated disclosures query per tag per window (about 240 plus
pagination), all cached in .massive_cache/, so a rerun is free and an interrupted run resumes.

Run from anywhere:  .venv/Scripts/python src/census.py      Output: data/census/ (never committed).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)                      # the notebook's cache and .env paths are relative to the repo root
sys.path.insert(0, str(ROOT / "src"))
import nb                           # noqa: E402

NB = nb.load()
pd = NB["pd"]
OUT = ROOT / "data" / "census"

WINDOWS = {
    "discovery": ("2022-01-01", "2023-12-31"),
    "in_sample": (NB["STUDY_START"], NB["STUDY_END"]),
}
for _name, (_s, _e) in WINDOWS.items():
    assert _s < _e < NB["OOS_START"], f"{_name} window reaches the out-of-sample period; refusing (CLAUDE.md rule 8)"


def top100_events(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per (filer, filing date) in TOP_100, as build_events collapses them (without acceptance times)."""
    if raw.empty or "tickers" not in raw:          # some tags' filers carry no tickers at all
        return pd.DataFrame(columns=["cik", "filing_date", "ticker", "accession_number"])
    ex = raw.explode("tickers").rename(columns={"tickers": "ticker"})
    ex["ticker"] = ex["ticker"].map(NB["normalize_ticker"])
    ex = ex[ex["ticker"].isin(NB["TOP_100"])]
    return (ex.sort_values(["cik", "filing_date"])
              .groupby(["cik", "filing_date"], as_index=False)
              .agg(ticker=("ticker", "first"), accession_number=("accession_number", "first")))


def main() -> None:
    taxonomy = pd.DataFrame(NB["api_get_all"]("/stocks/taxonomies/vX/disclosures", {"limit": 1000}))
    tags = taxonomy.sort_values(["primary_category", "secondary_category", "tertiary_category"])
    print(f"{len(tags)} tags x {len(WINDOWS)} windows: about {len(tags) * len(WINDOWS) + 1} requests plus pagination "
          f"(cached in {NB['CACHE_DIR']}).", flush=True)

    rows, events = [], []
    for i, t in enumerate(tags.itertuples(index=False), 1):
        for window, (start, end) in WINDOWS.items():
            raw = NB["fetch_disclosures"](t.tertiary_category, start, end)
            ev = top100_events(raw)
            rows.append({
                "primary": t.primary_category, "secondary": t.secondary_category, "tertiary": t.tertiary_category,
                "window": window,
                "disclosures_all": len(raw),
                "filings_all": raw["accession_number"].nunique() if not raw.empty else 0,
                "filers_all": raw["cik"].nunique() if not raw.empty else 0,
                "events_top100": len(ev),
                "tickers_top100": ev["ticker"].nunique(),
            })
            events.append(ev.assign(window=window, primary=t.primary_category,
                                    secondary=t.secondary_category, tertiary=t.tertiary_category))
        print(f"  [{i}/{len(tags)}] {t.tertiary_category}", flush=True)

    by_tag = pd.DataFrame(rows)
    ev_all = pd.concat(events, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    by_tag.to_csv(OUT / "by_tertiary.csv", index=False)
    ev_all.to_csv(OUT / "top100_events.csv", index=False)     # every tagged TOP_100 event: the "any 8-K" list

    # Families: unique events across their member tags (a filing can carry several tags in one family).
    for level in ("secondary", "primary"):
        fam = (ev_all.drop_duplicates(["window", level, "cik", "filing_date"])
                     .groupby([level, "window"]).agg(events_top100=("cik", "size"), tickers_top100=("ticker", "nunique"))
                     .unstack("window", fill_value=0))
        fam.columns = [f"{a}_{b}" for a, b in fam.columns]
        fam.to_csv(OUT / f"by_{level}.csv")

    wide = by_tag.pivot_table(index=["primary", "secondary", "tertiary"], columns="window",
                              values="events_top100", aggfunc="sum").fillna(0).astype(int)
    any_8k = ev_all.drop_duplicates(["window", "cik", "filing_date"]).groupby("window").size()
    print("\nTOP_100 events per category per window (counts only):")
    with pd.option_context("display.max_rows", 500, "display.width", 200):
        print(wide.sort_values("discovery", ascending=False).to_string())
    print(f"\nDistinct TOP_100 8-K filing events with any tag: {any_8k.to_dict()}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
