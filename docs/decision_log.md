# Decision log

Entries are added in time order. Anything that changes the design after it was committed is recorded here,
with the reason and what had been looked at.

## 2026-10-03

- **Hypothesis** (`docs/hypothesis.md`) and **test plan** (`docs/test_plan.md`) committed before any
  post-event option price or return was analysed for the hypothesis.
- **Five events dropped** (BK, NEE, TMUS, CVX, HD): each shares a ticker and filing date with an earlier
  filing, and the entry date in the download tables belonged to the earlier filing, so entry would have
  preceded the news. Listed in `data/oldnews/dropped_discovery.csv`.
- **Frozen standardisation constants.** The first draft standardised the gap inputs with ordinary days from the
  whole sample; replaced by constants computed once on discovery ordinary days (gap inputs only) and applied
  unchanged to every window.
- **Coverage rules for the gap inputs.** With a fresh 1-month mark and five baseline sessions required, 49 of
  258 late people-news events could be measured, because the cached strikes were chosen at the filing date.
  Fallbacks added (other expiry buckets for spot, a mark up to 3 sessions stale, a shorter volume baseline,
  `M` as the mean of the available inputs). Result: 237 of 258.
- **Volume input in log form**, because the raw ratio is heavy-tailed.
- **What had been looked at before these amendments:** event, lag and cue counts; measurement coverage and
  input distributions (events and ordinary days separately, no classification); and pooled medians of
  `iv0`, `rv`, `y` and the cash-secured-put result at 10 sessions on 358 development rows, events and ordinary
  days together, with no old-news / surprise split and no comparison. No old-versus-surprise or
  event-versus-ordinary-day comparison of any outcome had been looked at.
- **Known limits:** 570 usable rows take their two spot prices from different expiry buckets, so dividends and
  early exercise can bias their gap return slightly; the 1-month expiry is often a weekly whose ATM pair trades
  on about 70% of sessions; 5 of the 758 events lie near the start of the sample, where the market median is
  built from fewer tickers.
