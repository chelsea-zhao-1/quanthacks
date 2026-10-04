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

- **Data window rule (Massive, relayed by the team):** "for this project we are only allowed to look at data from
  2024-2025". Before this was known, 2022-23 had been used as an exploratory discovery window: its events were
  measured, classified and tested once, and the constants were frozen from it. Follow-up question to Massive posted
  (is 2022-23 permitted; is the sealed window inside it?). Strict default adopted: 2022-23 results are not used in
  any deliverable; the test window is 2024-25; constants re-frozen from 2024-25 gap inputs before any 2024-25
  outcome is analysed. Full EDGAR text is allowed by Massive and added as a variant for 2024-25 only. The 2022-23
  outputs stay on disk, unused; the discovery constants file remains in Git history (commit 6c7ecdf).
- **Sealed window defined from the notebook.** Cell 10: HOLDOUT placeholder 2023-06-01..2023-08-31; cell 48:
  options history covers only the in-sample and out-of-sample windows, "do not move them earlier". Because the
  earlier 2022-23 discovery window therefore overlapped the sealed placeholder, all 2022-23 outputs were moved to
  an unused folder, the 2022-23 labels were retired in code (every module refuses them), and no 2022-23 result
  appears in any deliverable. The 2022-23 analysis was exploratory and its results did not set any 2024-25
  parameter: the hypothesis and test plan were committed before it, and the three amendments made afterwards
  concerned input coverage, a log transform and the constants, based on input distributions only. The constants
  are re-frozen from 2024-25 gap inputs.
- **Full-text cues are primary** (amendment to the test plan, before any 2024-25 outcome was analysed): Massive allows
  full EDGAR text; the excerpt-based cues fired in 19 of 271 late people-news filings against 68 for full text. The
  excerpt stays as a sensitivity variant and the fallback. Only cue frequencies were looked at.

## 2026-10-03 night: the real 2024-25 result
- The committed test was run once on 2024-25 with the frozen constants (commit 5bbfd9d) and no tuning: pooled H1 at 10
  sessions (1-month bucket) has the wrong sign and is not significant (45 old, 101 surprise events; difference +0.097,
  95% interval -0.069 to +0.262, one-sided p = 0.86). H1b has the predicted sign without significance; the placebo shows no
  difference; no horizon passes the multiple-testing correction; the excerpt and exhibit variants are not significant.
  The result is reported as a null. No rule, threshold or definition is changed because of it.
- **Updated sealed-window prediction** (written after this result and before the sealed window is run): we expect no
  detectable old-versus-surprise difference there. The earlier prediction in the test plan (the sign holds without
  significance) is superseded by this one; the original text stays in the plan's history.
- Whether to run the one-time 2026 out-of-sample test is left to a human (default off).
- **Exploratory diagnostics added after the primary result** (power and minimum detectable effect, whether the classifier
  separates the pre-entry inputs, leave-one-ticker-out and trimmed means, levels behind the difference). They are labelled
  exploratory, logged in the ledger as "diagnostic", and do not change the primary conclusion. An independent recomputation
  of the primary result and a lookahead audit are run by a separate reviewer.
- **Independent audit passed** (`src/oldnews/audit.py`, separate code): labels, H1 and the placebo reproduce to 4 decimals;
  no duplicate events; entry timing of all 751 events reproduces; no date outside 2024-25; trades recompute at 1x and 2x.
  Disclosed: (1) the earnings exclusion uses earnings filings up to 5 sessions after entry, as the test plan states; earnings
  dates are scheduled in advance, but this is information after entry and is disclosed as such; (2) the related-filing cue
  also matches on the other filing's event date, slightly wider than the plan's wording (one placebo filing differs, no label
  changes). The 5% trimmed difference (+0.081) is the pre-committed robustness figure; the 10% version (+0.071) is an
  exploratory diagnostic. The trade's mean per trade is +0.121% (1x, h=10), +0.294% without its largest loss; the top 3
  trades account for 167% of total P&L.
- **QA end-to-end run found a bug** (fixed, not a design change): the notebook pipeline did not pass the full-text setting to
  the event builder, so the notebook used the excerpt word score while the committed plan (since commit 34a11a4, before any
  2024-25 result) makes full text primary. The notebook is fixed to match the plan; the excerpt result is already reported as a
  sensitivity variant (H1 -0.012, one-sided p 0.46; also not significant).
- **Incident in the QA scratch copy:** the starter's own worked-example cells made 3,283 uncached API requests (over our
  3,000 ask-first limit) and marked late-2025 exits with 2026 option prices (hard rule 8). This happened only in a scratch copy
  that was deleted; none of it entered our test, and nothing was committed. The notebook now clips every in-sample computation
  before 2026-01-01 unless the human-only out-of-sample switch is on.
- **Final QA passed** (commit b5a6a39): the whole notebook runs from a clean kernel with only the API key and the cache, zero
  errors, about 15 minutes; all 26 old-news result files are identical to the reported ones (H1 +0.0966, p 0.8615, 45 old;
  49 trades); no computation uses a date on or after 2026-01-01; the sealed cell is not run and the out-of-sample section is off.

## 2026-10-04: the one-time out-of-sample run
- The official track pages require it (Massive sub-track: the notebook's 2026-01-01..2026-08-31 window and the judges'
  sealed window replace the track's 20% holdout; checklist: every fixed horizon in-sample and out-of-sample; Systematic
  track: "run the out-of-sample period once and report the result, good or bad").
- A human (Sadie, SADIES COMPUTER) gave the GO at about 00:55 ET and switches `RUN_OOS` on herself. Everything is frozen:
  hypothesis, test plan, frozen 2024-25 constants (5bbfd9d), code. The run happens once; its result is reported whatever it
  shows, and nothing is changed afterwards. `RUN_OOS` is switched back off in the shipped notebook after the run.
- **Disclosure: the earlier dry run of the judges' path** (before the data rule was known) used 2023-07-01..2023-12-31 on
  the discarded 2022-23 setup. That window overlaps July and August of the notebook's sealed placeholder (2023-06-01..
  2023-08-31). Its outputs were archived unused and never informed any 2024-25 or 2026 choice; its ledger rows (label
  dryrun) are kept for disclosure. The judges set their own sealed dates; if they keep the placeholder, this overlap applies.
