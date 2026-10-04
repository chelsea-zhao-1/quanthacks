# Old news, new news

**When an executive or director 8-K arrives days after the market already knows the news, do options overpay for a move that has already happened?**

We committed this question to Git before seeing any result, tested it once, and found no evidence for it.

## Inspiration

Companies have four business days to file an 8-K after an executive or director change, and at the largest US companies most of these filings arrive days late. Sometimes the market already knows. Sometimes the filing is the first word. We expected both kinds of filing to lift implied volatility, but after old news that price would be paying for a move that is already over.

## What it does

Each late people-news filing is labelled **old news** or **surprise news** using only what was known at the last close before acceptance. The label combines a math score (the stock's gap move against the market, the change in implied volatility, option volume) with a word score from the full EDGAR text. The pipeline then measures log(realised / implied volatility) after entry, relative to matched ordinary days for the same stock, and tests a cash-secured put sold on old-news filings.

**Result (2024–25: 146 filings, 59 tickers, 45 old news vs 101 surprise).** Our primary test used a 10-session horizon and 1-month options, and predicted a negative difference. We got **+0.097** (95% interval −0.069 to +0.262, one-sided p = 0.86): the wrong sign, and not significant. Old news against ordinary days with equally large gap moves: −0.036 (p = 0.25). The placebo (late scheduled filings with no people news) gave +0.099 (p = 0.40), about the same as the main test. No horizon survives the Benjamini-Hochberg correction, and none of the 22 sensitivity variants is significant.

**The trade.** 49 old-news puts, closed after 10 sessions, made **+1.19%** of collateral net of costs and **−0.84% at double costs**, with a −4.68% maximum drawdown. Compared with the same put on matched ordinary days the difference is −0.09 points (interval −0.86 to +0.64). There is no edge.

## How we built it

- **Hypothesis first.** The hypothesis and the full test plan (weights, cutoff, horizons, placebo, costs) were committed before any outcome existed. Every later change is logged with its reason.
- **No lookahead.** Entry is the first close after the EDGAR acceptance time. Filings accepted after 15:30 ET enter the next session.
- **Frozen constants.** We computed the z-score constants once, from 1,245 ordinary days and gap inputs only, and never changed them.
- **Statistics.** Permutation tests, bootstrap intervals, BH q-values across every fixed horizon, leave-one-ticker-out checks, and all 337 evaluated variants logged and disclosed.
- **Costs and capacity.** The larger of 5% of premium or $0.05 per share, each way; every result is repeated at double costs, and size is capped at 10% of entry-day put volume.
- **Engineering.** A team of Claude Code agents built the pipeline, each owning one module under hard rules: never touch the API key, never compute on 2026 data. Each module has synthetic tests. One command reproduces every number, and a report script writes those numbers to a single file that the README, the note and this page all quote.

## Challenges

- The starter data has filing dates but no times, and about half of 8-Ks are accepted after the close. We rebuilt entry timing from EDGAR acceptance times in every window, including the judges' sealed window.
- Partway through, Massive told us only 2024–25 data may be used. We set aside our 2022–23 exploration, retired those dates in code, and re-froze the constants from 2024–25 gap inputs before analysing any outcome.
- The short text excerpt misses most prior disclosures. We switched the word score to the full filing text before looking at outcomes, which changed the score for 61 of 251 filings.

## Accomplishments that we're proud of

We ran a clean, pre-registered test and kept its answer. Date guards refuse the sealed and 2026 windows unless a human switches them on. Our updated prediction for the sealed window is already written down: no detectable difference.

## What we learned

The options market seems to price these filings efficiently. The placebo looks like the main test, so the small differences we did see are not specific to people news. We also learned that pre-registration is what makes a null result credible.

## What's next

The judges' sealed-window run, the one-time 2026 test (left to a human), a larger sample if Massive allows 2022–23, and quotes in place of last-trade marks.

## Built with

Python, pandas, NumPy, matplotlib, requests, Jupyter, the Massive API (8-K disclosures, option contracts, daily option bars), SEC EDGAR, Claude Code, Git.
