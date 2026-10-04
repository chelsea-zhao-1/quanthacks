# Old news or surprise news after late executive 8-Ks

Sarah Spellman, Lalitha Kantam, Chelsea Zhao, Shayaan Nesargi · University of Florida · Gator Quant Hacks 2026 (Systematic Trading, Massive Trade the 8-K)

## 1. Hypothesis

**A null result: a specific mechanism, rules fixed in advance, and 2024-25 data that do not support it.** Executive and director changes (Item 5.02) at the 100 largest US firms are mostly filed days late. We posited two kinds: *old news*, which the market heard first (an unusual stock move between event and filing, or the filing says the news was public), and *surprise news* (neither). Either way the filing is a dated headline that draws option demand and lifts implied volatility; unlike earnings, nothing follows to bring it down. After surprise news that premium pays for a move to come; after old news, for one already made. We predicted realised volatility would fall short of implied by more after old news (H1: 10 sessions, 1-month options), so a cash-secured put sold on old news would beat the same put elsewhere (H2). The other side is the option buyer the headline draws; our put seller supplies that demand. Such a premium could persist because it is small, event-specific, short-lived and hard to separate from ordinary volatility risk; our data did not find it.

Every rule, the frozen constants and the test plan were committed before any 2024-25 outcome was seen (hypothesis 07d3ddb, 2026-10-03 19:28 ET; test plan 79c001d, 19:49; constants 5bbfd9d, 22:18; first 2024-25 result 4ba177e, 22:20; amendments in `docs/decision_log.md`), so 2024-25 is a single pre-registered test with no tuning period. The notebook's 2026 out-of-sample section ships off, as its signed organiser warning requires, so no 2026 result exists; the judges' sealed-window rerun is the replication.

## 2. Method

2024-01-01 to 2025-12-31, pooled; each year a check. **Events:** late if the cover-page event date is at least one business day before acceptance; dropped if the ticker has an earnings filing within ±5 sessions. Entry `t_0` is the first close after EDGAR acceptance (next session if after 15:30 ET); `t_pre` is the last close before it. The gap runs from the session before the event to `t_pre`.

**Classification:** S = M + T; old news if S ≥ 1. M averages three gap inputs standardised with constants frozen from 1,245 ordinary days: the gap move |r_gap − r_mkt| / (σ√n) (r_mkt the TOP_100 median), the implied-volatility change, and log option volume over its prior 5-session average. T counts full-text cues: a dated prior announcement, "previously announced/disclosed/reported", an earlier people-news 8-K within 30 days.

**Outcome:** Y_h = log(RV_h / IV_0): realised volatility from `t_0` to h over 1-month ATM implied volatility at `t_0`, minus its mean on matched ordinary days (same ticker, ±60 sessions, over 5 sessions from any 8-K, same gap length).

**Tests:** H1, old minus surprise, one-sided label permutation (10,000 draws, seed 20261003); H1b, old events against ordinary days with gap moves as large; P (placebo), H1 on late governance and payout filings; H2, the put net of costs against ordinary days. Every horizon gets a bootstrap 95% interval and a Benjamini-Hochberg q.

![Figure 1](../data/oldnews/figures/insample/fig_method.png)

*Figure 1. Method: from filing to label, outcome and trade.*

## 3. Result

Of 410 filings, 286 were late, 271 clear of earnings and 146 usable: 45 old, 101 surprise.

H1 has the wrong sign and is not significant: +0.0966 (95% CI −0.0685 to +0.2621), one-sided p = 0.8615; 2024 +0.0263, 2025 +0.1636. H1b has the predicted sign, not significant: −0.0364 (−0.1435 to +0.0708), p = 0.2546. Placebo: +0.0990 (−0.1092 to +0.3081), two-sided p = 0.4013. No horizon survives the correction (Figure 2; Table A1). The committed 5% trimmed difference is +0.0811. An independent reimplementation reproduced H1 and found no lookahead in entry timing.

![Figure 2](../data/oldnews/figures/insample/fig_results.png)

*Figure 2. Mean Y_h by group; old minus surprise by horizon with 95% intervals; the placebo difference.*

**Sensitivity** (23 one-at-a-time changes to weights, cutoff, bucket, strike, category, text source, entry session): 7 negative, range −0.1150 to +0.1536, smallest one-sided p 0.1999. Entry one session later: +0.0388 (−0.1354 to +0.2129), p = 0.6681 (47 old, 86 surprise).

**Exploratory diagnostics** (after the result): minimum detectable effect 0.2109 (80% power); effects below −0.0429 are ruled out, small negative ones are not. Dropping any one of 59 tickers never flips the sign (+0.0553 to +0.1173).

## 4. What would break it

- **Classifier.** Math and word labels barely agree (Cohen's kappa 0.097); 35 of 45 old calls are events the math alone calls surprise.
- **Event date and selection.** The cover-page date is the company's, not the market's; late filing is a choice; the universe is today's top 100.
- **Information after entry.** Constants use later 2024-25 gap inputs (never outcomes); the earnings exclusion looks up to 5 sessions past entry, at dates scheduled in advance.
- **Thin options.** Weekly 1-month expiries miss sessions; 16 gap-start marks are 1 to 3 sessions old.
- **Sealed window.** We predict no detectable difference; below 30 qualifying events the run reports a descriptive result.

## 5. How to trade it

We would not trade it. As tested: sell a 1-month put 3% below spot at `t_0` on each old-news filing (acceptance lag for the 251 trade-eligible filings: median 2 sessions, IQR 1 to 3; 64% enter the next session); close after 10 sessions; at most five positions; collateral strike × 100. The committed cost rule is the larger of 5% of premium or $0.05 a share, each way (median round trip at h = 10: 16.4 bps of collateral and 1,031 bps of premium at 1x; 32.8 and 2,062 at 2x), in line with typical quoted half-spreads on liquid large-cap options; without quotes this is an assumption, so every result is repeated at 2x.

H2: −0.09 points per trade against ordinary days (−0.86 to +0.64). The mean trade is +0.121%, +0.294% without its largest loss (C, −8.17%), and the top three trades exceed the whole P&L (167%).

## 6. Performance evidence

Table 1 is the committed strategy, shown because the track asks. Daily returns come from the marked equity curve; idle days count as zero; no risk-free rate or collateral interest.

| Old-news book, h = 10 | 2024-25, 1x | 2024-25, 2x |
|---|---|---|
| Trades | 49 | 49 |
| Total return, % of book | +1.19 | −0.84 |
| Annualised return, % | +0.60 | −0.43 |
| Sharpe | 0.20 | −0.10 |
| Annualised volatility, % | 3.27 | 3.53 |
| Maximum drawdown (marked), % | −4.68 | −5.32 |
| Worst month, % | −1.91 (2025-03) | −2.12 (2025-03) |
| Skew of trade P&L | −1.72 | −1.85 |
| Hit rate, % | 67.35 | 63.27 |
| Turnover, collateral per year / book | 5.01 | 5.01 |

*Table 1. The old-news book in 2024-25, at 1x and 2x costs.*

By year the book made +1.52% in 2024 and −0.33% in 2025 at 1x (+0.49% and −1.33% at 2x; Figure A2). The same put has a higher 1x Sharpe on matched ordinary days (0.51 and 0.34 in two draws) and on all late people-news filings (0.62).

![Figure 3](../data/oldnews/figures/insample/equity_curve.png)

*Figure 3. Cumulative P&L and drawdown of the old-news book at 1x and 2x costs, against the same put on matched ordinary days and on all late people-news filings.*

Book returns at other horizons (1x): −4.91% at 1 session, −2.03% at 5, +2.52% held to expiry.

| h | n | Mean net P&L per trade, 1x (%) | 95% CI (%) | 2x mean (%) |
|---|---|---|---|---|
| 1 | 49 | −0.50 | −0.97, −0.17 | −0.72 |
| 2 | 51 | −0.36 | −0.68, −0.07 | −0.57 |
| 3 | 51 | −0.36 | −0.76, −0.03 | −0.58 |
| 5 | 51 | −0.20 | −0.70, +0.20 | −0.40 |
| 10 | 49 | +0.12 | −0.48, +0.66 | −0.09 |
| 21 (descriptive) | 17 | +0.14 | −1.91, +1.87 | −0.13 |
| 42 | 0 | n/a | n/a | n/a |
| 63 | 0 | n/a | n/a | n/a |
| expiry | 45 | +0.28 | −0.74, +1.14 | +0.07 |

*Table 2. Old-news put trades taken by the book, net P&L per trade in % of collateral; bootstrap 95% CI (10,000 draws); a 1-month put expires before 42 and 63 sessions.*

The put loses at 1 to 3 sessions, intervals below zero. At 10 sessions, +0.12% is indistinguishable from zero and below the same put on matched ordinary days (+0.23%, CI −0.22 to +0.64).

## 7. Risk management

**Limits.** Each of the five slots, filled in order of entry, holds a fifth of the book as cash: no leverage or margin call. The book averaged 1.00 open position, at most 4. The largest single loss was −1.63% of the book at 1x (C, −8.17% of its collateral), −1.73% at 2x.

**Stops.** None beyond the position cap; every position is held 10 sessions. Untested candidates: a strike-breach exit and pausing entries when market-wide implied volatility jumps.

**Market exposure.** Beta to the median TOP_100 daily return is 0.018 (R² 0.005, 492 days), understated: open puts are re-marked only at the fixed horizons, so daily volatility, Sharpe and beta use stale marks. Only market beta is estimated: the permitted data have no stock or factor returns (§10).

**Tail and regime.** Trade P&L skew is −1.72. The worst quarter by entry, 2025Q1, had 9 trades, −2.68% (−3.12% at 2x), hit rate 44.44%. Four of the five worst trades entered between 2024-12-05 and 2025-03-03 (C, INTC, AAPL, BAC; CSCO is the fifth); no stop guards against such clustering.

## 8. Liquidity and capital

Each trade is capped at 10% of the put's entry-day volume, so the median trade takes $73,500 of collateral, and a five-slot book of about $367,500 fits half the trades; 8 of 49 trades allow no contract. In total the cap allows $25,704,500 of collateral and $306,932 of P&L at 1x. Capital turns over 5.01 times a year against the book, 25.13 against capital in use.

Marks are daily closes; entry requires put volume above zero.

## 9. Data and integrity

**Sources:** Massive and SEC EDGAR (References). With no stock prices, spot comes from ATM put-call parity on the same chain, so contracts adjusted for splits, which carry their own tickers, need no price adjustment; parity ignores dividends. The notebook runs cleanly from a fresh kernel with only the API key and cache.

**Survivorship.** The universe is the top 100 as of September 2026, firms that survived and grew: short-put P&L is biased upward for every book; the effect on the old-versus-surprise contrast cannot be signed.

**Variants tested.** The ledger (`data/oldnews/ledger.csv`) has 420 rows: 242 for 2024-25 in five runs (113 primary tests, years and sensitivity grid; 1 entry at `t_0 + 1`; 96 trade books and H2 over two runs; 32 exploratory diagnostics) and 178 from four runs of the discarded 2022-23 setup (labels `discovery`, `dryrun`; commit 6c7ecdf), none used here. The dry run (2023-07-01 to 2023-12-31) overlaps July and August of the sealed placeholder (2023-06-01 to 2023-08-31); archived unused, it informed no 2024-25 choice. A 2026 run was stopped during the filing-list download (2026-10-04 01:16-01:17 ET), before any 2026 event was built.

**Decision-log disclosures** (besides §4). (1) The related-filing cue also matches the other filing's event date, slightly wider than the plan; it changes one placebo filing and no label. (2) The notebook path once used the excerpt word score, not the plan's primary full text; fixed, the excerpt result is a sensitivity (H1 −0.0118, one-sided p 0.4574). (3) A deleted scratch run of the starter's example cells made 3,283 uncached requests (ask-first limit 3,000) and marked late-2025 exits with 2026 prices; nothing from it was used, and in-sample work is clipped before 2026-01-01.

## 10. What we would test with more time

- A longer sample: detecting a 0.10 difference needs about 296 events per group, against 45 and 101; 2022-23, if Massive allows, under the frozen rules.
- Sharper tests, each committed first: math and word scores as separate hypotheses; a delta-hedged straddle for the volatility channel; the §7 stops.
- Better inputs: quotes and daily marks, an advance earnings calendar, and stock and factor returns for a momentum and value regression.

## References

*References and the appendix are outside the five-page body.*

- Massive REST API: 8-K disclosures, `/stocks/filings/8-K/vX/disclosures`; option contracts as of a date, `/v3/reference/options/contracts`; option daily bars, `/v2/aggs/ticker/O:…/range/1/day/…`.
- U.S. Securities and Exchange Commission, EDGAR (sec.gov): filing acceptance times and full 8-K text.
- Benjamini, Y. and Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society, Series B* 57(1), 289-300. (The q-values in §2-3.)
- Software: Python; pandas; NumPy; matplotlib; requests; Jupyter (ipykernel, JupyterLab). Versions in `requirements.txt`.

## Appendix

| h | Old − surprise (95% CI) | BH q |
|---|---|---|
| 1 | −0.1844 (−0.7212, +0.3464) | 0.9501 |
| 2 | +0.1400 (−0.1968, +0.4901) | 0.9501 |
| 3 | +0.1433 (−0.1466, +0.4471) | 0.9501 |
| 5 | +0.1913 (−0.0024, +0.3914) | 0.9501 |
| 10 | +0.0966 (−0.0685, +0.2621) | 0.9501 |
| 21 | −0.0579 (−0.4224, +0.2548) | 0.9501 |
| 42 | n/a | n/a |
| 63 | n/a | n/a |
| expiry | −0.0131 (−0.1597, +0.1354) | 0.9501 |

*Table A1. H1 by horizon, 1-month bucket; 33 events at 21 sessions, none at 42 or 63 (past expiry).*

![Figure A1](../data/oldnews/figures/insample/walkthrough.png)

*Figure A1. One filing end to end: event, acceptance, gap, label, entry and outcome.*

![Figure A2](../data/oldnews/figures/insample/by_year.png)

*Figure A2. Results by year, 2024 and 2025.*
