# Old news or surprise news after late executive 8-Ks

## 1. Hypothesis

**This is a null result: a specific mechanism, rules fixed in advance, and 2024-25 data that do not support it.** Executive and director departures and appointments (Item 5.02) at the 100 largest US companies are mostly filed days after the event. We believed they come in two kinds: *old news*, which the market heard first (the stock moved unusually between event and filing, or the filing says the news was public), and *surprise news* (no sign of either). Either way the filing is a dated headline that draws option demand and lifts implied volatility, and unlike earnings nothing follows to bring it down. After surprise news that premium pays for a move to come; after old news, for one already made. We predicted realised volatility would fall short of implied by more after old news (H1: 10 sessions, 1-month options), so a cash-secured put sold on old news would beat the same put elsewhere (H2).

Rules were committed on 2026-10-03; every amendment (`docs/decision_log.md`) preceded any 2024-25 outcome. An earlier exploratory run on 2022-23 was discarded because the challenge allows only 2024-25 data.

## 2. Method

Filings from 2024-01-01 to 2025-12-31, pooled, each year a check; spot from ATM put-call parity. **Events:** late if the cover-page event date is at least one business day before acceptance; dropped if an earnings filing for the ticker lies within ±5 sessions. Entry `t_0` is the first close after EDGAR acceptance (next session if after 15:30 ET); `t_pre` is the last close before it. The gap runs from the session before the event date to `t_pre`.

**Classification:** S = M + T, old news if S ≥ 1. M averages three gap inputs standardised with constants frozen from 1,245 ordinary days. The inputs are the gap move |r_gap − r_mkt| / (σ√n) (r_mkt the TOP_100 median), the implied-volatility change, and log option volume over its prior 5-session average. T counts full-text cues: a dated prior announcement, "previously announced/disclosed/reported", an earlier people-news 8-K by the ticker within 30 days.

**Outcome:** Y_h = log(RV_h / IV_0), parity-spot realised volatility from `t_0` to h over 1-month ATM implied volatility at `t_0`, minus its mean on matched ordinary days (same ticker, ±60 sessions, over 5 sessions from any 8-K, same gap length).

**Tests:** H1, old minus surprise, one-sided label permutation (10,000 draws, seed 20261003); H1b, old events against ordinary days with gap moves at least as large; P, H1 on late governance and payout filings; H2, the put net of costs against ordinary days. Every horizon gets a bootstrap 95% interval and Benjamini-Hochberg q. Every variant is logged (§9).

## 3. Result

Of 410 filings, 286 were late, 271 clear of earnings and 146 usable: 45 old, 101 surprise.

H1 has the opposite sign and is not significant: +0.0966 (95% CI −0.0685 to +0.2621), one-sided p = 0.8615; 2024 +0.0263, 2025 +0.1636. H1b has the predicted sign without significance: −0.0364 (−0.1435 to +0.0708), p = 0.2546. The placebo shows no difference: +0.0990 (−0.1092 to +0.3081), two-sided p = 0.4013. No horizon survives the correction. The pre-committed 5% trimmed difference is +0.0811. An independent reimplementation reproduced H1 and found no lookahead in entry timing.

![Figure 1](../data/oldnews/figures/insample/fade_curve.png)

*Figure 1. Mean Y_h, old versus surprise.*

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

*Table 1. H1 by horizon, 1-month bucket; 33 events at 21 sessions, none at 42 or 63 (past expiry).*

**Sensitivity** (23 settings changed one at a time from the primary: weights, cutoff, bucket, strike, category, text source, entry session; grid in the notebook): 7 negative, range −0.1150 to +0.1536, smallest one-sided p 0.1999. Entering one session later gives +0.0388 (−0.1354 to +0.2129), p = 0.6681 (47 old, 86 surprise).

**Exploratory diagnostics** (after the result): the minimum detectable effect is 0.2109 (80% power); the null rules out a true effect below −0.0429 but not a small negative one. Leaving out each of 59 tickers never flips the sign (+0.0553 to +0.1173).

## 4. What would break it

- **Classifier.** Math and word labels barely agree (Cohen's kappa 0.097), and 35 of 45 old calls are events the math alone calls surprise.
- **Event date and selection.** The cover-page date is the company's, not when the market heard; late filing is a choice; the universe is today's top 100.
- **Information after entry.** Constants use later 2024-25 gap inputs (never outcomes); the earnings exclusion looks up to 5 sessions past entry, at scheduled dates whose filings come later.
- **Thin options.** Weekly 1-month expiries miss sessions; 16 gap-start marks are 1 to 3 sessions old; parity ignores dividends.
- **Sealed window.** We predict no detectable old-versus-surprise difference. If the judges choose a window with thin option coverage, the run reports a descriptive result (fewer than 30 qualifying events) rather than failing.

## 5. How to trade it

We would not trade it. As tested: sell a 1-month put 3% below spot at `t_0` on each old-news filing and close after 10 sessions; at most five positions; collateral strike × 100; costs the larger of 5% of premium or $0.05 a share, each way.

The 49-trade book returns +1.19%, marked drawdown −4.68%; −0.84% at double costs. H2: −0.09 points per trade against ordinary days (−0.86 to +0.64). The mean trade is +0.121%, +0.294% without its largest loss (C, −8.17%), and the top three trades exceed the whole P&L (167%). Capacity at 10% of entry-day put volume: median $73,500 collateral per trade.


## 6. Performance evidence

The book below is the strategy as committed: old-news filings only, 10-session hold, 2024-25. It is shown because the track asks for it, not because we recommend it; H2 already finds it indistinguishable from the same put on ordinary days. Daily returns come from the marked equity curve, idle days count as zero, the risk-free rate is zero and no interest is credited on collateral.

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

*Table 2. The old-news book in 2024-25, at 1x and 2x costs.*

**Out of sample.** The 2024-25 test was itself a one-shot test: the hypothesis, test plan and frozen constants were committed to Git before any 2024-25 outcome was seen. The 2026 out-of-sample section is implemented and tested but shipped off, as the notebook's signed warning requires; no 2026 result was computed or is reported. The judges' sealed-window rerun is the true out-of-sample check, and the notebook runs it with only a start and an end date.

By year, the book made +1.52% in 2024 and −0.33% in 2025 at 1x (+0.49% and −1.33% at 2x). The same put on matched ordinary days has a higher Sharpe, 0.51 and 0.34 in the two draws, and on all late people-news filings 0.62, all at 1x. Selecting old-news filings adds nothing measurable (H2).

![Figure 2](../data/oldnews/figures/insample/equity_curve.png)

*Figure 2. Cumulative P&L and drawdown of the old-news book at 1x and 2x costs, against the same put on matched ordinary days and on all late people-news filings.*

At other horizons the old-news book loses: −4.91% at 1 session, −2.03% at 5, and gains +2.52% when held to expiry (1x).

## 7. Risk management

**Limits.** At most five positions open, taken in order of entry; each gets one fifth of the book as cash collateral (strike × 100), so there is no leverage and no margin call. In 2024-25 the book averaged 1.00 open position and never held more than 4. The largest single loss was −1.63% of the book at 1x (C, −8.17% of its collateral), −1.73% at 2x.

**Stops and de-risking.** There are none beyond the position cap. A position is held for 10 sessions whatever happens. If we ran the book we would add, and test first under committed rules: closing a put whose strike is breached by more than a set amount, and pausing new entries when market-wide implied volatility jumps. Neither was tested; neither is claimed.

**Market exposure.** Beta to the median TOP_100 daily return is 0.018 (R² 0.005, 492 days). This understates the true exposure: open puts are re-marked only at the fixed horizons (1, 2, 3, 5 and 10 sessions) and carried flat in between, so daily volatility, Sharpe and beta are measured on stale marks.

**Tail and regime.** A short put has a capped gain and a long left tail; trade P&L skew is −1.72. The worst quarter by entry, 2025Q1, had 9 trades and returned −2.68% (−3.12% at 2x), with a 44.44% hit rate. Four of the five worst trades entered between 2024-12-05 and 2025-03-03 (C, INTC, AAPL, BAC; CSCO is the fifth). Losses cluster in one period, and five equal positions with no stop give no protection against it.

## 8. Liquidity and capital

We size each trade at no more than 10% of the put's volume on the entry day. At that limit the median trade takes $73,500 of collateral, so a five-slot book of about $367,500 fits half the trades; 8 of 49 trades allow no contract at all. Summed over all trades the limit allows $25,704,500 of collateral and $306,932 of P&L at 1x. Capital turns over 5.01 times a year against the whole book and 25.13 times against the capital actually in use, because the book is mostly idle.

The one-month expiry is often a weekly option. Its ATM pair does not trade every session, and we mark from daily closes, not quotes. We require put volume above zero at entry and charge the larger of 5% of premium or $0.05 per share each way; the small 1x gain does not survive doubled costs.

## 9. Data and integrity

**Sources.** Massive: 8-K disclosures (`/stocks/filings/8-K/vX/disclosures`), option contracts as of a date (`/v3/reference/options/contracts`), option daily bars (`/v2/aggs/ticker/O:…/range/1/day/…`). SEC EDGAR: acceptance times from filing headers and the full 8-K text. No stock prices: spot is recovered from ATM put-call parity. Every response is cached; the final check ran the notebook from a clean kernel with only the API key and the cache, without errors.

**Survivorship.** The universe is the static top 100 as of September 2026. These are firms that survived and grew, so selling puts on them in 2024-25 avoids names that later fell out after large declines. That biases short-put P&L upward for every book, ours and the benchmarks alike. Its effect on the old-versus-surprise contrast cannot be signed.

**Variants tested.** The test ledger (`data/oldnews/ledger.csv`) has 420 rows. 242 belong to 2024-25, in five runs: 113 for the primary tests, years and sensitivity grid; 1 for entry at `t_0 + 1`; 96 for the trade books and H2 in two runs (the second added risk metrics); and 32 exploratory diagnostics. The other 178 rows come from four earlier runs (labels `discovery` and `dryrun`, commit 6c7ecdf) made with the discarded 2022-23 setup before the 2024-25 data rule; none is used here. The dry run covered 2023-07-01 to 2023-12-31, which overlaps July and August of the notebook's sealed placeholder (2023-06-01 to 2023-08-31); its outputs were archived unused and never informed any 2024-25 choice. A start of the 2026 run was stopped after about one minute, during the filing-list download (2026-10-04 01:16-01:17 ET), before any 2026 event was built, measured or tested; see the decision log.

**Disclosures from the decision log.** (1) The earnings exclusion uses earnings filings up to 5 sessions after entry; earnings dates are scheduled in advance, but this is information after entry. (2) The related-filing cue also matches on the other filing's event date, slightly wider than the plan's wording; one placebo filing differs and no label changes. (3) An end-to-end check found that the notebook path used the excerpt word score instead of the full text the plan makes primary; it was fixed, and the excerpt result is reported as a sensitivity (H1 −0.0118, one-sided p 0.4574). (4) In a scratch copy that was then deleted, the starter's example cells made 3,283 uncached API requests, over our 3,000 ask-first limit, and marked late-2025 exits with 2026 option prices. Nothing from it entered the test or the repository, and the notebook now clips all in-sample work before 2026-01-01.

## 10. What we would test with more time

- A longer sample. Detecting a 0.10 difference needs about 296 events per group; 2024-25 gives 45 and 101. If Massive confirms 2022-23 may be used, it would be an independent sample under the same frozen rules.
- A classifier the two scores agree on. Math and words barely agree (kappa 0.097); testing the math score and the word score as separate committed hypotheses would show which, if either, carries information.
- The volatility channel directly: implied-volatility change after the filing and a delta-hedged straddle, so the outcome does not mix volatility with direction.
- Quotes instead of closes for costs, and daily marks for every open position.
- An earnings calendar known in advance, to remove the only input that looks past entry.
- The stop and volatility-pause rules in §7, each committed before it is run.
