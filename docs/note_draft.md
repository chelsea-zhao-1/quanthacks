# Old news or surprise news after late executive 8-Ks

## 1. Hypothesis

**This is a null result: a specific mechanism, rules fixed in advance, and 2024-25 data that do not support it.** Executive and director departures and appointments (Item 5.02) at the 100 largest US companies are mostly filed days after the event. We believed they come in two kinds: *old news*, which the market heard first (the stock moved unusually between event and filing, or the filing says the news was public), and *surprise news* (no sign of either). Either way the filing is a dated headline that draws option demand and lifts implied volatility, and unlike earnings nothing follows to bring it down. After surprise news that premium pays for a move to come; after old news, for one already made. We predicted realised volatility would fall short of implied by more after old news (H1: 10 sessions, 1-month options), so a cash-secured put sold on old news would beat the same put elsewhere (H2).

Rules were committed on 2026-10-03; every amendment (`docs/decision_log.md`) preceded any 2024-25 outcome. An earlier exploratory run on 2022-23 was discarded because the challenge allows only 2024-25 data.

## 2. Method

Filings from 2024-01-01 to 2025-12-31, pooled, each year a check; spot from ATM put-call parity. **Events:** late if the cover-page event date is at least one business day before acceptance; dropped if an earnings filing for the ticker lies within ±5 sessions. Entry `t_0` is the first close after EDGAR acceptance (next session if after 15:30 ET); `t_pre` is the last close before it. The gap runs from the session before the event date to `t_pre`.

**Classification:** S = M + T, old news if S ≥ 1. M averages three gap inputs standardised with constants frozen from 1,245 ordinary days. The inputs are the gap move |r_gap − r_mkt| / (σ√n) (r_mkt the TOP_100 median), the implied-volatility change, and log option volume over its prior 5-session average. T counts full-text cues: a dated prior announcement, "previously announced/disclosed/reported", an earlier people-news 8-K by the ticker within 30 days.

**Outcome:** Y_h = log(RV_h / IV_0), parity-spot realised volatility from `t_0` to h over 1-month ATM implied volatility at `t_0`, minus its mean on matched ordinary days (same ticker, ±60 sessions, over 5 sessions from any 8-K, same gap length).

**Tests:** H1, old minus surprise, one-sided label permutation (10,000 draws, seed 20261003); H1b, old events against ordinary days with gap moves at least as large; P, H1 on late governance and payout filings; H2, the put net of costs against ordinary days. Every horizon gets a bootstrap 95% interval and Benjamini-Hochberg q. The ledger holds 337 variants in 6 runs.

## 3. Result

Of 410 filings, 286 were late, 271 clear of earnings and 146 usable: 45 old, 101 surprise.

H1 has the opposite sign and is not significant: +0.0966 (95% CI −0.0685 to +0.2621), one-sided p = 0.8615; 2024 +0.0263, 2025 +0.1636. H1b has the predicted sign without significance: −0.0364 (−0.1435 to +0.0708), p = 0.2546. The placebo shows no difference: +0.0990 (−0.1092 to +0.3081), two-sided p = 0.4013. No horizon survives the correction. The pre-committed 5% trimmed difference is +0.0811. An independent reimplementation reproduced H1 and found no lookahead in entry timing.

![Figure 1](../data/oldnews/figures/insample/fade_curve.png)

*Figure 1. Mean Y_h, old versus surprise.*

| h | Old − surprise (95% CI) | BH q | 2026 |
|---|---|---|---|
| 1 | −0.1844 (−0.7212, +0.3464) | 0.9501 | {{oos_1}} |
| 2 | +0.1400 (−0.1968, +0.4901) | 0.9501 | {{oos_2}} |
| 3 | +0.1433 (−0.1466, +0.4471) | 0.9501 | {{oos_3}} |
| 5 | +0.1913 (−0.0024, +0.3914) | 0.9501 | {{oos_5}} |
| 10 | +0.0966 (−0.0685, +0.2621) | 0.9501 | {{oos_10}} |
| 21 | −0.0579 (−0.4224, +0.2548) | 0.9501 | {{oos_21}} |
| 42 | n/a | n/a | {{oos_42}} |
| 63 | n/a | n/a | {{oos_63}} |
| expiry | −0.0131 (−0.1597, +0.1354) | 0.9501 | {{oos_exp}} |

*Table 1. H1 by horizon, 1-month bucket; 33 events at 21 sessions, none at 42 or 63 (past expiry). 2026: one-time run, pending.*

**Sensitivity** (24 one-at-a-time settings: weights, cutoff, bucket, strike, category, text source, entry session; grid in the notebook): 7 negative, range −0.1150 to +0.1536, smallest one-sided p 0.1999. Entering one session later gives +0.0388 (−0.1354 to +0.2129), p = 0.6681 (47 old, 86 surprise).

**Exploratory diagnostics** (after the result): the minimum detectable effect is 0.2109 (80% power); the null rules out a true effect below −0.0429 but not a small negative one. Leaving out each of 59 tickers never flips the sign (+0.0553 to +0.1173).

## 4. What would break it

- **Classifier.** Math and word labels barely agree (Cohen's kappa 0.097), and 35 of 45 old calls are events the math alone calls surprise.
- **Event date and selection.** The cover-page date is the company's, not when the market heard; late filing is a choice; the universe is today's top 100.
- **Information after entry.** Constants use later 2024-25 gap inputs (never outcomes); the earnings exclusion looks up to 5 sessions past entry, at scheduled dates whose filings come later.
- **Thin options.** Weekly 1-month expiries miss sessions; 16 gap-start marks are 1 to 3 sessions old; parity ignores dividends.
- **Sealed window.** We predict no detectable old-versus-surprise difference. If thin option coverage leaves fewer than 30 events, the result is labelled descriptive.

## 5. How to trade it

We would not trade it. As tested: sell a 1-month put 3% below spot at `t_0` on each old-news filing and close after 10 sessions; at most five positions; collateral strike × 100; costs the larger of 5% of premium or $0.05 a share, each way.

The 49-trade book returns +1.19%, marked drawdown −4.68%; −0.84% at double costs. H2: −0.09 points per trade against ordinary days (−0.86 to +0.64). The mean trade is +0.121%, +0.294% without its largest loss (C, −8.17%), and the top three trades exceed the whole P&L (167%). Capacity at 10% of entry-day put volume: median $73,500 collateral per trade.

---

## Placeholders and sources

Delete this section before submission. `RN` = `data/oldnews/README_numbers.md` (generated 2026-10-03 22:20, run `20261003T221852-bc564b`, git `5bbfd9d49e`); `RS` = `data/oldnews/results_insample/summary.md`; `TS` = `data/oldnews/trade_insample/summary.md`.

**Still open**

| Placeholder | Source |
|---|---|
| `{{oos_h}}` (h = 1, 2, 3, 5, 10, 21, 42, 63, exp) | `data/oldnews/results_oos/profile.csv`, rows `test` = H1: `effect` (`ci_lo`, `ci_hi`). Only after a human runs the one-time 2026 test; otherwise delete the column and say 2026 was not run |
| (no others besides `{{oos_h}}`) | The diagnostics placeholders are filled below. No file gives the trade mean without its most influential trade, so the note states only the mean trade against the C loss. |

**Filled from the exploratory diagnostics** (`D` = `data/oldnews/results_insample/diagnostics/`, run `20261003T223046-a26cb6`, git `2a67f5b61a`)

| Value in the note | Source |
|---|---|
| MDE 0.2109; rules out below −0.0429 | `D/diagnostics.md` §1 (`D/power.csv`: `mde`, `rules_out_below`) |
| 59 tickers left out, 0 sign flips, range +0.0553 to +0.1173 | `D/diagnostics.md` §3 (`D/influence.csv`: `n_left_out_runs`, `sign_flips`, `loto_min`, `loto_max`) |
| Cohen's kappa 0.097; 35 of 45 old calls with math-only surprise and words-only old | `D/diagnostics.md` §2 (`D/agreement.csv`: row math_only = surprise, words_only = old, `of_which_primary_old`) |

**Filled (2024-25 only)**

| Value in the note | Source |
|---|---|
| 1,245 ordinary days (constants) | RN §1 and §7 |
| 337 variants in 6 runs | RN §8 |
| 410, 286, 271, 146, 45, 101 (sample) | RN §2 (`people` column); RS "Counts" |
| 59 tickers; H1 +0.0966, −0.0685, +0.2621, p 0.8615; 5% trimmed +0.0811 (RN shows it as "trimmed effect" without the fraction; audit §5 confirms 5%) | RN §3 (H1 row) |
| 2024 +0.0263, 2025 +0.1636 | RS "By year" (`results_insample/by_year.csv`) |
| H1b −0.0364, −0.1435, +0.0708, p 0.2546, n 49 | RN §3 (H1b row) |
| P +0.0990, −0.1092, +0.3081, two-sided p 0.4013 | RN §3 (P row) |
| Table 1: n, effect, 95% CI, q per horizon | RN §4, H1 table (`results_insample/profile.csv`) |
| Figure 1 | `data/oldnews/figures/insample/fade_curve.png` |
| 24 settings, 7 negative, −0.1150 to +0.1536, smallest p 0.1999 | `results_insample/sensitivity.csv` (24 rows counted, modified 2026-10-03 23:50; negatives: math_only ×3, math_heavy ×3, excerpt equal; RN §5 shows the first 23) |
| entry `t_0 + 1`: +0.0388 (−0.1354, +0.2129), p 0.6681, 47 old, 86 surprise | `results_insample/sensitivity.csv`, row `dimension` = "entry t0+1": `effect`, `ci_lo`, `ci_hi`, `p_one_sided`, `n_old`, `n_comp` |
| 16 stale gap-start marks | RS "Coverage" (`results_insample/coverage.csv`, `people`) |
| +1.19%, −4.68%, −0.84% (old book, h = 10) | RN §6 "Books at h = 10" |
| H2 −0.09 (−0.86, +0.64) | RN §6 H2 table (1x rows) |
| C −8.17% | RN §6 books table and "Five worst" |
| $73,500 | RN §6 capacity table |
| 45 old-news events (small sample) | RN §3 |

**Filled from the independent audit** (`A` = `data/oldnews/audit_insample.md`, generated 2026-10-03 22:40; also `docs/decision_log.md`, latest entry)

| Value in the note | Source |
|---|---|
| reimplementation reproduced H1; no lookahead in entry timing | `A` Verdict table (tasks 1 and 3 PASS), §1 H1 table, §3 rules on all 751 events |
| 5% trimmed difference +0.0811 is the pre-committed figure | `A` §5 |
| mean trade +0.121%; +0.294% without the largest loss (C, −8.17%); top three trades 167% of P&L | `A` §6 (+0.1209, +0.2937, 167.0%) |
| earnings exclusion looks up to 5 sessions after entry | `A` §3 ("Sample filter, not a classification input"); decision log, audit entry (1) |
