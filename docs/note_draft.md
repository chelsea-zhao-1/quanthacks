# Old news or surprise news: implied volatility after late executive 8-Ks

## 1. Hypothesis

We study Item 5.02 filings at the 100 largest US companies: executive and director departures and appointments. Most are filed days after the event. We believe they are of two kinds. *Old news*: the market heard first, so the stock moved unusually between event and filing, or the filing says the news was public. *Surprise news*: the stock was quiet and the filing shows no earlier disclosure.

Either way we believe the filing is a new, dated headline that draws option demand and lifts implied volatility, and unlike earnings nothing follows to bring it back down. After surprise news that premium pays for a move to come; after old news, for one already made. We expect realised volatility to fall short of implied by more after old news (H1: 10 sessions, 1-month options, prediction negative), and trade it by selling cash-secured puts on old-news filings (H2).

Rules were committed on 2026-10-03; every amendment (`docs/decision_log.md`) preceded any 2024-25 outcome.

## 2. Method

Window: filings from 2024-01-01 to 2025-12-31; pooled is primary, each year a check. Spot is from ATM put-call parity.

**Events.** One row per accession. Late: cover-page event date at least one business day before acceptance. Dropped: filings with an earnings filing for the ticker within ±5 sessions ({{n_earn_dropped}}; tags are incomplete). Entry `t_0` is the first close after EDGAR acceptance, the next session if after 15:30 ET; `t_pre` is the last close before it. The gap runs from the last session before the event date to `t_pre`.

**Classification.** S = M + T; old news if S ≥ 1. M averages three inputs standardised with constants frozen from 2024-25 ordinary days (gap inputs only; `zref_frozen_insample.json`): gap move |r_gap − r_mkt| / (σ√n), r_mkt the TOP_100 median return, σ the gap-start implied volatility; implied-volatility change over the gap; log gap option volume over its prior 5-session average. T counts cues in the full EDGAR text: a dated prior announcement, "previously announced/disclosed/reported", an earlier people-news 8-K by the ticker within 30 days.

**Outcome.** Y_h = log(RV_h / IV_0): realised volatility of parity-spot returns from `t_0` to h over 1-month ATM implied volatility at `t_0`, minus its mean on two matched ordinary days (same ticker, ±60 sessions, over 5 sessions from any 8-K, same gap length; seed 20261003).

**Tests.** H1: old minus surprise, one-sided label permutation (10,000 draws, seed 20261003), 5% level. H1b: old events against ordinary days with a gap move at least as large. P: H1 on late governance and payout filings, expecting no difference. H2: the put below, net of costs, against ordinary days and all late filings. Every fixed horizon, expiry included, gets a bootstrap 95% interval and Benjamini-Hochberg q-value. The ledger holds all {{n_variants}} variants.

## 3. Result

{{n_late}} late filings remain: {{n_old}} old news, {{n_surprise}} surprise. H1: {{H1_diff}} (95% CI {{H1_lo}} to {{H1_hi}}), p = {{H1_p}}, {{H1_verdict}}; 2024 {{H1_2024}}, 2025 {{H1_2025}}. H1b: {{H1b_diff}} ({{H1b_lo}} to {{H1b_hi}}). Placebo: {{P_diff}} ({{P_lo}} to {{P_hi}}), p = {{P_p}}.

{{FIG_fade_curve}}

*Figure 1. Mean Y_h by horizon, old versus surprise.*

| h | Old − surprise (95% CI) | BH q | 2026 |
|---|---|---|---|
| 1 | {{d_1}} ({{ci_1}}) | {{q_1}} | {{oos_1}} |
| 2 | {{d_2}} ({{ci_2}}) | {{q_2}} | {{oos_2}} |
| 3 | {{d_3}} ({{ci_3}}) | {{q_3}} | {{oos_3}} |
| 5 | {{d_5}} ({{ci_5}}) | {{q_5}} | {{oos_5}} |
| 10 | {{d_10}} ({{ci_10}}) | {{q_10}} | {{oos_10}} |
| 21 | {{d_21}} ({{ci_21}}) | {{q_21}} | {{oos_21}} |
| 42 | {{d_42}} ({{ci_42}}) | {{q_42}} | {{oos_42}} |
| 63 | {{d_63}} ({{ci_63}}) | {{q_63}} | {{oos_63}} |
| expiry | {{d_exp}} ({{ci_exp}}) | {{q_exp}} | {{oos_exp}} |

*Table 1. H1 by horizon, 1-month bucket. 2026 run once, after the freeze.*

**Sensitivity.** H1 re-run one choice at a time (weights, cutoff, expiry bucket, strike, entry session, category set, text source): {{sens_n_neg}} of {{sens_n}} keep the negative sign; effects span {{sens_min}} to {{sens_max}}. Full grid in the notebook.

## 4. What would break it

- **Classifier.** The split may proxy for large moves in volatile names. H1b and P test this: a placebo gap as large as H1 would mean late filings, not old news.
- **Private event date.** The cover-page date is the company's own, perhaps a board date, not when the market heard; it mislabels events and should shrink the difference.
- **Selection.** Firms choose what to file late and the universe is today's top 100; we cannot sign either effect.
- **Constants.** A 2024 event is standardised partly with later gap inputs. No outcome enters, but a live trader could not have had them.
- **Thin options.** The 1-month expiry is often a weekly whose ATM pair misses sessions. Gap-start marks up to 3 sessions old are allowed ({{n_stale}} filings), closes are not quotes and parity ignores dividends.
- **Small sample.** {{n_old}} old-news events, at most five positions; under 30 events is labelled descriptive. One crash can dominate a short-put record.
- **Sealed window.** We predict H1's sign holds without significance in a 3 to 8 month window, larger when market-wide implied volatility is high.

## 5. How to trade it

On each late people-news filing scored old news, sell a 1-month put 3% below spot at `t_0` and close after 10 sessions. At most five positions, in order of entry; collateral is strike × 100. Costs: the larger of 5% of premium or $0.05 a share, each way. Size is capped at 10% of the put's entry-day volume: median {{cap_median}} of collateral per trade, a book near {{cap_book}}.

Net of costs: {{n_trades}} trades, {{ret_total}}% total ({{ret_ann}}% annualised), hit rate {{hit}}%, maximum drawdown {{maxdd}}%; {{ret_total_2x}}% at double costs. H2: {{H2_null_diff}} points per trade against ordinary days ({{H2_null_lo}} to {{H2_null_hi}}), {{H2_all_diff}} against all late filings. Worst five: {{worst5}}.

---

## Placeholders

Delete this section before submission. Every value comes from a run of the committed rules on 2024-25; `<insample>` is `data/oldnews/results_insample/`, `<trade>` is `data/oldnews/trade_insample/`. Nothing here is filled until the architect says the inputs are final.

| Placeholder | File | Column or source |
|---|---|---|
| `{{n_late}}`, `{{n_earn_dropped}}` | `<insample>/counts.csv` | column `people`; step "outcome usable ..." for `n_late`; the drop from "late" to "no earnings filing within 5 sessions" for `n_earn_dropped` |
| `{{n_old}}`, `{{n_surprise}}` | `<insample>/h1.csv` | `n_old`, `n_comp` |
| `{{n_variants}}` | `data/oldnews/ledger.csv` | row count (`ledger.variant_count`) |
| `{{H1_diff}}`, `{{H1_lo}}`, `{{H1_hi}}`, `{{H1_p}}` | `<insample>/h1.csv` | `effect`, `ci_lo`, `ci_hi`, `p_one_sided` |
| `{{H1_verdict}}` | `<insample>/h1.csv` | "confirmed" if `effect` < 0 and `p_one_sided` < 0.05, else "not confirmed" (write what the numbers say) |
| `{{H1_2024}}`, `{{H1_2025}}` | `<insample>/by_year.csv` | rows `test` = H1, `period` = 2024 / 2025; `effect` with CI and p |
| `{{H1b_diff}}`, `{{H1b_lo}}`, `{{H1b_hi}}` | `<insample>/h1b.csv` | `effect`, `ci_lo`, `ci_hi` |
| `{{P_diff}}`, `{{P_lo}}`, `{{P_hi}}`, `{{P_p}}` | `<insample>/placebo.csv` | `effect`, `ci_lo`, `ci_hi`, `p_two_sided` |
| `{{FIG_fade_curve}}` | `data/oldnews/figures/fade_curve.png` | image from `src/oldnews/figures.py` |
| `{{d_h}}`, `{{ci_h}}`, `{{q_h}}` for h = 1, 2, 3, 5, 10, 21, 42, 63, exp | `<insample>/profile.csv` | rows `test` = H1, `horizon` = h; `effect`, `ci_lo` to `ci_hi`, `q_bh` |
| `{{oos_h}}` for the same h | `data/oldnews/results_oos/profile.csv` | same columns. Filled only by a human after the one-time 2026 run. If it is not run, delete the column and its caption sentence and say 2026 was not run |
| `{{sens_n_neg}}`, `{{sens_n}}`, `{{sens_min}}`, `{{sens_max}}` | `<insample>/sensitivity.csv` | count of rows with `effect` < 0; row count; min and max `effect` |
| `{{n_stale}}` | `<insample>/coverage.csv` | row "gap-start mark stale (1 to 3 sessions old)", column `people` |
| `{{cap_median}}`, `{{cap_book}}` | `<trade>/capacity.csv` | `trade.capacity_summary`: `median_collateral_usd`, `book_at_median_usd` |
| `{{n_trades}}`, `{{ret_total}}`, `{{ret_ann}}`, `{{hit}}`, `{{maxdd}}` | `<trade>/summary.csv` | `book` = old, `horizon` = 10, `cost` = 1x: `n_trades`, `total_return_pct`, `annualised_pct`, `hit_rate_pct`, `max_dd_mtm_pct` |
| `{{ret_total_2x}}` | `<trade>/summary.csv` | same row with `cost` = 2x, `total_return_pct` |
| `{{H2_null_diff}}`, `{{H2_null_lo}}`, `{{H2_null_hi}}` | `<trade>/h2.csv` | `comparison` = old_minus_null, `horizon` = 10, `cost` = 1x: `diff_pct`, `ci_lo_pct`, `ci_hi_pct` |
| `{{H2_all_diff}}` | `<trade>/h2.csv` | `comparison` = old_minus_all_late, `horizon` = 10, `cost` = 1x: `diff_pct` |
| `{{worst5}}` | `<trade>/worst.csv` | `ticker`, `pnl_pct` of the five rows |
