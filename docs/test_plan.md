# Test plan

Committed 2026-10-03, before any post-event option price or return has been computed for the hypothesis in
`docs/hypothesis.md`. Any change after this commit is made in a new commit that states the reason, before
the outcome it affects is looked at. Every variant run is recorded in the test ledger.

## Windows

- Discovery: filings from 2022-01-01 to 2023-12-31.
- Confirmation: filings from 2024-01-01 to 2025-12-31, every exit before 2026-01-01, run once with
  everything below unchanged.
- Out-of-sample: 2026, run once by a human after everything is frozen.
- Dry run of the judges' path: 2023-07-01 to 2023-12-31 (discovery data only).

## Events

- Item 5.02 people-news tags: `ceo_departure`, `cfo_departure`, `executive_officer_departure`,
  `director_departure`, `ceo_appointment`, `cfo_appointment`, `executive_officer_appointment`,
  `director_appointment`. One row per filing (accession number), labelled by the most senior role: CEO,
  CFO, other officer, director.
- Late: the cover-page event date (`CONFORMED PERIOD OF REPORT`) is at least one business day before the
  acceptance date.
- Placebo: late filings tagged `annual_meeting_results`, `shareholder_proposal_outcome`,
  `executive_compensation_change`, `bylaw_amendment`, `charter_amendment`, `dividend_declaration` or
  `equity_compensation_grant`, with no people-news tag.
- Excluded: a filing tagged `quarterly_earnings`, `annual_earnings` or `preliminary_results` for the same
  ticker on the same filing or within ±5 sessions of entry. Earnings tagging is incomplete, so some earnings
  dates will be missed; this is disclosed. Events whose 1-month put or ATM pair did not trade at entry, or
  whose gap endpoints have no usable price, are dropped. Every exclusion is counted.
- Entry: `t_0`, the first close after the EDGAR acceptance time (filings after 15:30 ET, or the same margin
  before an early close, enter the next session). The gap runs from the last session before the event
  date to `t_pre`, the last close before acceptance.

## Ordinary days

Matched non-event days as already drawn: same ticker, within ±60 sessions, more than 5 sessions from any
8-K, two per event, seed 20261003. Each ordinary day gets a pseudo-gap with the same number of sessions as
its event's gap.

## Classification: old news or surprise news

**Math score M**, the mean of three inputs, each standardised by the mean and standard deviation of the
same input on the discovery-window ordinary days (gap inputs only, never outcomes). These six constants are
computed once, written to `src/oldnews/zref_frozen.json` and committed, then applied unchanged to
discovery, confirmation, the dry run and the sealed window, so no event is standardised with data from
after its own entry outside discovery. Amended before any outcome was computed (the first draft used
ordinary days from the whole sample):

1. Gap move: `|r_gap − r_mkt| / (σ × √n)`. `r_gap` is the stock's log return over the gap from parity spot;
   `r_mkt` is the median log return over the same dates across all cached TOP_100 tickers; `σ` is the
   1-month ATM implied volatility at the start of the gap divided by √252; `n` is the number of sessions in
   the gap.
2. Implied volatility change over the gap: 1-month ATM implied volatility at `t_pre` minus at the start of
   the gap.
3. Option volume in the gap: total volume of the ATM pair over the gap sessions divided by its average over
   the 5 sessions before the gap.

**Volume input in log form** (amended before any outcome was looked at: the raw ratio is heavy-tailed, with a
95th percentile near 47 and a maximum near 1,600, so standardising it raw would let a few events dominate
`M`): the volume input is `log(volume ratio)`; it is standardised like the others.

**Coverage rules for the gap inputs** (amended before any outcome was looked at; the first draft required a
fresh mark on the 1-month ATM pair at the gap start and five cached baseline sessions, which left 49 of 258
late people-news events measurable because the cached strikes were chosen at the filing date):

- Spot at the gap start and at `t_pre` is the parity spot from the 1-month ATM pair; if that pair has no
  mark on the day, from the 2-month, then the 3-6 month ATM pair (spot does not depend on expiry); if no
  pair has a mark on the gap start, from the latest mark on or before it and no more than 3 sessions
  earlier, in which case `n` in the gap-move formula is the number of sessions from that mark to `t_pre`.
  `sigma` comes from the same bucket as the spot.
- Volume baseline: the mean over the cached sessions among the 5 before the gap start, at least 2 of them;
  otherwise the volume input is missing.
- The implied-volatility change needs a 1-month mark at both ends; otherwise it is missing.
- `M` is the mean of the available standardised inputs. The gap move is required: an event without it is
  not scored, and is counted and reported.

**Word score T**, the number of these cues present (0 to 3):

1. Dated prior announcement: the filing text dates an announcement or press release before the filing date.
2. Prior-disclosure wording: "previously announced", "previously disclosed" or "previously reported".
3. Related earlier filing: the same ticker filed another people-news 8-K in the 30 calendar days before
   this event date.

Text source: Massive's `supporting_text`. If Massive permits full EDGAR filing text, the same cues are also
computed on the full text and reported as a variant.

**Total score** `S = w_m · M + w_t · T`. Old news if `S ≥ 1`; otherwise surprise news.

| Weight set | w_m | w_t | Role |
|---|---|---|---|
| Equal | 1 | 1 | **Primary** |
| Math only | 1 | 0 | Sensitivity |
| Words only | 0 | 1 | Sensitivity |
| Math heavy | 1 | 0.5 | Sensitivity |
| Words heavy | 0.5 | 1 | Sensitivity |

Weights are fixed here and never chosen by looking at outcomes.

## Outcome

`Y_h = log(RV_h / IV_0)`. `RV_h` is the annualised realised volatility of daily parity-spot log returns
from entry to horizon `h`. `IV_0` is the 1-month ATM implied volatility at entry. Each event's outcome is
its `Y_h` minus the mean `Y_h` of its matched ordinary days.

## Tests

| ID | Test | Prediction |
|---|---|---|
| H1 (primary) | Mean outcome of old-news events minus surprise events, `h = 10` sessions, 1-month bucket. One-sided permutation test of the old/surprise labels, 10,000 permutations, seed 20261003, 5% level | Negative |
| H1b | Old-news events' `Y_10` minus that of ordinary days whose gap-move input is at least as large as the event's | Negative |
| P (placebo) | H1 computed on the placebo filings | No difference |
| H2 | Cash-secured put on old-news events, net of costs, minus the same put on matched ordinary days, and minus the same put on all late people-news events | Positive |

All fixed horizons (1, 2, 3, 5, 10, 21, 42, 63 sessions and expiry) are reported with bootstrap 95%
intervals (10,000 resamples, seed 20261003) and Benjamini-Hochberg q-values across horizons. H1 is
confirmed on 2024–25 if the sign matches and the one-sided p-value is below 0.05.

## Trade

- Cash-secured put, 1-month bucket, the strike 3% below spot at entry, sold at `t_0` on old-news events
  only, closed after 10 sessions. Results at every horizon are also reported.
- Costs: the larger of 5% of premium or $0.05 per share, on entry and on exit. Every result is repeated at
  double costs.
- At most 5 positions open at once, taken in order of entry. Collateral is strike × 100. P&L is reported
  as a percentage of collateral, with maximum drawdown, the five worst events and the 2022 period
  separately.
- Capacity: contracts per trade limited to 10% of the put's volume on the entry day; the total capital this
  allows is stated.

## Sensitivity (each reported and logged)

Weight sets above; expiry bucket (1m, 2m, 3–6m); strike (3, 5, 10% below spot); entry session (`t_0`,
`t_0 + 1`); every horizon; categories (all eight tags; departures only; adding `deal_termination`,
`business_update`, `share_repurchase_program`); total-score cutoff (0.5, 1, 1.5); text source (excerpt,
full text if permitted).

## Sealed window

Each run prints the number of events, the H1 effect with its interval and p-value, and labels the result
descriptive when fewer than 30 events qualify. Prediction: the sign of H1 holds without significance in a
3 to 8 month window, with a larger effect when market-wide implied volatility is high.
