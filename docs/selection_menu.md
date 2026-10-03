# Selection rules: the decisions to make before the scan

`selection_rules.json` has a `TBD` for every choice below. The scan and the explorer refuse to run
until every `TBD` is filled in and the file is committed. That commit, made before anyone sees a
result, is our evidence that the hypotheses came first.

Nothing here comes from results. The only numbers are event counts from the census (2022–2023,
TOP_100 companies).

Already fixed by CLAUDE.md and pre-filled:
- every category plus its taxonomy families (secondary, primary)
- the earnings split: all / news alone / co-filed with earnings
- all five strategies and all three expiry buckets
- the organizers' horizons and the starter's OTM grid (3/5/10%)
- tradeable "post" entry only
- the four measurements
- a fixed seed

---

## 1. `earnings_tags`: what counts as "co-filed with earnings"
Earnings releases are filed under Item 2.02. The taxonomy's closest tags, with their 2022–2023
TOP_100 counts:

| Tag | Events |
|---|---|
| `quarterly_earnings` | 129 |
| `preliminary_results` | 23 |
| `annual_earnings` | 9 |
| `nav_per_share` | 0 |

Note: 129 quarterly results across 100 companies over two years is far fewer than the roughly 800
that exist. Earnings tagging is sparse, so "alone" will still contain some earnings filings. Say so
in the write-up.

| Option | Meaning |
|---|---|
| A | `["quarterly_earnings", "annual_earnings", "preliminary_results"]` (all results announcements) |
| B | `["quarterly_earnings"]` only (narrowest) |
| C | A plus `guidance_issuance_or_update` (guidance usually comes with results; 72 events) |

## 2. `min_sets` and `low_sample_n`: how small is too small
A "set" is one event plus its two matched ordinary days. Categories with at least *n* events:

| n | Categories | Secondary families | Primary families |
|---|---|---|---|
| 10 | 43 | 20 | 8 |
| 20 | 33 | 18 | 7 |
| 30 | 23 | 18 | 6 |
| 50 | 17 | 13 | 6 |

- `min_sets` means tests below this are not run at all. Common choices: 10 or 20.
- `low_sample_n` means results below this are shown but flagged, and (option in §6) cannot be
  candidates. Common choices: 30 or 50.

## 3. How p-values and false discoveries are handled
The full grid is about 513 tests per group and subset. With about 58 groups of 20+ events and about
2 non-empty subsets each, that's roughly **60,000 tests**.

Benjamini–Hochberg (BH) controls the share of false discoveries. To flag even one test out of 60,000
at q = 0.05, it needs a p-value below about 0.0000008.

- `p_value`:
  - `p_perm` is exact, but it can never be smaller than 1/(permutations+1). With 2,000
    permutations the floor is 0.0005, so with a large family **nothing can ever pass**.
  - `p_z` is a normal approximation built from the same permutation draws. It can go as small as
    needed. It assumes the permutation distribution is roughly bell-shaped, which is reasonable with
    20+ sets.
- `fdr_family`, meaning which tests are corrected together:
  - `all`: everything at once. Most conservative.
  - `per_level`: separately for categories, secondary families and primary families.
  - `per_metric`: strategies separately from each measurement.
  - `headline`: BH only inside a small grid we declare now (`headline` field). Everything else is
    reported as exploratory, with no q-value, and **cannot become a candidate**.

  Example headline grid: `{"bucket": ["3-6m"], "otm": [0.05], "horizon": [5, 21, 42, "exp"], "metric": ["net_pnl"]}`
  gives about 2,300 tests.
- `fdr_q`: 0.05 is standard; 0.10 is more permissive. Report whichever we choose.
- `permutations`: 2,000 is fast. 20,000 or more makes `p_perm` usable in small families but takes
  longer, roughly ten times as long per test.
- `headline`: needed only if `fdr_family` is `headline`; otherwise write `{}`.

## 4. Robustness: what a candidate must survive
Each one is true/false, except where noted:

| Field | What it checks | Common choices |
|---|---|---|
| `trimmed_mean_frac` | Fraction trimmed from each end before averaging | 0.05 or 0.10 |
| `trimmed_mean_same_sign` | The trimmed mean must agree in sign | |
| `leave_one_ticker_out` | Dropping any single company must not flip the sign | |
| `leave_one_quarter_out` | Dropping any single quarter must not flip the sign | |
| `plateau_min_same_sign_neighbours` | Neighbouring horizons and OTM levels that must agree in sign | 0 (off), 1 or 2 |
| `doubled_cost_must_hold` | Still positive with costs doubled (CLAUDE.md requires the doubled-cost *check*; this decides whether failing it rules a candidate out) | |

## 5. `selection`: what makes something a trade candidate
- `direction`:
  - `event_better_and_net_positive`: beats its ordinary days **and** makes money after costs.
  - `event_better`: only beats its ordinary days.
- `allow_low_sample`: whether a flagged low-sample result can be a candidate.
- `ranking`: `q_then_effect` (most certain first) or `effect_then_q` (largest first).
- `max_candidates`: how many go forward to in-sample confirmation (2024–2025). Fewer candidates
  means fewer tests on in-sample, which is cleaner. Common choices: 3, 5 or 10.

## 6. `confirmation`: what counts as confirmed on 2024–2025
Each candidate gets **one** test on the in-sample window (2024–2025), with the same matched-null
design. Nothing dated on or after 2026-01-01 is fetched or computed. Late-2025 events therefore lose
the horizons that would end in 2026, the same way unresolved horizons are absent elsewhere.

| Field | Meaning | Common choices |
|---|---|---|
| `p_max_one_sided` | Largest one-sided p-value, in the candidate's own direction, that counts as confirmed | 0.05 or 0.10 |
| `multiple_testing` | Correction across the candidates | `holm` (stricter, still has power); `bonferroni` (strictest); `none` (each candidate on its own) |
| `net_positive` | The in-sample event mean must make money after costs | true or false |
| `doubled_cost_must_hold` | Still positive versus ordinary days with costs doubled | true or false |
| `min_sets` | Fewest in-sample matched sets for a valid confirmation; below this the candidate is "untestable", not "failed" | 10 or 20 |

---

### After choosing
1. Fill in every `TBD` in `selection_rules.json`.
2. Commit it with a message that says it was chosen before results.
3. Run `.venv\Scripts\python src\playground\scan.py`.

The scan logs every test to `data/playground/ledger.csv`. The total count is the number we disclose.
