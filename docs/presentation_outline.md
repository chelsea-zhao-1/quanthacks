# Presentation outline: Old news, new news (5-minute pitch plus Q&A, 5 slides)

Figures come from `data/oldnews/figures/insample/`. Numbers come from `data/oldnews/README_numbers.md`. Never retype them from memory.

## Slide 1 · The idea (about 50 s) · no figure
- Executive and director 8-Ks at large companies usually arrive days after the event.
- Sometimes the market already knows (old news); sometimes the filing is the first word (surprise news).
- Claim: either way the filing lifts implied volatility, but after old news the move it pays for is already over.
- Prediction: realised minus implied volatility is lower after old news than after surprise news, so selling a put should pay.

## Slide 2 · How we label a filing without looking ahead (about 70 s) · `walkthrough.png`
- Walk through one filing (SBUX): event date, a 3-session gap, acceptance at 17:24 ET, entry at the next close.
- Math score from the gap: stock move against the market, change in implied volatility, option volume.
- Word score from the full EDGAR text: "previously announced", a dated earlier release, a related recent filing.
- Every input is known by the last close before acceptance; constants frozen once from 1,245 ordinary days.
- Hypothesis, weights, cutoff and tests all committed to Git before any outcome existed.

## Slide 3 · The result: a null (about 65 s) · `fade_curve.png`
- 146 filings, 59 tickers: 45 old news, 101 surprise news, 2024–25.
- Primary test (10 sessions, 1-month options): +0.097, interval −0.069 to +0.262, one-sided p = 0.86.
- Predicted negative; observed positive and not significant.
- No horizon survives Benjamini-Hochberg; none of the 23 sensitivity settings is significant.

## Slide 4 · Why we trust the null (about 60 s) · `placebo.png` (with `by_year.png` as backup)
- Placebo (late scheduled filings, no people news) looks like the main test: +0.099, p = 0.40.
- 2024 and 2025 each show the same pattern on their own.
- Matched ordinary days: same stock, within ±60 sessions, more than 5 sessions from any 8-K.
- Every variant is in the test ledger: 420 rows, 242 for the 2024–25 test and 178 from the 2022–23 exploration we set aside under Massive's data rule.

## Slide 5 · The trade and what comes next (about 55 s) · `equity_curve.png`
- 49 old-news puts: +1.19% net, −0.84% at double costs, max drawdown −4.68%.
- No edge over the same put on ordinary days: −0.09 points, interval −0.86 to +0.64.
- One command reproduces everything. The 2024–25 test was one-shot; the 2026 section ships off and is not run, so the judges' sealed-window rerun (start and end date only) is the true out-of-sample check.
- Prediction for the sealed window, written down in advance: no detectable difference.

## 30-second version
- When a CEO or director change is filed days late, we asked whether options overpay for news the market already had.
- We committed the test to Git first, labelled 146 filings using only information from before entry, and ran the test once.
- The answer is no: +0.097 where we predicted a negative, p = 0.86, and the placebo looks the same.
- The put trade makes +1.19% net and −0.84% at double costs, which is no edge. We expect the sealed window to show the same null.
