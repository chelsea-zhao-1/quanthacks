"""Figures for the old-news test, in the notebook's own style (section 7 palette and axes).

    1  fade_curve     the outcome by horizon: old and surprise news, and old minus surprise with its 95% interval
    2  placebo        old minus surprise for the people-news filings, next to the same on the placebo filings
    3  by_year        the same, split by the calendar year of entry (2024 and 2025)
    4  equity_curve   the cash-secured put on old news: cumulative P&L at 1x and 2x costs against the same put on
                      matched ordinary days and on all late filings, and its drawdown
    5  walkthrough    one filing end to end: event date, gap, filing, entry, option-implied versus realised move

No statistic is recomputed here. Figures 1 to 3 plot results_<label>/profile.csv and profile_by_year.csv exactly as
src/oldnews/tests.py wrote them (group means, the old-minus-surprise effect, its bootstrap 95% interval, the
permutation p-value and the Benjamini-Hochberg q); figure 4 plots trade_<label>/equity.csv and quotes
trade_<label>/summary.csv (its drawdown is the curve's own running peak-to-trough, checked against the table). The
profile has an interval for the difference only, so the group means are drawn without bands. The outcome is Y_h = log(RV_h / IV_0), each event minus the mean of its matched ordinary days; below zero
means realised volatility fell short of implied by more than on that stock's ordinary days. Horizons 42 and 63
sessions have no observations in the 1-month bucket (the option expires first); they stay on the axis, marked
n = 0, because the fixed horizons are reported in full.

Figure 2 shows one illustrative filing, chosen by a rule that never looks at outcomes (`walkthrough_candidates`):
among the old-news filings in the primary test's sample that have all three math inputs, the one whose gap length
is closest to the median gap length of that set (earliest entry, then row id, on ties).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HORIZONS = ["1", "2", "3", "5", "10", "21", "42", "63", "expiry"]
PRIMARY_H, PRIMARY_BUCKET, PRIMARY_OTM = "10", "1m", 3

# The notebook's tokens (section 7), so these figures sit beside its own.
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
OLD, SURPRISE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2 (validated: CVD dE 24.7, contrast >= 3:1)
YEARS = {"2024": ("#4a3aa7", "o"), "2025": ("#1baf7a", "s")}   # categorical slots 7 and 3 (validated: CVD dE 26.6); marker is the second cue
LABEL_BOX = dict(boxstyle="square,pad=0.15", facecolor=SURFACE, edgecolor="none")


@dataclass
class Inputs:
    """The tables a run produced. `profile` is tests.run()["profile"] (results_<label>/profile.csv); the rest only
    serve the walkthrough: `classified` (classify's table), `outcome` and `nulls` (00_shared.md columns)."""
    profile: pd.DataFrame
    classified: pd.DataFrame | None = None
    outcome: pd.DataFrame | None = None
    nulls: pd.DataFrame | None = None
    profile_by_year: pd.DataFrame | None = None     # results_<label>/profile_by_year.csv
    equity: pd.DataFrame | None = None              # trade_<label>/equity.csv
    trade_summary: pd.DataFrame | None = None       # trade_<label>/summary.csv


def flag(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes"])


def norm_h(h) -> str:
    """"exp" / "expiry" -> "expiry"; 10, "10", "10.0" -> "10"."""
    t = str(h).strip().lower()
    if t in ("exp", "expiry"):
        return "expiry"
    try:
        return str(int(float(t)))
    except ValueError:
        return t


# ---------------------------------------------------------------------------------------------------------
# Data, straight from the tables
# ---------------------------------------------------------------------------------------------------------
def read_tables(label: str, data_dir: Path) -> Inputs:
    """Read what a finished run wrote (results_<label>/profile.csv and the tables behind the walkthrough)."""
    from oldnews import pipeline

    pipeline.check_label(label)
    d = Path(data_dir)
    kw = dict(keep_default_na=False, na_values=["", "nan", "NaN"])
    data = Inputs(profile=pd.read_csv(d / f"results_{label}" / "profile.csv", dtype={"horizon": str}, **kw),
                  classified=pd.read_csv(d / f"classified_{label}.csv", **kw),
                  outcome=primary_rows(pd.read_csv(d / f"outcome_{label}.csv", **kw)),
                  nulls=pd.read_csv(d / f"nulls_{label}.csv", **kw))
    return attach_files(data, d, label)


def attach_files(data: Inputs, data_dir: Path, label: str) -> Inputs:
    """Add the tables the year split and the equity figure plot, when the run wrote them."""
    d = Path(data_dir)
    kw = dict(keep_default_na=False, na_values=["", "nan", "NaN"], dtype={"horizon": str})
    files = {"profile_by_year": d / f"results_{label}" / "profile_by_year.csv",
             "equity": d / f"trade_{label}" / "equity.csv", "trade_summary": d / f"trade_{label}" / "summary.csv"}
    for name, path in files.items():
        if path.exists() and getattr(data, name) is None:
            setattr(data, name, pd.read_csv(path, **kw))
    return data


def primary_rows(outcome: pd.DataFrame) -> pd.DataFrame:
    """The outcome rows of the primary test's cell: the 1-month bucket, the 3% put."""
    return outcome[(outcome["bucket"].astype(str) == PRIMARY_BUCKET) & (pd.to_numeric(outcome["otm"]) == PRIMARY_OTM)]


def from_profile(prof: pd.DataFrame, test: str) -> pd.DataFrame:
    """One test's rows from the stats module's horizon profile, indexed by horizon (all nine, in order)."""
    t = prof[prof["test"] == test].copy()
    t["horizon"] = t["horizon"].map(norm_h)
    keep = ["n", "n_old", "n_comp", "mean_old", "mean_comp", "effect", "ci_lo", "ci_hi", "p", "q_bh"]
    return t.set_index("horizon").reindex(HORIZONS)[keep].apply(pd.to_numeric, errors="coerce")


def walkthrough_candidates(c: pd.DataFrame, outcome: pd.DataFrame, nulls: pd.DataFrame) -> pd.DataFrame:
    """Old-news filings in the primary test's sample with all three math inputs, in the order of the rule: gap
    length closest to the median gap length of the set, then earliest entry, then row id. No outcome enters the
    order; an outcome must merely exist (the h = 10 outcome of the 1-month, 3% put row, and a matched ordinary day,
    as in the test) so the figure is complete."""
    o = outcome.assign(horizon=outcome["horizon"].map(norm_h))
    ok = o[(o["horizon"] == PRIMARY_H) & flag(o["usable"]) & np.isfinite(pd.to_numeric(o["y"], errors="coerce"))]
    has_null = set(nulls.loc[nulls["row_id"].isin(ok["row_id"]), "event_row_id"])
    ev = c[(c["group"] == "people") & flag(c["late"]) & ~flag(c["earnings_excluded"]) & flag(c["scored"])
           & flag(c["old"]) & (pd.to_numeric(c["n_inputs"]) == 3) & c["row_id"].isin(ok["row_id"]) & c["row_id"].isin(has_null)]
    if ev.empty:
        return ev
    ev = ev.merge(ok[["row_id", "iv0", "rv", "y", "entry_date", "exit_date"]], on="row_id", how="left")
    ev = ev.assign(_dist=(pd.to_numeric(ev["n_gap"]) - pd.to_numeric(ev["n_gap"]).median()).abs(),
                   _t0=pd.to_datetime(ev["t_0"]))
    return ev.sort_values(["_dist", "_t0", "row_id"]).drop(columns=["_dist", "_t0"]).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------------------------------------
def style(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """The notebook's axes: no box, recessive y grid, ink text."""
    ax.figure.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.tick_params(axis="x", labelsize=getattr(ax, "_xlabelsize", 9))
    if title:
        ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def _under_title(ax, note: str) -> None:
    """A note (one or two lines) between a panel's title and its plot area, where no data can collide with it."""
    lines = note.count("\n") + 1
    ax.set_title(ax.get_title(loc="left"), loc="left", color=INK, fontsize=11, fontweight="bold", pad=12 + 12 * lines)
    ax.annotate(note, (0, 1), xycoords="axes fraction", xytext=(0, 6), textcoords="offset points", ha="left",
                va="bottom", fontsize=8.5, color=INK2, linespacing=1.35)


def _headline(fig, title: str, subtitle: str) -> float:
    """Title and subtitle at the top left; returns the figure fraction left for the axes below them."""
    h = fig.get_figheight()
    fig.suptitle(title, x=0.01, y=1 - 0.08 / h, ha="left", va="top", color=INK, fontsize=12.5, fontweight="bold")
    fig.text(0.01, 1 - 0.36 / h, subtitle, ha="left", va="top", color=INK2, fontsize=9, linespacing=1.5)
    return 1 - (0.36 + 0.21 * (subtitle.count("\n") + 1) - 0.25) / h     # tight_layout adds its own padding


def _tick_labels(t: pd.DataFrame, sizes: bool = True) -> list[str]:
    """Horizon on the first line, the group sizes under it (n = 0 where the 1-month option has expired)."""
    labels = []
    for h, r in t.iterrows():
        name = "exp" if h == "expiry" else h
        if not sizes:
            labels.append(name)
            continue
        n_old, n_comp = r["n_old"], r["n_comp"]
        labels.append(f"{name}\n" + ("n = 0" if not np.isfinite(r["n"]) or r["n"] == 0 else f"{int(n_old)}|{int(n_comp)}"))
    return labels


def _horizon_axis(ax, x: np.ndarray, t: pd.DataFrame, primary: bool = True, sizes: bool = True) -> None:
    ax._xlabelsize = 7.3 if sizes else 9           # style() applies it after the ticks are set
    ax.set_xticks(x, _tick_labels(t, sizes))
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
    if primary:
        i = HORIZONS.index(PRIMARY_H)
        ax.axvspan(i - 0.35, i + 0.35, color=GRID, alpha=0.55, linewidth=0, zorder=0)
        ax.annotate("primary test", (i, 1), xycoords=("data", "axes fraction"), xytext=(0, -2),
                    textcoords="offset points", ha="center", va="top", fontsize=8, color=INK2)


def _line(ax, x, mean, color: str, label: str = "", lo=None, hi=None, marker: str = "o", band: bool = True) -> None:
    if lo is not None:
        if band:
            ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0, zorder=2)
        ax.vlines(x, lo, hi, color=color, linewidth=1.2, zorder=2)
    ax.plot(x, mean, color=color, linewidth=2, marker=marker, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
            solid_capstyle="round", solid_joinstyle="round", label=label, zorder=3)


def _primary_note(t: pd.DataFrame, predicted: str = "negative", sided: str = "one-sided") -> str:
    """The primary cell in two lines, as the table states it: the effect with its interval, the p-value, and the
    prediction against what was observed (the sign, for a directional prediction). No judgement beyond that."""
    r = t.loc[PRIMARY_H]
    if not np.isfinite(r["effect"]):
        return ""
    first = f"h = {PRIMARY_H}: {r['effect']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
    second = f"{sided} p = {r['p']:.4f}" + (f", BH q = {r['q_bh']:.3f}" if np.isfinite(r["q_bh"]) else "")
    seen = "positive" if r["effect"] > 0 else "negative"
    second += f"; predicted {predicted}" + (f", observed {seen}" if predicted in ("negative", "positive") else "")
    return first + "\n" + second


# ---------------------------------------------------------------------------------------------------------
# Figure 1 · the fade curve
# ---------------------------------------------------------------------------------------------------------
def fade_curve(h1: pd.DataFrame, path: Path | None = None, subtitle: str = "") -> plt.Figure:
    """h1: from_profile(profile, "H1")."""
    x = np.arange(len(HORIZONS))
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.9), gridspec_kw={"width_ratios": [1, 1]})
    # left: where each group sits
    _line(a, x, h1["mean_old"].to_numpy(float), OLD, "Old news")
    _line(a, x, h1["mean_comp"].to_numpy(float), SURPRISE, "Surprise news")
    _horizon_axis(a, x, h1)
    style(a, "Mean outcome by group", "sessions after entry (exp = expiry); events: old | surprise",
          "log(realised / implied vol), event minus ordinary days")
    a.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="lower right")
    _under_title(a, "event Y minus the mean Y of its matched ordinary days\n(the stats table gives an interval for the difference only)")
    # right: the tested difference with its interval
    _line(b, x, h1["effect"].to_numpy(float), INK2, "", h1["ci_lo"].to_numpy(float), h1["ci_hi"].to_numpy(float))
    _horizon_axis(b, x, h1)
    style(b, "Old minus surprise, 95% interval", "sessions after entry (exp = expiry); events: old | surprise",
          "difference in log(realised / implied vol)")
    _under_title(b, _primary_note(h1, "negative", "one-sided"))
    top = _headline(fig, "The fade curve: does old news over-price the move?",
                    subtitle or "Late executive and director 8-Ks, 1-month options. Prediction: old news below surprise news.")
    fig.tight_layout(rect=(0, 0, 1, top))
    if path:
        fig.savefig(path, dpi=160, facecolor=SURFACE)
    return fig


# ---------------------------------------------------------------------------------------------------------
# Figure 2 · one event, end to end
# ---------------------------------------------------------------------------------------------------------
def _pct(v) -> str:
    return "n/a" if pd.isna(v) else f"{v:.0%}"


def walkthrough(row: pd.Series, spot: pd.Series, path: Path | None = None, horizon: int = int(PRIMARY_H),
                note: str = "") -> plt.Figure:
    """Price path through the gap and the filing, then the option-implied +/-1 sigma cone against what happened."""
    spot = spot.dropna()
    t0, t_pre, g0 = pd.Timestamp(row["t_0"]), pd.Timestamp(row["t_pre"]), pd.Timestamp(row["gap_start"])
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    style(ax, xlabel="", ylabel="stock price from put-call parity ($)")

    ax.axvspan(g0, t_pre, color=GRID, alpha=0.6, linewidth=0, zorder=0)
    n_gap = int(row["n_gap"])
    ax.annotate(f"gap: {n_gap} session{'s' if n_gap != 1 else ''}, from the last close before the event to the last "
                "close before the filing", (g0, 0.02), xycoords=("data", "axes fraction"), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color=INK2, va="bottom", bbox=LABEL_BOX, zorder=5)

    # the option-implied 1-sigma cone from entry, sized by the 1-month ATM implied vol at entry
    y_lo, y_hi = float(spot.min()), float(spot.max())
    if t0 in spot.index and np.isfinite(pd.to_numeric(row.get("iv0"), errors="coerce")):
        after = spot.loc[t0:]
        k = np.arange(len(after))
        s0, iv0 = float(spot.loc[t0]), float(row["iv0"])
        up, dn = s0 * np.exp(iv0 * np.sqrt(k / 252)), s0 * np.exp(-iv0 * np.sqrt(k / 252))
        ax.fill_between(after.index, dn, up, color=OLD, alpha=0.10, linewidth=0, zorder=1)
        ax.plot(after.index, up, color=OLD, linewidth=1, zorder=1)
        ax.plot(after.index, dn, color=OLD, linewidth=1, zorder=1)
        ax.annotate(f"option-implied ±1σ (IV₀ = {_pct(iv0)})", (after.index[-1], up[-1]), xytext=(-4, 6),
                    textcoords="offset points", ha="right", fontsize=8.5, color=INK2)
        y_lo, y_hi = min(y_lo, float(dn.min())), max(y_hi, float(up.max()))

    ax.plot(spot.index, spot.to_numpy(float), color=INK, linewidth=2, solid_capstyle="round", zorder=3)
    exit_d = pd.Timestamp(row["exit_date"]) if pd.notna(row.get("exit_date")) else None
    for d, txt in [(t0, "entry"), (exit_d, f"exit, {horizon} sessions")]:
        if d is not None and d in spot.index:
            ax.plot([d], [spot.loc[d]], marker="o", markersize=8, color=INK, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, zorder=4)
            if txt != "entry":
                ax.annotate(txt, (d, spot.loc[d]), xytext=(0, -14), textcoords="offset points", ha="center",
                            fontsize=8, color=INK2, bbox=LABEL_BOX, zorder=5)

    # the three dates, labels staggered so close dates never collide
    accepted = pd.Timestamp(row["accepted_at"]) if pd.notna(row.get("accepted_at")) else None
    events = [(pd.Timestamp(row["event_date"]), f"event (cover page) {pd.Timestamp(row['event_date']):%d %b %Y}")]
    if accepted is not None:
        events.append((accepted, f"filing accepted {accepted:%d %b %H:%M} ET"))
    events.append((t0, f"entry t₀: close of {t0:%d %b}"))
    for j, (d, txt) in enumerate(events):
        ax.axvline(d, color=MUTED, linewidth=1, zorder=2)
        ax.annotate(txt, (d, 0.98 - 0.07 * j), xycoords=("data", "axes fraction"), xytext=(4, 0),
                    textcoords="offset points", fontsize=8.5, color=INK2, va="top", bbox=LABEL_BOX, zorder=5)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
    span = (y_hi - y_lo) or 1.0
    ax.set_ylim(y_lo - 0.12 * span, y_hi + 0.30 * span)       # room below for the gap label, above for the dates

    S, T = pd.to_numeric(row.get("S"), errors="coerce"), pd.to_numeric(row.get("T"), errors="coerce")
    gm, dv, vr = (pd.to_numeric(row.get(k), errors="coerce") for k in ("gap_move", "d_iv_gap", "vol_gap_ratio"))
    why = (f"gap move {gm:.1f}σ · implied vol {100 * dv:+.1f} pts in the gap · option volume ×{vr:.1f} · "
           f"word cues {int(T) if pd.notna(T) else 'n/a'} of 3" + (f" → old news (S = {S:.2f})" if pd.notna(S) else " → old news"))
    what = (f"next {horizon} sessions: realised {_pct(row.get('rv'))} vs implied {_pct(row.get('iv0'))} → "
            f"Y = log(RV/IV₀) = {pd.to_numeric(row.get('y'), errors='coerce'):+.2f}")
    role = str(row.get("role", "") or "people-news")
    top = _headline(fig, f"One filing, end to end: {row['ticker']}, {role} 8-K",
                    f"{why}\n{what}" + (f"\n{note}" if note else ""))
    fig.tight_layout(rect=(0, 0, 1, top))
    if path:
        fig.savefig(path, dpi=160, facecolor=SURFACE)
    return fig


# ---------------------------------------------------------------------------------------------------------
# Figure 3 · the placebo beside the main result
# ---------------------------------------------------------------------------------------------------------
def placebo(h1: pd.DataFrame, p: pd.DataFrame, path: Path | None = None, subtitle: str = "") -> plt.Figure:
    """h1, p: from_profile(profile, "H1") and from_profile(profile, "P")."""
    x = np.arange(len(HORIZONS))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.7), sharey=True)
    for ax, t, title, sided, pred in ((axes[0], h1, "People-news 8-Ks (test H1)", "one-sided", "negative"),
                                      (axes[1], p, "Placebo: late scheduled 8-Ks (test P)", "two-sided", "no difference")):
        _line(ax, x, t["effect"].to_numpy(float), INK2, "", t["ci_lo"].to_numpy(float), t["ci_hi"].to_numpy(float))
        _horizon_axis(ax, x, t)
        style(ax, title, "sessions after entry (exp = expiry); events: old | surprise",
              "old minus surprise, log(RV / IV₀)" if ax is axes[0] else "")
        _under_title(ax, _primary_note(t, pred, sided))
    top = _headline(fig, "The placebo: the same split on filings that carry no people news",
                    subtitle or "Prediction: negative on the left, about zero on the right.")
    fig.tight_layout(rect=(0, 0, 1, top))
    if path:
        fig.savefig(path, dpi=160, facecolor=SURFACE)
    return fig


# ---------------------------------------------------------------------------------------------------------
# Figure 4 · the year split
# ---------------------------------------------------------------------------------------------------------
def by_year(prof_y: pd.DataFrame, path: Path | None = None, subtitle: str = "") -> plt.Figure:
    """prof_y: results_<label>/profile_by_year.csv (the stats module's own per-year horizon profile)."""
    x = np.arange(len(HORIZONS))
    periods = sorted(str(p) for p in prof_y["period"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.7), sharey=True)
    for ax, test, title, sided, pred in ((axes[0], "H1", "People-news 8-Ks (test H1)", "one-sided", "negative"),
                                         (axes[1], "P", "Placebo: late scheduled 8-Ks (test P)", "two-sided", "no difference")):
        notes = []
        for k, period in enumerate(periods):
            t = from_profile(prof_y[prof_y["period"].astype(str) == period], test)
            color, marker = YEARS.get(period, (INK2, "o"))
            dx = (k - (len(periods) - 1) / 2) * 0.22
            r = t.loc[PRIMARY_H]
            _line(ax, x + dx, t["effect"].to_numpy(float), color,
                  f"{period}: {int(r['n_old'])} old | {int(r['n_comp'])} surprise at h = {PRIMARY_H}",
                  t["ci_lo"].to_numpy(float), t["ci_hi"].to_numpy(float), marker=marker, band=False)
            notes.append(f"{period}: {r['effect']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}], {sided} p = {r['p']:.3f}")
        _horizon_axis(ax, x, t, sizes=False)
        style(ax, title, "sessions after entry (exp = expiry); no events at 42 and 63",
              "old minus surprise, log(RV / IV₀)" if ax is axes[0] else "")
        ax.legend(frameon=False, labelcolor=INK2, fontsize=8.5, loc="lower right")
        _under_title(ax, f"h = {PRIMARY_H} by year of entry; predicted {pred}\n" + "\n".join(notes))
    top = _headline(fig, "The same split, by calendar year of entry",
                    subtitle or "Each year run on its own with the same rules; bars are bootstrap 95% intervals.")
    fig.tight_layout(rect=(0, 0, 1, top))
    if path:
        fig.savefig(path, dpi=160, facecolor=SURFACE)
    return fig


# ---------------------------------------------------------------------------------------------------------
# Figure 5 · the trade
# ---------------------------------------------------------------------------------------------------------
def _drawdown(mtm_pct: pd.Series) -> pd.Series:
    """Running peak-to-trough of the book's value, in %: wealth = 1 + cumulative return, as trade.max_drawdown."""
    wealth = 1 + mtm_pct.to_numpy(float) / 100
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    return pd.Series(100 * (wealth / peak - 1), index=mtm_pct.index)


def equity_curve(eq: pd.DataFrame, summary: pd.DataFrame | None = None, path: Path | None = None,
                 subtitle: str = "", horizon: str = PRIMARY_H) -> plt.Figure:
    """eq: trade_<label>/equity.csv; summary: trade_<label>/summary.csv (totals and drawdowns are quoted from it)."""
    eq = eq.assign(date=pd.to_datetime(eq["date"]), horizon=eq["horizon"].map(norm_h))
    eq = eq[eq["horizon"] == horizon]
    summ = None
    if summary is not None:
        summ = summary.assign(horizon=summary["horizon"].map(norm_h))
        summ = summ[summ["horizon"] == horizon].set_index(["book", "cost"])

    def quote(book: str, cost: str, what: str) -> str:
        if summ is None or (book, cost) not in summ.index:
            return ""
        v = summ.loc[(book, cost), what]
        return f" ({v:+.2f}%)" if what == "total_return_pct" else f"{v:.2f}%"

    fig, (a, b) = plt.subplots(2, 1, figsize=(8.8, 5.9), sharex=True, gridspec_kw={"height_ratios": [2.3, 1]})
    spec = [("old", "1x", "Old news, costs 1x", OLD, "-", 2.2), ("old", "2x", "Old news, costs 2x (double)", OLD, "--", 1.8),
            ("null_r1", "1x", "Same put on matched ordinary days", MUTED, "-", 1.2), ("null_r2", "1x", None, MUTED, "-", 1.2),
            ("all_late", "1x", "Same put on all late people-news filings", INK2, ":", 1.6)]
    for book, cost, label, color, ls, lw in spec:
        d = eq[(eq["book"] == book) & (eq["cost"] == cost)].sort_values("date")
        if d.empty:
            continue
        name = None if label is None else label + quote(book, cost, "total_return_pct")
        a.step(d["date"], d["mtm_pct"], where="post", color=color, linestyle=ls, linewidth=lw, label=name, zorder=3 if book == "old" else 2)
        if book == "old":
            dd = _drawdown(d["mtm_pct"])
            b.step(d["date"], dd, where="post", color=color, linestyle=ls, linewidth=lw, zorder=3,
                   label=f"Old news, costs {cost}: max {dd.min():.2f}%")
            if summ is not None and (book, cost) in summ.index:
                table_dd = float(summ.loc[(book, cost), "max_dd_mtm_pct"])
                if abs(table_dd - dd.min()) > 0.05:
                    warnings.warn(f"drawdown from the curve ({dd.min():.2f}%) differs from the table ({table_dd:.2f}%)")
    a.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
    b.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
    style(a, "", "", "cumulative P&L, % of collateral (marked)")
    style(b, "", "", "drawdown, %")
    a.legend(frameon=False, labelcolor=INK2, fontsize=8.5, loc="upper left")
    b.legend(frameon=False, labelcolor=INK2, fontsize=8.5, loc="lower left")
    b.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    b.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    top = _headline(fig, f"The trade: a cash-secured put on old-news filings, {horizon}-session hold",
                    subtitle or "Strike 3% below spot, 1-month options, at most 5 positions open; costs max(5% of premium, $0.05 "
                                "per share) on entry and exit.")
    fig.tight_layout(rect=(0, 0, 1, top))
    if path:
        fig.savefig(path, dpi=160, facecolor=SURFACE)
    return fig


# ---------------------------------------------------------------------------------------------------------
# All of them
# ---------------------------------------------------------------------------------------------------------
def make_all(data: Inputs, folder: Path, label: str, title_suffix: str = "",
             walk: tuple[pd.Series, pd.Series] | None = None, walk_note: str = "", show: bool = False) -> dict[str, Path]:
    """Write the figures for `label` to `folder`; return name -> path. The profile is the only source of the first
    two; the year split and the equity curve are drawn when their tables exist. Refuses the retired labels
    (discovery, dryrun): nothing is drawn from 2022-2023."""
    from oldnews import pipeline

    pipeline.check_label(label)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    h1, pl = from_profile(data.profile, "H1"), from_profile(data.profile, "P")
    tail = f" · {title_suffix}" if title_suffix else ""
    path = {n: folder / f"{n}.png" for n in ("fade_curve", "placebo", "by_year", "equity_curve", "walkthrough")}
    figs = {"fade_curve": fade_curve(h1, path["fade_curve"],
                                     "Late executive and director 8-Ks, earnings excluded, 1-month options. Prediction: old "
                                     "news below surprise news.\nBelow zero: realised volatility fell short of implied by more "
                                     "than on the same stock's ordinary days.\n"
                                     f"From results/profile.csv (tests.py); bands: bootstrap 95%{tail}"),
            "placebo": placebo(h1, pl, path["placebo"],
                               "Prediction: negative on the left, no difference on the right.\nFrom results/profile.csv "
                               f"(tests.py); bands: bootstrap 95%{tail}")}
    if data.profile_by_year is not None and len(data.profile_by_year):
        figs["by_year"] = by_year(data.profile_by_year, path["by_year"],
                                  f"Each year on its own with the same rules; bars: bootstrap 95%.\nFrom results/profile_by_year.csv{tail}")
    if data.equity is not None and len(data.equity):
        figs["equity_curve"] = equity_curve(data.equity, data.trade_summary, path["equity_curve"],
                                            "Strike 3% below spot, 1-month options, at most 5 positions open; costs max(5% of premium, "
                                            f"$0.05 per share) each way.\nFrom trade/equity.csv and summary.csv (trade.py){tail}")
    if walk is not None and len(walk[1].dropna()):
        figs["walkthrough"] = walkthrough(walk[0], walk[1], path["walkthrough"], note=walk_note)
    out = {}
    for name, fig in figs.items():
        out[name] = path[name]
        if not show:
            plt.close(fig)
    if show:
        plt.show()
    return out
