"""Figures for the old-news test, in the notebook's own style (section 7 palette and axes).

    1  fade_curve     mean outcome by horizon, old news versus surprise news, 95% bootstrap intervals
    2  walkthrough    one event: event date, gap, filing, entry, and the option-implied versus realised move
    3  placebo        old minus surprise for the people-news filings, next to the same on the placebo filings

Outcome (test plan): Y_h = log(RV_h / IV_0), each event minus the mean of its matched ordinary days. Below
zero means realised volatility fell short of implied by more than on that stock's ordinary days.

The samples are the stats module's own (src/oldnews/tests.py, `universe` and `cell`): late filings, no earnings
filing nearby, a usable score, a usable outcome in the 1-month bucket on the 3% put's row, and at least one
usable matched ordinary day. The placebo figure plots the stats module's horizon profile (H1 and P) when it is
given, so every plotted difference, interval and p-value is a reported number; the fade curve's group means
are checked against the same table.

The walkthrough event is chosen without looking at outcomes: the old-news event whose total score S is
closest to the median S of old-news events (earliest entry on ties).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

SEED = 20261003
N_BOOT = 10_000
HORIZONS = ["1", "2", "3", "5", "10", "21", "42", "63", "expiry"]
PRIMARY_H, PRIMARY_BUCKET, PRIMARY_OTM = "10", "1m", 3

# The notebook's tokens (section 7), so these figures sit beside its own.
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
OLD, SURPRISE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2 (validated: CVD dE 24.7, contrast >= 3:1)
GROUP_LABEL = {True: "Old news", False: "Surprise news"}


@dataclass
class Inputs:
    """One run's tables (00_shared.md columns). `labels` is classify's table (every event row with its inputs,
    M, T, S, `old` and `scored`); `profile` is tests.run()["profile"] when the stats step has run."""
    events: pd.DataFrame
    nulls: pd.DataFrame
    outcome: pd.DataFrame
    labels: pd.DataFrame
    gap: pd.DataFrame | None = None
    profile: pd.DataFrame | None = None


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
# Data
# ---------------------------------------------------------------------------------------------------------
def event_table(data: Inputs, group: str = "people") -> pd.DataFrame:
    """The test's filings for one group (people | placebo): late, no earnings filing nearby, usable score."""
    c = data.labels
    if "group" not in c:                                  # a bare label table: bring in the event columns
        c = data.events.merge(c, on="row_id", how="inner", suffixes=("", "_label"))
    scored = flag(c["scored"]) if "scored" in c else c["old"].notna()
    ev = c[(c["group"] == group) & flag(c["late"]) & ~flag(c["earnings_excluded"]) & scored].copy()
    ev["old"] = flag(ev["old"])
    return ev.reset_index(drop=True)


def outcome_y(outcome: pd.DataFrame, bucket: str = PRIMARY_BUCKET, otm: int = PRIMARY_OTM) -> pd.DataFrame:
    """One usable y per (row_id, horizon), from the bucket's rows for the primary put (usable also needs the
    put to have traded at entry, so the 3% row is the test's own cell)."""
    o = outcome[(outcome["bucket"].astype(str) == bucket) & (pd.to_numeric(outcome["otm"], errors="coerce") == otm)
                & flag(outcome["usable"])].copy()
    o["horizon"] = o["horizon"].map(norm_h)
    o["y"] = pd.to_numeric(o["y"], errors="coerce")
    o = o[np.isfinite(o["y"])].drop_duplicates(["row_id", "horizon"])
    return o.reindex(columns=["row_id", "horizon", "y", "iv0", "rv", "entry_date", "exit_date"])


def matched(data: Inputs, group: str = "people", bucket: str = PRIMARY_BUCKET) -> pd.DataFrame:
    """Per event and horizon: d = event y minus the mean y of its usable matched ordinary days."""
    ev = event_table(data, group)
    y = outcome_y(data.outcome, bucket)
    ey = ev[["row_id", "old"]].merge(y[["row_id", "horizon", "y"]], on="row_id")
    ny = (data.nulls[["row_id", "event_row_id"]].merge(y[["row_id", "horizon", "y"]], on="row_id")
          .groupby(["event_row_id", "horizon"], as_index=False)["y"].mean().rename(columns={"y": "y_null"}))
    m = ey.merge(ny, left_on=["row_id", "horizon"], right_on=["event_row_id", "horizon"], how="inner")
    m["d"] = m["y"] - m["y_null"]
    return m.dropna(subset=["d"])[["row_id", "old", "horizon", "y", "y_null", "d"]].reset_index(drop=True)


def _boot_means(x: np.ndarray, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    return x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)


def profile(m: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """Mean d by group and horizon with a 95% bootstrap interval of the mean."""
    rng, rows = np.random.default_rng(seed), []
    for old in (True, False):
        for h in HORIZONS:
            x = m.loc[(m["old"] == old) & (m["horizon"] == h), "d"].to_numpy(float)
            lo = hi = np.nan
            if len(x) >= 2:
                lo, hi = np.percentile(_boot_means(x, rng, n_boot), [2.5, 97.5])
            rows.append({"old": old, "horizon": h, "n": len(x), "mean": x.mean() if len(x) else np.nan,
                         "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def difference(m: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """Old minus surprise by horizon, each group resampled on its own, with a 95% bootstrap interval
    (used only when the stats module's profile is not available)."""
    rng, rows = np.random.default_rng(seed), []
    for h in HORIZONS:
        a = m.loc[m["old"] & (m["horizon"] == h), "d"].to_numpy(float)
        b = m.loc[~m["old"] & (m["horizon"] == h), "d"].to_numpy(float)
        row = {"horizon": h, "n_old": len(a), "n_surprise": len(b), "diff": np.nan, "ci_lo": np.nan,
               "ci_hi": np.nan, "p": np.nan}
        if len(a) >= 2 and len(b) >= 2:
            boot = _boot_means(a, rng, n_boot) - _boot_means(b, rng, n_boot)
            row.update(diff=a.mean() - b.mean(), ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5))
        rows.append(row)
    return pd.DataFrame(rows)


def from_profile(prof: pd.DataFrame, test: str) -> pd.DataFrame:
    """The stats module's horizon profile for one test (H1 or P) in this module's difference format."""
    t = prof[prof["test"] == test].copy()
    t["horizon"] = t["horizon"].map(norm_h)
    t = t.rename(columns={"effect": "diff", "n_comp": "n_surprise"})      # p: one-sided for H1, two-sided for P
    return t.reindex(columns=["horizon", "n_old", "n_surprise", "diff", "ci_lo", "ci_hi", "p", "q_bh"])


def check_against(prof_groups: pd.DataFrame, prof: pd.DataFrame) -> None:
    """Warn if the fade curve's group means differ from the stats module's H1 means (same sample, same d)."""
    t = prof[prof["test"] == "H1"].assign(horizon=lambda d: d["horizon"].map(norm_h)).set_index("horizon")
    for old, col in ((True, "mean_old"), (False, "mean_comp")):
        mine = prof_groups[prof_groups["old"] == old].set_index("horizon")["mean"]
        both = pd.concat([mine, t[col]], axis=1, keys=["mine", "stats"]).dropna()
        if len(both) and not np.allclose(both["mine"], both["stats"], atol=1e-9):
            warnings.warn(f"fade curve {GROUP_LABEL[old]} means differ from tests.py's H1 {col}; check the samples")


def pick_walkthrough(data: Inputs) -> pd.Series | None:
    """The old-news primary event whose S is closest to the median S of old-news events (no outcome used,
    except that its 1-month, 10-session outcome must exist so the figure is complete)."""
    ev = event_table(data, "people")
    ev = ev[ev["old"]]
    y = outcome_y(data.outcome)
    y10 = y[y["horizon"] == PRIMARY_H].drop(columns="horizon")
    ev = ev.drop(columns=[c for c in y10.columns if c != "row_id" and c in ev]).merge(y10, on="row_id", how="inner")
    if ev.empty:
        return None
    if "S" in ev and ev["S"].notna().any():
        ev = ev.assign(_dist=(pd.to_numeric(ev["S"]) - pd.to_numeric(ev["S"]).median()).abs())
        ev = ev.sort_values(["_dist", "t_0"])
    else:
        ev = ev.sort_values("t_0")
    return ev.iloc[0].drop(labels=["_dist"], errors="ignore")


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
    if title:
        ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def _headline(fig, title: str, subtitle: str) -> float:
    """Title and subtitle at the top left; returns the figure fraction left for the axes below them."""
    h = fig.get_figheight()
    fig.suptitle(title, x=0.01, y=1 - 0.08 / h, ha="left", va="top", color=INK, fontsize=12.5, fontweight="bold")
    fig.text(0.01, 1 - 0.36 / h, subtitle, ha="left", va="top", color=INK2, fontsize=9, linespacing=1.5)
    return 1 - (0.36 + 0.21 * (subtitle.count("\n") + 1) - 0.25) / h     # tight_layout adds its own padding


LABEL_BOX = dict(boxstyle="square,pad=0.15", facecolor=SURFACE, edgecolor="none")


def _horizon_axis(ax, x: np.ndarray, primary: bool = True) -> None:
    ax.set_xticks(x, [h if h != "expiry" else "exp" for h in HORIZONS])
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
    if primary:
        i = HORIZONS.index(PRIMARY_H)
        ax.axvspan(i - 0.35, i + 0.35, color=GRID, alpha=0.55, linewidth=0, zorder=0)
        ax.annotate("primary test", (i, 1), xycoords=("data", "axes fraction"), xytext=(0, -2),
                    textcoords="offset points", ha="center", va="top", fontsize=8, color=INK2)


def _line(ax, x, mean, lo, hi, color: str, label: str) -> None:
    ax.fill_between(x, lo, hi, color=color, alpha=0.10, linewidth=0, zorder=2)
    ax.plot(x, mean, color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=1.5, solid_capstyle="round", solid_joinstyle="round", label=label, zorder=3)


# ---------------------------------------------------------------------------------------------------------
# Figure 1 · the fade curve
# ---------------------------------------------------------------------------------------------------------
def fade_curve(prof: pd.DataFrame, path: Path | None = None, subtitle: str = "") -> plt.Figure:
    x = np.arange(len(HORIZONS))
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ends = {}
    for old, color in ((True, OLD), (False, SURPRISE)):
        t = prof[prof["old"] == old].set_index("horizon").reindex(HORIZONS)
        n = int(t["n"].max()) if t["n"].notna().any() else 0
        _line(ax, x, t["mean"].to_numpy(float), t["ci_lo"].to_numpy(float), t["ci_hi"].to_numpy(float),
              color, f"{GROUP_LABEL[old]} (n = {n})")
        last = t["mean"].dropna()
        if len(last):
            ends[old] = (HORIZONS.index(last.index[-1]), float(last.iloc[-1]))
    _horizon_axis(ax, x)
    style(ax, xlabel="sessions after entry (exp = option expiry)",
          ylabel="log(realised / implied vol), event minus ordinary days")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="lower left")
    lo, hi = ax.get_ylim()
    if len(ends) == 2 and abs(ends[True][1] - ends[False][1]) > 0.06 * (hi - lo):   # direct labels only if they clear
        for old, (i, v) in ends.items():
            ax.annotate(GROUP_LABEL[old], (i, v), xytext=(8, 0), textcoords="offset points", va="center",
                        fontsize=9, color=INK2)
        ax.set_xlim(-0.4, len(HORIZONS) - 0.1 + 0.9)
    ax.annotate("below zero: realised volatility fell short of implied\nby more than on the same stock's ordinary days",
                (0.99, 0.02), xycoords="axes fraction", ha="right", va="bottom", fontsize=8, color=MUTED)
    top = _headline(fig, "The fade curve: does old news over-price the move?",
                    subtitle or "Late executive and director 8-Ks, 1-month options.\n"
                                f"Bands: 95% bootstrap intervals ({N_BOOT:,} resamples).")
    fig.tight_layout(rect=(0, 0, 1, top))
    if path:
        fig.savefig(path, dpi=160, facecolor=SURFACE)
    return fig


# ---------------------------------------------------------------------------------------------------------
# Figure 2 · one event, end to end
# ---------------------------------------------------------------------------------------------------------
def _pct(v) -> str:
    return "n/a" if pd.isna(v) else f"{v:.0%}"


def walkthrough(row: pd.Series, spot: pd.Series, path: Path | None = None, horizon: int = int(PRIMARY_H)) -> plt.Figure:
    """Price path through the gap and the filing, then the option-implied +/-1 sigma cone against what happened."""
    spot = spot.dropna()
    t0, t_pre, g0 = pd.Timestamp(row["t_0"]), pd.Timestamp(row["t_pre"]), pd.Timestamp(row["gap_start"])
    fig, ax = plt.subplots(figsize=(11, 5.6))
    style(ax, xlabel="", ylabel="stock price from put-call parity ($)")

    ax.axvspan(g0, t_pre, color=GRID, alpha=0.6, linewidth=0, zorder=0)
    n_gap = int(row["n_gap"])
    ax.annotate(f"gap: {n_gap} session{'s' if n_gap != 1 else ''}, from the last close before the event to the last close "
                "before the filing", (g0, 0.02), xycoords=("data", "axes fraction"), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color=INK2, va="bottom", bbox=LABEL_BOX, zorder=5)

    # the option-implied 1-sigma cone from entry, sized by the 1-month ATM implied vol at entry
    y_lo, y_hi = float(spot.min()), float(spot.max())
    if t0 in spot.index and np.isfinite(row.get("iv0", np.nan)):
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
    exit_i = spot.index.searchsorted(t0) + horizon
    marks = [(t0, "entry")] + ([(spot.index[exit_i], f"exit, {horizon} sessions")] if exit_i < len(spot) else [])
    for d, txt in marks:
        if d in spot.index:
            ax.plot([d], [spot.loc[d]], marker="o", markersize=8, color=INK, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, zorder=4)
            if txt != "entry":
                ax.annotate(txt, (d, spot.loc[d]), xytext=(0, -14), textcoords="offset points", ha="center",
                            fontsize=8, color=INK2, bbox=LABEL_BOX, zorder=5)

    # the three dates, labels staggered so close dates never collide
    accepted = pd.Timestamp(row["accepted_at"]) if pd.notna(row.get("accepted_at")) else None
    events = [(pd.Timestamp(row["event_date"]), f"event (cover page) {pd.Timestamp(row['event_date']):%d %b}")]
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

    S, T = row.get("S", np.nan), row.get("T", np.nan)
    why = (f"gap move {row.get('gap_move', np.nan):.1f}σ · implied vol {100 * row.get('d_iv_gap', np.nan):+.1f} pts in the gap · "
           f"option volume ×{row.get('vol_gap_ratio', np.nan):.1f} · "
           f"word cues {int(T) if pd.notna(T) else 'n/a'} of 3"
           + (f" → old news (S = {S:.2f})" if pd.notna(S) else " → old news"))
    what = (f"next {horizon} sessions: realised {_pct(row.get('rv'))} vs implied {_pct(row.get('iv0'))} → "
            f"Y = log(RV/IV₀) = {row.get('y', np.nan):+.2f}")
    role = str(row.get("role", "") or "people-news")
    top = _headline(fig, f"One filing, end to end: {row['ticker']}, {role} 8-K", f"{why}\n{what}")
    fig.tight_layout(rect=(0, 0, 1, top))
    if path:
        fig.savefig(path, dpi=160, facecolor=SURFACE)
    return fig


# ---------------------------------------------------------------------------------------------------------
# Figure 3 · the placebo beside the main result
# ---------------------------------------------------------------------------------------------------------
def placebo(diff_people: pd.DataFrame, diff_placebo: pd.DataFrame, path: Path | None = None,
            subtitle: str = "") -> plt.Figure:
    x = np.arange(len(HORIZONS))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    panels = ((axes[0], diff_people, "People-news 8-Ks (test H1)", "one-sided"),
              (axes[1], diff_placebo, "Placebo: late scheduled 8-Ks (test P)", "two-sided"))
    for ax, d, title, sided in panels:
        t = d.set_index("horizon").reindex(HORIZONS)
        _line(ax, x, t["diff"].to_numpy(float), t["ci_lo"].to_numpy(float), t["ci_hi"].to_numpy(float), INK2, "")
        _horizon_axis(ax, x)
        n_old, n_sur = t.loc[PRIMARY_H, ["n_old", "n_surprise"]].fillna(0).astype(int)
        p = t.loc[PRIMARY_H, "p"] if "p" in t else np.nan
        style(ax, f"{title}", "sessions after entry (exp = option expiry)",
              "old minus surprise, log(RV / IV₀)" if ax is axes[0] else "")
        note = f"at h = {PRIMARY_H}: {n_old} old, {n_sur} surprise"
        if pd.notna(p):
            note += f" · {sided} permutation p = {p:.3f}"
        ax.annotate(note, (0.01, 0.02), xycoords="axes fraction", fontsize=8, color=INK2)
    top = _headline(fig, "The placebo: the same split on filings that carry no people news",
                    subtitle or "Prediction: negative on the left, about zero on the right.\n"
                                f"Bands: 95% bootstrap intervals ({N_BOOT:,} resamples).")
    fig.tight_layout(rect=(0, 0, 1, top))
    if path:
        fig.savefig(path, dpi=160, facecolor=SURFACE)
    return fig


# ---------------------------------------------------------------------------------------------------------
# All three
# ---------------------------------------------------------------------------------------------------------
def make_all(data: Inputs, folder: Path, title_suffix: str = "", walk: tuple[pd.Series, pd.Series] | None = None,
             show: bool = False) -> dict[str, Path]:
    """Write the three figures (and the tables behind them, as CSV) to `folder`; return name -> path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    m_people, m_placebo = matched(data, "people"), matched(data, "placebo")
    prof = profile(m_people)
    if data.profile is not None and len(data.profile):
        check_against(prof, data.profile)
        d_people, d_placebo = from_profile(data.profile, "H1"), from_profile(data.profile, "P")
    else:
        d_people, d_placebo = difference(m_people), difference(m_placebo)
    prof.to_csv(folder / "fade_curve.csv", index=False)
    pd.concat([d_people.assign(group="people"), d_placebo.assign(group="placebo")]).to_csv(
        folder / "placebo.csv", index=False)
    bands = f"Bands: 95% bootstrap intervals ({N_BOOT:,} resamples) · {title_suffix}"
    figs = {"fade_curve": fade_curve(prof, folder / "fade_curve.png",
                                     "Late executive and director 8-Ks, earnings excluded, 1-month options. "
                                     f"Prediction: old news below surprise news.\n{bands}"),
            "placebo": placebo(d_people, d_placebo, folder / "placebo.png",
                               f"Prediction: negative on the left, about zero on the right.\n{bands}")}
    if walk is not None and len(walk[1].dropna()):
        figs["walkthrough"] = walkthrough(walk[0], walk[1], folder / "walkthrough.png")
    for name, fig in figs.items():
        out[name] = folder / f"{name}.png"
        if not show:
            plt.close(fig)
    if show:
        plt.show()
    return out
