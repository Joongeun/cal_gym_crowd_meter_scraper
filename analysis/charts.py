"""Matplotlib figures for the semester report.

Colours come from a validated design-system palette: one blue hue light->dark
for magnitude (the heatmap), the fixed categorical slot order for identity (the
phase overlay).  Every figure is rendered twice, light and dark, so the report
reads correctly under either GitHub theme via a <picture> element.

Every value plotted here also appears in a markdown table in SUMMARY.md, so no
number is reachable only through colour.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .stats import WEEKDAY_NAMES, HourEstimate, WeekdayResult

# Sequential blue ramp, steps 100 -> 700.
BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    text_primary: str
    text_secondary: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, ...]
    ramp: tuple[str, ...]


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a"),
    ramp=tuple(BLUE_RAMP),
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series=("#3987e5", "#d95926", "#199e70"),
    # On a dark surface "near zero" has to recede toward the surface, so the
    # same one-hue ramp runs dark -> light instead of light -> dark.
    ramp=tuple(reversed(BLUE_RAMP)),
)

THEMES = (LIGHT, DARK)


def hour_label(hour: int) -> str:
    suffix = "a" if hour < 12 else "p"
    display = hour % 12 or 12
    return f"{display}{suffix}"


def _apply(theme: Theme) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": theme.surface,
            "axes.facecolor": theme.surface,
            "savefig.facecolor": theme.surface,
            "text.color": theme.text_primary,
            "axes.labelcolor": theme.text_secondary,
            "axes.edgecolor": theme.axis,
            "xtick.color": theme.muted,
            "ytick.color": theme.muted,
            "grid.color": theme.grid,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "figure.dpi": 130,
        }
    )


def _cmap(theme: Theme) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(f"gym-{theme.name}", list(theme.ramp), N=256)


def _ink_on(rgba) -> str:
    """Black or white, whichever the fill underneath can actually carry."""
    red, green, blue = rgba[:3]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#0b0b0b" if luminance > 0.55 else "#ffffff"


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def render_both(builder, out_dir: Path, stem: str) -> dict[str, str]:
    """Run ``builder(theme)`` once per theme and write ``stem.png`` / ``stem-dark.png``."""
    paths = {}
    for theme in THEMES:
        _apply(theme)
        fig = builder(theme)
        name = f"{stem}.png" if theme.name == "light" else f"{stem}-dark.png"
        _save(fig, out_dir / name)
        paths[theme.name] = name
    return paths


# ---------------------------------------------------------------------------
# 1. Weekday x hour heatmap
# ---------------------------------------------------------------------------
def heatmap(cells, results: dict[int, WeekdayResult], title: str, subtitle: str):
    active = cells[cells["status"].isin(("open", "partial", "sparse"))]
    if active.empty:
        hours = list(range(6, 23))
    else:
        hours = list(range(int(active["hour"].min()), int(active["hour"].max()) + 1))
    lookup = {(int(r.weekday), int(r.hour)): r for r in cells.itertuples()}
    vmax = max(100.0, float(active["mean"].max()) if not active.empty else 100.0)

    def build(theme: Theme):
        fig, ax = plt.subplots(figsize=(1 + 0.62 * len(hours), 4.6))
        cmap = _cmap(theme)
        grid = np.full((7, len(hours)), np.nan)
        for row in range(7):
            for col, hour in enumerate(hours):
                cell = lookup.get((row, hour))
                if cell is not None and cell.status in ("open", "partial"):
                    grid[row, col] = cell.mean

        mesh = ax.imshow(
            grid, cmap=cmap, vmin=0, vmax=vmax, aspect="auto",
            extent=(0, len(hours), 7, 0), interpolation="nearest",
        )

        for row in range(7):
            for col, hour in enumerate(hours):
                cell = lookup.get((row, hour))
                status = cell.status if cell is not None else "closed"
                if status in ("closed", "sparse") or cell is None:
                    # A 2px surface gap keeps unfilled cells reading as absent
                    # rather than as a fill of some colour.
                    ax.add_patch(
                        plt.Rectangle(
                            (col + 0.02, row + 0.02), 0.96, 0.96,
                            facecolor=theme.surface,
                            # In matplotlib the hatch inherits the edge colour,
                            # so this is what keeps it recessive rather than black.
                            edgecolor=theme.grid if status == "sparse" else "none",
                            hatch="///" if status == "sparse" else None,
                            linewidth=0.8 if status == "sparse" else 0,
                        )
                    )

        # Direct-label only the emptiest open hour on each day.
        for row in range(7):
            res = results.get(row)
            if res is None or res.best_hour is None or res.best_hour not in hours:
                continue
            col = hours.index(res.best_hour)
            ax.add_patch(
                plt.Rectangle(
                    (col + 0.06, row + 0.06), 0.88, 0.88,
                    facecolor="none", edgecolor=theme.surface, linewidth=2.6,
                )
            )
            ax.add_patch(
                plt.Rectangle(
                    (col + 0.06, row + 0.06), 0.88, 0.88,
                    facecolor="none", edgecolor=theme.text_primary, linewidth=1.4,
                )
            )
            best = next(e for e in res.estimates if e.hour == res.best_hour)
            ax.text(
                col + 0.5, row + 0.5, f"{best.mean:.0f}",
                ha="center", va="center", fontsize=8.5, fontweight="bold",
                color=_ink_on(cmap(best.mean / vmax)),
            )

        ax.set_xticks(np.arange(len(hours)) + 0.5)
        ax.set_xticklabels([hour_label(h) for h in hours], fontsize=8.5)
        ax.set_yticks(np.arange(7) + 0.5)
        ax.set_yticklabels([n[:3] for n in WEEKDAY_NAMES], fontsize=9)
        ax.set_xlabel("hour of day (campus local time)", color=theme.text_secondary, labelpad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

        bar = fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02)
        bar.set_label("mean % full", color=theme.text_secondary, fontsize=9)
        bar.outline.set_visible(False)
        bar.ax.tick_params(length=0, colors=theme.muted, labelsize=8.5)

        ax.set_title(title, loc="left", fontsize=13, fontweight="bold",
                     color=theme.text_primary, pad=30)
        ax.text(0, 1.045, subtitle, transform=ax.transAxes, fontsize=9,
                color=theme.text_secondary, va="bottom")
        ax.text(
            0, -0.24,
            "Blank = closed.  Hatched = too few samples.  Outlined = emptiest open hour that day.",
            transform=ax.transAxes, fontsize=8.5, color=theme.muted, va="top",
        )
        return fig

    return build


# ---------------------------------------------------------------------------
# 2. Per-weekday hourly profile with bootstrap intervals
# ---------------------------------------------------------------------------
def weekday_profiles(results: dict[int, WeekdayResult], title: str, subtitle: str):
    def build(theme: Theme):
        fig, axes = plt.subplots(2, 4, figsize=(15.5, 6.6), sharey=True)
        flat = axes.ravel()
        colour = theme.series[0]

        for row in range(7):
            ax = flat[row]
            res = results[row]
            ax.set_title(WEEKDAY_NAMES[row], loc="left", fontsize=11, fontweight="bold",
                         color=theme.text_primary, pad=6)
            ax.set_xlim(5.5, 23.5)
            ax.set_ylim(0, 110)
            ax.set_xticks(range(6, 24, 3))
            ax.set_xticklabels([hour_label(h) for h in range(6, 24, 3)], fontsize=8.5)
            ax.yaxis.grid(True, linewidth=0.8, color=theme.grid, linestyle="-")
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(length=0, labelsize=8.5)

            if not res.estimates:
                ax.text(0.5, 0.5, "not enough data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color=theme.muted)
                continue

            hours = np.array([e.hour for e in res.estimates])
            means = np.array([e.mean for e in res.estimates])
            lows = np.array([e.ci_low for e in res.estimates])
            highs = np.array([e.ci_high for e in res.estimates])

            # Break the line wherever an hour is missing, so a gap never reads
            # as a measured decline.
            for seg in _contiguous(hours):
                idx = np.isin(hours, seg)
                ax.fill_between(hours[idx], lows[idx], highs[idx], color=colour, alpha=0.16, linewidth=0)
                ax.plot(hours[idx], means[idx], color=colour, linewidth=2.0, solid_capstyle="round")

            tied = [e for e in res.estimates if e.tied_with_best and e.hour != res.best_hour]
            if tied:
                ax.scatter([e.hour for e in tied], [e.mean for e in tied], s=64,
                           facecolor=theme.surface, edgecolor=colour, linewidth=1.8, zorder=4)

            best = next(e for e in res.estimates if e.hour == res.best_hour)
            ax.scatter([best.hour], [best.mean], s=110, facecolor=colour,
                       edgecolor=theme.surface, linewidth=2.0, zorder=5)
            # Keep the label inside the panel instead of letting it run off
            # the left or right edge on early-morning / late-night winners.
            align, dx = "center", 0
            if best.hour <= 8:
                align, dx = "left", -6
            elif best.hour >= 21:
                align, dx = "right", 6
            ax.annotate(
                f"{hour_label(best.hour)} · {best.mean:.0f}%",
                (best.hour, best.mean), textcoords="offset points", xytext=(dx, -22),
                ha=align, fontsize=9.5, fontweight="bold", color=theme.text_primary,
            )

        legend_ax = flat[7]
        legend_ax.axis("off")
        # transAxes matters here: sharey=True gives this panel the 0-110 y-limits
        # of the real plots, so bare coordinates would place both blocks on top
        # of each other near the bottom.
        legend_ax.text(0, 1.0, "How to read", transform=legend_ax.transAxes,
                       fontsize=11, fontweight="bold", color=theme.text_primary, va="top")
        legend_ax.text(
            0, 0.88,
            "Line: mean % full by hour.\n"
            "Band: bootstrap 95% interval —\n"
            "how much that mean could move,\n"
            "given how few samples an hour has.\n\n"
            "Filled dot: the emptiest hour.\n"
            "Open dots: hours the data cannot\n"
            "tell apart from it — equally good\n"
            "choices, not runners-up.",
            transform=legend_ax.transAxes,
            fontsize=9, color=theme.text_secondary, va="top", linespacing=1.5,
        )

        flat[0].set_ylabel("% full", color=theme.text_secondary)
        flat[4].set_ylabel("% full", color=theme.text_secondary)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        fig.suptitle(title, x=0.006, y=0.995, ha="left", va="top", fontsize=14,
                     fontweight="bold", color=theme.text_primary)
        fig.text(0.006, 0.945, subtitle, ha="left", va="top", fontsize=9.5,
                 color=theme.text_secondary)
        return fig

    return build


def _contiguous(hours: np.ndarray) -> list[np.ndarray]:
    if len(hours) == 0:
        return []
    breaks = np.where(np.diff(hours) > 1)[0] + 1
    return [seg for seg in np.split(hours, breaks) if len(seg) > 0]


# ---------------------------------------------------------------------------
# 3. Weekly level across the semester
# ---------------------------------------------------------------------------
def weekly_trend(levels, phases, quiet, busy, title: str, subtitle: str):
    def build(theme: Theme):
        fig, ax = plt.subplots(figsize=(11, 3.9))
        colour = theme.series[0]
        weeks = levels["week"].to_numpy() + 1
        values = levels["level"].to_numpy()

        ax.axhline(1.0, color=theme.axis, linewidth=1.0)
        ax.plot(weeks, values, color=colour, linewidth=2.0, solid_capstyle="round", zorder=3)
        ax.scatter(weeks, values, s=42, facecolor=colour, edgecolor=theme.surface,
                   linewidth=2.0, zorder=4)

        # Dips get their label above the marker and surges below it, so neither
        # lands on the axis ticks.
        marked = {w + 1: ("quiet week", 16) for w in quiet}
        marked.update({w + 1: ("busy week", -20) for w in busy})
        for week, value in zip(weeks, values):
            if week not in marked:
                continue
            caption, offset = marked[week]
            ax.scatter([week], [value], s=90, facecolor=theme.surface,
                       edgecolor=theme.muted, linewidth=1.8, zorder=5)
            align = "center"
            if week == weeks[0]:
                align = "left"
            elif week == weeks[-1]:
                align = "right"
            ax.annotate(
                caption, (week, value), textcoords="offset points",
                xytext=(0, offset), ha=align, fontsize=8.5, color=theme.muted,
                # Surface halo so the caption stays readable where the line runs
                # underneath it.
                bbox=dict(facecolor=theme.surface, edgecolor="none", pad=1.5),
                zorder=6,
            )

        for phase in phases[1:]:
            boundary = phase.week_start + 0.5
            ax.axvline(boundary, color=theme.axis, linewidth=1.0, zorder=1)
            ax.text(boundary + 0.15, ax.get_ylim()[1], phase.label.split(" (")[0],
                    fontsize=8.5, color=theme.muted, va="top")

        ax.margins(y=0.18)
        ax.set_xlabel("week of semester", color=theme.text_secondary, labelpad=6)
        ax.set_ylabel("busyness vs. typical week", color=theme.text_secondary)
        ax.set_xticks(weeks[:: max(1, len(weeks) // 18)])
        ax.yaxis.grid(True, linewidth=0.8, color=theme.grid, linestyle="-")
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0, labelsize=8.5)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold",
                     color=theme.text_primary, pad=34)
        ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=9,
                color=theme.text_secondary, va="bottom")
        fig.tight_layout()
        return fig

    return build


# ---------------------------------------------------------------------------
# 4. Phase comparison (only drawn when the semester actually splits)
# ---------------------------------------------------------------------------
def phase_profiles(profiles: dict[str, dict[int, float]], title: str, subtitle: str):
    def build(theme: Theme):
        fig, ax = plt.subplots(figsize=(9.5, 4.2))
        for i, (label, profile) in enumerate(profiles.items()):
            colour = theme.series[i % len(theme.series)]
            hours = np.array(sorted(profile))
            values = np.array([profile[h] for h in hours])
            for j, seg in enumerate(_contiguous(hours)):
                idx = np.isin(hours, seg)
                ax.plot(hours[idx], values[idx], color=colour, linewidth=2.0,
                        solid_capstyle="round", label=label if j == 0 else None)
            ax.annotate(label.split(" (")[0], (hours[-1], values[-1]),
                        textcoords="offset points", xytext=(8, 0), fontsize=9.5,
                        fontweight="bold", color=colour, va="center")

        ax.axhline(1.0, color=theme.axis, linewidth=1.0)
        ax.set_xticks(range(6, 24, 2))
        ax.set_xticklabels([hour_label(h) for h in range(6, 24, 2)], fontsize=8.5)
        ax.set_xlabel("hour of day (campus local time)", color=theme.text_secondary, labelpad=6)
        ax.set_ylabel("relative to that phase's average", color=theme.text_secondary)
        ax.yaxis.grid(True, linewidth=0.8, color=theme.grid, linestyle="-")
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0, labelsize=8.5)
        legend = ax.legend(frameon=False, fontsize=9, loc="upper left")
        for text in legend.get_texts():
            text.set_color(theme.text_secondary)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold",
                     color=theme.text_primary, pad=34)
        ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=9,
                color=theme.text_secondary, va="bottom")
        fig.tight_layout()
        return fig

    return build
