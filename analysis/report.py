"""Build the end-of-semester report: charts, a markdown summary, and JSON.

Run at the end of spring / summer / fall (see
``.github/workflows/semester_report.yml``), or by hand:

    python -m analysis.report --semester fall-2025
    python -m analysis.report --semester all
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from . import charts
from .loader import (
    FACILITIES,
    completed_semesters,
    attach_semester,
    load_readings,
    semester_bounds,
)
from .stats import (
    WEEKDAY_NAMES,
    busy_weeks,
    cell_table,
    emptiest_hours,
    find_phases,
    quiet_weeks,
    rng,
    usable_readings,
    weekly_levels,
)

FACILITY_LABELS = {
    "RSF": "RSF weight room",
    "CMS": "CMS fitness centre",
}
SEMESTER_TITLES = {"spring": "Spring", "summer": "Summer", "fall": "Fall"}


def hour_range_label(hour: int) -> str:
    def part(h: int) -> str:
        return f"{h % 12 or 12}{'AM' if h % 24 < 12 else 'PM'}"

    return f"{part(hour)}–{part((hour + 1) % 24)}"


def semester_title(slug: str) -> str:
    name, year = slug.rsplit("-", 1)
    return f"{SEMESTER_TITLES.get(name, name.title())} {year}"


def picture(stem: str, alt: str) -> str:
    return (
        "<picture>\n"
        f'  <source media="(prefers-color-scheme: dark)" srcset="{stem}-dark.png">\n'
        f'  <img alt="{alt}" src="{stem}.png">\n'
        "</picture>\n"
    )


def _profile_index(frame: pd.DataFrame) -> dict[int, float]:
    """Hour-of-day profile normalised so the average hour is 1.0."""
    by_hour = frame.groupby("hour")["pct"].agg(["count", "mean"])
    by_hour = by_hour[by_hour["count"] >= 5]
    if by_hour.empty:
        return {}
    overall = float((by_hour["mean"] * by_hour["count"]).sum() / by_hour["count"].sum())
    if overall <= 0:
        return {}
    return {int(h): float(m / overall) for h, m in by_hour["mean"].items()}


def _answer_table(results) -> list[str]:
    lines = [
        "| Day | Go at | Mean % full | 95% CI | Just as good | Samples | Confidence |",
        "| --- | --- | ---: | ---: | --- | ---: | --- |",
    ]
    for weekday in range(7):
        res = results[weekday]
        if res.best_hour is None:
            lines.append(f"| {WEEKDAY_NAMES[weekday]} | — | — | — | — | 0 | not enough data |")
            continue
        best = next(e for e in res.estimates if e.hour == res.best_hour)
        shown = [hour_range_label(h) for h in res.tied_hours[:4]]
        if len(res.tied_hours) > 4:
            shown.append(f"+{len(res.tied_hours) - 4} more")
        ties = ", ".join(shown) or "—"
        note = " ¹" if res.note else ""
        lines.append(
            f"| {WEEKDAY_NAMES[weekday]} | **{hour_range_label(best.hour)}**{note} | "
            f"{best.mean:.0f}% | {best.ci_low:.0f}–{best.ci_high:.0f}% | {ties} | "
            f"{best.n} | {res.confidence} |"
        )
    if any(r.note for r in results.values() if r.best_hour is not None):
        lines.append("")
        lines.append("¹ closing hour — the room shuts partway through, so it is genuinely quiet but briefly so.")
    return lines


def _hourly_table(results) -> list[str]:
    """The table-view twin of the profile charts — every plotted number, in text."""
    hours = sorted({e.hour for r in results.values() for e in r.estimates})
    if not hours:
        return []
    lines = [
        "| Day | " + " | ".join(charts.hour_label(h) for h in hours) + " |",
        "| --- | " + " | ".join("---:" for _ in hours) + " |",
    ]
    for weekday in range(7):
        by_hour = {e.hour: e for e in results[weekday].estimates}
        cells = []
        for hour in hours:
            est = by_hour.get(hour)
            cells.append("·" if est is None else f"{est.mean:.0f}")
        lines.append(f"| {WEEKDAY_NAMES[weekday][:3]} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Mean % full per hour. `·` = closed, or too few samples to estimate.")
    return lines


def build_facility_section(
    facility: str,
    frame: pd.DataFrame,
    slug: str,
    out_dir: Path,
    generator,
) -> tuple[list[str], dict]:
    label = FACILITY_LABELS[facility]
    title = semester_title(slug)
    lines: list[str] = [f"## {label}", ""]
    payload: dict = {"facility": facility, "readings": int(len(frame))}

    if frame.empty:
        lines += ["No usable readings for this facility this semester.", ""]
        return lines, payload

    cells = cell_table(frame)
    usable = usable_readings(frame, cells)
    results = emptiest_hours(frame, cells, generator)
    span = f"{frame['ts'].min():%b %-d} – {frame['ts'].max():%b %-d, %Y}"
    subtitle = f"{label} · {len(frame):,} readings · {span} · campus local time"

    if usable.empty:
        lines += [
            f"{len(frame):,} readings, but none land in an hour with enough samples "
            "to estimate an open-hours pattern.",
            "",
        ]
        return lines, payload

    # --- the answer -------------------------------------------------------
    lines += _answer_table(results) + [""]

    stem = f"weekday-profiles-{facility.lower()}"
    charts.render_both(
        charts.weekday_profiles(results, f"{title} · when is {label} empty?", subtitle),
        out_dir, stem,
    )
    lines += [picture(stem, f"{title} {label} hourly occupancy by weekday with 95% intervals"), ""]

    stem = f"heatmap-{facility.lower()}"
    charts.render_both(
        charts.heatmap(cells, results, f"{title} · {label} occupancy by day and hour", subtitle),
        out_dir, stem,
    )
    lines += [picture(stem, f"{title} {label} weekday-by-hour occupancy heatmap"), ""]

    lines += ["<details><summary>Table view — mean % full per hour</summary>", ""]
    lines += _hourly_table(results)
    lines += ["", "</details>", ""]

    # --- does the pattern hold all semester? ------------------------------
    levels = weekly_levels(usable)
    quiet = quiet_weeks(levels)
    busy = busy_weeks(levels)
    phases, tests = find_phases(usable, generator)

    lines += ["### Does the pattern hold all semester?", ""]
    if len(phases) == 1:
        verdict = (
            "The shape of the day is stable across the semester "
            f"(best candidate split p = {tests[0]['p_value']:.3f}), so the table above "
            "applies throughout."
            if tests
            else "Not enough data to test whether the shape of the day changes mid-semester."
        )
    else:
        verdict = (
            f"The shape of the day **does** change mid-semester — it splits into "
            f"{len(phases)} phases (p = {min(t['p_value'] for t in tests if t['significant']):.3f}). "
            "Per-phase answers are below."
        )
    lines += [verdict, ""]

    if len(levels) >= 3:
        stem = f"weekly-trend-{facility.lower()}"
        charts.render_both(
            charts.weekly_trend(
                levels, phases, quiet, busy,
                f"{title} · how busy each week ran",
                "1.0 = a typical week this semester, corrected for when the scraper happened to sample.",
            ),
            out_dir, stem,
        )
        lines += [picture(stem, f"{title} {label} weekly busyness"), ""]

    if busy:
        weeks = ", ".join(str(w + 1) for w in busy)
        peak = float(levels.loc[levels["week"].isin(busy), "level"].max())
        lines += [
            f"Weeks running well *above* the semester norm: **{weeks}** — the busiest of them at "
            f"**{peak:.2f}×** a typical week. The start-of-semester rush is the usual cause, and it "
            "fades rather than holding: the trend above is the honest picture of how much a "
            "September answer is worth in November.",
            "",
        ]
    if quiet:
        weeks = ", ".join(str(w + 1) for w in quiet)
        lines += [
            f"Weeks running well *below* it (breaks, holidays, post-finals): **{weeks}**. "
            "They pull the averages down a little but barely move *which* hour is emptiest, so they are kept in.",
            "",
        ]

    phase_payloads = []
    if len(phases) > 1:
        profiles = {}
        for phase in phases:
            sub = frame[(frame["week"] >= phase.week_start) & (frame["week"] <= phase.week_end)]
            sub_cells = cell_table(sub)
            sub_results = emptiest_hours(sub, sub_cells, generator)
            start = sub["ts"].min()
            end = sub["ts"].max()
            heading = f"{phase.label} · {start:%b %-d} – {end:%b %-d}"
            lines += [f"#### {heading}", ""] + _answer_table(sub_results) + [""]
            profiles[phase.label] = _profile_index(usable_readings(sub, sub_cells))
            phase_payloads.append(
                {
                    "label": phase.label,
                    "week_start": phase.week_start,
                    "week_end": phase.week_end,
                    "start": str(start.date()),
                    "end": str(end.date()),
                    "readings": int(len(sub)),
                    "p_value": phase.p_value,
                    "emptiest": {
                        WEEKDAY_NAMES[d]: (None if r.best_hour is None else r.best_hour)
                        for d, r in sub_results.items()
                    },
                }
            )
        profiles = {k: v for k, v in profiles.items() if v}
        if len(profiles) > 1:
            stem = f"phase-profiles-{facility.lower()}"
            charts.render_both(
                charts.phase_profiles(
                    profiles,
                    f"{title} · shape of the day, phase by phase",
                    "Each phase normalised to its own average, so only the shape is being compared — not how busy it was.",
                ),
                out_dir, stem,
            )
            lines += [picture(stem, f"{title} {label} hour-of-day shape per phase"), ""]

    payload.update(
        {
            "span": [str(frame["ts"].min().date()), str(frame["ts"].max().date())],
            "emptiest": {
                WEEKDAY_NAMES[d]: {
                    "hour": r.best_hour,
                    "mean": None if r.best_hour is None else round(
                        next(e for e in r.estimates if e.hour == r.best_hour).mean, 1),
                    "ci": None if r.best_hour is None else [
                        round(next(e for e in r.estimates if e.hour == r.best_hour).ci_low, 1),
                        round(next(e for e in r.estimates if e.hour == r.best_hour).ci_high, 1),
                    ],
                    "tied_hours": r.tied_hours,
                    "n": None if r.best_hour is None else next(
                        e for e in r.estimates if e.hour == r.best_hour).n,
                    "confidence": r.confidence,
                }
                for d, r in results.items()
            },
            "phase_tests": tests,
            "phases": phase_payloads,
            "quiet_weeks": [w + 1 for w in quiet],
            "busy_weeks": [w + 1 for w in busy],
        }
    )
    return lines, payload


def build_semester_report(frame: pd.DataFrame, slug: str, out_root: Path, load_report) -> Path:
    out_dir = out_root / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    generator = rng()
    title = semester_title(slug)
    start, end = semester_bounds(slug)

    body: list[str] = [
        f"# {title} — Berkeley gym crowd report",
        "",
        f"Semester window **{start:%b %-d, %Y} – {end:%b %-d, %Y}** · "
        f"{len(frame):,} readings · generated {datetime.now():%Y-%m-%d}.",
        "",
        "All times are campus local (America/Los_Angeles); the scraper logs UTC and this "
        "report converts. An hour labelled *7AM–8AM* covers 7:00–7:59.",
        "",
    ]

    payloads = []
    for facility in FACILITIES:
        sub = frame[frame["facility"] == facility].copy()
        lines, payload = build_facility_section(facility, sub, slug, out_dir, generator)
        body += lines
        payloads.append(payload)

    body += [
        "## Method & caveats",
        "",
        "- **Closed hours are excluded.** An hour whose median reading is ≤5% is the gym being "
        "shut, not the gym being quiet; without this, 3AM wins every day.",
        "- **Opening and closing hours** are kept but have their zero readings dropped first, "
        "and are marked ¹ — genuinely quiet, but only for part of the hour.",
        "- **Intervals are bootstrap 95% intervals** on each hour's mean (4,000 resamples, fixed seed). "
        "The *Just as good* column lists every hour whose difference from the winner has an "
        "interval straddling zero — with this few samples these are ties, not runners-up.",
        "- **The phase test** is a block permutation test: whole days are shuffled along the "
        "timeline and the best split found in the real data is compared against the best split "
        "found in 1,000 shuffled timelines. The semester is only cut where that test clears p < 0.05.",
        "- **The phase test has limited power on thin data.** A stable verdict means no "
        "*large* shape change was detectable, not that nothing shifted; a semester with a few "
        "hundred readings can only reveal shifts on the scale of the morning/evening balance flipping.",
        "- **Sampling is uneven.** The scraper runs on GitHub Actions cron, which is throttled and "
        "delayed, so some hours are sampled far more than others. Sample counts are shown; treat "
        "any *weak* confidence row as a hint rather than an answer.",
        "",
        "### Data quality this run",
        "",
        f"- Rows read: **{load_report.rows_read:,}** · kept: **{load_report.rows_kept:,}** · "
        f"unparseable: **{load_report.rows_malformed:,}** · out-of-range values dropped: "
        f"**{load_report.rows_out_of_range:,}**",
        "",
    ]
    if load_report.examples:
        body += ["<details><summary>Examples of dropped rows</summary>", ""]
        body += ["```"] + load_report.examples + ["```", "", "</details>", ""]

    (out_dir / "SUMMARY.md").write_text("\n".join(body) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "semester": slug,
                "generated": datetime.now().isoformat(timespec="seconds"),
                "window": [str(start), str(end)],
                "load": load_report.as_dict(),
                "facilities": payloads,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return out_dir


def write_index(out_root: Path) -> None:
    entries = []
    for path in sorted(out_root.glob("*/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        slug = data["semester"]
        readings = sum(f.get("readings", 0) for f in data["facilities"])
        entries.append((slug, semester_title(slug), readings, data["generated"][:10]))

    def sort_key(entry):
        slug = entry[0]
        name, year = slug.rsplit("-", 1)
        return (int(year), {"spring": 0, "summer": 1, "fall": 2}.get(name, 3))

    entries.sort(key=sort_key, reverse=True)
    lines = [
        "# Semester reports",
        "",
        "One report per completed semester, regenerated automatically at the end of spring, "
        "summer and fall (see `.github/workflows/semester_report.yml`).",
        "",
        "| Semester | Readings | Generated | Report |",
        "| --- | ---: | --- | --- |",
    ]
    for slug, title, readings, generated in entries:
        lines.append(f"| {title} | {readings:,} | {generated} | [SUMMARY.md]({slug}/SUMMARY.md) |")
    lines.append("")
    (out_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="crowd_meter_data.csv")
    parser.add_argument("--out", default="reports")
    parser.add_argument(
        "--semester",
        default="auto",
        help="'auto' (the most recently completed semester), 'all', or a slug like fall-2025",
    )
    parser.add_argument("--today", default=None, help="override today's date (YYYY-MM-DD), for testing")
    args = parser.parse_args(argv)

    today = date.fromisoformat(args.today) if args.today else date.today()
    frame, load_report = load_readings(args.data)
    frame = attach_semester(frame)
    print(
        f"read {load_report.rows_read:,} rows · kept {load_report.rows_kept:,} · "
        f"dropped {load_report.rows_malformed:,} unparseable, {load_report.rows_out_of_range:,} out of range"
    )

    done = completed_semesters(frame, today)
    if args.semester == "auto":
        if not done:
            print("no completed semester in the data yet — nothing to report")
            return 0
        targets = [done[-1]]
    elif args.semester == "all":
        targets = done
    else:
        targets = [args.semester]

    out_root = Path(args.out)
    for slug in targets:
        sub = frame[frame["semester"] == slug].copy()
        if sub.empty:
            print(f"{slug}: no readings, skipped")
            continue
        out_dir = build_semester_report(sub, slug, out_root, load_report)
        print(f"{slug}: wrote {out_dir}/SUMMARY.md ({len(sub):,} readings)")

    if out_root.exists():
        write_index(out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
