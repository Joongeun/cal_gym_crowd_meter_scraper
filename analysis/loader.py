"""Read and clean crowd_meter_data.csv.

The scraper writes a naive UTC timestamp plus two "percent full" strings that
come straight off the RecWell page.  Both value columns are noisy:

  * ``">7%"``   -- a stray character from the page markup got captured with the
                   number.  The digits are still correct.
  * ``'"%=\n"'`` -- an MHTML quoted-printable soft line break landed inside the
                   captured slice.  No digits survive; the row is unusable.
  * ``"104%"``  -- genuinely over capacity.  Real, kept as-is.

Everything here is deliberately forgiving: pull the digits out when they exist,
drop the row when they don't, and report how many were dropped so the summary
can say so out loud.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

UTC = ZoneInfo("UTC")
CAMPUS_TZ = ZoneInfo("America/Los_Angeles")

FACILITIES = ("RSF", "CMS")

_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
_PCT_RE = re.compile(r"(\d{1,3})\s*%")

# A reading above this is treated as a scrape artefact rather than a real
# occupancy.  The meters do legitimately report over 100% when a room is past
# its comfortable capacity; 150% is well beyond anything observed.
MAX_PLAUSIBLE_PCT = 150


@dataclass
class LoadReport:
    """What happened while reading the CSV -- surfaced in the written summary."""

    rows_read: int = 0
    rows_kept: int = 0
    rows_malformed: int = 0
    rows_duplicate: int = 0
    rows_out_of_range: int = 0
    examples: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "rows_read": self.rows_read,
            "rows_kept": self.rows_kept,
            "rows_malformed": self.rows_malformed,
            "rows_duplicate": self.rows_duplicate,
            "rows_out_of_range": self.rows_out_of_range,
        }


def parse_percent(raw: str) -> int | None:
    """Pull an integer percentage out of a scraped cell, or None if there isn't one."""
    if raw is None:
        return None
    match = _PCT_RE.search(raw.strip())
    if not match:
        return None
    return int(match.group(1))


def load_readings(csv_path: str | Path) -> tuple[pd.DataFrame, LoadReport]:
    """Return a long-format frame of (timestamp, facility, pct) plus a load report.

    Timestamps are converted from the scraper's UTC clock to campus local time,
    because "the emptiest hour" is a question about when a person walks in the
    door, not about UTC.
    """
    report = LoadReport()
    records: list[tuple[datetime, str, int]] = []
    seen: set[tuple[datetime, str]] = set()

    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{csv_path} is empty")

        for row in reader:
            report.rows_read += 1
            if len(row) != 3 or not _TS_RE.match(row[0].strip()):
                report.rows_malformed += 1
                if len(report.examples) < 5:
                    report.examples.append(",".join(row)[:60])
                continue

            stamp = (
                datetime.strptime(row[0].strip(), "%Y-%m-%d %H:%M:%S")
                .replace(tzinfo=UTC)
                .astimezone(CAMPUS_TZ)
            )
            parsed_any = False
            for facility, raw in zip(FACILITIES, row[1:]):
                pct = parse_percent(raw)
                if pct is None:
                    continue
                parsed_any = True
                if pct > MAX_PLAUSIBLE_PCT:
                    report.rows_out_of_range += 1
                    continue
                key = (stamp, facility)
                if key in seen:
                    # The scraper occasionally logs the same second twice; that
                    # is a repeat, not a broken row.
                    report.rows_duplicate += 1
                    continue
                seen.add(key)
                records.append((stamp, facility, pct))
            if parsed_any:
                report.rows_kept += 1
            else:
                report.rows_malformed += 1
                if len(report.examples) < 5:
                    report.examples.append(",".join(row)[:60])

    frame = pd.DataFrame(records, columns=["ts", "facility", "pct"])
    if frame.empty:
        return frame.assign(date=[], weekday=[], hour=[]), report

    frame = frame.sort_values("ts", kind="stable").reset_index(drop=True)
    frame["date"] = frame["ts"].dt.date
    frame["weekday"] = frame["ts"].dt.weekday  # Monday = 0
    frame["hour"] = frame["ts"].dt.hour
    return frame, report


# --------------------------------------------------------------------------
# Semester windows
# --------------------------------------------------------------------------
# (name, first (month, day), last (month, day)) in campus local time.  These are
# generous brackets around the UC Berkeley academic calendar rather than exact
# instruction dates, so the same table keeps working year after year.  The dates
# a report is generated on (.github/workflows/semester_report.yml) sit one day
# after each window closes.
SEMESTER_WINDOWS: tuple[tuple[str, tuple[int, int], tuple[int, int]], ...] = (
    ("spring", (1, 13), (5, 20)),
    ("summer", (5, 21), (8, 17)),
    ("fall", (8, 18), (12, 21)),
)


def semester_of(day: date) -> str | None:
    """Return a slug like ``"fall-2025"``, or None for winter break."""
    for name, (m0, d0), (m1, d1) in SEMESTER_WINDOWS:
        if date(day.year, m0, d0) <= day <= date(day.year, m1, d1):
            return f"{name}-{day.year}"
    return None


def semester_bounds(slug: str) -> tuple[date, date]:
    name, year = slug.rsplit("-", 1)
    for candidate, (m0, d0), (m1, d1) in SEMESTER_WINDOWS:
        if candidate == name:
            return date(int(year), m0, d0), date(int(year), m1, d1)
    raise ValueError(f"unknown semester slug: {slug!r}")


def attach_semester(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``semester`` and ``week`` (0-based week index within the semester)."""
    if frame.empty:
        return frame.assign(semester=[], week=[])
    out = frame.copy()
    out["semester"] = [semester_of(d) for d in out["date"]]
    weeks = []
    for slug, day in zip(out["semester"], out["date"]):
        if not isinstance(slug, str):  # pandas turns the None gaps into NaN
            weeks.append(-1)
            continue
        start, _ = semester_bounds(slug)
        # Anchor weeks to the Monday on or before the window start so that a
        # "week" never straddles a weekend boundary mid-week.
        anchor = start - timedelta(days=start.weekday())
        weeks.append((day - anchor).days // 7)
    out["week"] = weeks
    return out


def completed_semesters(frame: pd.DataFrame, today: date) -> list[str]:
    """Semester slugs whose window has fully closed, oldest first.

    Sorted by end date, not by slug: alphabetically "fall-2026" sorts before
    "spring-2026", which would make ``--semester auto`` report the wrong one.
    """
    slugs = {s for s in frame["semester"].dropna().unique() if isinstance(s, str)}
    done = [s for s in slugs if semester_bounds(s)[1] < today]
    return sorted(done, key=lambda slug: semester_bounds(slug)[1])
