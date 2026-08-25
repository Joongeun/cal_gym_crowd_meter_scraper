"""Tests for the semester analysis.

The point of these is that the answer the report prints is checkable: synthetic
data with a known emptiest hour, a known changepoint, and known junk rows, and
the pipeline has to recover each one.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import report
from analysis.loader import (
    attach_semester,
    completed_semesters,
    load_readings,
    parse_percent,
    semester_of,
)
from analysis.stats import (
    _best_split_fast,
    _shape_distance,
    _split_arrays,
    cell_table,
    emptiest_hours,
    find_phases,
    quiet_weeks,
    rng,
    usable_readings,
    weekly_levels,
)

FAST_PERM = 200


# ---------------------------------------------------------------------------
# Parsing / loading
# ---------------------------------------------------------------------------
class ParsePercentTests(unittest.TestCase):
    def test_plain_and_prefixed(self):
        self.assertEqual(parse_percent("64%"), 64)
        self.assertEqual(parse_percent(">7%"), 7)  # stray markup char, digits fine
        self.assertEqual(parse_percent(" 100% "), 100)
        self.assertEqual(parse_percent("104%"), 104)  # genuinely over capacity

    def test_unparseable(self):
        for junk in ('"%=', "", "   ", '"=\n%"', None):
            self.assertIsNone(parse_percent(junk))


class LoaderTests(unittest.TestCase):
    def _write(self, rows) -> Path:
        handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
        writer = csv.writer(handle)
        writer.writerow(["Time", "RSF_percent_full", "CMS_percent_full"])
        for row in rows:
            writer.writerow(row)
        handle.close()
        return Path(handle.name)

    def test_utc_is_converted_to_campus_time(self):
        path = self._write([["2025-09-10 18:30:00", "64%", "50%"]])
        frame, _ = load_readings(path)
        # 18:30 UTC in September is 11:30 PDT.
        self.assertEqual(frame["ts"].iloc[0].hour, 11)
        self.assertEqual(frame["hour"].iloc[0], 11)

    def test_malformed_rows_are_dropped_and_counted(self):
        path = self._write(
            [
                ["2025-09-10 18:30:00", "64%", "50%"],
                ["2025-09-10 18:40:00", "%=\n", "\n"],  # both columns junk
                ["not a timestamp", "10%", "10%"],
                ["2025-09-10 19:00:00", ">3%", "900%"],  # CMS out of plausible range
            ]
        )
        frame, load = load_readings(path)
        self.assertEqual(load.rows_read, 4)
        self.assertEqual(load.rows_kept, 2)
        self.assertEqual(load.rows_malformed, 2)
        self.assertEqual(load.rows_out_of_range, 1)
        self.assertEqual(sorted(frame["pct"].tolist()), [3, 50, 64])

    def test_dst_shift_moves_a_fixed_utc_scrape_into_a_different_local_hour(self):
        """The scraper's cron is fixed in UTC, so the same slot lands an hour
        earlier locally once PDT ends. Binning must happen in campus time."""
        path = self._write(
            [
                ["2025-10-15 14:00:00", "40%", "40%"],  # PDT -> 7am
                ["2025-11-15 14:00:00", "40%", "40%"],  # PST -> 6am
            ]
        )
        frame, _ = load_readings(path)
        self.assertEqual(sorted(set(frame["hour"])), [6, 7])

    def test_duplicate_timestamps_are_collapsed(self):
        path = self._write(
            [
                ["2025-09-10 18:30:00", "64%", "50%"],
                ["2025-09-10 18:30:00", "64%", "50%"],
            ]
        )
        _, load = load_readings(path)
        self.assertEqual(load.rows_duplicate, 2)


class SemesterWindowTests(unittest.TestCase):
    def test_slugs(self):
        self.assertEqual(semester_of(date(2025, 9, 10)), "fall-2025")
        self.assertEqual(semester_of(date(2026, 3, 2)), "spring-2026")
        self.assertEqual(semester_of(date(2026, 6, 15)), "summer-2026")

    def test_winter_break_belongs_to_no_semester(self):
        self.assertIsNone(semester_of(date(2025, 12, 28)))
        self.assertIsNone(semester_of(date(2026, 1, 5)))

    def test_windows_do_not_overlap_and_boundaries_land_where_expected(self):
        self.assertEqual(semester_of(date(2026, 5, 20)), "spring-2026")
        self.assertEqual(semester_of(date(2026, 5, 21)), "summer-2026")
        self.assertEqual(semester_of(date(2026, 8, 17)), "summer-2026")
        self.assertEqual(semester_of(date(2026, 8, 18)), "fall-2026")


class CompletedSemesterTests(unittest.TestCase):
    def _frame(self, slugs):
        return pd.DataFrame({"semester": slugs})

    def test_ordered_by_end_date_not_alphabetically(self):
        frame = self._frame(["spring-2026", "fall-2026", "fall-2025", "summer-2026"])
        self.assertEqual(
            completed_semesters(frame, date(2027, 1, 5)),
            ["fall-2025", "spring-2026", "summer-2026", "fall-2026"],
        )

    def test_excludes_a_semester_still_running(self):
        frame = self._frame(["fall-2025", "spring-2026"])
        # Mid-spring: fall is done, spring is not.
        self.assertEqual(completed_semesters(frame, date(2026, 3, 1)), ["fall-2025"])


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------
def make_frame(rows) -> pd.DataFrame:
    """rows: iterable of (datetime, pct). Builds the frame shape the stats expect."""
    frame = pd.DataFrame(rows, columns=["ts", "pct"])
    frame["facility"] = "RSF"
    frame["date"] = frame["ts"].dt.date
    frame["weekday"] = frame["ts"].dt.weekday
    frame["hour"] = frame["ts"].dt.hour
    frame["week"] = ((frame["ts"] - frame["ts"].min()).dt.days // 7).astype(int)
    return frame


def synthetic_semester(shape, weeks=14, per_hour=6, seed=7, level=lambda w: 1.0):
    """Generate readings whose true hourly profile is ``shape`` (hour -> % full)."""
    generator = np.random.default_rng(seed)
    rows = []
    start = datetime(2025, 9, 1, 0, 0)
    for week in range(weeks):
        for day_offset in range(7):
            day = start + timedelta(days=week * 7 + day_offset)
            for hour, base in shape.items():
                for _ in range(per_hour):
                    minute = int(generator.integers(0, 60))
                    value = base * level(week) + generator.normal(0, 3)
                    rows.append((day.replace(hour=hour, minute=minute), max(0.0, value)))
    return make_frame(rows)


# ---------------------------------------------------------------------------
# Open / closed detection
# ---------------------------------------------------------------------------
class CellTableTests(unittest.TestCase):
    def setUp(self):
        generator = np.random.default_rng(1)
        rows = []
        day = datetime(2025, 9, 1)
        for offset in range(28):
            base = day + timedelta(days=offset)
            for _ in range(8):
                rows.append((base.replace(hour=3), 0.0))                       # closed
                rows.append((base.replace(hour=10), 60 + generator.normal(0, 4)))  # open
            for i in range(10):
                # closes partway through hour 22: a third of readings are zeros
                rows.append((base.replace(hour=22, minute=i * 5), 0.0 if i < 3 else 30.0))
            rows.append((base.replace(hour=15), 70.0))  # only 1 sample per day-of-week cell
        self.cells = cell_table(make_frame(rows))

    def _status(self, weekday, hour):
        row = self.cells[(self.cells["weekday"] == weekday) & (self.cells["hour"] == hour)]
        return row["status"].iloc[0]

    def test_all_zero_hour_is_closed(self):
        self.assertEqual(self._status(0, 3), "closed")

    def test_busy_hour_is_open(self):
        self.assertEqual(self._status(0, 10), "open")

    def test_hour_straddling_closing_time_is_partial_and_drops_its_zeros(self):
        self.assertEqual(self._status(0, 22), "partial")
        row = self.cells[(self.cells["weekday"] == 0) & (self.cells["hour"] == 22)]
        self.assertEqual(int(row["n_raw"].iloc[0]), 40)
        self.assertEqual(int(row["n"].iloc[0]), 28)  # the 12 closed zeros are gone
        self.assertAlmostEqual(float(row["mean"].iloc[0]), 30.0, places=6)

    def test_thin_cell_is_sparse(self):
        self.assertEqual(self._status(0, 15), "sparse")

    def test_usable_readings_excludes_closed_and_partial_zeros(self):
        generator = np.random.default_rng(1)
        rows = []
        day = datetime(2025, 9, 1)
        for offset in range(28):
            base = day + timedelta(days=offset)
            for _ in range(8):
                rows.append((base.replace(hour=3), 0.0))
                rows.append((base.replace(hour=10), 60 + generator.normal(0, 4)))
        frame = make_frame(rows)
        usable = usable_readings(frame, cell_table(frame))
        self.assertEqual(set(usable["hour"]), {10})
        self.assertTrue((usable["pct"] > 0).all())


# ---------------------------------------------------------------------------
# Emptiest-hour recovery
# ---------------------------------------------------------------------------
class EmptiestHourTests(unittest.TestCase):
    def test_recovers_an_injected_minimum(self):
        shape = {7: 25.0, 10: 60.0, 13: 70.0, 16: 85.0, 20: 55.0}
        frame = synthetic_semester(shape)
        cells = cell_table(frame)
        results = emptiest_hours(frame, cells, rng(3), n_boot=800)
        for weekday in range(7):
            self.assertEqual(results[weekday].best_hour, 7, f"weekday {weekday}")
            self.assertEqual(results[weekday].tied_hours, [], "distinct levels should not tie")
            self.assertEqual(results[weekday].confidence, "strong")

    def test_closed_hours_never_win(self):
        shape = {7: 25.0, 10: 60.0, 16: 85.0}
        frame = synthetic_semester(shape)
        # bolt on a genuinely closed 3AM that reads 0%
        closed = [(t.replace(hour=3), 0.0) for t in frame["ts"].head(400)]
        frame = make_frame(list(zip(frame["ts"], frame["pct"])) + closed)
        cells = cell_table(frame)
        results = emptiest_hours(frame, cells, rng(3), n_boot=800)
        self.assertTrue(all(results[d].best_hour == 7 for d in range(7)))

    def test_indistinguishable_hours_are_reported_as_ties(self):
        shape = {7: 40.0, 8: 40.4, 16: 85.0}
        frame = synthetic_semester(shape, per_hour=3)
        cells = cell_table(frame)
        results = emptiest_hours(frame, cells, rng(5), n_boot=1500)
        for weekday in range(7):
            best = results[weekday].best_hour
            self.assertIn(best, (7, 8))
            other = 8 if best == 7 else 7
            self.assertIn(other, results[weekday].tied_hours,
                          "two hours 0.4pp apart must not be called apart")

    def test_reports_nothing_when_the_day_has_no_open_hours(self):
        rows = [(datetime(2025, 9, 1) + timedelta(days=d, hours=3), 0.0) for d in range(30)]
        frame = make_frame(rows)
        results = emptiest_hours(frame, cell_table(frame), rng(1), n_boot=200)
        self.assertTrue(all(r.best_hour is None for r in results.values()))


# ---------------------------------------------------------------------------
# Phase detection
# ---------------------------------------------------------------------------
class SplitSearchEquivalenceTests(unittest.TestCase):
    """The numpy search must return exactly what the readable pandas version does."""

    def test_fast_split_matches_reference(self):
        frame = synthetic_semester({7: 30.0, 10: 60.0, 13: 70.0, 16: 85.0, 20: 55.0},
                                   weeks=12, per_hour=5, seed=21)
        usable = usable_readings(frame, cell_table(frame))
        weeks = usable["week"].to_numpy()
        week_lo, week_hi = int(weeks.min()), int(weeks.max())
        codes, n_cells, values = _split_arrays(usable)
        fast_cut, fast_stat = _best_split_fast(
            codes, n_cells, values, (weeks - week_lo), week_hi - week_lo + 1, week_lo
        )
        slow_cut, slow_stat = None, 0.0
        for cut in range(week_lo + 3, week_hi - 1):
            left, right = usable[usable["week"] < cut], usable[usable["week"] >= cut]
            if len(left) < 120 or len(right) < 120:
                continue
            stat = _shape_distance(left, right)
            if stat > slow_stat:
                slow_cut, slow_stat = cut, stat
        self.assertEqual(fast_cut, slow_cut)
        self.assertAlmostEqual(fast_stat, slow_stat, places=10)


class PhaseTests(unittest.TestCase):
    def test_no_split_when_the_shape_is_stable(self):
        frame = synthetic_semester({7: 30.0, 10: 60.0, 13: 70.0, 16: 85.0, 20: 55.0})
        usable = usable_readings(frame, cell_table(frame))
        phases, tests = find_phases(usable, rng(11), n_perm=FAST_PERM)
        self.assertEqual(len(phases), 1)
        self.assertTrue(tests and tests[0]["p_value"] >= 0.05)

    def test_a_level_change_alone_is_not_a_shape_change(self):
        # Second half is 40% busier at every hour: same shape, different level.
        frame = synthetic_semester(
            {7: 30.0, 10: 60.0, 13: 70.0, 16: 85.0, 20: 55.0},
            level=lambda w: 1.0 if w < 7 else 1.4,
        )
        usable = usable_readings(frame, cell_table(frame))
        phases, _ = find_phases(usable, rng(11), n_perm=FAST_PERM)
        self.assertEqual(len(phases), 1, "a uniform busyness shift must not split the semester")

    def test_detects_an_injected_shape_change(self):
        generator = np.random.default_rng(4)
        rows = []
        early = {7: 30.0, 10: 60.0, 13: 70.0, 16: 85.0, 20: 55.0}
        late = {7: 80.0, 10: 60.0, 13: 55.0, 16: 40.0, 20: 30.0}  # morning/evening flip
        start = datetime(2025, 9, 1)
        for week in range(14):
            shape = early if week < 7 else late
            for day_offset in range(7):
                day = start + timedelta(days=week * 7 + day_offset)
                for hour, base in shape.items():
                    for _ in range(6):
                        rows.append((day.replace(hour=hour,
                                                 minute=int(generator.integers(0, 60))),
                                     max(0.0, base + generator.normal(0, 3))))
        frame = make_frame(rows)
        usable = usable_readings(frame, cell_table(frame))
        phases, tests = find_phases(usable, rng(11), n_perm=FAST_PERM)
        self.assertGreater(len(phases), 1)
        self.assertTrue(any(t["significant"] for t in tests))
        self.assertEqual(phases[1].week_start, 7, "changepoint should land at the real flip")

    def test_weekly_levels_and_quiet_weeks(self):
        frame = synthetic_semester(
            {7: 30.0, 10: 60.0, 16: 85.0},
            level=lambda w: 0.4 if w == 5 else 1.0,
        )
        usable = usable_readings(frame, cell_table(frame))
        levels = weekly_levels(usable)
        self.assertEqual(len(levels), 14)
        self.assertEqual(quiet_weeks(levels), [5])


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
class EndToEndTests(unittest.TestCase):
    def test_report_writes_charts_summary_and_index(self):
        generator = np.random.default_rng(2)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv_path = tmp / "data.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Time", "RSF_percent_full", "CMS_percent_full"])
                shape = {14: 25, 17: 60, 20: 70, 23: 85}  # UTC hours -> 7a/10a/1p/4p PDT
                day = datetime(2025, 9, 1)
                # 60 days keeps the whole run inside PDT; crossing into PST
                # would legitimately move these into the 6AM bin.
                for offset in range(60):
                    current = day + timedelta(days=offset)
                    for hour, base in shape.items():
                        for _ in range(4):
                            stamp = current.replace(hour=hour, minute=int(generator.integers(0, 60)))
                            value = int(max(0, base + generator.normal(0, 4)))
                            writer.writerow([stamp.strftime("%Y-%m-%d %H:%M:%S"), f"{value}%", f"{value}%"])
                writer.writerow(["2025-09-02 12:00:00", '"%=', '"'])  # junk row, must be dropped

            out = tmp / "reports"
            code = report.main([
                "--data", str(csv_path), "--out", str(out),
                "--semester", "fall-2025", "--today", "2026-01-05",
            ])
            self.assertEqual(code, 0)

            summary = out / "fall-2025" / "SUMMARY.md"
            self.assertTrue(summary.exists())
            text = summary.read_text()
            self.assertIn("RSF weight room", text)
            self.assertIn("7AM–8AM", text)  # the injected minimum, in campus time
            self.assertIn("unparseable: **1**", text)

            for stem in ("weekday-profiles-rsf", "heatmap-rsf", "weekly-trend-rsf"):
                self.assertTrue((out / "fall-2025" / f"{stem}.png").exists(), stem)
                self.assertTrue((out / "fall-2025" / f"{stem}-dark.png").exists(), stem)

            data = json.loads((out / "fall-2025" / "summary.json").read_text())
            rsf = next(f for f in data["facilities"] if f["facility"] == "RSF")
            self.assertEqual(rsf["emptiest"]["Monday"]["hour"], 7)

            index = (out / "README.md").read_text()
            self.assertIn("Fall 2025", index)


if __name__ == "__main__":
    unittest.main()
