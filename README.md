# cal_gym_crowd_meter_scraper

Scraped realtime RSF and CMS gym capacities (percentages) from
<https://recwell.berkeley.edu/facilities/recreational-sports-facility-rsf/rsf-weight-room-crowd-meter/>

Note: the `Time` column in `crowd_meter_data.csv` is **UTC** (7–8 hours ahead of campus
time, depending on daylight saving). The analysis below converts it; nothing else should
read those timestamps as local.

## Reports — when is the gym actually empty?

**[→ reports/](reports/README.md)** — one report per semester, with the emptiest hour for
each day of the week.

A report is generated automatically the day after each semester window closes:

| Semester | Window | Report generated |
| --- | --- | --- |
| Spring | Jan 13 – May 20 | May 21 |
| Summer | May 21 – Aug 17 | Aug 18 |
| Fall | Aug 18 – Dec 21 | Dec 22 |

Each report contains, per facility:

- **The answer** — a table of the emptiest hour per weekday, with a 95% interval, the
  sample count behind it, and every other hour that is *statistically tied* with it.
- **Hourly profiles** — one small-multiple panel per weekday with bootstrap intervals.
- **A weekday × hour heatmap**, marking closed hours and under-sampled cells rather than
  quietly averaging over them.
- **A weekly trend**, showing how busyness moves across the semester (the start-of-term
  rush, breaks, finals).
- **A phase test** — if the *shape* of a typical day changes partway through the semester,
  the report splits and gives per-phase answers instead of one blurred average.

### Running it by hand

```bash
pip install -r requirements-analysis.txt

python -m analysis.report                          # last completed semester
python -m analysis.report --semester all           # every completed semester
python -m analysis.report --semester fall-2025     # one specific semester
```

`--out` chooses the output directory (default `reports/`), and `--today YYYY-MM-DD`
overrides the clock when deciding which semesters have finished.

### Verifying it

```bash
./scripts/verify.sh
```

Runs the unit tests (synthetic data with a known emptiest hour, a known mid-semester
changepoint, and known junk rows — the pipeline has to recover each) and then builds
every report against the real CSV in a scratch directory. The workflow runs the same
tests before it publishes anything.

### How the numbers are produced

- **Closed hours are excluded.** An hour whose median reading is ≤5% is the gym being
  shut, not the gym being quiet — without this, 3AM wins every day. Exact-zero readings
  inside an otherwise-busy hour are treated as mid-hour closure and dropped; if more than
  10% of an hour's readings are zeros it is additionally marked as a closing hour.
- **Everything is binned in campus local time.** The scraper's cron is fixed in UTC, so
  the same slot lands an hour earlier locally once PDT ends.
- **Intervals are bootstrap 95% intervals** (4,000 resamples, fixed seed). Because some
  hours have only a handful of samples, the report names every hour tied with the winner
  rather than pretending the single lowest mean is meaningful.
- **The phase test is a block permutation test** over whole days, comparing the best split
  found in the real timeline against the best split found in 1,000 shuffled ones. The
  semester is only cut where that clears p < 0.05.

### Layout

```
analysis/
  loader.py    read + clean the CSV, convert UTC -> campus time, assign semesters
  stats.py     open-hours detection, bootstrap intervals, changepoint test
  charts.py    matplotlib figures (light + dark, from a validated palette)
  report.py    CLI: builds charts, SUMMARY.md and summary.json
  tests/       the checks scripts/verify.sh runs
reports/       generated output, committed by the workflow
```

Semester windows live in `SEMESTER_WINDOWS` in `analysis/loader.py` and must stay in step
with the cron schedule in `.github/workflows/semester_report.yml`.
