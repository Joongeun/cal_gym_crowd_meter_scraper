# Summer 2026 — Berkeley gym crowd report

Semester window **May 21, 2026 – Aug 17, 2026** · 406 readings · generated 2026-08-25.

All times are campus local (America/Los_Angeles); the scraper logs UTC and this report converts. An hour labelled *7AM–8AM* covers 7:00–7:59.

## RSF weight room

| Day | Go at | Mean % full | 95% CI | Just as good | Samples | Confidence |
| --- | --- | ---: | ---: | --- | ---: | --- |
| Monday | — | — | — | — | 0 | not enough data |
| Tuesday | — | — | — | — | 0 | not enough data |
| Wednesday | — | — | — | — | 0 | not enough data |
| Thursday | — | — | — | — | 0 | not enough data |
| Friday | — | — | — | — | 0 | not enough data |
| Saturday | — | — | — | — | 0 | not enough data |
| Sunday | **4PM–5PM** | 37% | 33–41% | — | 5 | weak |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="weekday-profiles-rsf-dark.png">
  <img alt="Summer 2026 RSF weight room hourly occupancy by weekday with 95% intervals" src="weekday-profiles-rsf.png">
</picture>


<picture>
  <source media="(prefers-color-scheme: dark)" srcset="heatmap-rsf-dark.png">
  <img alt="Summer 2026 RSF weight room weekday-by-hour occupancy heatmap" src="heatmap-rsf.png">
</picture>


<details><summary>Table view — mean % full per hour</summary>

| Day | 4p |
| --- | ---: |
| Mon | · |
| Tue | · |
| Wed | · |
| Thu | · |
| Fri | · |
| Sat | · |
| Sun | 37 |

Mean % full per hour. `·` = closed, or too few samples to estimate.

</details>

### Does the pattern hold all semester?

Not enough data to test whether the shape of the day changes mid-semester.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="weekly-trend-rsf-dark.png">
  <img alt="Summer 2026 RSF weight room weekly busyness" src="weekly-trend-rsf.png">
</picture>


## CMS fitness centre

206 readings, but none land in an hour with enough samples to estimate an open-hours pattern.

## Method & caveats

- **Closed hours are excluded.** An hour whose median reading is ≤5% is the gym being shut, not the gym being quiet; without this, 3AM wins every day.
- **Opening and closing hours** are kept but have their zero readings dropped first, and are marked ¹ — genuinely quiet, but only for part of the hour.
- **Intervals are bootstrap 95% intervals** on each hour's mean (4,000 resamples, fixed seed). The *Just as good* column lists every hour whose difference from the winner has an interval straddling zero — with this few samples these are ties, not runners-up.
- **The phase test** is a block permutation test: whole days are shuffled along the timeline and the best split found in the real data is compared against the best split found in 1,000 shuffled timelines. The semester is only cut where that test clears p < 0.05.
- **The phase test has limited power on thin data.** A stable verdict means no *large* shape change was detectable, not that nothing shifted; a semester with a few hundred readings can only reveal shifts on the scale of the morning/evening balance flipping.
- **Sampling is uneven.** The scraper runs on GitHub Actions cron, which is throttled and delayed, so some hours are sampled far more than others. Sample counts are shown; treat any *weak* confidence row as a hint rather than an answer.

### Data quality this run

- Rows read: **6,886** · kept: **6,884** · unparseable: **2** · out-of-range values dropped: **0**

<details><summary>Examples of dropped rows</summary>

```
2026-08-17 23:41:48,%=
,%=

2026-08-19 22:01:03,=
%,=
%
```

</details>

