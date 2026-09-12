"""The statistics behind "when is it actually empty?".

Three jobs live here:

1. Decide which (weekday, hour) cells the gym is even open for.  A closed gym
   reads 0% and would otherwise win "emptiest hour" every time.
2. Estimate each open cell's mean occupancy with a bootstrap confidence
   interval, then report not just the single lowest hour but every hour that is
   statistically indistinguishable from it.  With ~5-20 samples per cell a bare
   argmin is mostly noise; the tie set is the honest answer.
3. Test whether the daily shape actually changes partway through the semester
   (a block-permutation changepoint test).  If it doesn't, one recommendation
   for the whole semester is the right output and splitting it would be fake
   precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --- open/closed detection -------------------------------------------------
# An hour whose median reading is at or below this is the gym being shut, not
# the gym being quiet: observed open hours never sit this low.
CLOSED_MEDIAN_PCT = 5.0
# An exact-zero reading inside an hour that otherwise runs at 40-70% is the
# room being shut mid-hour, not the room being empty -- so zeros are dropped
# from every open hour, always.  Above this share of them the hour is *labelled*
# partial as well: "empty because it closes at 10:30" is a real answer, but the
# reader should know that's why.
PARTIAL_ZERO_FRAC = 0.10
MIN_CELL_N = 5

# --- bootstrap / permutation ----------------------------------------------
N_BOOT = 4000
N_PERM = 1000
RANDOM_SEED = 20260101

# --- changepoint -----------------------------------------------------------
MIN_PHASE_WEEKS = 3
MIN_PHASE_READINGS = 120
MIN_CELL_N_FOR_SHAPE = 3
PHASE_ALPHA = 0.05
MAX_PHASES = 3

WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def rng(seed: int = RANDOM_SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Open-hours detection
# ---------------------------------------------------------------------------
def cell_table(frame: pd.DataFrame, min_n: int = MIN_CELL_N) -> pd.DataFrame:
    """One row per (weekday, hour) with a status of open / partial / closed / sparse.

    ``n`` and ``mean`` are computed on the *usable* readings: for a partial hour
    the exact-zero (closed) readings are removed first.
    """
    columns = ["weekday", "hour", "n_raw", "n", "mean", "median", "zero_frac", "status"]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for (weekday, hour), group in frame.groupby(["weekday", "hour"], sort=True):
        values = group["pct"].to_numpy(dtype=float)
        zero_frac = float((values == 0).mean())
        median = float(np.median(values))

        if median <= CLOSED_MEDIAN_PCT:
            status, usable = "closed", values
        else:
            status = "partial" if zero_frac > PARTIAL_ZERO_FRAC else "open"
            usable = values[values > 0]

        if status != "closed" and len(usable) < min_n:
            status = "sparse"

        rows.append(
            {
                "weekday": int(weekday),
                "hour": int(hour),
                "n_raw": int(len(values)),
                "n": int(len(usable)),
                "mean": float(usable.mean()) if len(usable) else float("nan"),
                "median": median,
                "zero_frac": zero_frac,
                "status": status,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def usable_readings(frame: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Drop readings from closed/sparse cells, and the closed zeros inside partial hours."""
    if frame.empty or cells.empty:
        return frame.iloc[0:0]
    status = {(r.weekday, r.hour): r.status for r in cells.itertuples()}
    keys = list(zip(frame["weekday"], frame["hour"]))
    stat = np.array([status.get(k, "sparse") for k in keys])
    keep = np.isin(stat, ("open", "partial")) & (frame["pct"].to_numpy() > 0)
    return frame.loc[keep].copy()


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def bootstrap_means(values: np.ndarray, generator: np.random.Generator, n_boot: int = N_BOOT) -> np.ndarray:
    """Bootstrap distribution of the mean of ``values``."""
    values = np.asarray(values, dtype=float)
    draws = generator.integers(0, len(values), size=(n_boot, len(values)))
    return values[draws].mean(axis=1)


@dataclass
class HourEstimate:
    hour: int
    n: int
    mean: float
    ci_low: float
    ci_high: float
    status: str
    tied_with_best: bool = False


@dataclass
class WeekdayResult:
    weekday: int
    best_hour: int | None
    estimates: list[HourEstimate] = field(default_factory=list)
    tied_hours: list[int] = field(default_factory=list)
    confidence: str = "none"
    note: str = ""

    @property
    def weekday_name(self) -> str:
        return WEEKDAY_NAMES[self.weekday]


def emptiest_hours(
    frame: pd.DataFrame,
    cells: pd.DataFrame,
    generator: np.random.Generator,
    n_boot: int = N_BOOT,
) -> dict[int, WeekdayResult]:
    """For each weekday, the emptiest open hour plus every hour tied with it.

    "Tied" means the bootstrap 95% interval for (that hour - the best hour)
    contains zero, i.e. the data cannot tell them apart.
    """
    results: dict[int, WeekdayResult] = {}
    eligible = cells[cells["status"].isin(("open", "partial"))]

    for weekday in range(7):
        day_cells = eligible[eligible["weekday"] == weekday].sort_values("hour")
        if day_cells.empty:
            results[weekday] = WeekdayResult(weekday, None, note="no open hours with enough samples")
            continue

        boots: dict[int, np.ndarray] = {}
        estimates: list[HourEstimate] = []
        for cell in day_cells.itertuples():
            values = frame[(frame["weekday"] == weekday) & (frame["hour"] == cell.hour)]["pct"].to_numpy(float)
            values = values[values > 0]
            if len(values) < MIN_CELL_N:
                continue
            draws = bootstrap_means(values, generator, n_boot)
            boots[cell.hour] = draws
            estimates.append(
                HourEstimate(
                    hour=int(cell.hour),
                    n=len(values),
                    mean=float(values.mean()),
                    ci_low=float(np.percentile(draws, 2.5)),
                    ci_high=float(np.percentile(draws, 97.5)),
                    status=cell.status,
                )
            )

        if not estimates:
            results[weekday] = WeekdayResult(weekday, None, note="no open hours with enough samples")
            continue

        best = min(estimates, key=lambda e: e.mean)
        tied = []
        for est in estimates:
            diff = boots[est.hour] - boots[best.hour]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            est.tied_with_best = bool(lo <= 0 <= hi)
            if est.tied_with_best and est.hour != best.hour:
                tied.append(est.hour)

        if best.n >= 15 and len(tied) <= 1:
            confidence = "strong"
        elif best.n >= 8 and len(tied) <= 3:
            confidence = "moderate"
        else:
            confidence = "weak"

        results[weekday] = WeekdayResult(
            weekday=weekday,
            best_hour=best.hour,
            estimates=sorted(estimates, key=lambda e: e.hour),
            tied_hours=sorted(tied),
            confidence=confidence,
            note="closing hour -- gym shuts partway through" if best.status == "partial" else "",
        )
    return results


# ---------------------------------------------------------------------------
# Weekly level ("how busy was the gym that week, adjusted for when we sampled")
# ---------------------------------------------------------------------------
def weekly_levels(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-week mean occupancy relative to that (weekday, hour)'s semester average.

    Dividing by the cell average removes the sampling bias: a week that happened
    to be sampled mostly at 4pm would otherwise look busier than one sampled at
    7am.  1.0 means "a typical week for this semester".
    """
    if frame.empty:
        return pd.DataFrame(columns=["week", "n", "level"])
    cell_mean = frame.groupby(["weekday", "hour"])["pct"].transform("mean")
    ratio = frame["pct"] / cell_mean.replace(0, np.nan)
    work = frame.assign(ratio=ratio).dropna(subset=["ratio"])
    out = work.groupby("week")["ratio"].agg(["count", "mean"]).reset_index()
    out.columns = ["week", "n", "level"]
    return out.sort_values("week").reset_index(drop=True)


def _outlier_weeks(levels: pd.DataFrame, compare, min_n: int) -> list[int]:
    usable = levels[levels["n"] >= min_n]
    if usable.empty:
        return []
    norm = float(usable["level"].median())
    return [int(w) for w, lvl in zip(usable["week"], usable["level"]) if compare(lvl, norm)]


def quiet_weeks(levels: pd.DataFrame, threshold: float = 0.75, min_n: int = 20) -> list[int]:
    """Weeks running well below the semester norm -- breaks, holidays, post-finals."""
    return _outlier_weeks(levels, lambda lvl, norm: lvl < threshold * norm, min_n)


def busy_weeks(levels: pd.DataFrame, threshold: float = 1.25, min_n: int = 20) -> list[int]:
    """Weeks running well above it -- the new-semester rush is the usual culprit."""
    return _outlier_weeks(levels, lambda lvl, norm: lvl > threshold * norm, min_n)


# ---------------------------------------------------------------------------
# Changepoint test on the shape of the day
# ---------------------------------------------------------------------------
def _shape_distance(left: pd.DataFrame, right: pd.DataFrame, min_cell: int = MIN_CELL_N_FOR_SHAPE) -> float:
    """How differently does a day *look* on either side of a split?

    Each side gets a (weekday, hour) profile normalised to its own mean, so a
    semester that simply gets busier overall scores zero.  Only the shape counts.
    """
    lg = left.groupby(["weekday", "hour"])["pct"].agg(["count", "mean"])
    rg = right.groupby(["weekday", "hour"])["pct"].agg(["count", "mean"])
    joined = lg.join(rg, how="inner", lsuffix="_l", rsuffix="_r")
    joined = joined[(joined["count_l"] >= min_cell) & (joined["count_r"] >= min_cell)]
    joined = joined[(joined["mean_l"] > 0) & (joined["mean_r"] > 0)]
    if len(joined) < 6:
        return 0.0

    weights = 2.0 / (1.0 / joined["count_l"] + 1.0 / joined["count_r"])  # harmonic mean
    norm_l = joined["mean_l"] / np.average(joined["mean_l"], weights=weights)
    norm_r = joined["mean_r"] / np.average(joined["mean_r"], weights=weights)
    return float(np.average(np.abs(np.log(norm_l / norm_r)), weights=weights))


def _split_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, int, np.ndarray]:
    """Encode each reading's (weekday, hour) cell as a small integer.

    The permutation test recomputes per-cell means thousands of times, so the
    search below works on integer-coded arrays and cumulative sums rather than
    repeated groupbys.  ``_shape_distance`` above is the readable reference
    implementation; ``test_fast_split_matches_reference`` pins them together.
    """
    raw = frame["weekday"].to_numpy() * 24 + frame["hour"].to_numpy()
    codes, uniques = pd.factorize(raw)
    return codes.astype(np.intp), len(uniques), frame["pct"].to_numpy(dtype=float)


def _best_split_fast(
    codes: np.ndarray,
    n_cells: int,
    values: np.ndarray,
    week_index: np.ndarray,
    n_weeks: int,
    week_lo: int,
    min_cell: int = MIN_CELL_N_FOR_SHAPE,
) -> tuple[int | None, float]:
    """Same statistic as ``_shape_distance``, maximised over candidate cut weeks."""
    counts = np.zeros((n_cells, n_weeks))
    totals = np.zeros((n_cells, n_weeks))
    np.add.at(counts, (codes, week_index), 1.0)
    np.add.at(totals, (codes, week_index), values)
    count_cum = counts.cumsum(axis=1)
    total_cum = totals.cumsum(axis=1)
    count_all = count_cum[:, -1]
    total_all = total_cum[:, -1]

    best_week, best_stat = None, 0.0
    for cut in range(week_lo + MIN_PHASE_WEEKS, week_lo + n_weeks - MIN_PHASE_WEEKS + 1):
        j = cut - week_lo - 1
        cl, sl = count_cum[:, j], total_cum[:, j]
        cr, sr = count_all - cl, total_all - sl
        if cl.sum() < MIN_PHASE_READINGS or cr.sum() < MIN_PHASE_READINGS:
            continue
        mask = (cl >= min_cell) & (cr >= min_cell) & (sl > 0) & (sr > 0)
        if mask.sum() < 6:
            continue
        cl_m, cr_m = cl[mask], cr[mask]
        mean_l, mean_r = sl[mask] / cl_m, sr[mask] / cr_m
        weights = 2.0 / (1.0 / cl_m + 1.0 / cr_m)
        norm_l = mean_l / np.average(mean_l, weights=weights)
        norm_r = mean_r / np.average(mean_r, weights=weights)
        stat = float(np.average(np.abs(np.log(norm_l / norm_r)), weights=weights))
        if stat > best_stat:
            best_week, best_stat = cut, stat
    return best_week, best_stat


@dataclass
class Phase:
    label: str
    week_start: int
    week_end: int
    n: int
    p_value: float | None = None


def find_phases(
    frame: pd.DataFrame,
    generator: np.random.Generator,
    n_perm: int = N_PERM,
    alpha: float = PHASE_ALPHA,
    max_phases: int = MAX_PHASES,
) -> tuple[list[Phase], list[dict]]:
    """Split the semester only where the daily shape genuinely changes.

    Significance comes from a block permutation test: whole days are shuffled
    along the timeline (keeping each day's readings together, which preserves
    the within-day correlation the readings actually have), and the observed
    best-split statistic is compared against the best split found in permuted
    timelines.  Taking the max over candidate splits in both the observed and
    permuted case is what keeps the search itself from manufacturing a p-value.
    """
    tests: list[dict] = []
    if frame.empty:
        return [], tests

    def split_recursive(sub: pd.DataFrame, depth: int) -> list[Phase]:
        weeks = sub["week"].to_numpy()
        week_lo, week_hi = int(weeks.min()), int(weeks.max())
        n_weeks = week_hi - week_lo + 1
        leaf = [Phase(label="", week_start=week_lo, week_end=week_hi, n=len(sub))]
        if depth <= 0 or len(sub) < 2 * MIN_PHASE_READINGS or n_weeks < 2 * MIN_PHASE_WEEKS:
            return leaf

        codes, n_cells, values = _split_arrays(sub)
        week_index = (weeks - week_lo).astype(np.intp)
        cut, observed = _best_split_fast(codes, n_cells, values, week_index, n_weeks, week_lo)
        if cut is None or observed <= 0:
            return leaf

        # Permute which date sits where in the timeline, whole days intact --
        # readings within a day are correlated, so the day is the exchangeable
        # unit, not the individual reading.
        dates, date_pos = np.unique(sub["date"].to_numpy(), return_inverse=True)
        week_of_date = np.empty(len(dates), dtype=np.intp)
        week_of_date[date_pos] = week_index

        null_stats = np.empty(n_perm)
        for i in range(n_perm):
            shuffled = week_of_date[generator.permutation(len(dates))][date_pos]
            _, stat = _best_split_fast(codes, n_cells, values, shuffled, n_weeks, week_lo)
            null_stats[i] = stat
        p_value = float((1 + np.sum(null_stats >= observed)) / (1 + n_perm))
        tests.append(
            {
                "weeks": [week_lo, week_hi],
                "cut_week": int(cut),
                "statistic": round(observed, 4),
                "p_value": round(p_value, 4),
                "significant": bool(p_value < alpha),
            }
        )
        if p_value >= alpha:
            return leaf

        left = sub[sub["week"] < cut]
        right = sub[sub["week"] >= cut]
        return split_recursive(left, depth - 1) + split_recursive(right, depth - 1)

    phases = split_recursive(frame, max_phases - 1)
    for i, phase in enumerate(phases, start=1):
        phase.label = f"Phase {i} (weeks {phase.week_start + 1}-{phase.week_end + 1})" if len(phases) > 1 else "Whole semester"
    for phase, test in zip(phases[1:], tests):
        phase.p_value = test["p_value"]
    return phases, tests
