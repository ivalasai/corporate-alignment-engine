"""
Longitudinal profile metric engine for individual-level event histories.

Computes distributional statistics from episodic career/assignment arcs:

  - Average event interval (mean tenure duration in days)
  - Duration variance (coefficient of variation of interval lengths)
  - Categorical transition frequencies (role-to-role movement matrix)

Designed as a standalone worker: accepts a longitudinal events table and
returns one row per individual with derived behavioral trace metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = Path("data/derived/individual_profiles.parquet")
MIN_INTERVAL_DAYS = 1
MAX_INTERVAL_DAYS = 300 * 30  # cap extreme durations (~300 months)


# ---------------------------------------------------------------------------
# Interval extraction
# ---------------------------------------------------------------------------


def compute_interval_durations(
    history: pd.DataFrame,
    individual_col: str = "individual_id",
    start_col: str = "start_date",
    end_col: str = "end_date",
    min_days: int = MIN_INTERVAL_DAYS,
    max_days: int = MAX_INTERVAL_DAYS,
) -> pd.DataFrame:
    """
    Derive per-event interval durations from a longitudinal history table.

    Parameters
    ----------
    history : pd.DataFrame
        Episodic records with start/end date columns.
    individual_col : str
        Person-level identifier.
    start_col, end_col : str
        Interval boundary columns.
    min_days, max_days : int
        Duration floor and ceiling (outliers capped at max_days).

    Returns
    -------
    pd.DataFrame
        Original rows augmented with ``duration_days`` column.
    """
    out = history.copy()
    out[start_col] = pd.to_datetime(out[start_col], errors="coerce")
    out[end_col] = pd.to_datetime(out[end_col], errors="coerce")

    raw_days = (out[end_col] - out[start_col]).dt.days
    out["duration_days"] = raw_days.clip(lower=min_days, upper=max_days)
    return out


# ---------------------------------------------------------------------------
# Statistical variation metrics
# ---------------------------------------------------------------------------


def average_event_interval(
    durations: pd.Series,
) -> float:
    """
    Compute mean interval length across an individual's event history.

    Returns NaN when no valid intervals exist.
    """
    valid = durations.dropna()
    if valid.empty:
        return np.nan
    return float(valid.mean())


def duration_variance(
    durations: pd.Series,
) -> float:
    """
    Compute the coefficient of variation (CV) of interval durations.

    CV = std / mean; captures pacing regularity independent of scale.
    Returns NaN when fewer than two intervals or mean is zero.
    """
    valid = durations.dropna()
    if len(valid) < 2:
        return np.nan
    mean = valid.mean()
    if mean == 0:
        return np.nan
    return float(valid.std(ddof=1) / mean)


def categorical_transition_frequencies(
    history: pd.DataFrame,
    individual_col: str = "individual_id",
    category_col: str = "role_code",
    start_col: str = "start_date",
) -> dict[str, float]:
    """
    Compute empirical transition probabilities for an individual's category sequence.

    Events are ordered chronologically; consecutive category pairs form
    transitions. Returns a flat dict keyed as ``{from}__to__{to}`` with
    transition probabilities summing to 1.0 within each origin category.

    Parameters
    ----------
    history : pd.DataFrame
        Single-individual event history (multiple individuals will be grouped).
    individual_col : str
        Person identifier (used when history spans one person only).
    category_col : str
        Categorical state column (e.g., role type).
    start_col : str
        Chronological ordering column.

    Returns
    -------
    dict[str, float]
        Transition probability map.
    """
    ordered = history.sort_values(start_col)
    categories = ordered[category_col].dropna().tolist()

    if len(categories) < 2:
        return {}

    transitions: dict[tuple[str, str], int] = {}
    for origin, destination in zip(categories[:-1], categories[1:]):
        key = (str(origin), str(destination))
        transitions[key] = transitions.get(key, 0) + 1

    origin_totals: dict[str, int] = {}
    for (origin, _), count in transitions.items():
        origin_totals[origin] = origin_totals.get(origin, 0) + count

    freq_map: dict[str, float] = {}
    for (origin, destination), count in transitions.items():
        total = origin_totals[origin]
        freq_map[f"{origin}__to__{destination}"] = count / total if total else 0.0

    return freq_map


# ---------------------------------------------------------------------------
# Per-individual profile assembly
# ---------------------------------------------------------------------------


def build_individual_profile(
    individual_id: str,
    history: pd.DataFrame,
    individual_col: str = "individual_id",
    category_col: str = "role_code",
    start_col: str = "start_date",
    end_col: str = "end_date",
) -> dict:
    """
    Assemble a single-row metric profile for one individual.

    Parameters
    ----------
    individual_id : str
        Target person identifier.
    history : pd.DataFrame
        Full events table (filtered internally to ``individual_id``).
    individual_col, category_col, start_col, end_col : str
        Schema column names.

    Returns
    -------
    dict
        Flat record with core metrics and serialized transition frequencies.
    """
    subset = history.loc[history[individual_col] == individual_id].copy()
    with_durations = compute_interval_durations(subset, individual_col, start_col, end_col)
    durations = with_durations["duration_days"]

    transitions = categorical_transition_frequencies(
        with_durations, individual_col, category_col, start_col
    )

    return {
        individual_col: individual_id,
        "n_events": len(subset),
        "avg_interval_days": average_event_interval(durations),
        "interval_cv": duration_variance(durations),
        "n_unique_categories": subset[category_col].nunique(),
        "transition_map": transitions,
    }


def build_all_profiles(
    history: pd.DataFrame,
    individual_col: str = "individual_id",
    category_col: str = "role_code",
    start_col: str = "start_date",
    end_col: str = "end_date",
) -> pd.DataFrame:
    """
    Vectorized orchestration: compute profiles for every individual in history.

    Parameters
    ----------
    history : pd.DataFrame
        Complete longitudinal events table.
    individual_col, category_col, start_col, end_col : str
        Schema column names.

    Returns
    -------
    pd.DataFrame
        One row per individual with scalar metrics. Transition maps are stored
        as JSON-serializable dicts in the ``transition_map`` column.
    """
    individuals = history[individual_col].dropna().unique()
    records = [
        build_individual_profile(
            ind_id, history, individual_col, category_col, start_col, end_col
        )
        for ind_id in individuals
    ]

    profiles = pd.DataFrame(records)
    logger.info(
        "Built %d individual profiles | mean events: %.1f",
        len(profiles),
        profiles["n_events"].mean() if not profiles.empty else 0,
    )
    return profiles


def write_profiles(
    profiles: pd.DataFrame,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Persist individual profile metrics to compressed Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_parquet(output_path, index=False)
    logger.info("Wrote %d profiles to %s", len(profiles), output_path)
    return output_path


# ---------------------------------------------------------------------------
# Mock data and CLI entry point
# ---------------------------------------------------------------------------


def _mock_longitudinal_history() -> pd.DataFrame:
    """Generate illustrative individual event histories."""
    return pd.DataFrame(
        {
            "individual_id": ["I01", "I01", "I01", "I02", "I02", "I03"],
            "entity_id": ["E001"] * 3 + ["E002"] * 2 + ["E003"],
            "role_code": ["R1", "R2", "R1", "R1", "R3", "R2"],
            "start_date": [
                "2010-01-01", "2014-06-01", "2018-03-15",
                "2008-05-01", "2016-01-01",
                "2012-09-01",
            ],
            "end_date": [
                "2014-05-31", "2018-03-14", "2023-12-31",
                "2015-12-31", "2024-06-30",
                "2020-08-31",
            ],
        }
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    history = _mock_longitudinal_history()
    profiles = build_all_profiles(history)
    write_profiles(profiles)
