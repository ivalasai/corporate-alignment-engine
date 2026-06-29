"""
Longitudinal panel synthesizer: episodic events → entity-year matrix.

Transforms irregular episodic role/assignment records into a fixed-interval
panel aligned with numeric outcome tables. Key operations:

  - Open-ended interval closure (far-future end dates → snapshot date)
  - Interval-based metric aggregation per entity and fiscal period
  - Left join with standard numeric outcome panels
  - Cross-sectional 1-year lag via within-entity ``.shift(1)``

Sorting by (entity_id, year) before lagging is mandatory for correctness.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_SNAPSHOT_DATE = date.today()
FAR_FUTURE_THRESHOLD = pd.Timestamp("2099-01-01")
DEFAULT_OUTPUT_PATH = Path("data/derived/panel_matrix.parquet")


# ---------------------------------------------------------------------------
# Date normalization
# ---------------------------------------------------------------------------


def normalize_open_end_dates(
    df: pd.DataFrame,
    end_col: str = "end_date",
    snapshot: date | None = None,
    far_future_threshold: pd.Timestamp = FAR_FUTURE_THRESHOLD,
) -> pd.DataFrame:
    """
    Replace open-ended (active) interval endpoints with a snapshot date.

    Any ``end_date`` on or after ``far_future_threshold`` is treated as
    "still active" and capped at ``snapshot`` (defaults to today's date).

    Parameters
    ----------
    df : pd.DataFrame
        Episodic records with at least ``end_col``.
    end_col : str
        Column containing interval end timestamps.
    snapshot : date, optional
        Reference date for active-record closure.
    far_future_threshold : pd.Timestamp
        Dates at or beyond this value are considered open-ended sentinels.

    Returns
    -------
    pd.DataFrame
        Copy of input with normalized end dates.
    """
    snapshot = snapshot or DEFAULT_SNAPSHOT_DATE
    out = df.copy()
    out[end_col] = pd.to_datetime(out[end_col], errors="coerce")
    open_mask = out[end_col] >= far_future_threshold
    out.loc[open_mask, end_col] = pd.Timestamp(snapshot)
    logger.debug("Closed %d open-ended intervals at snapshot %s", open_mask.sum(), snapshot)
    return out


# ---------------------------------------------------------------------------
# Episodic → interval aggregation
# ---------------------------------------------------------------------------


def episodic_to_annual_panel(
    events: pd.DataFrame,
    entity_col: str = "entity_id",
    start_col: str = "start_date",
    end_col: str = "end_date",
    year_end_month: int = 12,
    year_end_day: int = 31,
    snapshot: date | None = None,
) -> pd.DataFrame:
    """
    Expand episodic intervals into entity-year observations with overlap counts.

    For each fiscal year-end snapshot, records whether an event interval
    overlaps that date and computes tenure duration within the year.

    Parameters
    ----------
    events : pd.DataFrame
        Episodic assignment records.
    entity_col : str
        Entity identifier column.
    start_col, end_col : str
        Interval boundary columns.
    year_end_month, year_end_day : int
        Fiscal year-end definition (default: calendar year-end).
    snapshot : date, optional
        Passed to ``normalize_open_end_dates``.

    Returns
    -------
    pd.DataFrame
        Entity-year panel with ``active_roles``, ``mean_tenure_days``.
    """
    events = normalize_open_end_dates(events, end_col=end_col, snapshot=snapshot)
    events[start_col] = pd.to_datetime(events[start_col], errors="coerce")

    min_year = events[start_col].dt.year.min()
    max_year = events[end_col].dt.year.max()
    years = range(int(min_year), int(max_year) + 1)

    rows: list[dict] = []
    for year in years:
        snapshot_ts = pd.Timestamp(year=year, month=year_end_month, day=year_end_day)
        for entity_id, group in events.groupby(entity_col):
            active = group[
                (group[start_col] <= snapshot_ts) & (group[end_col] >= snapshot_ts)
            ]
            if active.empty:
                continue

            tenure_days = (active[end_col] - active[start_col]).dt.days.clip(lower=0)
            rows.append(
                {
                    entity_col: entity_id,
                    "year": year,
                    "active_roles": len(active),
                    "mean_tenure_days": tenure_days.mean(),
                }
            )

    if not rows:
        return pd.DataFrame(columns=[entity_col, "year", "active_roles", "mean_tenure_days"])

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Outcome merge and temporal lag
# ---------------------------------------------------------------------------


def merge_outcome_panel(
    event_panel: pd.DataFrame,
    outcomes: pd.DataFrame,
    entity_col: str = "entity_id",
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Left-join episodic aggregates onto a numeric outcome panel.

    Parameters
    ----------
    event_panel : pd.DataFrame
        Entity-year episodic aggregates.
    outcomes : pd.DataFrame
        Numeric outcome panel (e.g., financial ratios).
    entity_col, year_col : str
        Join key columns present in both frames.

    Returns
    -------
    pd.DataFrame
        Merged panel; outcome rows without event coverage retain NaN event fields.
    """
    return outcomes.merge(
        event_panel,
        on=[entity_col, year_col],
        how="left",
        validate="m:1",
    )


def apply_entity_lag(
    panel: pd.DataFrame,
    lag_columns: list[str],
    entity_col: str = "entity_id",
    year_col: str = "year",
    lag_periods: int = 1,
) -> pd.DataFrame:
    """
    Apply a within-entity temporal lag using ``.shift(lag_periods)``.

    The panel MUST be sorted by (entity_col, year_col) before calling.

    Parameters
    ----------
    panel : pd.DataFrame
        Entity-year panel (unsorted input is sorted internally).
    lag_columns : list[str]
        Columns to lag (creates ``{col}_lag{lag_periods}`` copies).
    entity_col, year_col : str
        Index columns defining the time series within each entity.
    lag_periods : int
        Number of annual periods to shift (default: 1).

    Returns
    -------
    pd.DataFrame
        Panel with lagged covariate columns appended.
    """
    out = panel.sort_values([entity_col, year_col]).copy()

    for col in lag_columns:
        if col not in out.columns:
            logger.warning("Lag column '%s' not found — skipping", col)
            continue
        lag_name = f"{col}_lag{lag_periods}"
        out[lag_name] = out.groupby(entity_col, sort=False)[col].shift(lag_periods)
        logger.debug("Created lagged column: %s", lag_name)

    return out


# ---------------------------------------------------------------------------
# End-to-end synthesis
# ---------------------------------------------------------------------------


def build_panel_matrix(
    events: pd.DataFrame | None = None,
    outcomes: pd.DataFrame | None = None,
    lag_columns: list[str] | None = None,
    snapshot: date | None = None,
) -> pd.DataFrame:
    """
    Orchestrate episodic consolidation, outcome merge, and 1-year lag.

    Parameters
    ----------
    events, outcomes : pd.DataFrame, optional
        Input tables. Mock data is generated when omitted (for demonstration).
    lag_columns : list[str], optional
        Covariate columns to lag. Defaults to episodic aggregate fields.
    snapshot : date, optional
        Active-record closure date.

    Returns
    -------
    pd.DataFrame
        Analysis-ready entity-year panel matrix.
    """
    events = events if events is not None else _mock_episodic_events()
    outcomes = outcomes if outcomes is not None else _mock_outcome_panel()

    event_panel = episodic_to_annual_panel(events, snapshot=snapshot)
    merged = merge_outcome_panel(event_panel, outcomes)
    lag_columns = lag_columns or ["active_roles", "mean_tenure_days"]
    panel = apply_entity_lag(merged, lag_columns=lag_columns)

    logger.info(
        "Panel matrix: %d rows | %d entities | years %d–%d",
        len(panel),
        panel["entity_id"].nunique(),
        panel["year"].min(),
        panel["year"].max(),
    )
    return panel


def write_panel_matrix(
    panel: pd.DataFrame,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Persist the synthesized panel matrix to compressed Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False)
    logger.info("Wrote panel matrix (%d rows) to %s", len(panel), output_path)
    return output_path


# ---------------------------------------------------------------------------
# Mock inputs for standalone execution
# ---------------------------------------------------------------------------


def _mock_episodic_events() -> pd.DataFrame:
    """Generate illustrative episodic assignment records."""
    return pd.DataFrame(
        {
            "entity_id": ["E001", "E001", "E002", "E003"],
            "individual_id": ["I01", "I02", "I03", "I04"],
            "role_code": ["R1", "R2", "R1", "R1"],
            "start_date": ["2018-06-01", "2020-01-15", "2015-03-01", "2019-01-01"],
            "end_date": ["2022-12-31", "2099-12-31", "2023-06-30", "2099-12-31"],
        }
    )


def _mock_outcome_panel() -> pd.DataFrame:
    """Generate illustrative numeric outcome panel."""
    entities = ["E001", "E002", "E003"]
    years = list(range(2018, 2025))
    rows = [
        {
            "entity_id": e,
            "year": y,
            "outcome_a": np.random.default_rng(42).uniform(0.01, 0.15),
            "outcome_b": np.random.default_rng(43).uniform(0.02, 0.08),
            "log_assets": np.random.default_rng(44).uniform(5, 12),
        }
        for e in entities
        for y in years
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    result = build_panel_matrix()
    write_panel_matrix(result)
