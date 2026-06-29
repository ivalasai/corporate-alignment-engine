"""
Multi-source entity resolution via sequential fallback key chain.

Resolves records from heterogeneous identifier systems into a unified
``firm_universe`` mapping table using a three-tier linkage strategy:

  1. Primary Key     — canonical internal entity identifier
  2. Alternative ID  — secondary registry crosswalk
  3. Asset Ticker    — public-market symbol (lowest precedence)

Unmatched rows are retained with a ``match_tier`` flag for downstream audit.
"""

from __future__ import annotations

import logging
from enum import IntEnum
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = Path("data/derived/firm_universe.parquet")


class MatchTier(IntEnum):
    """Linkage precedence tier (lower value = higher confidence)."""

    PRIMARY_KEY = 1
    ALTERNATIVE_ID = 2
    ASSET_TICKER = 3
    UNMATCHED = 99


# ---------------------------------------------------------------------------
# Mock source tables (replace with parquet/csv reads from staging layer)
# ---------------------------------------------------------------------------


def load_mock_source_tables() -> dict[str, pd.DataFrame]:
    """
    Return in-memory mock tables representing three independent registries.

    In production, each table is loaded from the compressed staging outputs
    produced by ``data_fetcher.py``.
    """
    registry_a = pd.DataFrame(
        {
            "primary_key": ["E001", "E002", "E003", "E004"],
            "alt_id": ["A100", "A200", None, "A400"],
            "ticker": ["TKR1", None, "TKR3", "TKR4"],
            "entity_name": ["Alpha Corp", "Beta Inc", "Gamma LLC", "Delta Co"],
        }
    )

    registry_b = pd.DataFrame(
        {
            "alt_id": ["A100", "A200", "A300", "A500"],
            "primary_key": ["E001", "E002", "E005", None],
            "ticker": [None, "TKR2", "TKR3", None],
            "source_label": ["reg_b"] * 4,
        }
    )

    registry_c = pd.DataFrame(
        {
            "ticker": ["TKR1", "TKR2", "TKR3", "TKR6"],
            "primary_key": ["E001", "E006", "E003", None],
            "alt_id": [None, "A600", None, "A700"],
            "market_cap_bucket": ["large", "mid", "large", "small"],
        }
    )

    return {"registry_a": registry_a, "registry_b": registry_b, "registry_c": registry_c}


# ---------------------------------------------------------------------------
# Fallback linkage helpers
# ---------------------------------------------------------------------------


def _merge_on_key(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key: str,
    tier: MatchTier,
    suffix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Left-merge on ``key`` and annotate matched rows with linkage tier.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (matched_subset, unmatched_left_rows)
    """
    right_cols = [c for c in right.columns if c not in left.columns or c == key]
    merged = left.merge(
        right[right_cols].drop_duplicates(subset=[key]),
        on=key,
        how="left",
        suffixes=("", f"_{suffix}"),
        indicator=True,
    )

    matched_mask = merged["_merge"] == "both"
    matched = merged.loc[matched_mask].copy()
    unmatched = merged.loc[~matched_mask, left.columns].copy()

    if not matched.empty:
        matched["match_tier"] = int(tier)
        matched["match_key"] = key

    return matched, unmatched


def resolve_entity(
    seed: pd.DataFrame,
    crosswalk: pd.DataFrame,
    seed_id_col: str = "primary_key",
) -> pd.DataFrame:
    """
    Resolve a seed registry against a crosswalk using the fallback key chain.

    Parameters
    ----------
    seed : pd.DataFrame
        Base entity table (one row per canonical record).
    crosswalk : pd.DataFrame
        Secondary registry with overlapping identifier columns.
    seed_id_col : str
        Column in ``seed`` treated as the canonical primary key.

    Returns
    -------
    pd.DataFrame
        Unified rows with ``match_tier`` and ``match_key`` annotations.
    """
    key_chain: list[tuple[str, MatchTier]] = [
        ("primary_key", MatchTier.PRIMARY_KEY),
        ("alt_id", MatchTier.ALTERNATIVE_ID),
        ("ticker", MatchTier.ASSET_TICKER),
    ]

    resolved_frames: list[pd.DataFrame] = []
    remaining = seed.copy()

    for key, tier in key_chain:
        if remaining.empty or key not in remaining.columns or key not in crosswalk.columns:
            continue

        # Only attempt linkage on rows where the key is non-null
        has_key = remaining[key].notna()
        attempt = remaining.loc[has_key]
        skip = remaining.loc[~has_key]

        matched, unmatched = _merge_on_key(attempt, crosswalk, key, tier, suffix=key)
        if not matched.empty:
            resolved_frames.append(matched)
        remaining = pd.concat([unmatched, skip], ignore_index=True)

    if not remaining.empty:
        remaining = remaining.copy()
        remaining["match_tier"] = int(MatchTier.UNMATCHED)
        remaining["match_key"] = None
        resolved_frames.append(remaining)

    if not resolved_frames:
        return pd.DataFrame()

    return pd.concat(resolved_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def build_firm_universe(
    sources: dict[str, pd.DataFrame] | None = None,
    anchor_registry: str = "registry_a",
) -> pd.DataFrame:
    """
    Sequentially merge all source registries into a unified firm universe.

    Parameters
    ----------
    sources : dict[str, pd.DataFrame], optional
        Named registry tables. Defaults to mock data.
    anchor_registry : str
        Registry used as the linkage seed (highest identifier precedence).

    Returns
    -------
    pd.DataFrame
        Deduplicated firm_universe with canonical ``entity_id`` column.
    """
    sources = sources or load_mock_source_tables()
    seed = sources[anchor_registry].copy()
    seed = seed.rename(columns={"primary_key": "entity_id"})

    all_resolved: list[pd.DataFrame] = [seed.assign(match_tier=MatchTier.PRIMARY_KEY, match_key="primary_key")]

    for name, table in sources.items():
        if name == anchor_registry:
            continue
        logger.info("Linking registry: %s", name)
        linked = resolve_entity(seed, table)
        all_resolved.append(linked)

    universe = pd.concat(all_resolved, ignore_index=True)
    universe = universe.drop_duplicates(subset=["entity_id"], keep="first")
    universe = universe.sort_values(["entity_id", "match_tier"]).reset_index(drop=True)

    logger.info(
        "Firm universe: %d entities | tier distribution:\n%s",
        universe["entity_id"].nunique(),
        universe["match_tier"].value_counts().to_string(),
    )
    return universe


def write_firm_universe(
    universe: pd.DataFrame,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Persist the firm universe mapping to compressed Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    universe.to_parquet(output_path, index=False)
    logger.info("Wrote firm universe (%d rows) to %s", len(universe), output_path)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    result = build_firm_universe()
    write_firm_universe(result)
