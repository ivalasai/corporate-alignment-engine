"""
Relational warehouse batch extractor with throttled query execution.

Demonstrates a generic pattern for:
  - Loading credentials from environment variables
  - Opening a PostgreSQL (or PostgreSQL-compatible) connection
  - Executing parameterized SQL in bounded batches
  - Enforcing explicit inter-batch delays to respect warehouse quotas
  - Persisting results to compressed columnar storage on local disk

This module is intentionally vendor-agnostic. Replace SQL templates and table
references with your own schema definitions before use.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extensions import connection as PgConnection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path("data/staging")
DEFAULT_BATCH_SIZE = 10_000
DEFAULT_THROTTLE_SECONDS = 1.5
DEFAULT_COMPRESSION = "snappy"


@dataclass(frozen=True)
class WarehouseConfig:
    """Connection and runtime parameters for the relational warehouse."""

    host: str
    user: str
    password: str
    port: int = 5432
    database: str = "analytics"
    batch_size: int = DEFAULT_BATCH_SIZE
    throttle_seconds: float = DEFAULT_THROTTLE_SECONDS
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)

    @classmethod
    def from_env(cls, env_path: str | Path | None = None) -> "WarehouseConfig":
        """
        Hydrate configuration from process environment.

        Expected variables: DB_HOST, DB_USER, DB_PASSWORD, DB_PORT (optional).
        """
        load_dotenv(env_path)
        host = os.getenv("DB_HOST", "")
        user = os.getenv("DB_USER", "")
        password = os.getenv("DB_PASSWORD", "")
        port = int(os.getenv("DB_PORT", "5432"))

        if not all([host, user, password]):
            raise ValueError(
                "Missing warehouse credentials. Copy .env.example to .env and "
                "populate DB_HOST, DB_USER, and DB_PASSWORD."
            )

        return cls(host=host, user=user, password=password, port=port)


@dataclass(frozen=True)
class ExtractionJob:
    """Descriptor for a single batched table extraction."""

    name: str
    sql_template: str
    partition_column: str
    partition_values: Sequence[Any]
    params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------


def open_connection(config: WarehouseConfig) -> PgConnection:
    """
    Establish a read-only PostgreSQL connection.

    Parameters
    ----------
    config : WarehouseConfig
        Resolved connection parameters.

    Returns
    -------
    psycopg2.extensions.connection
        Open database connection (caller must close).
    """
    logger.info("Connecting to warehouse at %s:%s", config.host, config.port)
    return psycopg2.connect(
        host=config.host,
        user=config.user,
        password=config.password,
        port=config.port,
        dbname=config.database,
    )


# ---------------------------------------------------------------------------
# Batched extraction with throttling
# ---------------------------------------------------------------------------


def _iter_batch_queries(
    job: ExtractionJob,
    batch_size: int,
) -> Iterator[tuple[str, list[Any]]]:
    """
    Yield (sql, bind_params) pairs for each partition slice.

    The SQL template must contain a ``{partition_filter}`` placeholder that
    expands to a parameterized IN-clause fragment.
    """
    values = list(job.partition_values)
    for offset in range(0, len(values), batch_size):
        chunk = values[offset : offset + batch_size]
        placeholders = ", ".join(["%s"] * len(chunk))
        partition_filter = f"{job.partition_column} IN ({placeholders})"
        sql = job.sql_template.format(partition_filter=partition_filter)
        yield sql, list(chunk)


def execute_batched_extraction(
    conn: PgConnection,
    job: ExtractionJob,
    config: WarehouseConfig,
) -> pd.DataFrame:
    """
    Run a batched extraction job with explicit inter-batch throttling.

    Each batch executes independently; results are concatenated in memory.
    For very large tables, call ``write_batch_to_parquet`` per batch instead.

    Parameters
    ----------
    conn : PgConnection
        Active database connection.
    job : ExtractionJob
        Job descriptor including SQL template and partition keys.
    config : WarehouseConfig
        Runtime configuration (batch size, throttle delay).

    Returns
    -------
    pd.DataFrame
        Concatenated result across all batches.
    """
    frames: list[pd.DataFrame] = []
    batches = list(_iter_batch_queries(job, config.batch_size))
    total = len(batches)

    for idx, (sql, bind_params) in enumerate(batches, start=1):
        logger.info("[%s] Executing batch %d/%d", job.name, idx, total)
        frame = pd.read_sql_query(sql, conn, params=bind_params)
        frames.append(frame)

        if idx < total:
            logger.debug(
                "[%s] Throttling %.1fs before next batch",
                job.name,
                config.throttle_seconds,
            )
            time.sleep(config.throttle_seconds)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def write_batch_to_parquet(
    df: pd.DataFrame,
    output_path: Path,
    compression: str = DEFAULT_COMPRESSION,
) -> Path:
    """
    Persist a DataFrame to compressed Parquet on local disk.

    Parameters
    ----------
    df : pd.DataFrame
        Batch result to persist.
    output_path : Path
        Destination file path (parent directories are created if missing).
    compression : str
        Parquet compression codec (default: snappy).

    Returns
    -------
    Path
        Resolved output path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, compression=compression)
    logger.info("Wrote %d rows to %s", len(df), output_path)
    return output_path


# ---------------------------------------------------------------------------
# Sample job definitions (replace with your schema)
# ---------------------------------------------------------------------------

SAMPLE_JOBS: list[ExtractionJob] = [
    ExtractionJob(
        name="numeric_outcomes",
        sql_template="""
            SELECT
                entity_id,
                fiscal_year,
                metric_a,
                metric_b,
                total_assets
            FROM staging.numeric_outcomes
            WHERE {partition_filter}
        """,
        partition_column="fiscal_year",
        partition_values=list(range(2000, 2025)),
    ),
    ExtractionJob(
        name="episodic_events",
        sql_template="""
            SELECT
                individual_id,
                entity_id,
                role_code,
                start_date,
                end_date
            FROM staging.role_events
            WHERE {partition_filter}
        """,
        partition_column="entity_id",
        partition_values=[],  # populate from a prior entity-universe query
    ),
]


def run_all_jobs(config: WarehouseConfig | None = None) -> dict[str, Path]:
    """
    Orchestrate sample extraction jobs and write compressed staging files.

    Returns
    -------
    dict[str, Path]
        Mapping of job name to output file path.
    """
    config = config or WarehouseConfig.from_env()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    with open_connection(config) as conn:
        for job in SAMPLE_JOBS:
            if not job.partition_values:
                logger.warning("[%s] Skipping — no partition values defined", job.name)
                continue

            df = execute_batched_extraction(conn, job, config)
            out_path = config.output_dir / f"{job.name}.parquet"
            write_batch_to_parquet(df, out_path)
            outputs[job.name] = out_path

    return outputs


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    run_all_jobs()
