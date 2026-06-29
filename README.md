# Multi-Vendor Corporate Entity Resolution & Longitudinal Panel Architecture

A modular, vendor-agnostic data engineering framework for constructing
analysis-ready entity-year panels from heterogeneous relational warehouses,
independent identifier registries, and episodic longitudinal event streams.

This repository provides **template infrastructure only**. It contains no
commercial data, no subscription credentials, and no proprietary schema
bindings. All SQL, table names, and mock inputs are placeholders intended
for adaptation to your own licensed data environment.

---

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Phase 1        │     │  Phase 2         │     │  Phase 3            │
│  Data Ingestion │────▶│  Entity Mapping  │────▶│  Event Consolidation│
│  (SQL / Batch)  │     │  Chain           │     │                     │
└─────────────────┘     └──────────────────┘     └──────────┬──────────┘
                                                              │
┌─────────────────┐     ┌──────────────────┐                  │
│  Phase 5        │     │  Phase 4         │◀─────────────────┘
│  Inference      │◀────│  Matrix          │
│  Gates          │     │  Synthesis       │
└─────────────────┘     └──────────────────┘
```

| Phase | Module | Output Artifact |
|-------|--------|-----------------|
| 1 — Data Ingestion | `src/data_fetcher.py` | Compressed staging Parquet |
| 2 — Entity Mapping | `src/entity_linker.py` | `firm_universe` crosswalk |
| 3 — Event Consolidation | `src/profile_metrics.py` | Individual-level profiles |
| 4 — Matrix Synthesis | `src/panel_aggregator.py` | Entity-year panel matrix |
| 5 — Inference Gates | External / downstream | Regression-ready frame |

---

## Phase 1: Data Ingestion (SQL / Batch)

**Objective:** Extract bounded slices from a relational warehouse without
triggering rate limits or connection pool exhaustion.

`data_fetcher.py` implements a batched extraction pattern:

1. **Credential hydration** — connection parameters loaded from `.env`
   via `python-dotenv` (see `.env.example`).
2. **Parameterized partition queries** — SQL templates accept a dynamic
   `IN`-clause filter on a partition column (e.g., fiscal year, entity batch).
3. **Explicit throttling** — configurable `time.sleep()` between batch
   executions to respect warehouse concurrency quotas.
4. **Compressed persistence** — each batch or consolidated result is written
   to Snappy-compressed Parquet under `data/staging/`.

**Design constraints:**

- Read-only connections; no DDL or DML in the extraction path.
- Batch size and throttle delay are independently tunable per environment.
- Large tables should stream batch-by-batch to disk rather than accumulate
  in memory.

```bash
cp .env.example .env   # populate DB_HOST, DB_USER, DB_PASSWORD, DB_PORT
python -m src.data_fetcher
```

---

## Phase 2: Entity Mapping Chain

**Objective:** Resolve records across independent identifier registries into
a single canonical `firm_universe` crosswalk.

`entity_linker.py` applies a **sequential fallback key chain**:

| Priority | Key Type | Description |
|----------|----------|-------------|
| 1 | Primary Key | Canonical internal entity identifier |
| 2 | Alternative ID | Secondary registry crosswalk code |
| 3 | Asset Ticker | Public-market trading symbol |

Each resolved row is annotated with `match_tier` (linkage confidence) and
`match_key` (which key type succeeded). Unmatched seed records are retained
with `match_tier = 99` for downstream audit and manual reconciliation.

Registries are merged sequentially against an anchor table. The first
successful match at the highest available tier wins; lower-tier attempts
are skipped for already-resolved entities.

```bash
python -m src.entity_linker
# → data/derived/firm_universe.parquet
```

---

## Phase 3: Longitudinal Event Consolidation

**Objective:** Transform irregular episodic assignment records into
distributional behavioral traces at the individual level.

`profile_metrics.py` operates as a **standalone worker** over longitudinal
event histories. For each individual it computes:

- **Average event interval** — mean tenure duration (days) across all
  observed assignment arcs.
- **Duration variance** — coefficient of variation (σ / μ) of interval
  lengths, capturing pacing regularity independent of scale.
- **Categorical transition frequencies** — empirical role-to-role movement
  probabilities ordered chronologically.

Interval durations are floored and capped to suppress data-quality artifacts
from missing dates or extreme outliers.

```bash
python -m src.profile_metrics
# → data/derived/individual_profiles.parquet
```

---

## Phase 4: Matrix Synthesis

**Objective:** Collapse episodic events and numeric outcomes into a
rectangular entity-year panel matrix with temporally consistent covariates.

`panel_aggregator.py` performs four operations in sequence:

1. **Open-ended date closure** — sentinel far-future end dates (≥ 2099-01-01)
   are replaced with a dynamic snapshot date (defaults to today), treating
   active assignments as censored at the observation window boundary.
2. **Interval overlap aggregation** — for each fiscal year-end snapshot,
   count active roles and compute mean tenure among overlapping intervals.
3. **Outcome panel merge** — left-join episodic aggregates onto a standard
   numeric outcome table (financial ratios, scale controls, etc.).
4. **Cross-sectional temporal lag** — within-entity `.shift(1)` applied after
   mandatory `(entity_id, year)` sort, producing `*_lag1` covariate columns
   aligned to a strict t−1 → t identification strategy.

```bash
python -m src.panel_aggregator
# → data/derived/panel_matrix.parquet
```

---

## Phase 5: Econometric Inference Gates

**Objective:** Enforce specification discipline before panel regression.

This phase is intentionally **not implemented** in the template repository.
Downstream analysis scripts should apply the following gates to the Phase 4
output before estimation:

| Gate | Rule |
|------|------|
| Sort order | `(entity_id, year)` ascending before any `.shift()` |
| Lag alignment | Independent variables at *t−1*; dependent variables at *t* |
| Missing outcomes | Do not impute null outcome values with zero |
| Estimator | `linearmodels.panel.PanelOLS` with entity + time fixed effects |
| Covariance | Heteroskedasticity-robust (`cov_type='robust'`) |
| Index | Multi-index on `['entity_id', 'year']` |

Example specification skeleton (adapt column names to your schema):

```python
from linearmodels.panel import PanelOLS

panel = panel.set_index(["entity_id", "year"]).sort_index()
formula = "outcome_a ~ covariate_lag1 + log_assets + EntityEffects + TimeEffects"
model = PanelOLS.from_formula(formula, data=panel.dropna(subset=["outcome_a"]))
result = model.fit(cov_type="robust")
```

---

## Repository Layout

```
.
├── .env.example              # Warehouse credential placeholders
├── .gitignore                # Strict data-artifact exclusions
├── requirements.txt          # Python dependencies
├── README.md
└── src/
    ├── __init__.py
    ├── data_fetcher.py       # Phase 1: batched SQL extraction
    ├── entity_linker.py      # Phase 2: fallback key-chain resolution
    ├── profile_metrics.py    # Phase 3: individual behavioral traces
    └── panel_aggregator.py   # Phase 4: entity-year matrix synthesis
```

---

## Security & Data Handling

- **Never commit** `.env`, raw extracts, or derived Parquet/CSV files.
  `.gitignore` enforces exclusions for `*.csv`, `*.parquet`, `*.txt`,
  `*.rds`, and virtual environments.
- Staging and derived artifacts are written to `data/staging/` and
  `data/derived/` locally; these directories are not tracked.
- Replace all SQL templates and mock tables with your own licensed schema
  definitions before connecting to a production warehouse.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # fill in warehouse credentials
```

Run modules individually in phase order, or chain them in a Makefile /
orchestration script of your own design.

---

## License

Template code is provided for educational and infrastructure scaffolding
purposes. Data rights, vendor licenses, and downstream publication
obligations remain the responsibility of the end user.
