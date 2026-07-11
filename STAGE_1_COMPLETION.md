# Stage 1 completion record

## Implemented

Configurable discovery, schema inspection, typed contracts, alias filtering, cardinality-safe composite joins, numeric parsing, robust normalization, smell/component/interaction features, 21 smell–recommendation compatibilities across 17 taxonomy entries, deterministic candidate generation, transparent applicability/benefit/risk/priority scoring, three ranking levels, diversity/MMR selection, CSV/JSON/HTML reporting, Typer CLI, scripts, fixtures, and unit/integration tests.

## Detected CSV schema and joins

- Affects: 10 columns; identity/edge fields include `project`, `versionId`, `fromId`, `toId`, `edgeId`.
- Metrics: 22 columns; component identifier is `vertexId`, with `name` and the required component metrics.
- Smells: 33 columns; smell identifier is `vertexId`, type is `smellType`, with the specified smell characteristics.
- Join keys: `(project, versionId, smells.vertexId=affects.fromId)`, followed by `(project, versionId, affects.toId=metrics.vertexId)`.
- Duplicate composite entity keys observed: 0 in smells and metrics. Duplicate affects edge IDs: 0.

The generated `outputs/processed/arcan_schema_report.json` is the authoritative machine-readable report once the CLI is run with dependencies installed.

## Real selected-smell counts

- `godComponent`: 1
- `unstableDep`: 12
- `hubLikeDep`: 12
- Excluded `cyclicDep`: 692

There are 25 selected smell records and 692 excluded records. The affects export contains 1, 12, and 12 respective selected edges.

## Validation performed here

- Inspected headers and sample values with Python's standard CSV library.
- Counted rows, smell types, and composite duplicates without changing inputs.
- Compiled all Python files with `compileall`.
- Dependency-based test and smoke commands are listed below; this development environment initially contained none of the declared Python packages, and no package download was authorized or performed.

## Known limitations

The baseline cannot infer code-level cohesion, API usage, deployment boundaries, or consistency needs from Arcan component exports. Missing evidence lowers confidence/risk interpretation but does not fabricate values. Recommendations require human validation. Stage 1 does no automatic weight tuning.

## Commands for the later HPC environment

These are CPU preparation/baseline commands; training belongs to later stages and must use the university's approved Slurm environment and job definitions, which Stage 1 intentionally does not invent.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
dsarp inspect --data-dir data --output-dir outputs
dsarp run-stage1 --data-dir data --output-dir outputs --top-k 20
pytest
ruff check .
mypy src/dsarp
```
