# Task 2 report: canonical config and stable identity

## Scope

Implemented only the E0 Task 2 foundation interfaces in `E:/SC-MV-DMER/.worktrees/e0-minimum-bootstrap`. No MERT loading/forward, data construction, downstream-dimension work, or research/design/plan document changes were made.

## RED/GREEN evidence

RED was run before production files existed:

```text
& .\.venv\Scripts\python.exe -m pytest tests\foundation -q
ModuleNotFoundError: No module named 'sc_mv_dmer.foundation'
2 collection errors
```

The failing tests specified canonical UTF-8 JSON and finite-number rejection, identity determinism/empty-component rejection, dual config hashing, formal and debug override policy, unknown-key rejection, and snapshot provenance/schema metadata.

GREEN after the minimal implementation:

```text
& .\.venv\Scripts\python.exe -m pytest tests\foundation -q
16 passed in 0.19s

& .\.venv\Scripts\python.exe -m pytest -q
17 passed in 0.18s

git diff --check
exit 0

& .\.venv\Scripts\python.exe -m json.tool schemas\resolved_config.schema.json > $null
exit 0
```

The task-local interpreter was verified as Python 3.10.11 and pytest 8.3.5.

## Files

- `src/sc_mv_dmer/foundation/__init__.py`
- `src/sc_mv_dmer/foundation/canonical.py`
- `src/sc_mv_dmer/foundation/config.py`
- `src/sc_mv_dmer/foundation/identity.py`
- `configs/research/defaults.yaml`
- `schemas/resolved_config.schema.json`
- `tests/foundation/test_canonical_config.py`
- `tests/foundation/test_stable_identity.py`

## Commit

Focused commit message: `feat: add canonical config and stable identity`.

## Self-review

- Canonical hashing sorts keys, emits compact non-ASCII-safe JSON, normalizes enums, and rejects non-finite floats.
- Stable IDs use a namespace-bound canonical SHA-256 payload and reject blank namespace/components.
- Semantic identity excludes runtime/local-path execution fields, while the resolved hash uses the complete merged configuration.
- YAML layers preserve merge order and file hashes with repository-relative provenance; unknown keys are rejected against the defaults shape.
- Formal CLI overrides are limited to the requested runtime allowlist; debug semantic changes are explicitly not paper-eligible.
- The snapshot schema is valid JSON, declares hashes and provenance, and rejects unknown top-level fields.

## Concerns

No blocking concerns. This foundation intentionally validates only the E0 configuration boundary; later approved work must add the research-specific cross-field invariants without silently changing frozen semantics.
