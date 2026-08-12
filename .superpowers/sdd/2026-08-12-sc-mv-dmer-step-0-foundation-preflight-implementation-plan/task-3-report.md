# Task 3 Report: immutable run and evidence lifecycle

## Scope

Implemented only the E0 lifecycle foundation in
`E:/SC-MV-DMER/.worktrees/e0-minimum-bootstrap`: immutable run records,
historical Gate evidence records, append-only invalidation/supersession, and
effective validity resolution. No model, data, MERT, real run, or Gate
evaluation was executed. No research/design/plan document was changed.

## RED/GREEN evidence

The initial lifecycle/validity tests were written before their production
modules. The task-local Python 3.10.11 command produced the expected RED:

```text
E   ModuleNotFoundError: No module named 'sc_mv_dmer.foundation.lifecycle'
2 collection errors
```

After the minimal lifecycle implementation, the same focused command passed:

```text
.venv\\Scripts\\python.exe -m pytest tests\\foundation\\test_run_lifecycle.py tests\\foundation\\test_effective_validity.py -q
10 passed in 0.27s
```

Schema parsing was then added test-first and produced RED because
`schemas/run_spec.schema.json` did not yet exist. The four strict JSON schemas
were added and the focused suite passed. A final serialization regression test
also went RED on `mappingproxy` JSON serialization, then GREEN after replacing
it with a JSON-serializable immutable mapping.

Final verification with the task-local interpreter:

```text
.venv\\Scripts\\python.exe -m pytest tests\\foundation\\test_run_lifecycle.py tests\\foundation\\test_effective_validity.py -q
12 passed in 0.23s

.venv\\Scripts\\python.exe -m pytest -q
29 passed in 0.23s

git diff --check
exit 0
```

Each of the four schemas also parsed through `python -m json.tool`.

## Fix round 1/5

Each reviewer finding was addressed test-first under the task-local Python
3.10.11 environment:

- Run mode RED: a `RunSpec` test failed because the raw string had no
  `RunMode` value. GREEN reuses `config.RunMode` and makes the schema enum
  exactly `formal` / `debug`.
- Manual-authority RED: new typed-metadata tests could not import the absent
  models. GREEN adds discriminated automatic/manual metadata. Manual `PASS`
  rejects pending/missing approval, and approved manual authority requires a
  non-Codex/non-automatic reviewer identity plus immutable approval-artifact
  SHA-256. `PENDING` remains allowed without approval.
- Lifecycle-store RED: the new explicit-store test could not import
  `LifecycleStore`. GREEN rejects repeated `run_id` registration both while
  active and after terminal sealing, preventing two valid handles from being
  finalized independently.
- Multi-predecessor RED: the result reported only `source-a` when two direct
  predecessors were invalidated. GREEN aggregates every invalidated source and
  every dependency path in `dependency_paths`.

Verification after the fixes:

```text
.venv\\Scripts\\python.exe -m pytest tests\\foundation\\test_run_lifecycle.py tests\\foundation\\test_effective_validity.py -q
20 passed in 0.22s

.venv\\Scripts\\python.exe -m pytest -q
37 passed in 0.25s
```

## Files

- `src/sc_mv_dmer/foundation/manifests.py`
- `src/sc_mv_dmer/foundation/lifecycle.py`
- `src/sc_mv_dmer/foundation/validity.py`
- `src/sc_mv_dmer/foundation/__init__.py`
- `schemas/run_spec.schema.json`
- `schemas/run_manifest.schema.json`
- `schemas/invalidation_record.schema.json`
- `schemas/gate_evaluation.schema.json`
- `tests/foundation/test_run_lifecycle.py`
- `tests/foundation/test_effective_validity.py`

## Commit

- `ad57fdcd89aba88f5c16b2f13928c531fee5a394` `feat: enforce immutable run and evidence lifecycle`

## Self-review

- Finalization seals an in-memory `ActiveRun` exactly once, accepts only the
  three terminal states, and has no path to reopen a terminal record.
- Continuation provenance requires a distinct `RunSpec.run_id` and records
  parent/continuation/checkpoint references rather than reopening history.
- Nested mapping/list aliases are copied into immutable values; the records
  remain JSON serializable.
- Gate verdicts and manual/PENDING human-criterion metadata are historical
  data only; no code assigns approval. Manual PASS is fail-closed until typed
  reviewer identity and immutable approval-artifact checksum are supplied.
- Run identity ownership is explicit through `LifecycleStore`; attempts to
  start an already active or terminal ID fail before another handle exists.
- Registry mutation is append-only. Direct invalidation/supersession marks the
  source invalid while descendants become stale without rewriting their stored
  records; unaffected branches stay valid.
- Unknown nodes and dependency cycles fail closed. Active Gate selection
  excludes superseded historical attempts, so an old PASS cannot unlock a
  future definition.

## Concerns

No blocking concerns. The `Registry` is intentionally in-memory; durable
storage, real Gate execution, and any domain-specific policies remain outside
this E0 task.
