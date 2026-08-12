# Task 6 report: E0 minimum bootstrap qualification

## RED → GREEN

- RED: `./.venv/Scripts/python.exe -m pytest tests/foundation/test_formal_preflight.py tests/foundation/test_step0_qualification.py -q` exited 1 because `sc_mv_dmer.foundation.preflight` did not exist.
- GREEN: focused preflight/qualification tests passed after minimal implementation.
- Supersession RED: the CLI rejected `--supersedes` and `--correction-reason`; the new focused test failed before those arguments were added.
- Trust-boundary RED: a caller-crafted `MertUpstreamStatus(... local_verification_status="VERIFIED")` yielded `READY`; the regression test failed before qualification was changed to invoke code-controlled `bind_mert_formal`.
- Final focused command: `./.venv/Scripts/python.exe -m pytest tests/foundation/test_formal_preflight.py tests/foundation/test_step0_qualification.py -q` passed with `15 passed in 0.43s`.
- Fresh final verification after evidence commit: focused `15 passed in 0.39s`; full suite `82 passed in 0.51s`; `git diff --check 5e59d05^..HEAD` exited 0.

## Commits and evidence

- `f34a0d63d6e6aa578f263fd94a7667adcff4e99d` — `feat: qualify E0 minimum bootstrap`
- `4a2d31a5cf94126f618b9299859bf38c95957dce` — `fix: preserve qualification supersession`
- `3e9378f6a1f7601cdf41137b428667eb126cec8e` — `docs: record E0 minimum qualification` (historical v1, which references the then-current v2 MERT status)
- `5e59d05d3fab3333224e0a265a2b6b0d24dd96f7` — `fix: enforce qualification trust boundaries`
- `evidence/qualifications/step-0-minimum-bootstrap-v2.json` is the current append-only successor. It supersedes v1 and references Task5's v3 MERT status.

## Current exact qualification state

- `e0_implementation_status = COMPLETE`
- `e1_1_readiness = BLOCKED`
- `formal_execution_ready = false`
- `paper_eligible = false`
- The v2 bundle was produced from a clean pre-write observation at implementation commit `5e59d05d3fab3333224e0a265a2b6b0d24dd96f7`.
- Current E1.1 block reasons are missing local `config`, `processor`, and `weights`, plus `MERT_FORMAL_BINDING_FAILED`: the code-controlled checksum registry is incomplete (`EXPECTED_CHECKSUMS_UNREGISTERED`).
- The full M0–M20 frozen-contract blocker is recorded as deferred beyond E1 minimum, not as an E0 implementation failure.

## Self-review

- `qualify-step` reads only artifacts/config/Git; it does not import or execute a model forward path.
- E1.1 consumes exactly the E0 predecessor and upstream/probe/probe-representation prerequisites. Full split, annotation, PMEmo, Sensor, Qwen, and CoT remain outside this predecessor set.
- Terminal qualification writes use atomic exclusive creation. Corrections are separate paths with explicit supersession metadata.
- A caller-provided MERT status is independently subjected to code-controlled formal binding; forged `VERIFIED` cannot make E1.1 ready.

## Review fix round 1

- RED: the strict schema omitted fields emitted by the qualification bundle, and `qualify-step` supplied a hard-coded four-`VALID` mapping. New tests failed for missing schema fields and for source/manifest verification that the old production API could not perform.
- GREEN: `./.venv/Scripts/python.exe -m pytest tests/foundation/test_formal_preflight.py tests/foundation/test_step0_qualification.py -q` passed after all prerequisite validity became derived from observed input only.
- `0b3891d7f9d824e1f1575d8d1f4af345c70207de` records the code fix. Schema properties and required fields now exactly match the immutable bundle model, with evidence key/const/enum/hash checks in focused tests.
- `step-0-minimum-bootstrap-v3.json` is append-only, supersedes v2, and was generated from clean formal observation plus `E:\DEAM` source checksum revalidation and MERT status v3. Its current status remains `COMPLETE/BLOCKED/false/false`; upstream MERT identity is `NOT_VALID` because the code-controlled registry is incomplete.

## Review fix round 2: schema evolution

- RED: no version-addressable legacy schema or artifact schema selector existed; historical v1/v2/v3 could only be compared against the mutable current schema.
- GREEN: `e0_minimum_qualification.v1.schema.json` preserves the strict 1.0 contract for historical evidence; the current Pydantic bundle and current schema are 1.1. The selector maps each artifact by `schema_version`, and tests validate v1/v2/v3 through the legacy schema without rewriting it.
- `e751dc13a22c676a0932593783c58c12eca75168` records schema evolution. `step-0-minimum-bootstrap-v4.json` is an append-only 1.1 successor to v3, generated from clean formal observation, `E:\DEAM` probe revalidation, and MERT status v3.
