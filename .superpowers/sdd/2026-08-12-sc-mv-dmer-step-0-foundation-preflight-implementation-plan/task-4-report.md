# Task 4 report: E1-minimum capability foundation

## RED → GREEN evidence

- RED: `./.venv/Scripts/python.exe -m pytest tests/foundation/test_capability_compiler.py tests/foundation/test_stage_authority.py -q` exited 1 at collection because `sc_mv_dmer.foundation.capabilities` did not exist.
- GREEN: the same focused command exited 0 with `11 passed in 0.27s` after the minimal implementation.
- Fresh final suite: `./.venv/Scripts/python.exe -m pytest -q` exited 0 with `49 passed in 0.30s`.
- Diff integrity: `git diff --check` exited 0.

## Delivered files

- `src/sc_mv_dmer/foundation/capabilities.py`: immutable profile/context/compiled-manifest models and fail-closed E1 compiler.
- `src/sc_mv_dmer/foundation/stages.py`: immutable E1-minimum stage authority rows and frozen-contract coverage audit.
- `configs/catalog/variants.yaml` and `configs/catalog/stages.yaml`: versioned, scoped E1-only catalogs.
- `schemas/variant_capability_profile.schema.json`, `schemas/compiled_execution_capability.schema.json`, and `schemas/stage_authority.schema.json`: machine-readable strict root schemas.
- `tests/foundation/test_capability_compiler.py` and `tests/foundation/test_stage_authority.py`: compiler, capability, immutability, catalog scope, and authority coverage tests.

## Self-review

- The catalog contains exactly one `E1_MERT_PROBE/E1-v1` profile and exactly E0/E1.1 authority rows.
- E0 does not activate forward, derivation, RG-01, training, or any late module.
- E1.1 activates only pinned identity/probe-representation plus MERT real-forward measurement; downstream derivation and RG-01 remain inactive.
- Requests for M0/M20 authority raise `IMPLEMENTATION_BLOCKED_BY_FROZEN_CONTRACT: SGA-M0M20-EV-3R+`.
- No model, dataset, training, Gate execution, or qualification was performed.

## Concerns

- The E1.1 row is an authority declaration only. It deliberately does not establish E0 readiness, verify pinned artifacts, enter the stage, or run a MERT forward.
- Full M0–M20/A1–A13 publication remains intentionally unavailable and fails closed.

## Commit

Commit message: `feat: compile E1 minimum capabilities`.
