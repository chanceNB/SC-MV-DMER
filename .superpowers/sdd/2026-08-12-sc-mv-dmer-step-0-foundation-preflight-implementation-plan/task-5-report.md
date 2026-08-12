# Task 5 Report: real DEAM probe and pinned MERT identity

## RED/GREEN

The new DEAM and MERT boundary tests were written first and produced RED:

```text
ModuleNotFoundError: No module named 'sc_mv_dmer.data'
ModuleNotFoundError: No module named 'sc_mv_dmer.features'
```

After the minimal read-only implementation, focused tests passed:

```text
.venv\\Scripts\\python.exe -m pytest tests\\data\\test_deam_probe.py tests\\features\\test_mert_upstream.py -q
7 passed in 0.22s
```

The CLI registration test was added test-first and went RED with:

```text
TypeError: main() takes 0 positional arguments but 1 was given
```

It then passed after adding explicit runtime `--data-root` probe registration
and deterministic no-overwrite artifact writing.

Final task-local Python 3.10.11 verification:

```text
.venv\\Scripts\\python.exe -m pytest -q
64 passed in 0.34s

git diff --check
exit 0
```

## Real artifact facts

Only the allowed read-only DEAM registration ran against runtime root
`E:/DEAM`; no absolute root occurs in the tracked manifest semantic identity.

- Probe: `manifests/probes/deam-e1-real-45s-v1.json`
- Selected source-relative path: `DEAM_audio/MEMD_audio/2.mp3`
- Source SHA-256: `e63489839b957bbd5a6d8b073bacdcb64b7a6b2883802c10c8148a93219dce7b`
- Source size: `600696` bytes; independently read Windows media duration: `45.0` seconds.
- Contract: 45 s, 24,000 Hz, 1,080,000 samples; `SPECIFIED_NOT_EXECUTED`.

Pinned MERT status is at `reports/preflight/mert-primary-local-status.json`:

- `m-a-p/MERT-v1-95M` at `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`, license `cc-by-nc-4.0`.
- Required config/processor/weights local bytes are absent; status is
  `BLOCKED_MISSING_LOCAL_BYTES`. No download, load, forward, checksum-incomplete
  formal bind, FPS/T_raw, TimesNet, downstream derivation, or RG-01 evaluation
  occurred.

## Self-review and blocker

Stable IDs use logical/versioned identity and source-relative paths; relocation
cannot affect them. Source mutation changes source SHA-256. MERT binding rejects
floating revisions and missing role inventory, but accepts a complete local
fixture without making local paths part of upstream identity.

Frozen-contract blocker: no verified local MERT snapshot exists, so formal MERT
binding remains correctly blocked pending provision of all pinned local bytes.

## Commit and workspace state

Implementation commit: `0560447659fa49879e6d980317c24e9b11f88745`
`feat: register E1 probe and MERT identity`.

`git diff --check` exited 0 before the focused commit. Task 5 staging used an
explicit scoped path list and did not include concurrent Task 4 capability or
stage changes.
