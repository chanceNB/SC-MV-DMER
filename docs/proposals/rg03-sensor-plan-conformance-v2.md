# RG-03 Sensor plan-conformance v2 proposal

This is a proposal only. It does not alter v3 artifacts, the historical formal
FAIL, thresholds, validation membership or model architecture identity.

## Evidence-based diagnosis

The current implementation is **not a faithful test of the intended research
plan Mode Sensor**. The strongest confirmed mismatch is the Mode pseudo-label:
the code reduces two fixed binary pitch masks to `pc4 - pc3 - pc10`, rather than
Krumhansl–Schmuckler tonic-template correlation. The Chroma frontend is also a
custom FFT projection rather than the specified `librosa.feature.chroma_cqt`.
The current encoder/head is `12/40/128→64` plus single linear heads, while the
plan specifies Chroma `12→256→1-layer Transformer→256→64→head`.

## Versioned repair

| Item | Required successor | v3 treatment |
|---|---|---|
| K-S math and KeyExtractor details | New Decision; freeze all numeric/template, tie, finite and segment rules | Preserve v3 as historical immutable |
| Pseudo-label semantics only | `pseudo-label-v4` and a new Sensor/RG-03 attempt | Do not overwrite v3 cache or evidence |
| Chroma extraction semantics change | `feature-cache-v4` plus `pseudo-label-v4` and downstream provenance bump | Preserve v3 feature cache |
| Encoder/head conformance | New sensor contract/version and implementation commit | Preserve A1-v1 architecture identity unless a Decision says otherwise |
| Formal result | New RG-03 attempt only after preflight and approval | Keep `rg-03-sensor-formal-v2.json` FAIL permanently |

## Minimal verification path

1. Freeze the K-S and Key contracts in a Decision.
2. Implement reference-only calculators and unit tests first.
3. Rebuild only the required versioned pseudo-label/feature caches, with full
   checksums and 896/99/58 isolation evidence.
4. Run a single-seed Mode-only prototype; compare against the current frozen
   diagnostics without selecting on test or changing the `CCC > 0.7` Gate.
5. Only after conformance and preflight pass, request authorization for a new
   five-seed formal attempt.

The current architecture-faithful prototype reached validation CCC `0.673502`
after 50 diagnostic epochs (one fixed seed, no checkpoint artifact). This is
directional evidence that the intended design may approach the Gate, not a
formal result or authorization to retrain.
