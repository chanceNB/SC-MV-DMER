# RG-03 research-plan versus current implementation audit

Status: `DEBUG_DIAGNOSTIC_ONLY` (2026-09-16). This audit does not invalidate or
rewrite `evidence/gates/rg-03/rg-03-sensor-formal-v2.json`; that historical
formal attempt remains FAIL. All payload diagnostics use the registered v3
feature/pseudo-label manifests and only the 896 optimization-train and 99
validation songs. No test payload is opened.

| Component | Research plan | Current implementation | Match? | Severity | Potential CCC impact |
|---|---|---|---|---|---|
| Chroma extraction | `librosa.feature.chroma_cqt`, hop 512, 12 chroma, 36 bins/octave | `features/handcrafted.py:_cqt_like` is a custom fixed FFT projection; `extract_handcrafted` calls it at lines 48–67 and 85 | **No** | High | Different frequency windows/aggregation can change the input-target relationship |
| Chroma normalization | L1 normalize each raw frame before bin mean | `_cqt_like` sums octave projections then normalizes the resulting 12-vector at lines 60–67; registered cache is v3 and must not be replaced in this audit | **Unconfirmed semantic difference** | High | Raw-frame versus post-aggregation normalization changes mode/key inputs |
| 0.5 s binning | half-open bins, mean over assigned raw frames | `features/binning.py:11-43` performs sample-domain half-open assignment and mean | Yes for registered binned cache | Low | Not a likely root cause by itself |
| Mode pseudo-label | Krumhansl–Schmuckler tonic-template correlation, max-major minus max-minor, mapped to [-1,1] | `pseudo_labels.py:39-43` uses two fixed binary masks and `chroma @ mask`; it is not a tonic-rotated 12-template correlation | **No: `RESEARCH_PLAN_IMPLEMENTATION_MISMATCH`** | Critical | Directly changes the supervised target |
| Fixed-mask reduction | N/A under K-S | `major-minor` masks cancel pitch classes 1, 5 and 7, leaving `+pc4 - pc3 - pc10` (0-based) | **No** | Critical | The label is a sparse pitch-class contrast, not mode evidence |
| Key pseudo-label | Essentia `KeyExtractor`, EDMA profile, 5 s windows | `pseudo_labels.py:44-45` uses frame `argmax(chroma)` then 5 s bin mean | **No: `RESEARCH_PLAN_IMPLEMENTATION_MISMATCH`** | High | Label semantics and ties/unknowns differ |
| Chroma encoder dimensionality | Linear 12→256 | `training/sensor_executor.py:47-58` defaults every encoder to hidden_dim=64; formal runner line 212 passes 64 | **No** | High | Lower capacity and different representation |
| Transformer encoder | One layer, d_model=256, 4 heads, FFN=512 | No Transformer module exists in `SensorOnlyModel`; only projection, GELU, LayerNorm | **No** | High | Missing temporal/context capacity |
| Sensor head topology | Mode `256→64→1`; Key `256→64→12`; Energy/Brightness analogous | `sensor_executor.py:53-58` uses a single `Linear(64, output)` per head | **No** | Medium/High | Head capacity and nonlinearity differ |
| Mode loss | Continuous mode-strength regression | `SensorLossContract._continuous` uses masked MSE (`sensor_executor.py:146-152`) | Yes as an optimization loss | Medium | MSE can shrink prediction amplitude, reducing CCC |
| Key loss | 5 s segment CE with frozen mask/reduction | `SensorLossContract.key_ce` pools 9×10 frames and applies CE (`154-183`) | Partial | Medium | It lacks explicit tie/unknown reason handling from the frozen contract |
| Shared Mode/Key encoder | Both heads on encoded E_chroma | `SensorOnlyModel.forward` sends both to `encoded["chroma"]` (`73-80`) | Yes structurally | Unknown | Possible multi-task interaction, not supported as conflict by diagnostic |
| Checkpoint selection | Validation CE only; earliest epoch then global step tie-break | `sensor_formal.py:240-243` selects lower validation CE with min_delta; curve records selection metadata | Yes at high level | Low/Medium | Diagnostic F finds small selection-vs-max-CCC deltas, not a Gate workaround |
| Early stopping | patience=10, min_delta=1e-4 | `sensor_formal.py:240-246` | Yes | Low | Not the main semantic discrepancy |
| CCC implementation | Frozen MEP-CCC2-S5-3R+, FP64 pooled moments/reason codes | `sensor_formal.py:146-163` computes FP64 pooled moments but has no explicit frozen reason-code path; `rg03.py:96-97` checks the evidence contract | Partial | Medium | Must retain frozen implementation/version in any repair |
| Normalization | Optimization-train only; raw-space metric | Formal evidence and `sensor_formal.py:97-118` bind train-only stats and inverse-transform metrics | Yes | Low | Not supported as leakage root cause |
| Train/validation/test isolation | Song-level; test excluded from fit, selection and Gate | `rg03.py:81-119` and formal evidence bind 896/99/58; diagnostic execution opened 0 test payloads | Yes | Low | No evidence for test leakage |

## Formula audit

With the registered v3 binned chroma `c`, the current code is exactly:

`mode_current = c[4] - c[3] - c[10]`.

This follows algebraically because the two masks are `[1,4,5,7]` and
`[1,3,5,7,10]` (0-based pitch-class indices). On the frozen optimization
train/validation populations, the formula-versus-frozen-target audit is
Pearson `0.559093`, CCC `0.553988`, MSE `0.018019`; it is therefore not a
K-S implementation and is not an exact reconstruction of the registered
frozen target.

The complete machine-readable statistics, regression ceilings, gradient
diagnostic, prototype and checkpoint comparison are in
`docs/diagnostics/rg03-mode-diagnostics-v1.json`.
