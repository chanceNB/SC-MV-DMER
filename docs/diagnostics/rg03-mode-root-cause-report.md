# RG-03 Mode root-cause report

## Executive conclusion

The five-seed v2 formal FAIL is real for the frozen v3 implementation, but it
must not be interpreted as evidence that the research-plan Mode Sensor cannot
reach CCC > 0.7. The implementation is not plan-conformant: fixed-mask Mode
labels, a non-`librosa.chroma_cqt` frontend, and a 64-dimensional linear Sensor
topology replace the specified K-S labels and 256/Transformer/MLP topology.

## Hypothesis ledger

| Hypothesis | Classification | Evidence | Counter-evidence | Confidence | Next action |
|---|---|---|---|---|---|
| H1 current Mode label is not K-S | **CONFIRMED** | `pseudo_labels.py:39-43`; algebra reduces to `pc4-pc3-pc10`; formula audit CCC `0.553988` | v3 target is frozen and internally checksum-valid | High | Freeze K-S math, then create versioned pseudo-label successor |
| H2 feature chroma and target chroma semantics mismatch | **LIKELY** | `handcrafted.py:48-67` custom FFT CQT-like projection; plan requires librosa CQT; target code separately computes another projection | Exact raw generation provenance for v3 is not fully specified in current source | Medium/High | Resolve frontend identity; if changed, build feature-cache-v4 |
| H3 64-d encoder is insufficient versus 256+Transformer | **LIKELY** | `sensor_executor.py:47-58`; no Transformer; plan requires one | One prototype reached only `0.673502` at 50 epochs, below Gate; no matched multi-seed comparison | Medium | Implement a versioned prototype after contract freeze |
| H4 single linear Mode head mismatches 256→64→1 MLP | **CONFIRMED** | `sensor_executor.py:57-58` | Capacity effect is not isolated from H1/H2 | High for mismatch; medium for causal CCC effect | Versioned topology test and prototype |
| H5 MSE causes variance shrinkage | **SUPPORTED** | Best current-chroma linear model prediction std `0.0872` vs target `0.1505`; formal/prototype outputs remain lower amplitude | Linear CCC only `0.5226`; objective alone does not explain semantic mismatch | Medium/High | Keep MSE contract; quantify with conformant labels/topology |
| H6 Key/Mode gradient conflict | **NOT SUPPORTED** | D1/D2 diagnostic mean cosine `0.1007`/`0.1098`, negative ratio `0.0` | Short fixed-batch diagnostic, not a full formal curve | Medium | Do not alter multitask loss based on this evidence |
| H7 CE checkpoint selection loses Mode optimum | **SUPPORTED AS SMALL EFFECT, NOT ROOT CAUSE** | Max-minus-selected CCC deltas `0.00261–0.01071` across five seeds | Max values remain `0.55966–0.57060`, far below `0.7`; selection rule is frozen | High | Preserve CE selection; report diagnostic only |
| H8 CCC implementation bug | **NOT SUPPORTED** | FP64 pooled calculation agrees with registered contract shape; all pairs finite | Current runner does not expose all frozen reason codes itself | Medium | Add conformance tests/reference implementation before any new formal attempt |

## Current formal CCC summary

`0.5489528009`, `0.5653557322`, `0.5567810050`, `0.5540305499`,
`0.5614885566`; all five fail strict `CCC > 0.7`. Energy, brightness and Key
criteria passed in the preserved evidence.

## Diagnostic conclusions

- Frozen target statistics are not center-collapsed: validation mean `-0.04612`,
  std `0.15052`, variance `0.022657`.
- Current fixed-mask formula is not the intended target semantics.
- Current 12-D chroma has a nontrivial but limited information ceiling:
  validation linear CCC `0.522581`; best 50-epoch conformant-topology DEBUG
  prototype CCC `0.673502`.
- The diagnostic used 0 test songs and did not write checkpoints or Gate
  evidence. Full machine output is in `docs/diagnostics/rg03-mode-diagnostics-v1.json`.

## Closure

**Current implementation is not a faithful test of the research plan.** No
formal retraining is authorized by this audit. The next required action is a
Decision freezing K-S/Key math and a versioned repair proposal, with v3 and the
historical FAIL retained unchanged.
