# Krumhansl–Schmuckler contract gap

Status: `STOP_FORMAL_IMPLEMENTATION` for any pseudo-label repair. The current
research documents state K-S/template correlation at the research-plan level,
but do not freeze the numerical details required to build a new formal cache.
The v3 pseudo-label cache is immutable and is not changed here.

The following must receive a new append-only Decision before `pseudo-label-v4`
or a new RG-03 attempt can be created:

1. Major and minor 12-tone template numeric values and their normalization.
2. Tonic rotation convention (pitch-class order and rotation direction).
3. Correlation definition (Pearson versus centered cosine, zero-variance rule).
4. Whether mode strength is `max_major - max_minor`, a normalized ratio, or
   another mapping, and the exact mapping/clipping to `[-1,1]`.
5. Tie behavior for equal tonic/template scores.
6. Invalid-frame and zero-energy behavior, including reason codes.
7. Raw-frame versus binned ordering and the exact 0.5 s aggregation rule.
8. KeyExtractor implementation identity, EDMA profile/version, window alignment,
   missing/unknown handling and 5-second segment reduction.

Until these values are frozen, a `DEBUG_KS_REFERENCE_ONLY` implementation may
be used for direction checks, but it cannot publish pseudo-labels, feature
caches, checkpoints or formal Gate evidence.
