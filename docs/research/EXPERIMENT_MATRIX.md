# SC-MV-DMER Experiment Matrix

> Authority: human-readable Experiment Catalog Source of Truth  
> Status: Section 6 semantic design `APPROVED / FROZEN`  
> Semantic closure: `COMPLETE`  
> Executable binding: `PENDING_DEPENDENCIES`  
> Machine-readable normalization: `PENDING`  
> Formal ready: `false`

## 1. Catalog Authority and Change Control

This catalog defines the frozen scientific meaning of the Primary A1–A13 Experiment Matrix. It is subordinate to `docs/research/RESEARCH_SPEC.md` for overall research semantics and is interpreted together with `docs/research/DECISIONS.md` and the Foundation Design Spec.

The catalog is closed-world for formal primary ablations. A new A-number, executable child, changed base, changed primary factor, changed required closure, or changed analysis anchor requires a new append-only Decision and version-change review. Debug/exploratory runs may exist outside this catalog only with `run_mode=debug` and are ineligible for formal aggregation.

Unresolved executable dependencies do not reopen frozen scientific definitions. They block PRE-FLIGHT, stage entry, artifact generation, Gate evaluation, aggregation, or paper reporting as identified in the Blocking Dependency Register.

## 2. Definition Architecture

Every executable entry is defined by:

- immutable `variant_id` and `variant_version`;
- `base_experiment_variant_id`, or `NONE` for an independent root;
- one frozen `primary_factor`;
- `direct_semantic_diff` containing only the intended scientific intervention;
- `required_dependency_closure` containing consequences forced by that intervention;
- `invariant_groups` that must remain unchanged;
- `analysis_anchor` describing the primary comparison;
- `forbidden_changes` preventing interpretation drift;
- Gate applicability and executable-binding status.

Formal PRE-FLIGHT must establish:

```text
actual_semantic_diff
= declared_direct_semantic_diff
+ approved_required_dependency_closure
```

Any undeclared difference or missing closure is a formal failure.

## 3. Scope and Execution Fields

`scope_tier` and `execution_obligation` are independent fields.

### 3.1 Scope tier

- `PRIMARY`: required for the main DEAM research question or its registered primary matrix.
- `SECONDARY`: registered supporting evaluation such as approved cross-dataset evidence.
- `OPTIONAL`: resource-dependent extension with no current execution commitment.
- `CONTINGENCY`: activated only by its frozen contingency policy.

### 3.2 Execution obligation

- `REQUIRED`: complete formal evidence is required by the frozen scope.
- `CONDITIONAL`: execution is controlled by a pre-registered trigger.
- `NOT_TRIGGERED`: the registered conditional experiment remains in the catalog, but its trigger resolved false.
- `OPTIONAL_NOT_COMMITTED`: optional scope exists but resources have not been committed.
- `BLOCKED`: execution is required or conditionally selected but cannot proceed because a registered dependency is unresolved or invalid.

The eventual schema may normalize enum spelling without changing these meanings. Scope and obligation cannot be changed in response to model results.

## 4. Global Seed and Evidence Policy

All trainable formal variants use the same frozen S5 and require `COMPLETE_S5` for final paper evidence. Staged execution may use debug S1, then formal pilot S1, then remaining S4; partial formal evidence cannot be reported as a complete variant. A8 children complete S5 independently. A12 completes S5 if triggered. PMEmo evaluates five corresponding DEAM checkpoints without retraining.

The frozen identity is `SC-MV-DMER-FORMAL-S5/v1`, ordered as `[52826381, 128866372, 1616929435, 1871035633, 1460830465]`. `SS-3+` governs project-identity SHA-256 indexed selection and `RNG-3+` governs namespaced RNG domains; `OQ-SEEDSET-S5-VALUES` is resolved.

Stage-2 CoT coverage follows `COTC-SN21-DSDP-3R+`: for `N` unique training songs, `N_syn=floor(2N/3+0.5)` are deterministically SHA-256-ranked `SYNTHETIC_COT_SUPERVISED`, and the rest are `NATIVE_DEAM_VA_ONLY`. Membership is fixed before training, S5-independent, target/difficulty/outcome-blind, and phase-invariant. The native role has no teacher CoT and is never “real CoT”; it may still be eligible for online consistency. A10/A13 are N/A.

## 5. Frozen Identity Table

| # | Variant ID | Version | Base | Scope | Obligation | Primary factor |
|---|---|---|---|---|---|---|
| A1 | `A1_FULL_7B` | `A1-v1` | `NONE` | PRIMARY | REQUIRED | Canonical full 7B system |
| A2 | `A2_DEEP_ONLY_COT` | `A2-v1` | `A1-v1` | PRIMARY | REQUIRED | Complete handcrafted-subsystem removal |
| A3 | `A3_INDEPENDENT_A` | `A3-v1` | `A1-v1` | PRIMARY | REQUIRED | Shared-to-independent transition matrices |
| A4 | `A4_STATE_BIAS_DISABLED` | `A4-v1` | `A1-v1` | PRIMARY | REQUIRED | Predictive state-bias removal |
| A5 | `A5_EVENT_INJECTION_DISABLED` | `A5-v1` | `A1-v1` | PRIMARY | REQUIRED | Prompt-side EventSequence removal |
| A6 | `A6_SENSOR_HEADS_DISABLED` | `A6-v1` | `A1-v1` | PRIMARY | REQUIRED | Sensor auxiliary supervision removal |
| A7 | `A7_VIEW_DROPOUT_DISABLED` | `A7-v1` | `A1-v1` | PRIMARY | REQUIRED | View Dropout removal |
| A8-MEL | `A8_DEEP_PLUS_MEL` | `A8-MEL-v1` | `A1-v1` | PRIMARY | REQUIRED | One complete Mel subsystem above Deep-only |
| A8-MFCC | `A8_DEEP_PLUS_MFCC` | `A8-MFCC-v1` | `A1-v1` | PRIMARY | REQUIRED | One complete MFCC subsystem above Deep-only |
| A8-CHROMA | `A8_DEEP_PLUS_CHROMA` | `A8-CHROMA-v1` | `A1-v1` | PRIMARY | REQUIRED | One complete Chroma subsystem above Deep-only |
| A9 | `A9_TIMESNET_DISABLED` | `A9-v1` | `A1-v1` | PRIMARY | REQUIRED | TimesNet structural removal |
| A10 | `A10_VA_ONLY_LLM` | `A10-v1` | `A1-v1` | PRIMARY | REQUIRED | Generative CoT/ANSWER subsystem removal |
| A11 | `A11_CONSENSUS_ENABLED` | `A11-v1` | `A1-v1` | PRIMARY | REQUIRED | Explicit Markov-posterior consensus loss |
| A12 | `A12_HSIC_DECORRELATION_ENABLED` | `A12-v1` | `A1-v1` | PRIMARY | CONDITIONAL | HSIC representation decorrelation |
| A13 | `A13_FEATURE_CONCAT_BASELINE` | `A13-v1` | `NONE` | PRIMARY | REQUIRED | Independent four-view feature concatenation baseline |

All 15 executable identities are unique, versioned, immutable once used by formal evidence, and governed by append-only invalidation/supersession.

`A8_SINGLE_HANDCRAFTED_VIEW` is a `NON_EXECUTABLE_GROUP`, not a run-producing variant. It has no seed instance, checkpoint, or paper row. `A8_FAMILY_COMPLETE` requires valid `COMPLETE_S5` evidence for all three executable children.

## 6. Shared Invariant Groups

Formal machine-readable normalization must expand every applicable field rather than using only `everything_else_same_as_A1`. The human-readable invariant groups are:

- `DATA_SPLIT`: registered DEAM dataset version, frozen song-level 80/10/10 primary split, labels, normalization provenance, masks, sample population.
- `FEATURE_TIME`: upstream manifests, feature/cache identities, 45s segment contract, 2Hz/90-bin TimeGrid, timestamps, masks, view dimensions.
- `ENCODER`: active-view frontend definitions, projection dimensions, and initialization policy except where direct diff requires removal.
- `MARKOV`: state count, prototypes, transition/initial-state semantics, gamma computation, state-index contract, beta parameterization except where explicitly changed.
- `SENSOR_EVENT`: Sensor class/mask/segment contract, event symbolizer/source/version and artifact lineage except where explicitly changed.
- `FUSION_LLM`: fusion, prompt, Qwen/LoRA, VA/LM heads, Parser, generation semantics except where explicitly changed.
- `TRAINING`: stage schedule, optimizer/scheduler, trainable/frozen components, loss definitions and weights, batch semantics, checkpoint-selection policy once frozen.
- `EVALUATION`: metric implementation, primary validation selection, test exclusion from model selection/Gates, aggregation and uncertainty protocol.
- `SEEDS`: frozen S5 values, order, seed-instance mapping and RNG-domain policy.
- `PROVENANCE`: semantic/resolved config identity, component/artifact manifests, checksums, effective-validity dependencies.

## 7. Variant Catalog

### 7.1 A1_FULL_7B / A1-v1

- Base: `NONE`; canonical root.
- Composition: Deep+Mel+MFCC+Chroma; shared Markov `A/π`; direct-view Sensor Heads; EventSequence; handcrafted-path View Dropout `p=0.15`; full Event-conditioned Qwen dual head.
- Default inactive mechanisms: consensus loss, HSIC, CoT-DPO, difficulty extension.
- Analysis anchor: main primary system, baselines, ablations, and cross-dataset source checkpoint family.
- Required closure: none; all components are affirmative canonical semantics.
- Invariants: all shared groups after binding.
- Forbidden: silent changes to any research-semantic field, result-driven baseline/threshold redefinition, No-LLM ensembling into Full LLM output.
- Gate notes: lifecycle mapping across RG-01–RG-10 remains dependency-bound; no PASS may be inferred without exact identity/applicability.
- Binding: `PENDING_DEPENDENCIES`; formal ready `false`.

### 7.2 A2_DEEP_ONLY_COT / A2-v1

- Direct diff: disable Mel, MFCC, and Chroma.
- Required closure: Sensor Heads/`L_sensor`, EventSequence, handcrafted Markov/Fusion branches, beta, and View Dropout are N/A; use event-free prompt and a separately versioned event-free CoT artifact.
- Preserved: Deep path, Qwen dual-head CoT/ANSWER behavior, shared data/time/evaluation/training contracts.
- Analysis anchor: A1; interpretation is total handcrafted-subsystem effect.
- Gate notes: independent RG-09 quality profile; Sensor/Event-related Gate applicability follows the absence closure.
- Forbidden: retaining hidden handcrafted-derived Event inputs, calling the result a marginal contribution of one feature, or reusing A1 event-grounded artifact as if event-free.

### 7.3 A3_INDEPENDENT_A / A3-v1

- Direct diff: one shared `A` becomes four per-view independent transition matrices.
- Required closure: distinct parameter/optimizer state; same numeric initialization permitted; `L_state` reduced equally across view matrices.
- Preserved: shared `π`, per-view prototypes, state index meaning, Sensor/Event/Fusion/LLM, all other contracts.
- Analysis anchor: A1; transition-sharing contribution.
- Gate notes: per-view-A RG-05 profile is frozen by `SGA-M0M20-EV-3R+`; RG-06 remains applicable.
- Forbidden: independent `π`, post-hoc state matching/permutation, or altering state count/prototype definition.

### 7.4 A4_STATE_BIAS_DISABLED / A4-v1

- Direct diff: remove predictive state-bias branch.
- Required closure: beta parameters absent; effective bias exactly zero; RG-06 N/A.
- Preserved: Markov definitions, transition/state diagnostics, gamma and applicable `L_state`, all non-bias A1 paths.
- Analysis anchor: A1; predictive state-bias contribution.
- Gate notes: RG-05 is diagnostic-only under `SGA-M0M20-EV-3R+`.
- Forbidden: describing A4 as no-Markov, leaving trainable/dummy beta, or adding capacity compensation.

### 7.5 A5_EVENT_INJECTION_DISABLED / A5-v1

- Direct diff: remove EventSequence from model-visible prompt.
- Required closure: prompt removes event block and mandatory citation; Event artifacts become shadow-only; model-visible-input firewall required.
- Preserved: Sensor Heads, `L_sensor`, Event generation/provenance, A1 event-derived CoT targets, all other architecture.
- Analysis anchor: A1; value of model-visible event injection.
- Gate notes: RG-07 N/A; RG-09 may reuse only the exact approved A1 target artifact; citation uses `ECA-ATOM-PREC-3R++` with A5's own shadow Event reference.
- Forbidden: deleting Sensor/Event artifacts, regenerating targets as event-free, or leaking Event through metadata/derived prompt fields.

### 7.6 A6_SENSOR_HEADS_DISABLED / A6-v1

- Direct diff: remove local Sensor Heads and `L_sensor`.
- Required closure: consume immutable, paired A1 Event Input Replay including source schedule and lineage; no local event generation; save view-gradient/beta diagnostics.
- Preserved: all four encoded views, Markov/Fusion, View Dropout, Full LLM and A1 CoT targets.
- Analysis anchor: A1; auxiliary sensor supervision effect under controlled event input.
- Gate notes: local RG-03 N/A but upstream Sensor bundle must remain effectively valid; RG-05/RG-06 are local; RG-08 uses the replay topology compiled by Batch 1.
- Forbidden: claiming deployable sensor-free semantics, generating replacement events, or using replay from a different seed/config/artifact identity.

### 7.7 A7_VIEW_DROPOUT_DISABLED / A7-v1

- Direct diff: remove View Dropout.
- Required closure: dropout embeddings, probability, component RNG and stochastic branch masks are N/A.
- Preserved: full-view A1 system and endogenous Sensor/Event behavior.
- Analysis anchor: A1 primary full-view evaluation plus a paired deterministic seven-condition missing-branch stress suite.
- Gate notes: the frozen stress suite is diagnostic/non-gating; RNG isolation follows `RNG-3+`; executable stress evidence remains runtime work.
- Forbidden: dropping Deep, applying stress masks to Sensor/Event branches, or tuning the stress suite after observing A1/A7 outcomes.

### 7.8 A8_SINGLE_HANDCRAFTED_VIEW children

- Executable children: Mel, MFCC, Chroma variants listed in the identity table.
- Direct diff per child: disable the two non-selected handcrafted views; Deep is always retained.
- Required closure: retain one complete Sensor/Event subsystem; active Deep+selected-view Markov with shared `A/π`; one beta/fusion branch; fusion concat 512→projector 256; one View Dropout embedding; view-scoped EventSequence and child-specific CoT artifact.
- Preserved: selected view's complete A1 component contract and all shared data/time/training/evaluation semantics.
- Analysis anchor: each child versus A2; conditional total subsystem contribution.
- Gate notes: active-set Gate/RG-08 profiles are compiled by `VCP-CCEW-3R++`/`SGA-M0M20-EV-3R+`; each child owns RG-09 artifact evidence.
- Forbidden: treating the parent as executable, reporting an incomplete subset as A8 family complete, interpreting results as Shapley values, or using an event/teacher capability unavailable to the child.

### 7.9 A9_TIMESNET_DISABLED / A9-v1

- Direct diff: remove TimesNet from the Deep path.
- Required closure: parameter-free exact RG-01 timestamp/bin mean converts raw gated MERT to `[90,768]`, followed by the same `W_deep` and post-bin Transformer.
- Preserved: RG-01 TimeGrid/mask/bin membership/mean, all handcrafted/Sensor/Event/Markov/Fusion/LLM semantics.
- Analysis anchor: A1; TimesNet temporal modeling contribution.
- Gate notes: exact RG-01/RG-02 reuse plus bypass routing evidence; RG-08 uses the TimesNet-disabled topology profile.
- Forbidden: adaptive pooling, interpolation, new temporal parameters/model, altered bins, or silent trim/pad.

### 7.10 A10_VA_ONLY_LLM / A10-v1

- Direct diff: remove generative CoT/ANSWER subsystem.
- Required closure: no LM logits/generation, CoT targets, ANSWER Parser, `Y_answer`, `L_CoT`, or `L_consist`; sanitize prompt for VA-only forward; retain local model-visible Event for VA conditioning.
- Preserved: Event-conditioned ChatTS, Qwen, LoRA, hidden states and VA Regression Head; upstream A1 multimodal path.
- Analysis anchor: A1; contribution of generative rationale/ANSWER training beyond Qwen hidden-state VA prediction.
- Gate notes: RG-04/RG-09/RG-10 N/A; RG-08 uses P1/P2 Stage 1, P1/P3 Stage 2 and FVA, with P4 N/A; epoch-5 terminal only.
- Forbidden: calling A10 No-LLM, reusing No-LLM identity/checkpoint/RG-04 verdict, or silently invoking generation.

### 7.11 A11_CONSENSUS_ENABLED / A11-v1

- Direct diff: enable `L_consensus = mean KL(gamma_deep || gamma_view)` for eligible handcrafted views.
- Required closure: View-dropped pairs excluded; no stop-gradient; FP32; epsilon `1e-8` clamp+renormalize; equal eligible-pair mean; zero-pair microbatch contributes zero and logs; Stage 1+2; fixed weight `0.005` outside uncertainty weighting.
- Preserved: A1 architecture/data/prompt and all other loss definitions.
- Analysis anchor: A1; explicit posterior consensus regularization.
- Gate notes: own RG-05/RG-06 attempts; accumulation reduction design is resolved by `HLAC-HDSA-3R+` / `DLC-P6-DAAI-3R+`; executable numerical qualification is pending.
- Forbidden: reverse/symmetric KL, detached teacher, unequal pair weighting, or result-driven weight tuning.

### 7.12 A12_HSIC_DECORRELATION_ENABLED / A12-v1

- Direct diff: enable per-song, per-eligible-view RBF-HSIC between `F_deep←v` and `E_deep`.
- Required closure: valid frames only; View-dropped pairs excluded; detached median bandwidth with `1e-6` floor; biased estimator; equal song-view mean; FP32; no stop-gradient on X/Y; Stage 1+2; fixed weight `0.005` outside uncertainty weighting.
- Preserved: A1 architecture and other losses.
- Analysis anchor: A1; representation decorrelation after a pre-registered A1-versus-A2 handcrafted-gain trigger.
- Gate notes: own RG-05/RG-06 attempts; trigger is frozen by `A12-HGST-Z0-3R++`, and accumulation reduction by `HLAC-HDSA-3R+`/`DLC-P6-DAAI-3R+`.
- Execution state: `WAITING_FOR_VALID_A1_A2_COMPLETE_S5`; if the frozen trigger resolves false, obligation becomes `NOT_TRIGGERED` without deleting A12.
- Forbidden: using A12 outcomes to define its trigger, selecting bandwidth/weight after results, or reporting non-execution as failure.

### 7.13 A13_FEATURE_CONCAT_BASELINE / A13-v1

- Family: independent formal comparison; base `NONE`; primary comparison anchor `A1_FULL_7B`.
- Composition: four independently trained contract-aligned encoded views `[90,256]`; fixed concat order `[deep,mel,mfcc,chroma]`; `[90,1024]`; independent-weight A1 post-concat projector contract `1024→256`; No-LLM VA head contract.
- Required closure: Markov, Sensors/`L_sensor`, Event, View Dropout, Qwen/ChatTS and CoT all absent.
- Training: one independent five-logical-epoch `BASELINE_TRAIN`; `L_A13=L_VA+0.05L_smooth`, fixed weights/no Kendall; OSNP/OSNS Stage-1 numeric profile; epoch-5 terminal only.
- Shared contract references: dataset/split, features/cache, Encoders, TimeGrid/mask, evaluation and training-budget contracts; no A1 checkpoint reuse.
- Analysis anchor: A1 full-stack comparison; not a single-factor ablation.
- Gate notes: RG-01/RG-02 reusable only under exact identities; RG-03/05/06/07/08/09/10 N/A; RG-04 is not an A13 Gate.
- Naming: `KANG_HERREMANS_INSPIRED`, `NOT_EXACT_REPRODUCTION`.
- Forbidden: A1-overlay semantics, Markov/Sensor/Event/ViewDropout/Qwen/CoT/KD/MTL, external chord/key inputs, extra dataset/temporal model, stronger projector, budget search, checkpoint reuse, dummy compensation, test-informed tuning, or claiming exact reproduction.

## 8. Frozen Non-Equivalence Rules

- A2 is complete handcrafted-subsystem removal; A8 is one complete subsystem's contribution over Deep-only.
- A3 changes transition-parameter sharing; A11 adds posterior-distribution consensus.
- A4 removes predictive state bias inside A1; A13 is an independent simple-concat family.
- A5 removes Event prompt injection; A10 removes generative CoT/ANSWER behavior.
- A6 removes local Sensor supervision under controlled A1 Event Replay; A13 has no Event, Markov, or LLM.
- A10 is Qwen-based VA-only; No-LLM is `F_fused → no_llm_head` with no ChatTS/Qwen/LoRA.

Event semantics are fixed as:

```text
A2: no handcrafted views, no Sensors, no EventSequence, event-free CoT target
A5: Sensor/Event artifacts retained, Event not model-visible, A1 event-derived target retained
A6: no local Sensor, paired A1 Event Input Replay model-visible, A1 target retained
```

Markov semantics are fixed as:

```text
A3: shared A → per-view independent A
A4: predictive state bias removed; Markov definitions retained
A11: explicit KL posterior consensus added
A12: HSIC representation decorrelation added
```

## 9. Gate Applicability Summary

Gate applicability is identity- and lifecycle-specific. Historical PASS cannot be inherited solely because a variant shares a base. Exact artifact reuse is permitted only where the catalog states it and all referenced evidence remains effectively valid.

| Variant/group | Frozen applicability note |
|---|---|
| A1 | Full M0–M20 lifecycle mapping frozen by `SGA-M0M20-EV-3R+` |
| A2 | Sensor/Event Gates follow absence closure; new event-free RG-09 profile |
| A3 | Per-view-A RG-05 profile required; RG-06 applies |
| A4 | RG-05 diagnostic-only; RG-06 N/A |
| A5 | RG-07 N/A; exact-target RG-09 reuse only; citation reference design resolved by `ECA-ATOM-PREC-3R++`; executable shadow-artifact/firewall evidence pending |
| A6 | Local RG-03 N/A; local RG-05/06; RG-08 applicability resolved by Batch 1/VCP; upstream Sensor/replay artifact binding and executable hardware evidence pending |
| A7 | A1 applicability plus paired non-gating stress evidence |
| A8 children | Active-set and RG-08 applicability profiles resolved by VCP/SGA; child-specific Event/CoT artifact binding and executable hardware evidence pending |
| A9 | RG-02 reuse and RG-08 applicability resolved; exact RG-01 bypass-routing evidence and executable hardware evidence pending |
| A10 | RG-04/09/10 N/A; RG-08 uses frozen VA-only profile; checkpoint selection frozen by `CS-3+`/Batch 1 |
| A11 | New RG-05/06 attempts |
| A12 | New RG-05/06 attempts if triggered |
| A13 | Exact RG-01/02 reuse only; RG-03/05/06/07/08/09/10 N/A; RG-04 not Gate |

## 10. Blocking Dependency Register

Status values are `OPEN`, `RESOLVED`, `DEFERRED_NON_BLOCKING`, `INVALIDATED`, or `SUPERSEDED`. Every `OPEN` item is an execution/provenance blocker with an explicit owner and required-before boundary. Frozen design questions are `RESOLVED`; deferred optional scopes do not block 7B Primary Foundation Closure.

| Dependency ID | Owner section | Affected variants | Blocking stage / required before | Description | Status | Resolution ref |
|---|---|---|---|---|---|---|
| `OQ-SEEDSET-S5-VALUES` | Training/Reproducibility | all trainable formal variants | any formal pilot | Five integers/order/version | RESOLVED | `SS-3+`; DEC-0023 |
| `S5_RNG_APPLICATION_CONTRACT` | Training/Reproducibility | all trainable formal variants | any formal pilot | Namespaced RNG domains | RESOLVED | `RNG-3+`; DEC-0023 |
| `OQ-A1-7B-LORA-RANK` | Model/Training | A1–A12 applicable | 7B binding | Rank/alpha/dropout/quantization | RESOLVED | `LQ7-R16A32-ZD-NF4-3R+` |
| `OQ-A10-QWEN-HIDDEN-BACKBONE-LOADING-CONTRACT` | Model/Training | A10 | design closure | Hidden-backbone semantics | RESOLVED | `A10-HBL-DSP-3R++` |
| `OQ-A2-EVENTFREE-COT-QUALITY-PROTOCOL` | CoT/Metrics | A2 | artifact design | Event-free target/prompt/QA | RESOLVED | `VCP-CCEW-3R++`; `COTC-SN21-DSDP-3R+` |
| `OQ-A8-VIEW_SCOPED-COT-TEACHER-ALLOWLIST` | CoT | A8 children | artifact design | Capability-scoped teacher evidence | RESOLVED | `VCP-CCEW-3R++`; Batch 1/2 |
| `OQ-A6-EVENT-REPLAY-CONTRACT` | Event/Artifact | A6 | design closure | Paired A1 replay semantics | RESOLVED | `TEB-FBC-3R+++`; `FI-A+` |
| `OQ-A6-GRADIENT-DIAGNOSTIC-CONTRACT` | Metrics | A6 | evidence design | Gradient/beta diagnostics | RESOLVED | `MEP-CCC2-S5-3R+` |
| `OQ-A3-RG05-PER_VIEW_A-EVIDENCE-PROFILE` | Gate Catalog | A3 | RG-05 design | Per-view A evidence, no rematching | RESOLVED | `SGA-M0M20-EV-3R+` |
| `OQ-A4-RG05-DIAGNOSTIC-APPLICABILITY-MAPPING` | Gate Catalog | A4 | applicability design | Diagnostic-only RG-05 | RESOLVED | `SGA-M0M20-EV-3R+` |
| `OQ-A6-RG08-APPLICABILITY` | Gate Catalog | A6 | applicability design | Replay-topology profile | RESOLVED | `VCP-CCEW-3R++`; Batch 1 |
| `OQ-A8-ACTIVE-SET-GATE-APPLICABILITY` | Gate Catalog | A8 | applicability design | Active two-view profiles | RESOLVED | `VCP-CCEW-3R++`; `SGA-M0M20-EV-3R+` |
| `OQ-A8-RG08-APPLICABILITY` | Gate Catalog | A8 | applicability design | Active topology memory profile | RESOLVED | `TEB-FBC-3R+++` |
| `OQ-A9-RG02-APPLICABILITY` | Gate Catalog | A9 | applicability design | RG-02 reuse + bypass evidence | RESOLVED | `SGA-M0M20-EV-3R+` |
| `OQ-A9-RG08-APPLICABILITY` | Gate Catalog | A9 | applicability design | TimesNet-disabled topology | RESOLVED | `TEB-FBC-3R+++` |
| `OQ-A10-RG08-VA_ONLY-APPLICABILITY` | Gate Catalog | A10 | applicability design | VA-only probe | RESOLVED | `A10-RG08-ASAND-FVA-DP-3R+++` |
| `STAGE_ENTRY_EXIT_APPLICABILITY_MAPPING` | Stage/Gate | A1–A13 | design closure | M0–M20 and effective validity | RESOLVED | `SGA-M0M20-EV-3R+` |
| `OQ-A5-EVENT-CITATION-REFERENCE-CONTRACT` | Metrics | A1,A5,A6 | design closure | Atom precision references | RESOLVED | `ECA-ATOM-PREC-3R++` |
| `OQ-A7-MISSING-VIEW-EVALUATION-CONTRACT` | Metrics | A1,A7 | design closure | Frozen deterministic stress suite | RESOLVED | A7-v1; `MEP-CCC2-S5-3R+` |
| `OQ-A10-CHECKPOINT-SELECTION-APPLICABILITY` | Evaluation | A10 | design closure | VA-only terminal selection | RESOLVED | `CS-3+`; `TEB-FBC-3R+++` |
| `OQ-A12-HANDCRAFTED-GAIN-SUFFICIENCY-TRIGGER` | Metrics | A12 | obligation resolution | Paired S5 trigger/tolerance | RESOLVED | `A12-HGST-Z0-3R++` |
| `OQ-A7-COMPONENT-RNG-ISOLATION-CONTRACT` | Reproducibility | A1,A7 | design closure | ViewDrop namespace isolation | RESOLVED | `RNG-3+`; `CRR-HRSC-3R++` |
| `OQ-A11-CONSENSUS-ACCUMULATION-REDUCTION` | Loss | A11 | design closure | Denominator-safe pair reduction | RESOLVED | `HLAC-HDSA-3R+`; `DLC-P6-DAAI-3R+` |
| `OQ-A12-HSIC-ACCUMULATION-REDUCTION` | Loss | A12 | design closure | Song-view accumulation | RESOLVED | `HLAC-HDSA-3R+`; `DLC-P6-DAAI-3R+` |
| `A1_FUSION_POST_CONCAT_PROJECTOR_CONTRACT` | Model | A1,A8,A13 | design closure | Exact projector architecture | RESOLVED | `FPP-VAH-MIN-3R+` |
| `NO_LLM_VA_HEAD_CONTRACT` | Model | No-LLM,A13 | design closure | Exact VA head/domain | RESOLVED | `FPP-VAH-MIN-3R+` |
| `NO_LLM_BASELINE_TRAINING_PROTOCOL` | Training/Loss | No-LLM,A13 | design closure | Objective/schedule/selection | RESOLVED | `LOR-FBC-3R+++`; `A13-NLBTP-VAS-3R+` |
| `METRICS_EVALUATION_PROTOCOL` | Metrics | A1–A13 | design closure | CCC/MSE/S5/test isolation | RESOLVED | `MEP-CCC2-S5-3R+`; `PER-EVAND-S5-3R++` |
| `PROMPT_GENERATION_PROTOCOL` | Prompt/CoT | applicable variants | design closure | Exact ANSWER/budget/citation profiles | RESOLVED | `GEN-DTB-VA90-3R++`; `ECA-ATOM-PREC-3R++` |
| `TRAINING_LOSS_PROTOCOL` | Training/Loss | all trainable | design closure | Stage/loss/reduction/optimizer | RESOLVED | `LOR-FBC-3R+++` |
| `PMEMO_SOURCE_VARIANT_MAPPING` | Scope/Metrics | PMEmo | design closure | A1-S5 zero-shot mapping | RESOLVED | `PME-ZST19-A1S5-3R++` |
| `OPTIONAL_CONTINGENCY_CATALOG_REGISTRATION` | Scope/Catalog | 14B,DPO,Audio,static MER | before any such run | Independent future authorization | DEFERRED_NON_BLOCKING | `PER-EVAND-S5-3R++` |
| `OQ-DATA-DEAM-TARGET-ARTIFACT-BINDING` | Data | all DEAM | any data build/formal run | Bind exact audio/annotation IDs, hashes, domains and split | OPEN | UNRESOLVED |
| `PINNED_QWEN_UPSTREAM_MANIFEST` | Upstream Model | A1–A12 applicable | loader/pre-flight | Exact Qwen revision/files/checksums/license | OPEN | UNRESOLVED |
| `QWEN_EXECUTABLE_LOADER_QUALIFICATION` | Model/Runtime | A1–A12 applicable | any Qwen run | Qualify QKB/LQ7 actual load and k-bit state | OPEN | UNRESOLVED |
| `A10_PROJECTION_EQUIVALENCE_QUALIFICATION` | Model/Runtime | A10 | A10 pre-flight | Verify hidden backbone/projection/absence of LM path | OPEN | UNRESOLVED |
| `TOKENIZER_ANSWER_BUDGET_QUALIFICATION` | Tokenizer/Prompt | generated variants | artifact/RG-08/RG-10 | Qualify B_answer/B_wrapper under pinned tokenizer | OPEN | UNRESOLVED |
| `PMEMO_ARTIFACT_HASH_DOMAIN_AUDIT` | Data/Evaluation | PMEmo | PMEmo evaluation | Hash timeline, first-15 rule, target domain | OPEN | UNRESOLVED |
| `DEAM_ANNOTATION_AUDIT` | Data/Evaluation | DEAM long58/primary | data binding | Audit coverage, target validity and domains | OPEN | UNRESOLVED |
| `CRR_EXECUTABLE_QUALIFICATION` | Reproducibility | activation-checkpoint stochastic contexts | before formal checkpointed training | Demonstrate original/recompute stochastic equivalence for per-invocation capsules; durable continuation is governed by TrainingState | OPEN | UNRESOLVED |
| `GRADIENT_FLOW_QUALIFICATION` | Training | all variants | formal training | Verify compiled trainability/modes/gradients | OPEN | UNRESOLVED |
| `NUMERICAL_POLICY_QUALIFICATION` | Training/Numerical | all variants | formal training | Verify precision, clipping, nonfinite and optimizer groups | OPEN | UNRESOLVED |
| `RG08_CANONICAL_HARDWARE_EVIDENCE` | Gate/Hardware | applicable 7B variants | LLM stage entry | Run required profiles on canonical hardware | OPEN | UNRESOLVED |
| `RG09_HUMAN_REVIEW_EXECUTION` | Gate/Human | applicable CoT artifacts | CoT Gate PASS | Real reviewers/adjudication/evidence | OPEN | UNRESOLVED |
| `MACHINE_READABLE_MANIFEST_CONFIG_BINDING` | Config/Schema | all | formal pre-flight | Realize catalog, invariants, schemas and manifests | OPEN | UNRESOLVED |
| `CLEAN_GIT_PROVENANCE_PREFLIGHT` | Provenance | all formal runs | formal pre-flight | Git repo/clean commit/dirty-state enforcement | OPEN | current directory is not a Git repository |
| `A5_MODEL_VISIBLE_INPUT_FIREWALL_QUALIFICATION` | Prompt/Testing | A5 | A5 pre-flight | Executably prove shadow Event cannot reach model input | OPEN | design resolved by `VCP-CCEW-3R++` |
| `A6_EVENT_REPLAY_BUNDLE_BINDING` | Event/Artifact | A6 | A6 pre-flight | Materialize paired replay bundle/checksum | OPEN | design resolved by `TEB-FBC-3R+++` |
| `COT_ARTIFACT_MATERIALIZATION` | CoT Artifact | A1/A2/A5/A6/A8 | Stage-2 training | Generate/version qualified artifacts | OPEN | design resolved by Batch 1/2 |
| `EVALUATION_INPUT_COVERAGE_EXECUTABLE_BINDING` | Data/Features | long58/PMEmo | secondary evaluation | Enforce three validity signals without fake events | OPEN | `EvaluationInputCoverageContract` |

The register deliberately separates resolved semantic questions from remaining execution/provenance work. No `OPEN` row authorizes reinterpretation of A1–A13 or of the three frozen Foundation batches.

## 11. Closed-World Boundary

The frozen primary-training closed world is exactly the 15 executable identities in Section 5 plus the non-executable A8 grouping identity. `PME-ZST19-A1S5-3R++` separately authorizes required A1-S5 zero-shot PMEmo evaluation without adding a trainable A-variant. 14B scaling, 14B CoT-DPO, Qwen2.5-Audio, and static MER contingency remain `DEFERRED_NON_BLOCKING` and require separately approved versioned catalog entries before any formal run.

## 12. Closure Verdict

```text
A1–A13 semantic coverage          = PASS
ID uniqueness                     = PASS
base-reference consistency        = PASS
factor uniqueness                 = PASS
research-plan coverage            = PASS
special semantic conflict audit   = PASS
primary matrix closed world       = PASS

machine-readable normalization    = PENDING
executable binding                = PENDING_DEPENDENCIES
formal experiment readiness       = NOT_READY

Section 6 semantic design         = APPROVED / FROZEN
```
