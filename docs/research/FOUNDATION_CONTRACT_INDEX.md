# SC-MV-DMER Foundation Contract Index

> Registry status: Foundation research semantics `APPROVED / FROZEN`; formal execution `BLOCKED`. This is a human-readable registry, not an executable schema. Current authority resolves from the rows marked `CURRENT`; historical records remain append-only.

## 1. Registry Fields

- **Derivation**: `RPD` = research-plan-derived; `ED` = pre-registered engineering decision; `EI` = engineering interpretation; `EC` = engineering correction; `DEV` = explicit non-blocking source-plan deviation.
- **Variants/stage**: closed-world applicability, not permission to run. Exact execution requires the blocker register in `EXPERIMENT_MATRIX.md` to be clear.
- **Source**: `FD` = Foundation Design; `RS` = Research Spec; `EM` = Experiment Matrix; `DL` = Decisions Log.

## 2. Research-Plan and Engineering Field Separation

| Contract family | Research-plan-derived fields | Engineering decisions / interpretations / corrections |
|---|---|---|
| Batch 1 / model execution | 7B primary stack, Stage 1/2 intent, Sensor/Event/LLM roles, QLoRA/checkpointing/memory reference intent | new-run typed handoff, epoch-3 phase boundary, pseudo→Stage-1-snapshot Event curriculum, epoch-5 terminal checkpoint, capability/readiness axes, FVA and topology-qualified reuse |
| Batch 2 / objectives | VA/state/Sensor/CoT/consistency/smoothness families, 2:1 statement, optimizer/scheduler intent | exact coefficients/active sets, denominator-safe reduction, Synthetic-CoT/Native-DEAM song interpretation and deterministic partition, A13 five-epoch objective, RNG/CRR/numeric semantics |
| Batch 3 / evaluation | CCC/MSE, S5 reporting intent, 300-token statement, long-song/PMEmo and Gate goals | FP64 pooled metrics, exact S5, THINK-only 300-token correction, exact ANSWER/schema budget, coverage masks, citation atom precision, A12 tau trigger, secondary timeline/window contracts, eligibility/claim firewall |
| RG-01–RG-10 | Gate subjects and source thresholds explicitly identified in each Gate section | deterministic identities, populations, tolerances, authority, evidence, fail-fast and attempt/version rules identified in each Gate Decision |

The field-level authority remains the cited Foundation Design/Research Spec section; this table prevents an engineering operationalization from being reported as an original research-plan fact.

## 3. Foundation Batch Authorities

| Contract ID | Version/status | Domain/owner | Derivation | Dependencies | Variants/stage | Open blocker refs | Source | Supersession |
|---|---|---|---|---|---|---|---|---|
| `TEB-FBC-3R+++` | CURRENT / FROZEN | Training execution | ED | VCP, FI, PB, CS, H3, SPP, MMP, LSAM | A1–A13; Stage 1/2/inference | loader, gradient, manifests | FD §7; RS §10.1; DL DEC-0022 | supersedes prior Batch-1 proposals |
| `VCP-CCEW-3R++` | CURRENT / FROZEN | Capability compiler | ED | Experiment Matrix | A1–A13, all stages | machine-readable binding | FD §7.3 | supersedes earlier VCP proposals |
| `A10-RG08-ASAND-FVA-DP-3R+++` | CURRENT / FROZEN | A10/RG-08 | ED | A10-HBL, RG-08 | A10 Stage 2 | A10 equivalence, hardware | FD §7.4 | supersedes earlier A10 applicability |
| `LOR-FBC-3R+++` | CURRENT / FROZEN | Loss/optimization/reproducibility | ED/EI | DLC, HLAC, TSC, RNG, CRR, OSNP/OSNS | all trainable variants | CRR, gradient, numeric | FD §8; RS §10.2; DL DEC-0023 | supersedes prior Batch-2 proposals |
| `COTC-SN21-DSDP-3R+` | CURRENT / FROZEN | Synthetic-CoT/Native-DEAM song coverage | RPD/EI | population/split/artifact lineage | A1-like CoT-capable variants, Stage 2 | partition/artifact materialization, human review | FD §8.3 | current; native role is not real CoT |
| `A13-NLBTP-VAS-3R+` | CURRENT / FROZEN | A13 baseline | ED | FPP, MEP, OSNS | A13 Stage 1/selection | manifests, gradient/numeric | FD §8.3 | current |
| `MEGPC-FBC-3R++++` | CURRENT / FROZEN | Metrics/evaluation/global stage | ED/EI/EC | MEP, ECA, A12, secondary eval, GEN, FPP, SGA, PER | A1–A13; M0–M20 | data/tokenizer/hardware/human/manifests | FD §9; RS §11–13; DL DEC-0024 | supersedes `MEGPC-FBC-3R+++` |

## 4. Research Gates

| Contract ID | Version/status | Domain/owner | Derivation | Dependencies | Variants/stage | Open blocker refs | Source | Supersession |
|---|---|---|---|---|---|---|---|---|
| `RG-01` | CURRENT / FROZEN | MERT time axis | RPD/ED | pinned MERT manifest | all Deep-view variants; feature gate | DEAM binding, manifests | FD §5.9; RS §8.9; DL DEC-0016 | current |
| `RG-02` | CURRENT / FROZEN | Handcrafted bin/alignment | RPD/ED + manual authority | valid RG-01 | handcrafted variants; feature gate | data audit, human review package | FD §5.10; RS §8.10; DL DEC-0017 | current |
| `RG-03` | CURRENT / FROZEN | Sensor independent validation | RPD/ED | Sensor artifact/contract | Sensor-capable profiles | data, gradient, manifests | FD §5.1; RS §8.1; DL DEC-0007/0009 | current |
| `RG-04` | CURRENT / FROZEN | No-LLM vs CRNN | ED threshold | frozen CRNN and S5 | No-LLM gate only | executable baselines/metrics | FD §5.2; RS §8.2; DL DEC-0008 | current |
| `RG-05` | CURRENT / FROZEN | Markov occupancy | RPD/ED | Markov/eval profile | applicable Markov variants | data/manifests | FD §5.4; RS §8.4; DL DEC-0011 | current |
| `RG-06` | CURRENT / FROZEN | beta stability | RPD/ED | >=3 valid checkpoints | beta-capable variants | training evidence | FD §5.3; RS §8.3; DL DEC-0010 | current |
| `RG-07` | CURRENT / FROZEN | Event token length | RPD/ED | exact serializer/tokenizer | Event-visible source profiles | tokenizer/model manifest | FD §5.5; RS §8.5; DL DEC-0012 | new serializer requires new attempt, threshold stable |
| `RG-08` | CURRENT / FROZEN | LLM memory feasibility | RPD/ED | VCP + StageDefinition + Context, QKB/LQ7, GEN | stage/context-compiled applicable LLM roles; P1 Stage1 excludes LoRA, P1 Stage2 includes applicable LoRA | loader, tokenizer, canonical hardware | FD §5.6/§7.4; RS §8.6; DL DEC-0013/0026 | profile-qualified attempts |
| `RG-09` | CURRENT / FROZEN | synthetic CoT artifact quality | RPD/ED + human authority | CoT artifact/profile | CoT-target variants | artifact materialization, human execution | FD §5.7; RS §8.7; DL DEC-0014 | not final-generation human eval |
| `RG-10-v2` | CURRENT / FROZEN | ANSWER parser stability | RPD/ED/authority sync | GEN, canonical TimeGrid, parser/schema/tokenizer | generated ANSWER profiles; parsed V/A counts each equal full T independent of target mask | tokenizer, loader, hardware | FD §5.8; RS §8.8; DL DEC-0026 | supersedes historical `RG-10-v1`/DEC-0015 count interpretation; no attempts created |
| `RG-10-v1` | HISTORICAL / SUPERSEDED | ANSWER parser stability | RPD/ED | DEC-0015 Sample/TimeGrid/mask count interpretation | historical only | N/A | DL DEC-0015 | superseded by `RG-10-v2`; no historical attempt is asserted |

## 5. Stage, Capability, and Training Contracts

| Contract ID | Version/status | Domain | Derivation | Dependencies | Variants/stage | Blocker refs | Source/supersession |
|---|---|---|---|---|---|---|---|
| `VERSIONED_STAGE_CONTRACT_GRAPH` | CURRENT / FROZEN | Stage graph | ED | SGA, Gates | A1–A13; Stage 1/2 | manifests | FD §7 |
| `SCHEME-B+` | CURRENT / FROZEN | Event curriculum/artifact boundary | ED | PB, Event artifacts | Stage 2 Event-capable | artifact materialization | FD §7.2 |
| `FI-A+` | CURRENT / FROZEN | Final inference Event source | ED | integrated Sensor, coverage | A1-like final inference | coverage/gradient | FD §7.2 |
| `PB-1+` | CURRENT / FROZEN | Stage-2 phase boundary | ED | logical epochs, CRR | Stage 2 | CRR qualification | FD §7.2 |
| `CS-3+` | CURRENT / FROZEN | Checkpoint selection | ED | complete curriculum | formal training | manifests | FD §7.2 |
| `H3+` | CURRENT / FROZEN | Stage handoff | ED | Stage 1 terminal artifact | Stage 1→2 | artifact binding | FD §7.2 |
| `SPP-3R+` | CURRENT / FROZEN | Parameter policy | ED | VCP | all trainable stages | gradient qualification | FD §7.3; supersedes `SPP-3R` |
| `MMP-GT-3R+` | CURRENT / FROZEN | Mode/gradient transparency | ED | VCP/SPP | all trainable stages | gradient qualification | FD §7.3 |
| `LSAM-3R+` | CURRENT / FROZEN | Loss activation matrix | ED | VCP/DLC | all stages | machine binding | FD §8.1 |
| `MTP-LRSI-3R+` | CURRENT / FROZEN | Markov transition parameterization | ED | RG-05/DLC-P4 | Markov variants | numerical qualification | FD §8 refs; replaces partially approved MTP-1+ |
| `TSC-GPESA-3R+` | CURRENT / FROZEN | Sampler | ED | S5/RNG | all trainable | CRR qualification | FD §8.4 |

## 6. Loss, Reduction, RNG, and Numerical Contracts

| Contract ID | Version/status | Domain | Derivation | Dependencies | Variants/stage | Blocker refs | Source/supersession |
|---|---|---|---|---|---|---|---|
| `DLC-P1-MESM-3R+` | CURRENT / FROZEN | VA/mask/smoothness | RPD/EI | canonical VA/masks | applicable stages | numeric/data | FD §8.2 |
| `DLC-P2-TFES-3R+` | CURRENT / FROZEN | Teacher-forced CoT | RPD/EI | COTC/GEN | CoT target training | artifact/tokenizer | FD §8.2 |
| `DLC-P3-ARSA-3R+` | CURRENT / FROZEN | Autoregressive consistency | RPD/ED | GEN/parser | generated profiles | tokenizer/hardware | FD §8.2 |
| `DLC-P4-NEHF-3R+` | CURRENT / FROZEN | Markov entropy/state | RPD/EI | MTP/RG-05 | Markov variants | numeric | FD §8.2 |
| `DLC-P5-FHST-3R+` | CURRENT / FROZEN | Flat-Head Sum Sensor Supervision | RPD/ED | Sensor Heads, DLC-P6/HLAC reduction | A1/A3/A4/A5/A7/A8/A9/A10/A11/triggered A12; A2/A6/A13 N/A | gradient/numeric | `L_sensor=L_rms+L_brightness+L_mode+L_key`, 12-way tonic, mode `[-1,1]`, lambda 0.1; FD §8.2; supersedes rejected `DLC-P5-VBFS-3R+` |
| `DLC-P6-DAAI-3R+` | CURRENT / FROZEN | Auxiliary accumulation | ED | HLAC/TSC | auxiliary-loss variants | numeric | FD §8.2 |
| `DLC-P7-KLV-CN-3R+` | CURRENT / FROZEN | Kendall log variance | RPD/EI | active eligible tasks | applicable train stages | optimizer/numeric | FD §8.2 |
| `HLAC-HDSA-3R+` | CURRENT / FROZEN | Hierarchical reduction | ED | TSC/DLC | all trainable | numeric | FD §8.1 |
| `RNG-3+` | CURRENT / FROZEN | RNG domains | ED | SS-3+ | all trainable | CRR | FD §8.4 |
| `SS-3+` | CURRENT / FROZEN | S5 selection | ED | project identity | all trainable | none semantic | FD §8.4 |
| `CRR-HRSC-3R++` | CURRENT / FROZEN | Per-checkpoint-invocation stochastic recomputation replay | ED | component RNG roles, activation checkpoint invocation | checkpoint original/recompute within an execution; not durable continuation | `CRR_EXECUTABLE_QUALIFICATION` | invocation/RNG entry-exit/shadow trace/capsule hashes; durable model/optimizer/scheduler/sampler/Stage state belongs to TrainingState; FD §8.4; DL DEC-0026 |
| `OSNP-SLAW-3R++` | CURRENT / FROZEN | Optimizer/scheduler/numeric | RPD/ED | DLC/HLAC | all trainable | numerical/gradient | FD §8.4 |
| `OSNS-DB10-3R+` | CURRENT / FROZEN | Numeric subcontract | RPD/ED | OSNP | all trainable | numerical | FD §8.4 |

## 7. Model Loading and Architecture Contracts

| Contract ID | Version/status | Domain | Derivation | Dependencies | Variants/stage | Blocker refs | Source/supersession |
|---|---|---|---|---|---|---|---|
| `LQ7-R16A32-ZD-NF4-3R+` | CURRENT / FROZEN | 7B Stage-2 QLoRA | RPD/ED | pinned Qwen, VCP/SPP stage role | applicable Qwen7B Stage 2 contexts; Stage 1 LoRA N/A | upstream/loader | FD §5.6/§8.4; DL DEC-0026 |
| `QKB-SPMS-3R++` | CURRENT / FROZEN | Base loading/k-bit prep | ED | upstream manifest | Qwen7B variants | upstream/loader | FD §8.4 |
| `A10-HBL-DSP-3R++` | CURRENT / FROZEN | A10 hidden backbone | ED | QKB/LQ7 | A10 | loader/equivalence | FD §7.4 |
| `FPP-VAH-MIN-3R+` | CURRENT / FROZEN | Projector/VA heads | ED | VCP | A1/A8/A10/A13/No-LLM | gradient/numeric | FD §9.8; RS §11.7 |

## 8. Metrics, Evaluation, and Eligibility Contracts

| Contract ID | Version/status | Domain | Derivation | Dependencies | Variants/stage | Blocker refs | Source/supersession |
|---|---|---|---|---|---|---|---|
| `MEP-CCC2-S5-3R+` | CURRENT / FROZEN | Metrics/S5 aggregation | RPD/ED | masks/S5 | all evidence | data/manifests | FD §9.2 |
| `ECA-ATOM-PREC-3R++` | CURRENT / FROZEN | Event citation | ED | Event serializer | A1/A5/A6 applicable | artifact/parser binding | FD §9.5; supersedes `ECA-ATOM-PREC-3R+` |
| `A12-HGST-Z0-3R++` | CURRENT / FROZEN | A12 trigger | ED | valid A1/A2 S5 | A12 obligation | A1/A2 evidence | FD §9.6; supersedes `A12-HGST-Z0-3R+` |
| `DEAM-L58-SW45-3R++` | CURRENT / FROZEN | Long-song evaluation | RPD/ED | A1 S5/coverage | A1 secondary | DEAM audit/coverage | FD §9.7; supersedes `DEAM-L58-SW45-3R+` |
| `PME-ZST19-A1S5-3R++` | CURRENT / FROZEN | PMEmo zero-shot | RPD/ED | A1 S5/coverage | A1 secondary | PMEmo audit | FD §9.7; supersedes `PME-ZST19-A1S5-3R+` |
| `GEN-DTB-VA90-3R++` | CURRENT / FROZEN | Generation budget/schema | EC/ED | pinned tokenizer | generated profiles | tokenizer qualification | FD §9.3; supersedes `GEN-DTB-VA90-3R+` |
| `EvaluationInputCoverageContract` | CURRENT / FROZEN | Secondary masks | ED | TimeGrid/masks | long58/PMEmo | executable coverage binding | FD §9.4 |
| `SGA-M0M20-EV-3R+` | CURRENT / FROZEN | Stage/Gate applicability | ED | Gates/VCP/effective validity | M0–M20 | manifests | FD §9.9 |
| `PER-EVAND-S5-3R++` | CURRENT / FROZEN | Paper eligibility | ED | SGA/S5/invalidation | aggregation/closure | all execution blockers | FD §9.9 |
| `EvaluationScopeDeviationManifest` | CURRENT / FROZEN | Source-plan deviation | DEV | claim firewall | reporting | deferred evidence | FD §9.10 |

## 9. Variant Applicability Summary

| Variant | Event semantics | Qwen/generation | Distinct Gate/evidence profile |
|---|---|---|---|
| A1 | endogenous Sensor Event; final FI-A+ | full dual head | full applicable profile; source for secondary eval |
| A2 | no Sensor/Event; event-free artifact | full generated profile | event-free RG-09; Sensor/Event Gates N/A |
| A3 | endogenous; four independent A | full | per-view-A RG-05; own RG-06 |
| A4 | endogenous; state bias absent | full | RG-05 diagnostic-only; RG-06 N/A |
| A5 | local shadow Event, model-invisible | full but no Event injection | RG-07 N/A; shadow citation reference |
| A6 | paired immutable A1 replay; no local Sensor | full | local RG-03 N/A; replay/gradient evidence |
| A7 | endogenous; no View Dropout | full | A1/A7 stress suite; isolated RNG |
| A8 children | active-view-scoped Sensor/Event | full | active-set Gates and RG-08 per child |
| A9 | endogenous; TimesNet bypass | full | RG-01 exact reuse + bypass evidence |
| A10 | Event-conditioned hidden backbone | no LM generation/parser | RG-04/09/10 N/A; own VA-only RG-08 |
| A11 | endogenous | full | own RG-05/06; KL diagnostics |
| A12 | endogenous | full | conditional trigger; own RG-05/06/HSIC diagnostics |
| A13 | no Sensor/Event/Markov | no Qwen/CoT | exact RG-01/02 reuse only; independent baseline |

## 10. Current Execution Blocker Classes

The canonical row-level register is `EXPERIMENT_MATRIX.md` §10. Current blocker classes are: DEAM/PMEmo artifact binding and audits; pinned Qwen manifest and loader qualification; A10 equivalence; tokenizer ANSWER budget; CRR replay; gradient and numerical qualification; RG-08 canonical hardware; RG-09 real-human execution; machine-readable manifests/configs; clean Git provenance; A5 firewall; A6 replay materialization; CoT artifacts; and executable evaluation coverage. They do not reopen frozen research semantics.
