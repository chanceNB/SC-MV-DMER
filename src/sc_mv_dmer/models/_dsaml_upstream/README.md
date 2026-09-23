# Pinned DSAML baseline source

Source: [Littleor/Personalized-DMER](https://github.com/Littleor/Personalized-DMER/tree/20b12ea1dd8a41ba0e3072a15a19f88d8fa28496), commit `20b12ea1dd8a41ba0e3072a15a19f88d8fa28496`, accompanying Dengming Zhang et al., *Personalized Dynamic Music Emotion Recognition with Dual-Scale Attention-Based Meta-Learning*, AAAI 2025.

The upstream Apache-2.0 license is included in `LICENSE`. `UPSTREAM.json` records original and local SHA-256 hashes. The seven Python source files keep the upstream layer arithmetic; imports are made relative and `sys.path` mutations/demo logger imports are removed. Modified source headers identify these changes. `dsaml.py` assembles the four components in upstream `models/PDMER.py` without importing or downloading ImageBind weights.

## Actual model and feature requirements

The wrapper follows the defaults used by upstream `train.py` and `utils/args.py`: 1024 hidden units, 3 transformer layers, 16 heads, dropout 0.1, `relative_key` positions, local radius 5 and global radius 30. The same encoder weights serve both attention scales. The upstream transformer constructor hardcodes intermediate size 2048; the command-line `intermediate_size=3072` does not reach it. Prediction is a BiLSTM plus linear layers, with original output order A/V; the wrapper reorders to V/A without adding a tanh.

Exact ordinary-DMER inputs are the 60 half-second steps over 15–45 seconds:

- `[B,60,51,128]` local log-Mel **patches**, not a pooled 128-vector: 44.1 kHz waveform, 60 ms FFT/window, 10 ms hop, 128 Mel bands, `torchaudio.transforms.MelSpectrogram` defaults, natural log(power + 1e-6). See upstream `script/dataset.py` and `utils/music/util.py`.
- `[B,1,1024]` frozen `imagebind_huge(pretrained=True)` global audio embedding of the 30-second supervised interval, broadcast over 60 steps. Upstream ImageBind audio preprocessing uses 2-second clips and 3 clips per sample, with 44.1 kHz waveform input. See upstream `PDMERModel.get_embedding`.

The current MERT/Mel/MFCC/Chroma cache supplies neither required object. MERT must never be renamed ImageBind, and pooled Mel must never be expanded or tiled into fake patches. `probe_readiness` reports `OFFICIAL_FEATURE_CACHE_MISSING` until an independently versioned, complete official-feature cache and checkpoint identity exist. No DSAML result has been produced by adding this wrapper.

## Objective and protocol adaptations

For ordinary DMER, upstream README uses `--not_using_maml`. The effective loss in `utils/train.py` is **1 minus the mean of the six values CCC_A, PCC_A, 1−RMSE_A, CCC_V, PCC_V, 1−RMSE_V**, plus attention regularization. It is not just CCC and not the `SmoothL1Loss` object declared in `train.py`. Attention loss takes the last local/global layer, averages heads and batch, and penalizes diagonal distances to 0.9 and 0.1. The wrapper exposes this objective and attention loss; finite gradient conventions at exact zero RMSE and constant series, and independent missing-label masks, are explicit extensions.

This project uses its approved 1395/174/175 split, validation-only mean V/A CCC selection, and 58 supplemental long songs. The wrapper takes `[B,90]` audio masks but uses the original 60-step supervised interval. Thus it has 30-second feature context, not the proposed model's full 45-second context; report this difference. Supplemental inference uses project windows rather than the upstream Arousal splice routine. Right-padded output features are trimmed before the source model and restored afterward; short-clip training stays on complete 60-step sequences. Future official-feature materialization must freeze its partial audio window policy and verify complete song/time coverage before training.

Use `model.eval()` and `torch.no_grad()` during validation/inference. Upstream `utils/train.py` comments out `model.eval()` and validates on its test set; that behavior is not reproduced. Baselines should be described as **pinned DSAML source under our protocol**, not reproduction of the paper's Table 1 scores.

## Runtime isolation

The upstream layers depend on BLIP2 QFormer components in `transformers==4.37.2`. The existing MERT environment uses 4.24 and is preserved. Use a separate process/runtime or isolated dependency directory for DSAML; do not change the active MERT process. Ordinary wrapper import and readiness probing work without loading BLIP2. Real model construction fails clearly if the pinned runtime is absent. CPU random-input forward tests validate software behavior, not dataset quality or model accuracy.
