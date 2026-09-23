"""Pinned upstream DSAML ordinary-DMER model, requiring its actual features.

The four-view MERT cache is deliberately incompatible. Vendored layer arithmetic
is unchanged; this wrapper adds package isolation, VA order, explicit tail
handling, and the repository's validation-only selection interface. It does not
claim to reproduce published scores or provide an ImageBind feature cache.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


UPSTREAM_COMMIT = "20b12ea1dd8a41ba0e3072a15a19f88d8fa28496"
FEATURE_CONTRACT = "dsaml-official-20b12ea1-features-v1"
REQUIRED_FEATURES = {
    "log_mel_spectrogram": {
        "shape": ["B", 60, 51, 128], "sample_rate": 44100,
        "segment_seconds": .5, "frame_length_ms": 60, "frame_shift_ms": 10,
        "n_fft": 2646, "hop_length": 441, "n_mels": 128,
        "transform": "torchaudio MelSpectrogram defaults; natural log(power + 1e-6)",
        "time_support": "60 patches over window_start+15s through window_start+45s",
    },
    "global_imagebind_audio_embedding": {
        "shape": ["B", 1, 1024], "model": "ImageBind imagebind_huge(pretrained=True), frozen eval",
        "audio_resample_rate": 44100, "clip_duration_seconds": 2,
        "clips_per_video": 3,
        "time_support": "global embedding of supervised 30-second audio interval; repeat over its 60 steps",
    },
}


def verify_upstream_sources() -> dict:
    root = Path(__file__).with_name("_dsaml_upstream")
    metadata = json.loads((root / "UPSTREAM.json").read_text(encoding="utf-8"))
    if metadata["commit"] != UPSTREAM_COMMIT or metadata["license"] != "Apache-2.0":
        raise ValueError("DSAML upstream identity mismatch")
    for filename, info in metadata["files"].items():
        if hashlib.sha256((root / filename).read_bytes()).hexdigest() != info["vendored_sha256"]:
            raise ValueError(f"DSAML upstream file checksum mismatch: {filename}")
    return {"status": "PASS", "upstream_commit": UPSTREAM_COMMIT, "file_count": len(metadata["files"])}


def runtime_ready() -> bool:
    try:
        import transformers
        return transformers.__version__ == "4.37.2" and hasattr(transformers, "Blip2QFormerConfig")
    except ImportError:
        return False


def probe_readiness(feature_metadata: dict | None = None) -> dict:
    """Report blockers without importing upstream BLIP2 or downloading weights."""
    verify_upstream_sources()
    metadata = feature_metadata or {}
    blockers = []
    if not runtime_ready(): blockers.append("ISOLATED_TRANSFORMERS_4_37_2_RUNTIME_REQUIRED")
    if (metadata.get("feature_contract_id") != FEATURE_CONTRACT
            or not set(REQUIRED_FEATURES).issubset(metadata.get("feature_keys", []))
            or metadata.get("complete") is not True
            or not metadata.get("imagebind_checkpoint_sha256")):
        blockers.append("OFFICIAL_FEATURE_CACHE_MISSING")
    return {"status": "BLOCKED" if blockers else "READY_FOR_VERIFICATION", "blockers": blockers,
            "upstream_commit": UPSTREAM_COMMIT, "feature_contract_id": FEATURE_CONTRACT,
            "required_features": REQUIRED_FEATURES, "maml": False,
            "published_score_reproduction_claimed": False,
            "selection": "validation macro mean(CCC_V,CCC_A); test never selects checkpoint"}


def attention_regularization(attention_maps: Tensor, output_mask: Tensor, *, layer_count: int = 3) -> Tensor:
    """Upstream last-layer diagonal targets: local=.9, global=.1.

    On fully valid training clips this is exactly the source's batch/head mean
    then diagonal MSE. A valid-time extension handles right-padded tail windows.
    """
    if attention_maps.ndim != 5 or attention_maps.shape[1] != 2 * layer_count:
        raise ValueError("expected [batch,2*layers,heads,time,time] attention maps")
    if output_mask.shape != (attention_maps.shape[0], attention_maps.shape[-1]):
        raise ValueError("attention mask/time shape mismatch")
    mean_heads = attention_maps.mean(2)
    valid_counts = output_mask.sum(0)
    present = valid_counts > 0
    if not present.any(): raise ValueError("attention loss needs valid output times")
    result = attention_maps.new_zeros(())
    for index, target in ((layer_count - 1, .9), (-1, .1)):
        diagonals = torch.diagonal(mean_heads[:, index], dim1=-2, dim2=-1)
        average_diagonal = diagonals.masked_fill(~output_mask, 0).sum(0) / valid_counts.clamp_min(1)
        result = result + (average_diagonal[present] - target).square().mean()
    return result


def official_objective(output: dict, target: Tensor, mask: Tensor) -> Tensor:
    """Source objective: 1-mean(CCC, PCC, 1-RMSE over A/V)+attention.

    Separate masks and finite zero-gradient limits for degenerate correlations
    or exactly zero RMSE are explicit extensions; no targets enter forward().
    """
    prediction = output["prediction"]
    if prediction.shape != target.shape or mask.shape != target.shape or target.ndim != 3 or target.shape[-1] != 2:
        raise ValueError("DSAML objective needs matching [batch,time,2] arrays")
    mask = mask.bool()
    if not torch.isfinite(prediction[mask]).all() or not torch.isfinite(target[mask]).all():
        raise ValueError("nonfinite valid DSAML prediction or target")
    scores = []
    for batch in range(target.shape[0]):
        for dimension in range(2):
            p, y = prediction[batch, :, dimension][mask[batch, :, dimension]], target[batch, :, dimension][mask[batch, :, dimension]]
            if not len(y): raise ValueError("zero valid song dimension in DSAML objective")
            pm, ym = p.mean(), y.mean()
            pc, yc = p - pm, y - ym
            pv, yv = pc.square().mean(), yc.square().mean()
            covariance = (pc * yc).mean()
            denominator = pv + yv + (pm - ym).square()
            ccc = 2 * covariance / denominator if denominator.detach().item() > 0 else p.sum() * 0 + 1
            product = pc.square().sum() * yc.square().sum()
            pcc = (pc * yc).sum() / (product.sqrt() + 1e-8) if product.detach().item() > 0 else p.sum() * 0
            mse = (p - y).square().mean()
            rmse = mse.sqrt() if mse.detach().item() > 0 else mse * 0
            scores.extend((ccc, pcc, 1 - rmse))
    return 1 - torch.stack(scores).mean() + output["attention_regularization"]


class DSAMLModel(nn.Module):
    """Official source layer assembly with train.py's actual default parameters.

    Inputs are upstream features for the 60 supervised steps, not pooled Mel or
    MERT. Short clips use the original 30-second context. For supplemental long
    songs, each approved project window supplies its own 30-second interval;
    right-padding is removed before the source model, then restored in output.
    """

    def __init__(self):
        super().__init__()
        verify_upstream_sources()
        if not runtime_ready():
            raise RuntimeError("DSAML requires isolated transformers==4.37.2; preserve the MERT runtime")
        from ._dsaml_upstream.feature_fusion import FeatureFusionModel
        from ._dsaml_upstream.multi_scale_attention import TransformerMultiScaleAttention
        from ._dsaml_upstream.predicton import SequenceValuePredictionModel
        from ._dsaml_upstream.spectrogram_feature_extractor import SpectrogramFeatureExtractor
        self.layer_count = 3
        self.spectrogram_feature_extractor = SpectrogramFeatureExtractor()
        self.fusion_model = FeatureFusionModel(476, 1024)
        self.multi_scale_attention = TransformerMultiScaleAttention(
            origin_feature_dim=1024, hidden_size=1024, num_attention_heads=16,
            num_hidden_layers=3, dropout_rate=.1, hidden_act="gelu",
            max_position_embeddings=512, position_embedding_type="relative_key",
            local_context_length=5, global_context_length=30, device="cpu")
        self.sequence_predict_model = SequenceValuePredictionModel(1024, 256, output_size=2, dropout_rate=.1)

    def forward(self, views: dict[str, Tensor], audio_mask: Tensor) -> dict:
        if not set(REQUIRED_FEATURES).issubset(views):
            raise ValueError("DSAML requires official log-Mel patches and ImageBind features; four-view cache is incompatible")
        if audio_mask.ndim != 2 or audio_mask.shape[1] != 90:
            raise ValueError("DSAML audio_mask must be [batch,90]")
        output_mask = audio_mask[:, 30:].bool()
        if not output_mask.any(-1).all() or ((~output_mask[:, :-1]) & output_mask[:, 1:]).any():
            raise ValueError("DSAML requires contiguous valid output audio then right padding")
        batch = audio_mask.shape[0]
        mel, imagebind = views["log_mel_spectrogram"], views["global_imagebind_audio_embedding"]
        if mel.shape != (batch, 60, 51, 128) or imagebind.shape != (batch, 1, 1024):
            raise ValueError("official DSAML requires [B,60,51,128] Mel and [B,1,1024] ImageBind")
        if not torch.isfinite(mel[output_mask]).all() or not torch.isfinite(imagebind).all():
            raise ValueError("nonfinite valid official DSAML features")
        self.multi_scale_attention.multi_scale_attention.device = mel.device
        predictions = mel.new_zeros(batch, 60, 2)
        attention_maps = mel.new_zeros(batch, 6, 16, 60, 60)
        lengths = output_mask.sum(-1)
        for length in lengths.unique().tolist():
            indices = (lengths == length).nonzero().flatten()
            patches = mel[indices, :length].contiguous()
            spectral = self.spectrogram_feature_extractor(patches.unsqueeze(2))
            fused = self.fusion_model(spectral, imagebind[indices])
            hidden, maps = self.multi_scale_attention(fused, output_attentions=True)
            source_av = self.sequence_predict_model(hidden)
            predictions[indices] = F.pad(source_av[..., [1, 0]], (0, 0, 0, 60 - length))
            attention_maps[indices] = F.pad(maps, (0, 60 - length, 0, 60 - length))
        return {"prediction": predictions, "attention_maps": attention_maps,
                "attention_regularization": attention_regularization(attention_maps, output_mask, layer_count=self.layer_count),
                "output_audio_mask": output_mask, "dimension_order": ["valence", "arousal"],
                "feature_source": FEATURE_CONTRACT, "upstream_commit": UPSTREAM_COMMIT,
                "upstream_feature_cache_required": True, "published_score_reproduction_claimed": False}
