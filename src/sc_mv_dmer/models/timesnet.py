"""Raw-layer frontend and a compact TimesNet-style period-convolution block.

This is an explicit FFT top-k / 2D-convolution implementation, not a verbatim
reproduction of the official TimesNet model. Its identity must be reported.
Raw time resolution is retained until after trainable processing.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TimesNetBlock(nn.Module):
    implementation_id = 'compact-fft-period-conv2d-v1-not-official-reproduction'

    def __init__(self, feature_dim: int = 768, channels: int = 32, top_k: int = 3):
        super().__init__()
        self.top_k = top_k
        self.input_projection = nn.Linear(feature_dim, channels)
        self.period_convolution = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                                               nn.GELU(),
                                               nn.Conv2d(channels, channels, 3, padding=1))
        self.output_projection = nn.Linear(channels, feature_dim)
        self.normalization = nn.LayerNorm(feature_dim)

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        clean = values.masked_fill(~mask[..., None].bool(), 0)
        hidden = self.input_projection(clean).masked_fill(~mask[..., None].bool(), 0)
        time = hidden.shape[1]
        if time < 2:
            raise ValueError('TimesNet requires at least two raw time frames')
        outputs = []
        # Per-song period selection avoids coupling predictions to batch peers.
        for song in hidden:
            amplitude = torch.fft.rfft(song.float(), dim=0).abs().mean(-1)
            frequencies = torch.topk(amplitude[1:], k=min(self.top_k, len(amplitude) - 1)).indices + 1
            branches = []
            for frequency in frequencies.detach().tolist():
                period = max(1, time // frequency)
                padded_time = math.ceil(time / period) * period
                padded = F.pad(song, (0, 0, 0, padded_time - time))
                grid = padded.reshape(padded_time // period, period, song.shape[-1]).permute(2, 0, 1)[None]
                encoded = self.period_convolution(grid)
                restored = encoded[0].permute(1, 2, 0).reshape(padded_time, -1)[:time]
                branches.append(restored)
            weights = amplitude[frequencies].softmax(0).to(song.dtype)
            outputs.append((torch.stack(branches, -1) * weights).sum(-1))
        result = self.normalization(clean + self.output_projection(torch.stack(outputs)))
        return result.masked_fill(~mask[..., None].bool(), 0)


class MertTemporalFrontend(nn.Module):
    def __init__(self, use_timesnet: bool = True, feature_dim: int = 768):
        super().__init__()
        self.feature_dim = feature_dim
        self.layer_logits = nn.Parameter(torch.zeros(2))
        self.timesnet = TimesNetBlock(feature_dim) if use_timesnet else None

    def forward(self, layers: Tensor, time_seconds: Tensor, raw_mask: Tensor,
                audio_mask: Tensor) -> Tensor:
        if layers.ndim != 4 or layers.shape[1] != 2 or layers.shape[-1] != self.feature_dim:
            raise ValueError('raw MERT layers must have shape [batch,2,raw_time,feature]')
        batch, _, raw_time, _ = layers.shape
        if raw_mask.shape != (batch, raw_time) or audio_mask.shape != (batch, 90):
            raise ValueError('raw MERT and 90-bin audio masks must match their time grids')
        if not raw_mask.bool().any(-1).all() or not audio_mask.bool().any(-1).all():
            raise ValueError('each sequence needs valid audio')
        time_seconds = torch.as_tensor(time_seconds, device=layers.device)
        if time_seconds.ndim == 1:
            time_seconds = time_seconds.expand(batch, -1)
        if time_seconds.shape != (batch, raw_time) or not torch.isfinite(time_seconds).all():
            raise ValueError('raw MERT time centers must be finite and match raw_time')
        if not (time_seconds[:, 1:] > time_seconds[:, :-1]).all():
            raise ValueError('raw MERT time centers must be strictly increasing')
        bin_index = torch.floor(time_seconds * 2).long()
        if ((bin_index < 0) | (bin_index >= 90)).any():
            raise ValueError('raw MERT frame centers must belong to the 45-second window')
        clean = layers.masked_fill(~raw_mask[:, None, :, None].bool(), 0)
        if not torch.isfinite(clean).all():
            raise ValueError('nonfinite MERT values on valid raw frames')
        mixed = (clean * self.layer_logits.softmax(0)[None, :, None, None]).sum(1)
        if self.timesnet is not None:
            mixed = self.timesnet(mixed, raw_mask)
        sums = mixed.new_zeros(batch, 90, self.feature_dim)
        counts = mixed.new_zeros(batch, 90, 1)
        sums.scatter_add_(1, bin_index[..., None].expand_as(mixed), mixed.masked_fill(~raw_mask[..., None].bool(), 0))
        counts.scatter_add_(1, bin_index[..., None], raw_mask[..., None].to(mixed.dtype))
        # A real audio tail can be shorter than one convolutional receptive
        # field (DEAM2053). Zero is an explicit unavailable deep feature, not
        # a reason to delete its target or the whole song. Other views and
        # temporal context still support the fixed-grid prediction.
        empty = (counts[..., 0] == 0) & audio_mask.bool()
        indices = torch.arange(90, device=layers.device)[None]
        last_audio_bin = indices.masked_fill(~audio_mask.bool(), -1).max(-1).values
        if (empty & (indices != last_audio_bin[:, None])).any():
            raise ValueError('empty MERT bin inside an audio-covered interval')
        return (sums / counts.clamp_min(1)).masked_fill(~audio_mask[..., None].bool(), 0)
