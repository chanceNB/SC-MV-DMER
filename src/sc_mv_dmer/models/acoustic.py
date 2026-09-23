"""Mask-aware acoustic comparisons on the fixed 90-input / 60-output grid."""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .markov import MarkovStateBridge
from .timesnet import MertTemporalFrontend


VIEW_DIMS = {'deep': 768, 'mel': 128, 'mfcc': 40, 'chroma': 12}
VARIANTS = ('mert_mlp', 'concat', 'shared_state', 'independent_state',
            'no_state_bias', 'deep_only', 'crnn', 'bilstm_attention')


def _positions(time: int, dimensions: int) -> Tensor:
    angles = torch.arange(time).float()[:, None] * torch.exp(
        torch.arange(0, dimensions, 2).float() * (-math.log(10000.) / dimensions))
    result = torch.zeros(time, dimensions)
    result[:, 0::2], result[:, 1::2] = angles.sin(), angles.cos()
    return result


class ViewEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.register_buffer('positions', _positions(90, hidden_dim), persistent=False)
        block = nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim * 2, dropout=0,
                                           activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(block, 1, enable_nested_tensor=False)

    def forward(self, value: Tensor, mask: Tensor) -> Tensor:
        hidden = self.projection(value.masked_fill(~mask[..., None], 0)) + self.positions.to(value.dtype)
        return self.transformer(hidden, src_key_padding_mask=~mask).masked_fill(~mask[..., None], 0)


class StateFusion(nn.Module):
    def __init__(self, hidden_dim: int, state_bias: bool):
        super().__init__()
        self.state_bias = state_bias
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.keys = nn.ModuleDict({v: nn.Linear(hidden_dim, hidden_dim) for v in ('mel', 'mfcc', 'chroma')})
        self.values = nn.ModuleDict({v: nn.Linear(hidden_dim, hidden_dim) for v in ('mel', 'mfcc', 'chroma')})
        self.beta = nn.ParameterDict({v: nn.Parameter(torch.tensor(.5)) for v in self.keys}) if state_bias else None
        self.project = nn.Sequential(nn.Linear(4 * hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))

    def forward(self, views: Mapping[str, Tensor], states: Mapping[str, Tensor], mask: Tensor) -> Tensor:
        deep = views['deep']
        query = self.query(deep)
        fused = [deep]
        for name in self.keys:
            logits = query @ self.keys[name](views[name]).transpose(1, 2) / math.sqrt(query.shape[-1])
            if self.state_bias:
                agreement = states['deep'] @ states[name].transpose(1, 2)
                logits = logits + self.beta[name] * agreement.clamp_min(1e-8).log().to(logits.dtype)
            weights = logits.masked_fill(~mask[:, None], -torch.inf).softmax(-1)
            fused.append(weights @ self.values[name](views[name]))
        return self.project(torch.cat(fused, -1)).masked_fill(~mask[..., None], 0)


class AcousticModel(nn.Module):
    """Baseline architectures are local comparisons, not published reproductions.

    Passing raw MERT layers activates trainable layer aggregation. TimesNet
    requires those raw layers; cached 2-Hz features are never labelled TimesNet.
    The returned output grid always contains all sixty requested timestamps.
    """
    def __init__(self, variant: str = 'shared_state', use_timesnet: bool = False,
                 hidden_dim: int = 256, view_dropout_p: float = .15):
        super().__init__()
        if variant not in VARIANTS or hidden_dim % 4:
            raise ValueError('unknown variant or hidden_dim not divisible by four')
        if not 0 <= view_dropout_p <= 1:
            raise ValueError('view_dropout_p must lie in [0,1]')
        if use_timesnet and variant in ('crnn', 'bilstm_attention'):
            raise ValueError('handcrafted baselines do not consume MERT/TimesNet')
        self.variant, self.use_timesnet, self.hidden_dim = variant, use_timesnet, hidden_dim
        self.view_dropout_p = view_dropout_p
        self.mert_frontend = MertTemporalFrontend(use_timesnet=use_timesnet)
        self.state_variant = variant in ('shared_state', 'independent_state', 'no_state_bias')
        four_views = self.state_variant or variant == 'concat'
        self.required_views = tuple(VIEW_DIMS) if four_views else (
            ('mel',) if variant == 'crnn' else ('mfcc',) if variant == 'bilstm_attention' else ('deep',))
        self.encoders = nn.ModuleDict()
        if four_views or variant == 'deep_only':
            self.encoders = nn.ModuleDict({v: ViewEncoder(VIEW_DIMS[v], hidden_dim) for v in self.required_views})
        if variant == 'mert_mlp':
            self.deep_mlp = nn.Sequential(nn.Linear(768, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        if variant == 'crnn':
            self.conv = nn.Sequential(nn.Conv1d(128, hidden_dim, 3, padding=1), nn.GELU(),
                                      nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1), nn.GELU())
            self.recurrent = nn.GRU(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        if variant == 'bilstm_attention':
            self.recurrent = nn.LSTM(40, hidden_dim // 2, batch_first=True, bidirectional=True)
            self.temporal_attention = nn.MultiheadAttention(hidden_dim, 4, dropout=0, batch_first=True)
        if four_views:
            self.sensor_heads = nn.ModuleDict({k: nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, d))
                                              for k, d in {'rms': 1, 'brightness': 1, 'mode': 1, 'key': 12}.items()})
            self.drop_embeddings = nn.ParameterDict({v: nn.Parameter(torch.zeros(hidden_dim)) for v in ('mel', 'mfcc', 'chroma')})
        if variant == 'concat':
            self.concat_project = nn.Sequential(nn.Linear(4 * hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        if self.state_variant:
            self.bridge = MarkovStateBridge(hidden_dim, independent_transitions=variant == 'independent_state')
            self.fusion = StateFusion(hidden_dim, state_bias=variant != 'no_state_bias')
        self.va_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 2), nn.Tanh())

    def forward(self, views: Mapping[str, Tensor], audio_mask: Tensor, *,
                mert_layers: Tensor | None = None, mert_time_seconds: Tensor | None = None,
                mert_mask: Tensor | None = None) -> dict[str, object]:
        if audio_mask.ndim != 2 or audio_mask.shape[1] != 90:
            raise ValueError('audio_mask must have shape [batch,90]')
        mask = audio_mask.bool()
        if not mask.any(-1).all():
            raise ValueError('each sequence needs valid audio')
        batch = mask.shape[0]
        inputs = dict(views)
        source = ('handcrafted_only_no_mert' if self.variant in ('crnn', 'bilstm_attention')
                  else 'mert_layers_5_6_2hz_cache_no_timesnet')
        if self.use_timesnet and mert_layers is None:
            raise ValueError('TimesNet requires separate raw MERT layers and time centers')
        if mert_layers is not None:
            if self.variant in ('crnn', 'bilstm_attention'):
                raise ValueError('handcrafted baselines do not consume raw MERT layers')
            if mert_time_seconds is None or mert_mask is None:
                raise ValueError('raw MERT requires time centers and frame validity mask')
            inputs['deep'] = self.mert_frontend(mert_layers, mert_time_seconds, mert_mask, mask)
            source = 'raw_mert_layers_5_6_trainable_gate' + ('_timesnet' if self.use_timesnet else '_no_timesnet')
        clean = {}
        for name in self.required_views:
            if name not in inputs or inputs[name].shape != (batch, 90, VIEW_DIMS[name]):
                raise ValueError(f'{name} must have shape [batch,90,{VIEW_DIMS[name]}]')
            clean[name] = inputs[name].masked_fill(~mask[..., None], 0)
            if not torch.isfinite(clean[name]).all():
                raise ValueError(f'nonfinite {name} values on valid audio')
        result = {'feature_source': source, 'timesnet_implementation':
                  self.mert_frontend.timesnet.implementation_id if self.use_timesnet else None}
        encoded = {v: encoder(clean[v], mask) for v, encoder in self.encoders.items()}
        if self.variant == 'mert_mlp':
            fused = self.deep_mlp(clean['deep'])
        elif self.variant == 'deep_only':
            fused = encoded['deep']
        elif self.variant in ('crnn', 'bilstm_attention'):
            value = self.conv(clean['mel'].transpose(1, 2)).transpose(1, 2) if self.variant == 'crnn' else clean['mfcc']
            # Native clips have contiguous real input followed by right padding.
            if ((~mask[:, :-1]) & mask[:, 1:]).any():
                raise ValueError('recurrent baselines require contiguous audio then right padding')
            packed = nn.utils.rnn.pack_padded_sequence(value, mask.sum(-1).cpu(), batch_first=True, enforce_sorted=False)
            packed_result, _ = self.recurrent(packed)
            fused, _ = nn.utils.rnn.pad_packed_sequence(packed_result, batch_first=True, total_length=90)
            if self.variant == 'bilstm_attention':
                attention, _ = self.temporal_attention(fused, fused, fused, key_padding_mask=~mask, need_weights=False)
                fused = fused + attention
        else:
            result['sensors'] = {name: self.sensor_heads[name](encoded[view])
                                 for name, view in {'rms': 'mel', 'brightness': 'mfcc', 'mode': 'chroma', 'key': 'chroma'}.items()}
            result['sensors']['mode'] = result['sensors']['mode'].tanh()
            result['encoded_views'] = encoded
            downstream = dict(encoded)
            dropped = torch.zeros(batch, 3, dtype=torch.bool, device=mask.device)
            if self.training and self.view_dropout_p:
                dropped = torch.rand(batch, 3, device=mask.device) < self.view_dropout_p
                for i, name in enumerate(('mel', 'mfcc', 'chroma')):
                    downstream[name] = torch.where(dropped[:, i, None, None], self.drop_embeddings[name][None, None], encoded[name])
                    downstream[name] = downstream[name].masked_fill(~mask[..., None], 0)
            result['dropped_views'] = dropped
            if self.state_variant:
                diagnostics = self.bridge(downstream, mask)
                fused = self.fusion(downstream, diagnostics['states'], mask)
                result.update(diagnostics)
            else:
                fused = self.concat_project(torch.cat([downstream[v] for v in VIEW_DIMS], -1))
        fused = fused.masked_fill(~mask[..., None], 0)
        result.update(fused=fused, prediction=self.va_head(fused[:, 30:90]), output_audio_mask=mask[:, 30:90])
        return result
