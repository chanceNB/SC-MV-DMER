"""Injectable causal-language-model adapter with separate VA and LM supervision.

No model is downloaded here. A caller supplies a pinned Qwen causal model.
Unit tests may supply a tiny randomly initialized causal model. Generated
numbers are evaluated after parsing; they never create a training loss.
"""
from __future__ import annotations

import re
from decimal import Decimal

import torch
from torch import Tensor, nn

from .acoustic import _positions


ANSWER_PAIRS = 60
_NUMBER = r'[+-]?(?:\d+(?:\.\d+)?|\.\d+)'
_PAIR = rf'\(\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\)'
_LIST = re.compile(rf'\s*VA:\s*\[\s*{_PAIR}(?:\s*,\s*{_PAIR})*\s*\]\s*')
_WRAPPER = re.compile(r'\s*(?:<THINK>(?P<think>.*?)</THINK>\s*)?<ANSWER>(?P<answer>.*?)</ANSWER>\s*', re.S)


def parse_answer(text: str) -> Tensor:
    """Parse exactly sixty ordered decimal V/A pairs, without repairs."""
    if not isinstance(text, str) or text.count('<ANSWER>') != 1 or text.count('</ANSWER>') != 1:
        raise ValueError('exactly one ANSWER wrapper is required')
    match = _WRAPPER.fullmatch(text)
    if match is None or text.count('<THINK>') > 1 or text.count('</THINK>') > 1:
        raise ValueError('invalid wrapper structure')
    body = match.group('answer')
    if _LIST.fullmatch(body) is None:
        raise ValueError('invalid decimal-pair ANSWER grammar')
    pairs = re.findall(_PAIR, body)
    if len(pairs) != ANSWER_PAIRS:
        raise ValueError('ANSWER must contain exactly 60 V/A pairs')
    if any(abs(Decimal(value)) > 1 for pair in pairs for value in pair):
        raise ValueError('ANSWER values must be finite and lie in [-1,1]')
    values = torch.tensor([[float(v), float(a)] for v, a in pairs], dtype=torch.float32)
    if not torch.isfinite(values).all() or (values.abs() > 1).any():
        raise ValueError('ANSWER values must be finite and lie in [-1,1]')
    return values


def format_answer(values: Tensor) -> str:
    values = torch.as_tensor(values).detach().cpu()
    if values.shape != (60, 2) or not torch.isfinite(values).all() or (values.abs() > 1).any():
        raise ValueError('formatting requires exactly 60 finite pairs in [-1,1]')
    pairs = ', '.join(f'({float(v):+.4f}, {float(a):+.4f})' for v, a in values)
    return f'<ANSWER>VA: [{pairs}]</ANSWER>'


def evaluate_answer_consistency(prediction: Tensor, text: str,
                                output_audio_mask: Tensor | None = None) -> dict[str, object]:
    """Detached diagnostic only. Parse failures stay explicit, never repaired."""
    if prediction.shape != (60, 2):
        raise ValueError('consistency prediction must be [60,2]')
    try:
        parsed = parse_answer(text)
    except ValueError as error:
        return {'parse_success': False, 'mse': None, 'error': str(error), 'metric_only': True}
    predicted = prediction.detach().float().cpu()
    valid = torch.isfinite(predicted)
    if output_audio_mask is not None:
        mask = output_audio_mask.detach().cpu().bool()
        if mask.shape not in ((60,), (60, 2)):
            raise ValueError('consistency mask must be [60] or [60,2]')
        valid &= mask[:, None] if mask.ndim == 1 else mask
    mse = float((predicted - parsed).square()[valid].mean()) if valid.any() else None
    return {'parse_success': True, 'mse': mse, 'error': None, 'metric_only': True,
            'valid_values': int(valid.sum())}


def generation_token_budget(tokenizer, *, explanation_tokens: int = 300) -> dict[str, object]:
    """Return measured examples plus a conservative ASCII-answer allowance.

    This is a development budgeting helper, not formal tokenizer qualification
    or a proof that a free-running model will finish in this number of tokens.
    Actual pinned-tokenizer teacher texts and generation must still be checked.
    """
    if explanation_tokens < 0:
        raise ValueError('explanation budget cannot be negative')
    examples = [format_answer(torch.full((60, 2), x)) for x in (-1., -.9999, 0., .9999, 1.)]
    encode = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    measured = max(map(encode, examples))
    # Canonical numeric answers use ASCII. Byte-level Qwen tokenization cannot
    # need more than one token per byte for this fixed serialization.
    answer_allowance = max(measured, max(len(x.encode('ascii')) for x in examples))
    wrappers = encode('<THINK></THINK>\n') + 8
    return {'answer_pairs': 60, 'explanation_tokens': explanation_tokens,
            'measured_answer_tokens': measured, 'answer_token_allowance': answer_allowance,
            'wrapper_tokens': wrappers,
            'max_new_tokens': explanation_tokens + answer_allowance + wrappers,
            'qualification': 'development_budget_requires_pinned_tokenizer_and_generation_validation'}


class QwenDualHeadAdapter(nn.Module):
    """Causal model sees prompt, ninety time tokens, then optional teacher text.

    Regression reads only the last sixty time-token positions. Because the
    backbone is causal it cannot read later explanation/ANSWER tokens. The
    pre-Qwen acoustic/patch encoders may themselves be bidirectional.
    """
    def __init__(self, backbone: nn.Module, acoustic_dim: int = 256,
                 freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone
        hidden_dim = getattr(backbone.config, 'hidden_size', None)
        if hidden_dim is None:
            hidden_dim = getattr(backbone.config, 'n_embd', None)
        if hidden_dim is None or acoustic_dim % 4:
            raise ValueError('backbone needs hidden_size and acoustic_dim must divide by four')
        self.acoustic_dim = acoustic_dim
        self.register_buffer('positions', _positions(90, acoustic_dim), persistent=False)
        layer = nn.TransformerEncoderLayer(acoustic_dim, 4, 2 * acoustic_dim,
                                           dropout=0, activation='gelu', batch_first=True)
        self.patch_encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.patch_projection = nn.Linear(acoustic_dim, hidden_dim)
        self.va_head = nn.Sequential(nn.Linear(hidden_dim, 512), nn.GELU(), nn.Linear(512, 2), nn.Tanh())
        if freeze_backbone:
            self.configure_stage('warmup')
        else:
            self.stage = 'explicit_full_backbone_debug'

    def configure_stage(self, stage: str) -> None:
        """Warmup freezes weights, not autograd; LoRA must already be injected."""
        if stage not in ('warmup', 'lora'):
            raise ValueError('stage must be warmup or lora')
        lora_names = [name for name, _ in self.backbone.named_parameters() if 'lora_' in name.lower()]
        if stage == 'lora' and not lora_names:
            raise ValueError('LoRA stage requires real, already-injected LoRA adapters')
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad_(stage == 'lora' and 'lora_' in name.lower())
        self.stage = stage

    def prepare_prefix(self, prompt_input_ids: Tensor, fused: Tensor, audio_mask: Tensor,
                       prompt_attention_mask: Tensor | None = None) -> dict[str, Tensor]:
        if fused.ndim != 3 or fused.shape[1:] != (90, self.acoustic_dim):
            raise ValueError('fused acoustic input must be [batch,90,acoustic_dim]')
        batch = fused.shape[0]
        if audio_mask.shape != (batch, 90) or not audio_mask.bool().any(-1).all():
            raise ValueError('each sequence needs valid audio on the 90-position grid')
        if prompt_input_ids.ndim != 2 or prompt_input_ids.shape[0] != batch or prompt_input_ids.shape[1] == 0:
            raise ValueError('prompt_input_ids must have nonempty [batch,prompt] shape')
        if prompt_attention_mask is None:
            prompt_attention_mask = torch.ones_like(prompt_input_ids, dtype=torch.bool)
        if prompt_attention_mask.shape != prompt_input_ids.shape or not prompt_attention_mask.bool().any(-1).all():
            raise ValueError('each prompt needs at least one valid context token')
        mask = audio_mask.bool()
        clean = fused.masked_fill(~mask[..., None], 0)
        if not torch.isfinite(clean).all():
            raise ValueError('nonfinite acoustic input on valid audio')
        encoded = self.patch_encoder(clean + self.positions.to(clean.dtype), src_key_padding_mask=~mask)
        time_tokens = self.patch_projection(encoded).masked_fill(~mask[..., None], 0)
        embeddings = self.backbone.get_input_embeddings()
        prompt_tokens = embeddings(prompt_input_ids)
        time_tokens = time_tokens.to(prompt_tokens.dtype)
        combined_mask = torch.cat([prompt_attention_mask.bool(), mask], dim=1)
        combined = torch.cat([prompt_tokens, time_tokens], dim=1)
        return {'inputs_embeds': combined, 'attention_mask': combined_mask,
                'position_ids': (combined_mask.long().cumsum(-1) - 1).clamp_min(0)}

    def forward(self, prompt_input_ids: Tensor, fused: Tensor, audio_mask: Tensor, *,
                prompt_attention_mask: Tensor | None = None,
                answer_input_ids: Tensor | None = None, answer_labels: Tensor | None = None,
                va_targets: Tensor | None = None, target_mask: Tensor | None = None) -> dict[str, object]:
        prefix = self.prepare_prefix(prompt_input_ids, fused, audio_mask, prompt_attention_mask)
        prefix_length = prefix['inputs_embeds'].shape[1]
        batch = fused.shape[0]
        labels = None
        if answer_input_ids is not None or answer_labels is not None:
            if answer_labels is None:
                answer_labels = answer_input_ids.clone()
            if answer_labels.ndim != 2 or answer_labels.shape[0] != batch:
                raise ValueError('teacher answer labels must have [batch,tokens] shape')
            teacher_mask = answer_labels != -100
            if not teacher_mask.any():
                raise ValueError('teacher answer has no supervised tokens')
            if answer_input_ids is None:
                pad_id = getattr(self.backbone.config, 'pad_token_id', None)
                answer_input_ids = answer_labels.masked_fill(~teacher_mask, 0 if pad_id is None else pad_id)
            if answer_input_ids.shape != answer_labels.shape:
                raise ValueError('teacher input IDs and labels must have the same shape')
            teacher_embeds = self.backbone.get_input_embeddings()(answer_input_ids)
            prefix['inputs_embeds'] = torch.cat([prefix['inputs_embeds'], teacher_embeds], dim=1)
            prefix['attention_mask'] = torch.cat([prefix['attention_mask'], teacher_mask], dim=1)
            prefix['position_ids'] = (prefix['attention_mask'].long().cumsum(-1) - 1).clamp_min(0)
            labels = torch.cat([torch.full((batch, prefix_length), -100, dtype=answer_labels.dtype,
                                           device=answer_labels.device), answer_labels], dim=1)
        output = self.backbone(**prefix, labels=labels, output_hidden_states=True,
                               return_dict=True, use_cache=False)
        start = prompt_input_ids.shape[1] + 30
        time_hidden = output.hidden_states[-1][:, start:start + 60]
        prediction = self.va_head(time_hidden.to(self.va_head[0].weight.dtype))
        losses = {}
        if labels is not None:
            losses['language'] = output.loss
        if va_targets is not None:
            if va_targets.shape != prediction.shape:
                raise ValueError('VA targets must match [batch,60,2]')
            valid = audio_mask[:, 30:90, None].bool().expand_as(prediction) & torch.isfinite(va_targets)
            if target_mask is not None:
                if target_mask.shape not in (prediction.shape, prediction.shape[:2]):
                    raise ValueError('target mask must be [batch,60] or [batch,60,2]')
                valid &= target_mask[..., None].bool() if target_mask.ndim == 2 else target_mask.bool()
            # Invalid target values never enter differentiable arithmetic.
            difference = prediction - va_targets.masked_fill(~valid, 0)
            count = valid.sum((1, 2))
            eligible = count > 0
            if eligible.any():
                per_song = difference.square().masked_fill(~valid, 0).sum((1, 2)) / count.clamp_min(1)
                losses['va'] = per_song[eligible].mean()
        return {'prediction': prediction, 'losses': losses, 'lm_loss': losses.get('language'),
                'va_loss': losses.get('va'), 'output_audio_mask': audio_mask[:, 30:90].bool(),
                'stage': self.stage, 'regression_uses_future_answer': False}
