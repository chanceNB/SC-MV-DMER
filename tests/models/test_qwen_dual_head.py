import pytest
import torch

from sc_mv_dmer.models.qwen import (
    QwenDualHeadAdapter, parse_answer, format_answer,
    evaluate_answer_consistency, generation_token_budget,
)


def tiny_backbone():
    try:
        from transformers import Qwen2Config, Qwen2ForCausalLM
        return Qwen2ForCausalLM(Qwen2Config(vocab_size=128, hidden_size=32,
            intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=4096,
            bos_token_id=1, eos_token_id=2, pad_token_id=0))
    except ImportError:
        from transformers import GPT2Config, GPT2LMHeadModel
        return GPT2LMHeadModel(GPT2Config(vocab_size=128, n_embd=32,
            n_layer=1, n_head=4, n_positions=4096, resid_pdrop=0,
            embd_pdrop=0, attn_pdrop=0, bos_token_id=1, eos_token_id=2, pad_token_id=0))


def sample():
    return torch.tensor([[1, 3, 4]]), torch.randn(1, 90, 16), torch.ones(1, 90, dtype=torch.bool)


def test_real_causal_backbone_has_separate_supervision_and_no_consistency_loss():
    model = QwenDualHeadAdapter(tiny_backbone(), acoustic_dim=16, freeze_backbone=False).eval()
    prompt, fused, mask = sample()
    answers = torch.tensor([[5, 6, 7, 2]])
    result = model(prompt, fused, mask, answer_labels=answers,
                   va_targets=torch.zeros(1, 60, 2), target_mask=torch.ones(1, 60, dtype=torch.bool))
    assert result['prediction'].shape == (1, 60, 2)
    assert set(result['losses']) == {'va', 'language'}
    assert result['losses']['language'].requires_grad
    sum(result['losses'].values()).backward()
    assert model.patch_projection.weight.grad.abs().sum() > 0
    assert model.va_head[0].weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.parameters())
    assert all('consist' not in name and 'sigma' not in name for name, _ in model.named_parameters())


def test_future_answer_does_not_change_regression_and_padding_values_are_ignored():
    model = QwenDualHeadAdapter(tiny_backbone(), acoustic_dim=16).eval()
    prompt, fused, mask = sample()
    mask[:, 87:] = False
    with torch.no_grad():
        first = model(prompt, fused, mask, answer_labels=torch.tensor([[5, 6, 2]]))['prediction']
        altered = fused.masked_fill(~mask[..., None], float('nan'))
        second = model(prompt, altered, mask, answer_labels=torch.tensor([[9, 8, 7, 2]]))['prediction']
    torch.testing.assert_close(first, second)


def test_frozen_backbone_still_backpropagates_to_time_tokens():
    model = QwenDualHeadAdapter(tiny_backbone(), acoustic_dim=16)
    prompt, fused, mask = sample()
    model(prompt, fused, mask)['prediction'].square().mean().backward()
    assert all(p.grad is None and not p.requires_grad for p in model.backbone.parameters())
    assert model.patch_projection.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='LoRA'):
        model.configure_stage('lora')


def test_output_schema_is_exactly_sixty_finite_pairs():
    values = torch.linspace(-1, 1, 120).reshape(60, 2)
    text = format_answer(values)
    parsed = parse_answer('<THINK>可核验的声学解释</THINK>\n' + text)
    torch.testing.assert_close(parsed, values, atol=5.1e-5, rtol=0)
    for bad in [text.replace('(\u002d1.0000', '(nan'), text.replace('-1.0000', '-1.1000'),
                text + text, format_answer(values).replace(', (+1.0000', ', (+2.0000'),
                '<ANSWER>VA: [(0,0)]</ANSWER>']:
        if bad != text:
            with pytest.raises(ValueError):
                parse_answer(bad)


def test_consistency_is_detached_evaluation_only_and_reports_parse_failure():
    prediction = torch.zeros(60, 2, requires_grad=True)
    result = evaluate_answer_consistency(prediction, format_answer(prediction))
    assert result['parse_success'] and result['mse'] == 0.0
    assert isinstance(result['mse'], float)
    bad = evaluate_answer_consistency(prediction, '<ANSWER>broken</ANSWER>')
    assert not bad['parse_success'] and bad['mse'] is None


def test_budget_includes_answer_and_wrapper_not_only_explanation():
    class CharacterTokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(text)
    result = generation_token_budget(CharacterTokenizer(), explanation_tokens=300)
    assert result['max_new_tokens'] > 300
    assert result['answer_pairs'] == 60


def test_adapter_rejects_entirely_missing_audio_and_empty_teacher_supervision():
    model = QwenDualHeadAdapter(tiny_backbone(), acoustic_dim=16)
    prompt, fused, mask = sample()
    with pytest.raises(ValueError, match='valid audio'):
        model(prompt, fused, torch.zeros_like(mask))
    with pytest.raises(ValueError, match='supervised tokens'):
        model(prompt, fused, mask, answer_labels=torch.full((1, 5), -100))


def test_out_of_range_decimal_cannot_be_repaired_by_float32_rounding():
    text = format_answer(torch.ones(60, 2)).replace('+1.0000', '+1.000000001', 1)
    with pytest.raises(ValueError, match='lie in'):
        parse_answer(text)


def test_complete_sixty_pair_teacher_text_is_actually_supervised():
    model = QwenDualHeadAdapter(tiny_backbone(), acoustic_dim=16).eval()
    prompt, fused, mask = sample()
    text = '<THINK>Energy rises.</THINK>\n' + format_answer(torch.zeros(60, 2))
    labels = torch.tensor([[ord(character) for character in text]])
    result = model(prompt, fused, mask, answer_labels=labels)
    assert torch.isfinite(result['lm_loss'])
    result['lm_loss'].backward()
    assert model.patch_projection.weight.grad.abs().sum() > 0
