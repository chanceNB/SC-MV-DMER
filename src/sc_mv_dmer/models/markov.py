"""Differentiable shared-transition filtering of per-view emission scores.

State-index agreement is an empirical hypothesis, not guaranteed by shared A.
These discriminative scores are not asserted to be generative HMM likelihoods.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn


class MarkovStateBridge(nn.Module):
    def __init__(self, hidden_dim: int = 256, state_count: int = 8,
                 independent_transitions: bool = False,
                 view_names: tuple[str, ...] = ('deep', 'mel', 'mfcc', 'chroma')):
        super().__init__()
        if state_count < 2:
            raise ValueError('at least two states are required')
        self.state_count = state_count
        self.view_names = view_names
        self.independent_transitions = independent_transitions
        self.prototypes = nn.ParameterDict({v: nn.Parameter(torch.randn(state_count, hidden_dim) * .02)
                                           for v in view_names})
        # At eight states: 0.65 on the diagonal, 0.05 elsewhere.
        transition = .6 * torch.eye(state_count) + .4 / state_count
        keys = view_names if independent_transitions else ('shared',)
        self.transition_logits = nn.ParameterDict({v: nn.Parameter(transition.log().clone()) for v in keys})
        # A3 isolates sharing of A; pi is shared in both conditions.
        self.initial_logits = nn.Parameter(torch.zeros(state_count))

    def transition_logits_for(self, view: str) -> Tensor:
        if view not in self.view_names:
            raise ValueError(f'unknown view {view}')
        return self.transition_logits[view if self.independent_transitions else 'shared']

    def transition(self, view: str) -> Tensor:
        return self.transition_logits_for(view).float().softmax(-1)

    def filter_log_emissions(self, log_emissions: Tensor, mask: Tensor, *, view: str) -> Tensor:
        if log_emissions.ndim != 3 or log_emissions.shape[-1] != self.state_count:
            raise ValueError('log emissions must be [batch,time,state]')
        if mask.shape != log_emissions.shape[:2] or not mask.bool().any(-1).all():
            raise ValueError('each sequence needs valid audio and a matching mask')
        batch, time, _ = log_emissions.shape
        log_a = self.transition_logits_for(view).float().log_softmax(-1)
        log_pi = self.initial_logits.float().log_softmax(-1).expand(batch, -1)
        previous = log_pi
        started = torch.zeros(batch, device=mask.device, dtype=torch.bool)
        results = []
        for index in range(time):
            prediction = torch.logsumexp(previous.unsqueeze(-1) + log_a, dim=-2)
            prior = torch.where(started[:, None], prediction, log_pi)
            evidence = log_emissions[:, index].float().masked_fill(~mask[:, index, None].bool(), 0)
            posterior = prior + evidence
            posterior = posterior - torch.logsumexp(posterior, dim=-1, keepdim=True)
            valid = mask[:, index].bool()
            previous = torch.where(valid[:, None], posterior, previous)
            started = started | valid
            results.append(previous.exp())
        return torch.stack(results, dim=1)

    def forward(self, encoded: Mapping[str, Tensor], mask: Tensor) -> dict[str, object]:
        raw, posterior, occupancy, entropy = {}, {}, {}, {}
        for name in self.view_names:
            value = encoded[name].float().masked_fill(~mask[..., None].bool(), 0)
            distance = (value.unsqueeze(-2) - self.prototypes[name].float()).square().sum(-1)
            emissions = (-distance).log_softmax(-1)
            raw[name] = emissions.exp()
            posterior[name] = self.filter_log_emissions(emissions, mask, view=name)
            valid_probabilities = posterior[name][mask.bool()]
            hard = torch.nn.functional.one_hot(valid_probabilities.argmax(-1), self.state_count).float()
            occupancy[name] = {'soft': valid_probabilities.mean(0), 'hard': hard.mean(0)}
            a = self.transition(name)
            entropy[name] = -(a * a.clamp_min(1e-30).log()).sum(-1) / math.log(self.state_count)
        return {'raw_states': raw, 'states': posterior, 'occupancy': occupancy,
                'row_entropy': entropy,
                'transitions': {name: self.transition(name) for name in self.view_names}}
