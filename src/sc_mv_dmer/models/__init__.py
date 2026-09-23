"""Executable research models; no downloads or training are performed on import."""

from .acoustic import AcousticModel
from .markov import MarkovStateBridge
from .timesnet import MertTemporalFrontend, TimesNetBlock
from .qwen import QwenDualHeadAdapter, parse_answer, format_answer, evaluate_answer_consistency, generation_token_budget

__all__ = ['AcousticModel', 'MarkovStateBridge', 'MertTemporalFrontend', 'TimesNetBlock',
           'QwenDualHeadAdapter', 'parse_answer', 'format_answer',
           'evaluate_answer_consistency', 'generation_token_budget']
