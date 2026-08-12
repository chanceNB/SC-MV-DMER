"""Runtime-only helpers for E1.1 MERT hidden-state identity observation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sc_mv_dmer.features.mert_frame_rate import E11VerificationError


def map_block_outputs_to_returned_hidden_states(
    *,
    returned_hidden_states: Sequence[object],
    captured_block_outputs: Mapping[int, object],
) -> dict[int, tuple[int, ...]]:
    """Match block outputs to returned tensors by Python object identity only."""

    candidates: dict[int, tuple[int, ...]] = {}
    for block_number in (5, 6):
        if block_number not in captured_block_outputs:
            raise E11VerificationError(f"research layer {block_number} was not captured")
        captured = captured_block_outputs[block_number]
        matches = tuple(
            index
            for index, returned in enumerate(returned_hidden_states)
            if returned is captured
        )
        if not matches:
            raise E11VerificationError(
                f"research layer {block_number} output is not present in returned hidden states"
            )
        candidates[block_number] = matches
    return candidates
