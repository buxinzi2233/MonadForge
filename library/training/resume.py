"""Helpers for translating optimizer-step resume state into loop positions."""

from __future__ import annotations


def resolve_resume_position(
    global_step: int,
    *,
    batches_per_epoch: int,
    gradient_accumulation_steps: int,
) -> tuple[int, int]:
    """Return ``(epoch_index, batch_offset)`` for a saved optimizer step.

    ``global_step`` is counted in optimizer updates, while DataLoader skipping
    is counted in micro-batches.  A partially filled accumulation window at the
    end of an epoch still produces one optimizer update, so simply multiplying
    the global step by ``gradient_accumulation_steps`` is not correct when the
    DataLoader length is not divisible by the accumulation factor.
    """
    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    if batches_per_epoch <= 0:
        raise ValueError("batches_per_epoch must be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")

    updates_per_epoch = (
        batches_per_epoch + gradient_accumulation_steps - 1
    ) // gradient_accumulation_steps
    epoch_index, update_offset = divmod(global_step, updates_per_epoch)
    batch_offset = min(
        update_offset * gradient_accumulation_steps,
        batches_per_epoch,
    )
    return epoch_index, batch_offset
