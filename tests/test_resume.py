"""Regression tests for optimizer-step to DataLoader resume conversion."""

import pytest

from library.training.resume import resolve_resume_position


def test_resume_position_with_gradient_accumulation_at_epoch_boundary():
    # 216 batches / accumulation 4 = 54 optimizer updates per epoch.
    # Step 1890 is therefore exactly the end of epoch 35, not epoch 140.
    assert resolve_resume_position(
        1890,
        batches_per_epoch=216,
        gradient_accumulation_steps=4,
    ) == (35, 0)


def test_resume_position_with_gradient_accumulation_mid_epoch():
    assert resolve_resume_position(
        1900,
        batches_per_epoch=216,
        gradient_accumulation_steps=4,
    ) == (35, 40)


def test_resume_position_handles_partial_accumulation_window_at_epoch_end():
    # ceil(218 / 4) = 55 updates. The 55th update consumes only two batches,
    # so the next step starts at the following epoch with no batch offset.
    assert resolve_resume_position(
        55,
        batches_per_epoch=218,
        gradient_accumulation_steps=4,
    ) == (1, 0)
    assert resolve_resume_position(
        54,
        batches_per_epoch=218,
        gradient_accumulation_steps=4,
    ) == (0, 216)


@pytest.mark.parametrize(
    ("global_step", "batches_per_epoch", "gradient_accumulation_steps"),
    [(-1, 10, 1), (0, 0, 1), (0, 10, 0)],
)
def test_resume_position_rejects_invalid_inputs(
    global_step, batches_per_epoch, gradient_accumulation_steps
):
    with pytest.raises(ValueError):
        resolve_resume_position(
            global_step,
            batches_per_epoch=batches_per_epoch,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
