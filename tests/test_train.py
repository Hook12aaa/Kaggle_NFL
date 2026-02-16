import torch
from src.train import masked_displacement_loss
from src.model import TrajectoryModel
from src.dataset import PlayBatch


def _make_batch(B=4, P=10, T=15):
    return PlayBatch(
        player_sequences=torch.randn(B, P, 20, 13),
        player_mask=torch.ones(B, P, dtype=torch.bool),
        target_player_idx=torch.zeros(B, dtype=torch.long),
        play_metadata=torch.randn(B, 5),
        last_input_xy=torch.randn(B, 2),
        target_displacements=torch.randn(B, T, 2),
        output_mask=torch.ones(B, T, dtype=torch.bool),
        num_output_frames=torch.full((B,), T, dtype=torch.long),
    )


def test_masked_displacement_loss():
    """Loss should only count frames where output_mask is True."""
    pred = torch.ones(2, 10, 2)
    target = torch.zeros(2, 10, 2)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[0, :5] = True
    mask[1, :3] = True
    loss = masked_displacement_loss(pred, target, mask)
    assert loss.item() > 0


def test_train_one_epoch_reduces_loss():
    """Training steps on same batch should reduce loss."""
    model = TrajectoryModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = _make_batch()

    model.train()
    loss1 = masked_displacement_loss(
        model(batch, teacher_forcing_ratio=1.0),
        batch.target_displacements,
        batch.output_mask,
    ).item()

    for _ in range(20):
        optimizer.zero_grad()
        pred = model(batch, teacher_forcing_ratio=1.0)
        loss = masked_displacement_loss(pred, batch.target_displacements, batch.output_mask)
        loss.backward()
        optimizer.step()

    loss2 = loss.item()
    assert loss2 < loss1, f"Loss did not decrease: {loss1} -> {loss2}"
