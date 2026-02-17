"""Training loop for NFL trajectory prediction model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import PlayBatch, collate_plays
from src.model import TrajectoryModel


def masked_displacement_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE loss on displacements, masked to valid output frames.

    Args:
        pred: (batch, max_frames, 2) predicted displacements.
        target: (batch, max_frames, 2) ground truth displacements.
        mask: (batch, max_frames) bool, True for valid frames.

    Returns:
        Scalar loss.
    """
    sq_err = (pred - target) ** 2
    sq_err = sq_err.sum(dim=-1)
    masked = sq_err * mask.float()
    return masked.sum() / mask.float().sum()


def train_one_epoch(
    model: TrajectoryModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    teacher_forcing_ratio: float = 1.0,
) -> float:
    """Train the model for one epoch.

    Args:
        model: TrajectoryModel instance.
        dataloader: DataLoader yielding PlayBatch objects.
        optimizer: PyTorch optimizer.
        device: Device to run on (cpu/mps/cuda).
        teacher_forcing_ratio: Probability of teacher forcing.

    Returns:
        Average loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        batch = _batch_to_device(batch, device)
        optimizer.zero_grad()
        pred = model(batch, teacher_forcing_ratio=teacher_forcing_ratio)
        loss = masked_displacement_loss(pred, batch.target_displacements, batch.output_mask)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def validate(
    model: TrajectoryModel,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Compute validation loss.

    Args:
        model: TrajectoryModel instance.
        dataloader: Validation DataLoader.
        device: Device.

    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        batch = _batch_to_device(batch, device)
        pred = model(batch, teacher_forcing_ratio=0.0)
        loss = masked_displacement_loss(pred, batch.target_displacements, batch.output_mask)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def _batch_to_device(batch: PlayBatch, device: torch.device) -> PlayBatch:
    """Move all batch tensors to the specified device.

    Args:
        batch: PlayBatch on CPU.
        device: Target device.

    Returns:
        New PlayBatch on the target device.
    """
    return PlayBatch(
        player_sequences=batch.player_sequences.to(device),
        player_mask=batch.player_mask.to(device),
        target_player_idx=batch.target_player_idx.to(device),
        play_metadata=batch.play_metadata.to(device),
        last_input_xy=batch.last_input_xy.to(device),
        target_displacements=batch.target_displacements.to(device),
        output_mask=batch.output_mask.to(device),
        num_output_frames=batch.num_output_frames.to(device),
    )
