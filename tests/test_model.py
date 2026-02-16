import torch
from src.model import TrajectoryModel
from src.dataset import PlayBatch


def _make_dummy_batch(batch_size=4, max_players=12, max_output=15, feat_dim=13):
    return PlayBatch(
        player_sequences=torch.randn(batch_size, max_players, 20, feat_dim),
        player_mask=torch.ones(batch_size, max_players, dtype=torch.bool),
        target_player_idx=torch.zeros(batch_size, dtype=torch.long),
        play_metadata=torch.randn(batch_size, 5),
        last_input_xy=torch.randn(batch_size, 2),
        target_displacements=torch.randn(batch_size, max_output, 2),
        output_mask=torch.ones(batch_size, max_output, dtype=torch.bool),
        num_output_frames=torch.full((batch_size,), max_output, dtype=torch.long),
    )


def test_model_forward_shape():
    """Forward pass should return (batch, max_output_frames, 2) displacements."""
    model = TrajectoryModel(tracking_dim=6, static_dim=7, meta_dim=5)
    batch = _make_dummy_batch()
    pred_displacements = model(batch)
    assert pred_displacements.shape == (4, 15, 2)


def test_model_forward_with_teacher_forcing():
    """With teacher forcing, output shape is the same."""
    model = TrajectoryModel(tracking_dim=6, static_dim=7, meta_dim=5)
    batch = _make_dummy_batch()
    pred = model(batch, teacher_forcing_ratio=1.0)
    assert pred.shape == (4, 15, 2)


def test_model_param_count():
    """Model should be in the 150K-300K param range."""
    model = TrajectoryModel(tracking_dim=6, static_dim=7, meta_dim=5)
    n_params = sum(p.numel() for p in model.parameters())
    assert 50_000 < n_params < 500_000, f"Param count {n_params} outside expected range"
