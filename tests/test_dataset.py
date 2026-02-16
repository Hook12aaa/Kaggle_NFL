import torch
from src.dataset import PlaySample, collate_plays


def test_play_sample_fields():
    """PlaySample is a dataclass holding all tensors for one (play, target_player)."""
    sample = PlaySample(
        player_sequences=torch.randn(10, 20, 13),
        player_mask=torch.ones(10, dtype=torch.bool),
        target_player_idx=2,
        play_metadata=torch.randn(5),
        last_input_xy=torch.tensor([50.0, 25.0]),
        target_displacements=torch.randn(11, 2),
        num_output_frames=11,
    )
    assert sample.player_sequences.shape == (10, 20, 13)
    assert sample.num_output_frames == 11


def test_collate_pads_players_and_output_frames():
    """Collate should pad to max players and max output frames in batch."""
    s1 = PlaySample(
        player_sequences=torch.randn(10, 20, 13),
        player_mask=torch.ones(10, dtype=torch.bool),
        target_player_idx=0,
        play_metadata=torch.randn(5),
        last_input_xy=torch.tensor([50.0, 25.0]),
        target_displacements=torch.randn(8, 2),
        num_output_frames=8,
    )
    s2 = PlaySample(
        player_sequences=torch.randn(12, 20, 13),
        player_mask=torch.ones(12, dtype=torch.bool),
        target_player_idx=1,
        play_metadata=torch.randn(5),
        last_input_xy=torch.tensor([40.0, 30.0]),
        target_displacements=torch.randn(14, 2),
        num_output_frames=14,
    )
    batch = collate_plays([s1, s2])
    assert batch.player_sequences.shape == (2, 12, 20, 13)
    assert batch.player_mask.shape == (2, 12)
    assert batch.target_displacements.shape == (2, 14, 2)
    assert batch.output_mask.shape == (2, 14)
    assert batch.output_mask[0, 7].item() == True
    assert batch.output_mask[0, 8].item() == False
    assert batch.output_mask[1, 13].item() == True


def test_dataset_from_real_data():
    """NFLDataset should build samples from actual week 1 data."""
    from src.data import load_train_week
    from src.dataset import NFLDataset

    inp, out = load_train_week(1)
    first_plays = inp.groupby(["game_id", "play_id"]).ngroup()
    mask = first_plays < 5
    inp_small = inp[mask]
    game_plays = inp_small[["game_id", "play_id"]].drop_duplicates()
    out_small = out.merge(game_plays, on=["game_id", "play_id"])

    ds = NFLDataset(inp_small, out_small)
    assert len(ds) > 0
    sample = ds[0]
    assert sample.player_sequences.ndim == 3
    assert sample.player_sequences.shape[1] == 20
    assert sample.target_displacements.shape[1] == 2
    assert sample.target_displacements.shape[0] == sample.num_output_frames
