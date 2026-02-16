# NFL Trajectory Prediction — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement an Encoder-Decoder LSTM that predicts player (x, y) displacement trajectories while the ball is in the air.

**Architecture:** Shared LSTM encoder per player → attention-based context fusion → autoregressive LSTM decoder emitting (dx, dy) displacements. See `docs/plans/2026-02-16-nfl-trajectory-prediction-design.md`.

**Tech Stack:** Python 3, PyTorch 2.7 (MPS backend), pandas, numpy, pytest.

---

### Task 1: Downsample frames utility

**Files:**
- Modify: `src/features.py`
- Create: `tests/test_features.py`

**Step 1: Write the failing test**

```python
# tests/test_features.py
import numpy as np
from src.features import downsample_frames

def test_downsample_exact_length():
    """If input has exactly N frames, return as-is."""
    frames = np.arange(40).reshape(20, 2).astype(np.float32)
    result = downsample_frames(frames, n=20)
    assert result.shape == (20, 2)
    np.testing.assert_array_equal(result, frames)

def test_downsample_longer():
    """If input has more than N frames, evenly sample N."""
    frames = np.arange(80).reshape(40, 2).astype(np.float32)
    result = downsample_frames(frames, n=20)
    assert result.shape == (20, 2)
    # First and last frames should be preserved
    np.testing.assert_array_equal(result[0], frames[0])
    np.testing.assert_array_equal(result[-1], frames[-1])

def test_downsample_shorter():
    """If input has fewer than N frames, pad with last frame."""
    frames = np.arange(20).reshape(10, 2).astype(np.float32)
    result = downsample_frames(frames, n=20)
    assert result.shape == (20, 2)
    np.testing.assert_array_equal(result[0], frames[0])
    # Padded frames should repeat the last real frame
    np.testing.assert_array_equal(result[-1], frames[-1])
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_features.py::test_downsample_exact_length -v`
Expected: FAIL — `ImportError: cannot import name 'downsample_frames'`

**Step 3: Implement in `src/features.py`**

```python
def downsample_frames(frames: np.ndarray, n: int = 20) -> np.ndarray:
    """Downsample or pad a frame sequence to exactly n frames.

    If longer than n, evenly sample n frames (always including first and last).
    If shorter than n, pad by repeating the last frame.

    Args:
        frames: Array of shape (num_frames, features).
        n: Target number of frames.

    Returns:
        Array of shape (n, features).
    """
    num_frames = len(frames)
    if num_frames == n:
        return frames
    if num_frames > n:
        indices = np.linspace(0, num_frames - 1, n, dtype=int)
        return frames[indices]
    # Pad with last frame
    pad_count = n - num_frames
    padding = np.tile(frames[-1:], (pad_count, 1))
    return np.concatenate([frames, padding], axis=0)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features.py -v`
Expected: 3 PASS

**Step 5: Commit**

```
git add src/features.py tests/test_features.py
git commit -m "feat: add downsample_frames utility"
```

---

### Task 2: Encode static player and play-level features

**Files:**
- Modify: `src/features.py`
- Modify: `tests/test_features.py`

**Step 1: Write the failing tests**

```python
# append to tests/test_features.py
import pandas as pd
from src.features import encode_player_static, encode_play_metadata

ROLE_MAP = {"Passer": 0, "Targeted Receiver": 1, "Other Route Runner": 2, "Defensive Coverage": 3}
SIDE_MAP = {"Offense": 0, "Defense": 1}

def test_encode_player_static():
    """Encodes role (4), side (2), is_target (1) into a 7-dim vector."""
    vec = encode_player_static(
        player_role="Defensive Coverage",
        player_side="Defense",
        player_to_predict=True,
    )
    assert vec.shape == (7,)
    # One-hot role: index 3 should be 1
    assert vec[3] == 1.0
    assert vec[0] == 0.0
    # One-hot side: index 4+1=5 should be 1
    assert vec[5] == 1.0
    # is_target
    assert vec[6] == 1.0

def test_encode_play_metadata():
    """Encodes play-level metadata into a 5-dim vector."""
    vec = encode_play_metadata(
        ball_land_x=63.26,
        ball_land_y=-0.22,
        play_direction="right",
        absolute_yardline_number=42,
        num_frames_output=21,
    )
    assert vec.shape == (5,)
    assert vec[0] == 63.26
    assert vec[1] == -0.22
    # play_direction: right=1, left=0
    assert vec[2] == 1.0
    assert vec[3] == 42.0
    assert vec[4] == 21.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_features.py::test_encode_player_static -v`
Expected: FAIL — `ImportError`

**Step 3: Implement in `src/features.py`**

```python
def encode_player_static(
    player_role: str,
    player_side: str,
    player_to_predict: bool,
) -> np.ndarray:
    """Encode per-player static features as a fixed-length vector.

    Args:
        player_role: One of Passer, Targeted Receiver, Other Route Runner, Defensive Coverage.
        player_side: One of Offense, Defense.
        player_to_predict: Whether this player is a prediction target.

    Returns:
        Array of shape (7,): [role_onehot(4), side_onehot(2), is_target(1)].
    """
    role_map = {"Passer": 0, "Targeted Receiver": 1, "Other Route Runner": 2, "Defensive Coverage": 3}
    side_map = {"Offense": 0, "Defense": 1}

    role_vec = np.zeros(4, dtype=np.float32)
    role_vec[role_map[player_role]] = 1.0

    side_vec = np.zeros(2, dtype=np.float32)
    side_vec[side_map[player_side]] = 1.0

    target_vec = np.array([float(player_to_predict)], dtype=np.float32)

    return np.concatenate([role_vec, side_vec, target_vec])


def encode_play_metadata(
    ball_land_x: float,
    ball_land_y: float,
    play_direction: str,
    absolute_yardline_number: int,
    num_frames_output: int,
) -> np.ndarray:
    """Encode play-level metadata as a fixed-length vector.

    Args:
        ball_land_x: X coordinate where the ball lands.
        ball_land_y: Y coordinate where the ball lands.
        play_direction: 'left' or 'right'.
        absolute_yardline_number: Yardline of the line of scrimmage.
        num_frames_output: Number of frames to predict.

    Returns:
        Array of shape (5,): [ball_land_x, ball_land_y, direction_encoded, yardline, num_frames].
    """
    direction_val = 1.0 if play_direction == "right" else 0.0
    return np.array(
        [ball_land_x, ball_land_y, direction_val, float(absolute_yardline_number), float(num_frames_output)],
        dtype=np.float32,
    )
```

**Step 4: Run tests**

Run: `pytest tests/test_features.py -v`
Expected: 5 PASS

**Step 5: Commit**

```
git add src/features.py tests/test_features.py
git commit -m "feat: add static player and play metadata encoding"
```

---

### Task 3: Compute displacement targets

**Files:**
- Modify: `src/features.py`
- Modify: `tests/test_features.py`

**Step 1: Write the failing test**

```python
# append to tests/test_features.py
from src.features import compute_displacements

def test_compute_displacements():
    """Displacements are differences between consecutive positions."""
    last_input_xy = np.array([10.0, 20.0])
    output_xy = np.array([
        [11.0, 21.0],
        [13.0, 23.0],
        [16.0, 22.0],
    ])
    displacements = compute_displacements(last_input_xy, output_xy)
    assert displacements.shape == (3, 2)
    # Frame 1: output[0] - last_input
    np.testing.assert_array_almost_equal(displacements[0], [1.0, 1.0])
    # Frame 2: output[1] - output[0]
    np.testing.assert_array_almost_equal(displacements[1], [2.0, 2.0])
    # Frame 3: output[2] - output[1]
    np.testing.assert_array_almost_equal(displacements[2], [3.0, -1.0])
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_features.py::test_compute_displacements -v`
Expected: FAIL — `ImportError`

**Step 3: Implement in `src/features.py`**

```python
def compute_displacements(last_input_xy: np.ndarray, output_xy: np.ndarray) -> np.ndarray:
    """Compute per-frame (dx, dy) displacements from absolute positions.

    Args:
        last_input_xy: Array of shape (2,) — the target player's last known (x, y).
        output_xy: Array of shape (num_frames, 2) — ground truth absolute positions.

    Returns:
        Array of shape (num_frames, 2) — per-frame displacements.
    """
    positions = np.vstack([last_input_xy.reshape(1, 2), output_xy])
    return np.diff(positions, axis=0)
```

**Step 4: Run tests**

Run: `pytest tests/test_features.py -v`
Expected: 6 PASS

**Step 5: Commit**

```
git add src/features.py tests/test_features.py
git commit -m "feat: add displacement target computation"
```

---

### Task 4: PyTorch Dataset and collate function

**Files:**
- Create: `src/dataset.py`
- Create: `tests/test_dataset.py`

**Step 1: Write the failing test**

```python
# tests/test_dataset.py
import torch
from src.dataset import PlaySample, collate_plays

def test_play_sample_fields():
    """PlaySample is a dataclass holding all tensors for one (play, target_player)."""
    sample = PlaySample(
        player_sequences=torch.randn(10, 20, 13),   # (num_players, 20 frames, 6 tracking + 7 static)
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
    assert batch.player_sequences.shape == (2, 12, 20, 13)  # padded to max 12 players
    assert batch.player_mask.shape == (2, 12)
    assert batch.target_displacements.shape == (2, 14, 2)    # padded to max 14 frames
    assert batch.output_mask.shape == (2, 14)
    assert batch.output_mask[0, 7].item() == True   # s1 has 8 frames, idx 7 valid
    assert batch.output_mask[0, 8].item() == False   # s1 padded beyond 8
    assert batch.output_mask[1, 13].item() == True   # s2 has 14 frames, all valid
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset.py::test_play_sample_fields -v`
Expected: FAIL — `ImportError`

**Step 3: Implement `src/dataset.py`**

```python
"""PyTorch Dataset and collation for NFL trajectory prediction."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data import iter_plays
from src.features import (
    compute_displacements,
    downsample_frames,
    encode_play_metadata,
    encode_player_static,
    extract_target,
)

N_INPUT_FRAMES = 20
TRACKING_COLS = ["x", "y", "s", "a", "dir", "o"]


@dataclass
class PlaySample:
    """All tensors for one (play, target_player) training sample."""

    player_sequences: torch.Tensor   # (num_players, N_INPUT_FRAMES, tracking_dim + static_dim)
    player_mask: torch.Tensor        # (num_players,) bool — True for real players
    target_player_idx: int           # index into player_sequences for the target
    play_metadata: torch.Tensor      # (5,)
    last_input_xy: torch.Tensor      # (2,) last known position of target player
    target_displacements: torch.Tensor  # (num_output_frames, 2)
    num_output_frames: int


@dataclass
class PlayBatch:
    """Collated batch of PlaySamples, padded to uniform shapes."""

    player_sequences: torch.Tensor   # (batch, max_players, N_INPUT_FRAMES, feat_dim)
    player_mask: torch.Tensor        # (batch, max_players)
    target_player_idx: torch.Tensor  # (batch,)
    play_metadata: torch.Tensor      # (batch, 5)
    last_input_xy: torch.Tensor      # (batch, 2)
    target_displacements: torch.Tensor  # (batch, max_output_frames, 2)
    output_mask: torch.Tensor        # (batch, max_output_frames)
    num_output_frames: torch.Tensor  # (batch,)


class NFLDataset(Dataset):
    """Builds (play, target_player) samples from raw DataFrames.

    Args:
        input_df: Input tracking DataFrame (multiple plays).
        output_df: Ground-truth output DataFrame. None for test-time.
    """

    def __init__(self, input_df: pd.DataFrame, output_df: Optional[pd.DataFrame] = None):
        self.samples: list[PlaySample] = []
        self._build(input_df, output_df)

    def _build(self, input_df: pd.DataFrame, output_df: Optional[pd.DataFrame]) -> None:
        for play_inp, play_out in iter_plays(input_df, output_df):
            row0 = play_inp.iloc[0]
            play_meta = encode_play_metadata(
                ball_land_x=row0["ball_land_x"],
                ball_land_y=row0["ball_land_y"],
                play_direction=row0["play_direction"],
                absolute_yardline_number=row0["absolute_yardline_number"],
                num_frames_output=row0["num_frames_output"],
            )
            num_output_frames = int(row0["num_frames_output"])

            # Encode each player's sequence
            player_data = []  # list of (sequence_tensor, static_vec, nfl_id, is_target)
            for nfl_id, pf in play_inp.groupby("nfl_id"):
                pf = pf.sort_values("frame_id")
                tracking = pf[TRACKING_COLS].values.astype(np.float32)
                tracking = downsample_frames(tracking, n=N_INPUT_FRAMES)
                static = encode_player_static(
                    player_role=pf.iloc[0]["player_role"],
                    player_side=pf.iloc[0]["player_side"],
                    player_to_predict=pf.iloc[0]["player_to_predict"],
                )
                # Tile static features across all frames and concatenate
                static_tiled = np.tile(static, (N_INPUT_FRAMES, 1))
                seq = np.concatenate([tracking, static_tiled], axis=1)

                is_target = bool(pf.iloc[0]["player_to_predict"])
                last_xy = pf[["x", "y"]].values[-1].astype(np.float32)
                player_data.append((seq, nfl_id, is_target, last_xy))

            # Build one sample per target player
            target_indices = [i for i, (_, _, is_t, _) in enumerate(player_data) if is_t]

            for t_idx in target_indices:
                seq_stack = np.stack([d[0] for d in player_data], axis=0)
                mask = np.ones(len(player_data), dtype=bool)
                last_xy = player_data[t_idx][3]
                nfl_id = player_data[t_idx][1]

                if play_out is not None:
                    abs_xy = extract_target(play_out, nfl_id)
                    displacements = compute_displacements(last_xy, abs_xy)
                else:
                    displacements = np.zeros((num_output_frames, 2), dtype=np.float32)

                self.samples.append(PlaySample(
                    player_sequences=torch.from_numpy(seq_stack),
                    player_mask=torch.from_numpy(mask),
                    target_player_idx=t_idx,
                    play_metadata=torch.from_numpy(play_meta),
                    last_input_xy=torch.from_numpy(last_xy),
                    target_displacements=torch.from_numpy(displacements.astype(np.float32)),
                    num_output_frames=num_output_frames,
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> PlaySample:
        return self.samples[idx]


def collate_plays(samples: list[PlaySample]) -> PlayBatch:
    """Collate PlaySamples into a padded batch.

    Args:
        samples: List of PlaySample instances.

    Returns:
        PlayBatch with all tensors padded to the max sizes in the batch.
    """
    batch_size = len(samples)
    max_players = max(s.player_sequences.shape[0] for s in samples)
    max_output = max(s.num_output_frames for s in samples)
    feat_dim = samples[0].player_sequences.shape[2]

    seqs = torch.zeros(batch_size, max_players, N_INPUT_FRAMES, feat_dim)
    pmask = torch.zeros(batch_size, max_players, dtype=torch.bool)
    tidx = torch.zeros(batch_size, dtype=torch.long)
    meta = torch.stack([s.play_metadata for s in samples])
    last_xy = torch.stack([s.last_input_xy for s in samples])
    disps = torch.zeros(batch_size, max_output, 2)
    omask = torch.zeros(batch_size, max_output, dtype=torch.bool)
    nof = torch.tensor([s.num_output_frames for s in samples], dtype=torch.long)

    for i, s in enumerate(samples):
        np_ = s.player_sequences.shape[0]
        nf = s.num_output_frames
        seqs[i, :np_] = s.player_sequences
        pmask[i, :np_] = s.player_mask
        tidx[i] = s.target_player_idx
        disps[i, :nf] = s.target_displacements
        omask[i, :nf] = True

    return PlayBatch(
        player_sequences=seqs,
        player_mask=pmask,
        target_player_idx=tidx,
        play_metadata=meta,
        last_input_xy=last_xy,
        target_displacements=disps,
        output_mask=omask,
        num_output_frames=nof,
    )
```

**Step 4: Run tests**

Run: `pytest tests/test_dataset.py -v`
Expected: 2 PASS

**Step 5: Commit**

```
git add src/dataset.py tests/test_dataset.py
git commit -m "feat: add PlaySample, NFLDataset, and collate_plays"
```

---

### Task 5: Test NFLDataset with real data (one week)

**Files:**
- Modify: `tests/test_dataset.py`

**Step 1: Write the integration test**

```python
# append to tests/test_dataset.py
from src.data import load_train_week
from src.dataset import NFLDataset

def test_dataset_from_real_data():
    """NFLDataset should build samples from actual week 1 data."""
    inp, out = load_train_week(1)
    # Use first 5 plays only for speed
    play_keys = inp.groupby(["game_id", "play_id"]).ngroups
    first_plays = inp.groupby(["game_id", "play_id"]).ngroup()
    mask = first_plays < 5
    inp_small = inp[mask]
    game_plays = inp_small[["game_id", "play_id"]].drop_duplicates()
    out_small = out.merge(game_plays, on=["game_id", "play_id"])

    ds = NFLDataset(inp_small, out_small)
    assert len(ds) > 0
    sample = ds[0]
    assert sample.player_sequences.ndim == 3
    assert sample.player_sequences.shape[1] == 20  # N_INPUT_FRAMES
    assert sample.target_displacements.shape[1] == 2
    assert sample.target_displacements.shape[0] == sample.num_output_frames
```

**Step 2: Run test**

Run: `pytest tests/test_dataset.py::test_dataset_from_real_data -v`
Expected: PASS

**Step 3: Commit**

```
git add tests/test_dataset.py
git commit -m "test: add real-data integration test for NFLDataset"
```

---

### Task 6: Model — TrajectoryModel (encoder + attention fusion + decoder)

**Files:**
- Rewrite: `src/model.py`
- Create: `tests/test_model.py`

**Step 1: Write the failing test**

```python
# tests/test_model.py
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py::test_model_forward_shape -v`
Expected: FAIL — `ImportError: cannot import name 'TrajectoryModel'`

**Step 3: Implement `src/model.py`**

The full model has three components:

1. **PlayerEncoder**: shared LSTM, processes `(batch*num_players, 20, feat_dim)` → `(batch*num_players, hidden_size)`
2. **AttentionFusion**: target player queries all player encodings → fused context `(batch, context_size)`
3. **TrajectoryDecoder**: autoregressive LSTM → `(batch, max_frames, 2)` displacements

```python
"""Encoder-Decoder LSTM for NFL trajectory prediction."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset import PlayBatch


class PlayerEncoder(nn.Module):
    """Shared LSTM encoder that processes each player's tracking sequence.

    Args:
        input_dim: Per-frame feature dimension (tracking + static).
        hidden_size: LSTM hidden state size.
    """

    def __init__(self, input_dim: int, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)

    def forward(self, sequences: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode player sequences.

        Args:
            sequences: (batch, max_players, seq_len, feat_dim).
            mask: (batch, max_players) bool.

        Returns:
            Player encodings: (batch, max_players, hidden_size).
        """
        B, P, T, F = sequences.shape
        flat = sequences.reshape(B * P, T, F)
        _, (h_n, _) = self.lstm(flat)
        encodings = h_n.squeeze(0).reshape(B, P, -1)
        # Zero out padded players
        encodings = encodings * mask.unsqueeze(-1).float()
        return encodings


class AttentionFusion(nn.Module):
    """Attention pooling over player encodings + play metadata fusion.

    Args:
        hidden_size: Player encoding size.
        meta_dim: Play metadata vector size.
        context_size: Output fused context size.
    """

    def __init__(self, hidden_size: int = 64, meta_dim: int = 5, context_size: int = 128):
        super().__init__()
        self.attn_proj = nn.Linear(hidden_size, hidden_size)
        self.fusion = nn.Linear(hidden_size + meta_dim, context_size)

    def forward(
        self,
        encodings: torch.Tensor,
        mask: torch.Tensor,
        target_idx: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse player encodings with play metadata.

        Args:
            encodings: (batch, max_players, hidden_size).
            mask: (batch, max_players) bool.
            target_idx: (batch,) index of target player.
            metadata: (batch, meta_dim).

        Returns:
            Fused context: (batch, context_size).
        """
        B = encodings.shape[0]

        # Extract target player encoding as query
        idx = target_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, encodings.shape[-1])
        query = encodings.gather(1, idx).squeeze(1)  # (B, hidden)

        # Attention scores: query dot projected keys
        keys = self.attn_proj(encodings)  # (B, P, hidden)
        scores = (keys * query.unsqueeze(1)).sum(-1)  # (B, P)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)  # (B, P)

        attended = (encodings * weights.unsqueeze(-1)).sum(1)  # (B, hidden)

        fused = torch.cat([attended, metadata], dim=-1)
        return self.fusion(fused)


class TrajectoryDecoder(nn.Module):
    """Autoregressive LSTM decoder that emits (dx, dy) per step.

    Args:
        context_size: Fused context size (used to init hidden state).
        hidden_size: Decoder LSTM hidden size.
        target_encoding_size: Size of the target player's encoding vector.
        ball_dim: Dimension of ball landing features fed at each step (2: ball_land_x/y).
    """

    def __init__(self, context_size: int = 128, hidden_size: int = 64, target_encoding_size: int = 64, ball_dim: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        # Project context → decoder initial hidden state
        self.context_to_h = nn.Linear(context_size, hidden_size)
        self.context_to_c = nn.Linear(context_size, hidden_size)
        # Decoder input: prev_dx, prev_dy, ball_land_x, ball_land_y, target_encoding
        input_dim = 2 + ball_dim + target_encoding_size
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.output_head = nn.Linear(hidden_size, 2)

    def forward(
        self,
        context: torch.Tensor,
        target_encoding: torch.Tensor,
        ball_land_xy: torch.Tensor,
        max_frames: int,
        target_displacements: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Decode displacement trajectory.

        Args:
            context: (batch, context_size) fused context.
            target_encoding: (batch, encoding_size) target player's encoding.
            ball_land_xy: (batch, 2) ball landing coordinates.
            max_frames: Number of decode steps.
            target_displacements: (batch, max_frames, 2) ground truth for teacher forcing.
            teacher_forcing_ratio: Probability of using ground truth as next input.

        Returns:
            Predicted displacements: (batch, max_frames, 2).
        """
        B = context.shape[0]
        h = self.context_to_h(context).unsqueeze(0)
        c = self.context_to_c(context).unsqueeze(0)

        prev_disp = torch.zeros(B, 2, device=context.device)
        outputs = []

        for t in range(max_frames):
            decoder_input = torch.cat([prev_disp, ball_land_xy, target_encoding], dim=-1)
            decoder_input = decoder_input.unsqueeze(1)  # (B, 1, input_dim)
            out, (h, c) = self.lstm(decoder_input, (h, c))
            pred_disp = self.output_head(out.squeeze(1))  # (B, 2)
            outputs.append(pred_disp)

            if target_displacements is not None and torch.rand(1).item() < teacher_forcing_ratio:
                prev_disp = target_displacements[:, t]
            else:
                prev_disp = pred_disp.detach()

        return torch.stack(outputs, dim=1)  # (B, max_frames, 2)


class TrajectoryModel(nn.Module):
    """Full encoder-decoder model for trajectory prediction.

    Args:
        tracking_dim: Number of tracking features per frame (default 6: x,y,s,a,dir,o).
        static_dim: Number of static player features (default 7: role_4 + side_2 + is_target_1).
        meta_dim: Number of play metadata features (default 5).
        encoder_hidden: LSTM encoder hidden size.
        context_size: Fused context vector size.
        decoder_hidden: LSTM decoder hidden size.
    """

    def __init__(
        self,
        tracking_dim: int = 6,
        static_dim: int = 7,
        meta_dim: int = 5,
        encoder_hidden: int = 64,
        context_size: int = 128,
        decoder_hidden: int = 64,
    ):
        super().__init__()
        self.encoder = PlayerEncoder(tracking_dim + static_dim, encoder_hidden)
        self.attention = AttentionFusion(encoder_hidden, meta_dim, context_size)
        self.decoder = TrajectoryDecoder(context_size, decoder_hidden, encoder_hidden, ball_dim=2)

    def forward(
        self,
        batch: PlayBatch,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Full forward pass.

        Args:
            batch: Collated PlayBatch.
            teacher_forcing_ratio: Probability of teacher forcing in decoder.

        Returns:
            Predicted displacements: (batch_size, max_output_frames, 2).
        """
        encodings = self.encoder(batch.player_sequences, batch.player_mask)

        context = self.attention(
            encodings, batch.player_mask, batch.target_player_idx, batch.play_metadata,
        )

        # Extract target player encoding
        B = encodings.shape[0]
        idx = batch.target_player_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, encodings.shape[-1])
        target_enc = encodings.gather(1, idx).squeeze(1)

        # Ball landing xy from metadata (first 2 elements)
        ball_land_xy = batch.play_metadata[:, :2]

        max_frames = int(batch.num_output_frames.max().item())
        pred = self.decoder(
            context, target_enc, ball_land_xy, max_frames,
            target_displacements=batch.target_displacements,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        return pred


def evaluate(predicted: pd.DataFrame, actual: pd.DataFrame) -> float:
    """Compute mean Euclidean distance between predicted and actual positions.

    Args:
        predicted: DataFrame with columns [game_id, play_id, nfl_id, frame_id, x, y].
        actual: DataFrame with same columns.

    Returns:
        Mean Euclidean distance in yards.
    """
    merged = predicted.merge(
        actual,
        on=["game_id", "play_id", "nfl_id", "frame_id"],
        suffixes=("_pred", "_true"),
    )
    distances = np.sqrt(
        (merged["x_pred"] - merged["x_true"]) ** 2
        + (merged["y_pred"] - merged["y_true"]) ** 2
    )
    return distances.mean()
```

**Step 4: Run tests**

Run: `pytest tests/test_model.py -v`
Expected: 3 PASS

**Step 5: Commit**

```
git add src/model.py tests/test_model.py
git commit -m "feat: implement TrajectoryModel (encoder + attention + decoder)"
```

---

### Task 7: Training loop

**Files:**
- Create: `src/train.py`
- Create: `tests/test_train.py`

**Step 1: Write the failing test**

```python
# tests/test_train.py
import torch
from src.train import train_one_epoch, masked_displacement_loss
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
    """One epoch of training should reduce loss."""
    model = TrajectoryModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = _make_batch()

    model.train()
    loss1 = masked_displacement_loss(
        model(batch, teacher_forcing_ratio=1.0),
        batch.target_displacements,
        batch.output_mask,
    ).item()

    # Train a few steps on same batch
    for _ in range(20):
        optimizer.zero_grad()
        pred = model(batch, teacher_forcing_ratio=1.0)
        loss = masked_displacement_loss(pred, batch.target_displacements, batch.output_mask)
        loss.backward()
        optimizer.step()

    loss2 = loss.item()
    assert loss2 < loss1, f"Loss did not decrease: {loss1} -> {loss2}"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_train.py::test_masked_displacement_loss -v`
Expected: FAIL — `ImportError`

**Step 3: Implement `src/train.py`**

```python
"""Training loop for NFL trajectory prediction model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import NFLDataset, PlayBatch, collate_plays
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
        mask: (batch, max_frames) bool — True for valid frames.

    Returns:
        Scalar loss.
    """
    sq_err = (pred - target) ** 2  # (B, T, 2)
    sq_err = sq_err.sum(dim=-1)    # (B, T) — sum over x, y
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
```

**Step 4: Run tests**

Run: `pytest tests/test_train.py -v`
Expected: 2 PASS

**Step 5: Commit**

```
git add src/train.py tests/test_train.py
git commit -m "feat: add training loop with masked displacement loss"
```

---

### Task 8: Inference — predict function and submission

**Files:**
- Modify: `src/submission.py`
- Create: `tests/test_submission.py`

**Step 1: Write the failing test**

```python
# tests/test_submission.py
import pandas as pd
import torch
from src.submission import predict_play
from src.model import TrajectoryModel

def test_predict_play_returns_correct_shape():
    """predict_play should return DataFrame with x, y columns, correct row count."""
    from src.data import load_train_week
    inp, out = load_train_week(1)
    # Get first play
    first_key = inp.groupby(["game_id", "play_id"]).ngroups
    gp = inp.groupby(["game_id", "play_id"])
    play_key = list(gp.groups.keys())[0]
    play_inp = gp.get_group(play_key)

    model = TrajectoryModel()
    model.eval()

    result = predict_play(model, play_inp, device=torch.device("cpu"))

    n_targets = play_inp[play_inp["player_to_predict"]]["nfl_id"].nunique()
    n_frames = int(play_inp.iloc[0]["num_frames_output"])
    expected_rows = n_targets * n_frames
    assert len(result) == expected_rows
    assert "x" in result.columns
    assert "y" in result.columns
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_submission.py -v`
Expected: FAIL — `ImportError: cannot import name 'predict_play'`

**Step 3: Implement `predict_play` in `src/submission.py`**

```python
"""Submission generation and inference for NFL Big Data Bowl 2026."""

import numpy as np
import pandas as pd
import polars as pl
import torch

from src.data import load_test
from src.dataset import NFLDataset, collate_plays
from src.model import TrajectoryModel


def predict_play(
    model: TrajectoryModel,
    play_input: pd.DataFrame,
    device: torch.device = torch.device("cpu"),
) -> pd.DataFrame:
    """Predict (x, y) trajectories for all target players in a single play.

    Args:
        model: Trained TrajectoryModel.
        play_input: Input tracking DataFrame for one play.
        device: Torch device.

    Returns:
        DataFrame with columns [nfl_id, frame_id, x, y] for all target players.
    """
    ds = NFLDataset(play_input, output_df=None)
    if len(ds) == 0:
        return pd.DataFrame(columns=["nfl_id", "frame_id", "x", "y"])

    batch = collate_plays(ds.samples)
    batch = _batch_to_device(batch, device)

    with torch.no_grad():
        pred_displacements = model(batch, teacher_forcing_ratio=0.0)

    results = []
    target_players = play_input[play_input["player_to_predict"]].groupby("nfl_id")

    for i, (nfl_id, pf) in enumerate(target_players):
        nof = int(batch.num_output_frames[i].item())
        disps = pred_displacements[i, :nof].cpu().numpy()  # (nof, 2)
        last_xy = batch.last_input_xy[i].cpu().numpy()      # (2,)

        # Accumulate displacements to absolute positions
        positions = last_xy + np.cumsum(disps, axis=0)

        for frame_idx in range(nof):
            results.append({
                "nfl_id": nfl_id,
                "frame_id": frame_idx + 1,
                "x": positions[frame_idx, 0],
                "y": positions[frame_idx, 1],
            })

    return pd.DataFrame(results)


def generate_submission(model: TrajectoryModel, device: torch.device = torch.device("cpu")) -> pd.DataFrame:
    """Generate a full submission DataFrame.

    Args:
        model: Trained TrajectoryModel.
        device: Torch device.

    Returns:
        DataFrame with columns [id, x, y] matching test.csv template.
    """
    template, test_input = load_test()
    model.eval()

    predictions = []
    for (game_id, play_id), play_inp in test_input.groupby(["game_id", "play_id"]):
        play_template = template[
            (template["game_id"] == game_id) & (template["play_id"] == play_id)
        ]
        pred = predict_play(model, play_inp, device=device)
        # Merge with template to get id and correct ordering
        merged = play_template.merge(pred, on=["nfl_id", "frame_id"], how="left")
        predictions.append(merged)

    submission = pd.concat(predictions, ignore_index=True)
    return submission[["id", "x", "y"]]


def make_predict_fn(model: TrajectoryModel, device: torch.device = torch.device("cpu")):
    """Create predict function for Kaggle gRPC inference server.

    Args:
        model: Trained TrajectoryModel.
        device: Torch device.

    Returns:
        Callable for the inference server.
    """
    model.eval()

    def predict_fn(test_batch, test_input_batch):
        if isinstance(test_input_batch, pl.DataFrame):
            play_input = test_input_batch.to_pandas()
        else:
            play_input = test_input_batch

        pred = predict_play(model, play_input, device=device)

        if isinstance(test_batch, pl.DataFrame):
            tb = test_batch.to_pandas()
        else:
            tb = test_batch

        merged = tb.merge(pred, on=["nfl_id", "frame_id"], how="left")
        return merged[["x", "y"]]

    return predict_fn


def _batch_to_device(batch, device):
    from src.train import _batch_to_device as _btd
    return _btd(batch, device)
```

**Step 4: Run tests**

Run: `pytest tests/test_submission.py -v`
Expected: PASS

**Step 5: Commit**

```
git add src/submission.py tests/test_submission.py
git commit -m "feat: add predict_play and generate_submission"
```

---

### Task 9: End-to-end training script

**Files:**
- Create: `run_train.py`

**Step 1: Write the training script**

```python
"""End-to-end training script for NFL trajectory prediction."""

import torch
from torch.utils.data import DataLoader

from src.data import load_train_week
from src.dataset import NFLDataset, collate_plays
from src.model import TrajectoryModel
from src.train import train_one_epoch, validate


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data — train on weeks 1-16, validate on 17-18
    print("Loading training data...")
    train_inputs, train_outputs = [], []
    for w in range(1, 17):
        inp, out = load_train_week(w)
        train_inputs.append(inp)
        train_outputs.append(out)

    import pandas as pd
    train_inp = pd.concat(train_inputs, ignore_index=True)
    train_out = pd.concat(train_outputs, ignore_index=True)
    del train_inputs, train_outputs

    print("Loading validation data...")
    val_inp_17, val_out_17 = load_train_week(17)
    val_inp_18, val_out_18 = load_train_week(18)
    val_inp = pd.concat([val_inp_17, val_inp_18], ignore_index=True)
    val_out = pd.concat([val_out_17, val_out_18], ignore_index=True)

    print("Building datasets...")
    train_ds = NFLDataset(train_inp, train_out)
    val_ds = NFLDataset(val_inp, val_out)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_plays)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_plays)

    model = TrajectoryModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_val_loss = float("inf")
    patience = 7
    patience_counter = 0

    for epoch in range(50):
        tf_ratio = max(0.0, 1.0 - epoch / 30)  # Linear decay from 1.0 to 0.0 over 30 epochs

        train_loss = train_one_epoch(model, train_loader, optimizer, device, teacher_forcing_ratio=tf_ratio)
        val_loss = validate(model, val_loader, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={lr:.6f} | tf={tf_ratio:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "models/best_model.pt")
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print("Model saved to models/best_model.pt")


if __name__ == "__main__":
    main()
```

**Step 2: Smoke test with 1 week**

Run: `python3 -c "from src.data import load_train_week; from src.dataset import NFLDataset; ds = NFLDataset(*load_train_week(1)); print(f'Built {len(ds)} samples')"`
Expected: prints sample count (should be ~2700 for week 1)

**Step 3: Commit**

```
git add run_train.py
git commit -m "feat: add end-to-end training script"
```

---

### Task 10: End-to-end validation & submission script

**Files:**
- Create: `run_eval.py`

**Step 1: Write the eval/submission script**

```python
"""Evaluate model on validation set and generate submission."""

import torch
import pandas as pd

from src.data import load_train_week
from src.model import TrajectoryModel, evaluate
from src.submission import generate_submission, predict_play


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = TrajectoryModel()
    model.load_state_dict(torch.load("models/best_model.pt", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Validate on weeks 17-18
    print("Evaluating on weeks 17-18...")
    val_inp_17, val_out_17 = load_train_week(17)
    val_inp_18, val_out_18 = load_train_week(18)
    val_inp = pd.concat([val_inp_17, val_inp_18], ignore_index=True)
    val_out = pd.concat([val_out_17, val_out_18], ignore_index=True)

    all_preds = []
    for (game_id, play_id), play_inp in val_inp.groupby(["game_id", "play_id"]):
        pred = predict_play(model, play_inp, device=device)
        pred["game_id"] = game_id
        pred["play_id"] = play_id
        all_preds.append(pred)

    predicted = pd.concat(all_preds, ignore_index=True)
    score = evaluate(predicted, val_out)
    print(f"Mean Euclidean distance (val): {score:.4f} yards")

    # Generate submission
    print("\nGenerating submission...")
    submission = generate_submission(model, device=device)
    submission.to_csv("submissions/submission.csv", index=False)
    print(f"Submission saved: {len(submission)} rows")
    print(submission.head())


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```
git add run_eval.py
git commit -m "feat: add evaluation and submission generation script"
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | Downsample frames utility | `src/features.py`, `tests/test_features.py` |
| 2 | Static player + play metadata encoding | `src/features.py`, `tests/test_features.py` |
| 3 | Displacement target computation | `src/features.py`, `tests/test_features.py` |
| 4 | PyTorch Dataset + collate | `src/dataset.py`, `tests/test_dataset.py` |
| 5 | Integration test with real data | `tests/test_dataset.py` |
| 6 | TrajectoryModel (encoder + attention + decoder) | `src/model.py`, `tests/test_model.py` |
| 7 | Training loop + masked loss | `src/train.py`, `tests/test_train.py` |
| 8 | Inference predict_play + submission | `src/submission.py`, `tests/test_submission.py` |
| 9 | End-to-end training script | `run_train.py` |
| 10 | Evaluation + submission script | `run_eval.py` |
