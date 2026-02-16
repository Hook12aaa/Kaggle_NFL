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

    player_sequences: torch.Tensor
    player_mask: torch.Tensor
    target_player_idx: int
    play_metadata: torch.Tensor
    last_input_xy: torch.Tensor
    target_displacements: torch.Tensor
    num_output_frames: int


@dataclass
class PlayBatch:
    """Collated batch of PlaySamples, padded to uniform shapes."""

    player_sequences: torch.Tensor
    player_mask: torch.Tensor
    target_player_idx: torch.Tensor
    play_metadata: torch.Tensor
    last_input_xy: torch.Tensor
    target_displacements: torch.Tensor
    output_mask: torch.Tensor
    num_output_frames: torch.Tensor


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

            player_data = []
            for nfl_id, pf in play_inp.groupby("nfl_id"):
                pf = pf.sort_values("frame_id")
                tracking = pf[TRACKING_COLS].values.astype(np.float32)
                tracking = downsample_frames(tracking, n=N_INPUT_FRAMES)
                static = encode_player_static(
                    player_role=pf.iloc[0]["player_role"],
                    player_side=pf.iloc[0]["player_side"],
                    player_to_predict=pf.iloc[0]["player_to_predict"],
                )
                static_tiled = np.tile(static, (N_INPUT_FRAMES, 1))
                seq = np.concatenate([tracking, static_tiled], axis=1)

                is_target = bool(pf.iloc[0]["player_to_predict"])
                last_xy = pf[["x", "y"]].values[-1].astype(np.float32)
                player_data.append((seq, nfl_id, is_target, last_xy))

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
