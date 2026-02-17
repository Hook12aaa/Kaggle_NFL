"""
NFL Big Data Bowl 2026 — Trajectory Prediction Submission
=========================================================
Encoder-Decoder LSTM with displacement prediction.
Predicts (x, y) positions of players whilst the ball is in the air.
"""

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

N_INPUT_FRAMES = 20
TRACKING_COLS = ["x", "y", "s", "a", "dir", "o"]


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
    pad_count = n - num_frames
    padding = np.tile(frames[-1:], (pad_count, 1))
    return np.concatenate([frames, padding], axis=0)


def encode_player_static(player_role: str, player_side: str, player_to_predict: bool) -> np.ndarray:
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


def encode_play_metadata(ball_land_x, ball_land_y, play_direction, absolute_yardline_number, num_frames_output) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class PlaySample:
    """Tensors for one (play, target_player) sample."""
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


def iter_plays(input_df, output_df=None):
    """Iterate over plays, yielding per-play DataFrames.

    Args:
        input_df: Input tracking data (may span multiple plays).
        output_df: Ground-truth output data. If None, yields (play_input, None).

    Yields:
        Tuple of (play_input_df, play_output_df) for each unique (game_id, play_id).
    """
    for (game_id, play_id), play_inp in input_df.groupby(["game_id", "play_id"]):
        play_out = None
        if output_df is not None:
            mask = (output_df["game_id"] == game_id) & (output_df["play_id"] == play_id)
            play_out = output_df[mask]
        yield play_inp, play_out


class NFLDataset(Dataset):
    """Builds (play, target_player) samples from raw DataFrames.

    Args:
        input_df: Input tracking DataFrame (multiple plays).
        output_df: Ground-truth output DataFrame. None for test-time inference.
    """

    def __init__(self, input_df, output_df=None):
        self.samples = []
        self._build(input_df, output_df)

    def _build(self, input_df, output_df):
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

                # No ground truth at inference time
                displacements = np.zeros((num_output_frames, 2), dtype=np.float32)

                self.samples.append(PlaySample(
                    player_sequences=torch.from_numpy(seq_stack),
                    player_mask=torch.from_numpy(mask),
                    target_player_idx=t_idx,
                    play_metadata=torch.from_numpy(play_meta),
                    last_input_xy=torch.from_numpy(last_xy),
                    target_displacements=torch.from_numpy(displacements),
                    num_output_frames=num_output_frames,
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_plays(samples):
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


# ---------------------------------------------------------------------------
# Model — Encoder-Decoder LSTM with attention
# ---------------------------------------------------------------------------

class PlayerEncoder(nn.Module):
    """Shared LSTM encoder that processes each player's tracking sequence.

    Args:
        input_dim: Per-frame feature dimension (tracking + static).
        hidden_size: LSTM hidden state size.
    """

    def __init__(self, input_dim, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)

    def forward(self, sequences, mask):
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
        encodings = encodings * mask.unsqueeze(-1).float()
        return encodings


class AttentionFusion(nn.Module):
    """Attention pooling over player encodings with play metadata fusion.

    Args:
        hidden_size: Player encoding size.
        meta_dim: Play metadata vector size.
        context_size: Output fused context size.
    """

    def __init__(self, hidden_size=64, meta_dim=5, context_size=128):
        super().__init__()
        self.attn_proj = nn.Linear(hidden_size, hidden_size)
        self.fusion = nn.Linear(hidden_size + meta_dim, context_size)

    def forward(self, encodings, mask, target_idx, metadata):
        """Fuse player encodings with play metadata via attention.

        Args:
            encodings: (batch, max_players, hidden_size).
            mask: (batch, max_players) bool.
            target_idx: (batch,) index of target player.
            metadata: (batch, meta_dim).

        Returns:
            Fused context: (batch, context_size).
        """
        idx = target_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, encodings.shape[-1])
        query = encodings.gather(1, idx).squeeze(1)

        keys = self.attn_proj(encodings)
        scores = (keys * query.unsqueeze(1)).sum(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)

        attended = (encodings * weights.unsqueeze(-1)).sum(1)
        fused = torch.cat([attended, metadata], dim=-1)
        return self.fusion(fused)


class TrajectoryDecoder(nn.Module):
    """Autoregressive LSTM decoder that emits (dx, dy) per step.

    Args:
        context_size: Fused context size (used to initialise hidden state).
        hidden_size: Decoder LSTM hidden size.
        target_encoding_size: Size of the target player's encoding vector.
        ball_dim: Dimension of ball landing features fed at each step.
    """

    def __init__(self, context_size=128, hidden_size=64, target_encoding_size=64, ball_dim=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.context_to_h = nn.Linear(context_size, hidden_size)
        self.context_to_c = nn.Linear(context_size, hidden_size)
        input_dim = 2 + ball_dim + target_encoding_size
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.output_head = nn.Linear(hidden_size, 2)

    def forward(self, context, target_encoding, ball_land_xy, max_frames,
                target_displacements=None, teacher_forcing_ratio=0.0):
        """Decode displacement trajectory autoregressively.

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
            decoder_input = decoder_input.unsqueeze(1)
            out, (h, c) = self.lstm(decoder_input, (h, c))
            pred_disp = self.output_head(out.squeeze(1))
            outputs.append(pred_disp)
            prev_disp = pred_disp.detach()

        return torch.stack(outputs, dim=1)


class TrajectoryModel(nn.Module):
    """Full encoder-decoder model for trajectory prediction.

    Args:
        tracking_dim: Number of tracking features per frame (default 6).
        static_dim: Number of static player features (default 7).
        meta_dim: Number of play metadata features (default 5).
        encoder_hidden: LSTM encoder hidden size.
        context_size: Fused context vector size.
        decoder_hidden: LSTM decoder hidden size.
    """

    def __init__(self, tracking_dim=6, static_dim=7, meta_dim=5,
                 encoder_hidden=64, context_size=128, decoder_hidden=64):
        super().__init__()
        self.encoder = PlayerEncoder(tracking_dim + static_dim, encoder_hidden)
        self.attention = AttentionFusion(encoder_hidden, meta_dim, context_size)
        self.decoder = TrajectoryDecoder(context_size, decoder_hidden, encoder_hidden, ball_dim=2)

    def forward(self, batch, teacher_forcing_ratio=0.0):
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

        idx = batch.target_player_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, encodings.shape[-1])
        target_enc = encodings.gather(1, idx).squeeze(1)
        ball_land_xy = batch.play_metadata[:, :2]

        max_frames = int(batch.num_output_frames.max().item())
        pred = self.decoder(
            context, target_enc, ball_land_xy, max_frames,
            target_displacements=batch.target_displacements,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        return pred


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def batch_to_device(batch, device):
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


def predict_play(model, play_input, device):
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
    batch = batch_to_device(batch, device)

    with torch.no_grad():
        pred_displacements = model(batch, teacher_forcing_ratio=0.0)

    results = []
    target_players = play_input[play_input["player_to_predict"]].groupby("nfl_id")

    for i, (nfl_id, pf) in enumerate(target_players):
        nof = int(batch.num_output_frames[i].item())
        disps = pred_displacements[i, :nof].cpu().numpy()
        last_xy = batch.last_input_xy[i].cpu().numpy()

        # Accumulate displacements to get absolute positions
        positions = last_xy + np.cumsum(disps, axis=0)

        for frame_idx in range(nof):
            results.append({
                "nfl_id": nfl_id,
                "frame_id": frame_idx + 1,
                "x": float(positions[frame_idx, 0]),
                "y": float(positions[frame_idx, 1]),
            })

    return pd.DataFrame(results)


def make_predict_fn(model, device):
    """Create the predict function for Kaggle's gRPC inference server.

    Args:
        model: Trained TrajectoryModel.
        device: Torch device.

    Returns:
        Callable that accepts (test_batch, test_input_batch) and returns
        a DataFrame with columns [x, y].
    """
    model.eval()

    def predict(test_batch, test_input_batch):
        # The gateway sends polars DataFrames — convert to pandas
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

    return predict


# ---------------------------------------------------------------------------
# Load model and start the inference server
# ---------------------------------------------------------------------------

import os
import glob

# Find model weights — path varies between initial run and competition rerun
candidates = glob.glob("/kaggle/input/**/best_model.pt", recursive=True)
MODEL_PATH = candidates[0] if candidates else "/kaggle/input/nfl-bdb-2026-trajectory-model/best_model.pt"
print(f"Loading model from: {MODEL_PATH}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TrajectoryModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.to(device)
model.eval()

predict_fn = make_predict_fn(model, device)
print("Model loaded and predict function ready")

# Register with the competition's gRPC inference server
import sys
sys.path.insert(0, "/kaggle/input/competitions/nfl-big-data-bowl-2026-prediction/")
from kaggle_evaluation.nfl_inference_server import NFLInferenceServer

server = NFLInferenceServer(predict_fn)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    server.serve()
else:
    server.run_local_gateway(
        data_paths=("/kaggle/input/competitions/nfl-big-data-bowl-2026-prediction/",)
    )
