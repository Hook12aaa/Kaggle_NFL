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
        context_size: Fused context size (used to init hidden state).
        hidden_size: Decoder LSTM hidden size.
        target_encoding_size: Size of the target player's encoding vector.
        ball_dim: Dimension of ball landing features fed at each step.
    """

    def __init__(self, context_size: int = 128, hidden_size: int = 64, target_encoding_size: int = 64, ball_dim: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.context_to_h = nn.Linear(context_size, hidden_size)
        self.context_to_c = nn.Linear(context_size, hidden_size)
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
            decoder_input = decoder_input.unsqueeze(1)
            out, (h, c) = self.lstm(decoder_input, (h, c))
            pred_disp = self.output_head(out.squeeze(1))
            outputs.append(pred_disp)

            if target_displacements is not None and torch.rand(1).item() < teacher_forcing_ratio:
                prev_disp = target_displacements[:, t]
            else:
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
