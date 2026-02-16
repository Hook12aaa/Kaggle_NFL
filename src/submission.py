"""Submission generation and inference for NFL Big Data Bowl 2026."""

import numpy as np
import pandas as pd
import polars as pl
import torch

from src.data import load_test
from src.dataset import NFLDataset, collate_plays
from src.model import TrajectoryModel
from src.train import _batch_to_device


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
        disps = pred_displacements[i, :nof].cpu().numpy()
        last_xy = batch.last_input_xy[i].cpu().numpy()

        positions = last_xy + np.cumsum(disps, axis=0)

        for frame_idx in range(nof):
            results.append({
                "nfl_id": nfl_id,
                "frame_id": frame_idx + 1,
                "x": float(positions[frame_idx, 0]),
                "y": float(positions[frame_idx, 1]),
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
