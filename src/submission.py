"""Submission generation and inference server for NFL Big Data Bowl 2026.

This is a code competition — the model runs via a gRPC inference server
that receives one play at a time as (test_batch, test_input_batch) and
returns a DataFrame with columns [x, y].

For local testing, generate_submission() produces the full submission
DataFrame. For Kaggle submission, the InferenceServer class wraps
your predict function for the competition runtime.
"""

import pandas as pd
import polars as pl

from src.data import load_test
from src.model import predict


def generate_submission(model: object) -> pd.DataFrame:
    """Generate a full submission DataFrame for local validation or upload.

    Args:
        model: Trained model object from model.train().

    Returns:
        DataFrame with columns [id, x, y] matching the test.csv template,
        one row per (play, player_to_predict, frame_id).
    """
    template, test_input = load_test()

    predictions = []
    for (game_id, play_id), play_inp in test_input.groupby(["game_id", "play_id"]):
        play_template = template[
            (template["game_id"] == game_id) & (template["play_id"] == play_id)
        ]
        pred = predict(model, play_inp)
        pred["id"] = play_template["id"].values
        predictions.append(pred)

    submission = pd.concat(predictions, ignore_index=True)
    return submission[["id", "x", "y"]]


def make_predict_fn(model: object):
    """Create a predict function compatible with the Kaggle inference server.

    The gateway sends (test_batch, test_input_batch) per play.
    test_batch is a polars DataFrame with [id, game_id, play_id, nfl_id, frame_id].
    test_input_batch is a polars DataFrame with all input tracking columns.
    Must return a DataFrame with columns [x, y] (no id column), same row count
    as test_batch.

    Args:
        model: Trained model object.

    Returns:
        Callable that takes (test_batch, test_input_batch) and returns
        a DataFrame with [x, y] predictions.
    """

    def predict_fn(test_batch, test_input_batch):
        if isinstance(test_input_batch, pl.DataFrame):
            play_input = test_input_batch.to_pandas()
        else:
            play_input = test_input_batch

        pred = predict(model, play_input)

        if isinstance(pred, pd.DataFrame):
            return pred[["x", "y"]]
        return pred

    return predict_fn
