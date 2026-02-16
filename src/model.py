"""Model training, prediction, and evaluation for NFL Big Data Bowl 2026.

The prediction target is (x, y) coordinates for each output frame, for
each player marked player_to_predict=True. Output frame counts vary per
play (5-30 frames).

Evaluation metric: mean Euclidean distance between predicted and actual
(x, y) positions across all frames and players (custom metric — confirmed
as location-based comparison via NGS).
"""

import numpy as np
import pandas as pd


def train(features: list, targets: list, **kwargs) -> object:
    """Train the prediction model.

    Args:
        features: List of feature dicts/arrays, one per (play, player) pair.
        targets: List of target arrays, each shape (num_frames, 2) for [x, y].
        **kwargs: Model-specific hyperparameters.

    Returns:
        Trained model object.
    """
    raise NotImplementedError


def predict(model: object, play_input: pd.DataFrame) -> pd.DataFrame:
    """Predict (x, y) trajectories for all player_to_predict players in a play.

    Args:
        model: Trained model object from train().
        play_input: Input tracking DataFrame for a single play (all players).

    Returns:
        DataFrame with columns [x, y] — one row per (player_to_predict,
        frame_id) pair, ordered to match the test.csv template rows for
        this play.
    """
    raise NotImplementedError


def evaluate(predicted: pd.DataFrame, actual: pd.DataFrame) -> float:
    """Compute mean Euclidean distance between predicted and actual positions.

    Args:
        predicted: DataFrame with columns [game_id, play_id, nfl_id,
            frame_id, x, y] of predicted positions.
        actual: DataFrame with same columns of ground-truth positions.

    Returns:
        Mean Euclidean distance in yards across all rows.
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
