"""Feature engineering for NFL Big Data Bowl 2026 Prediction.

Transforms raw tracking data into model-ready features. Each play provides
pre-throw frames for ~9-14 players at 10Hz with (x, y, s, a, dir, o) plus
metadata (player info, ball landing coords, num_frames_output).

The model must predict future (x, y) positions for players marked
player_to_predict=True.
"""

import numpy as np
import pandas as pd


def extract_play_features(play_input: pd.DataFrame) -> dict:
    """Extract features from a single play's input data.

    Args:
        play_input: DataFrame of pre-throw tracking data for one play.
            Contains all players (both to_predict and context players).

    Returns:
        Dictionary of features for this play. Structure is up to your
        model design — flat dict for tree models, nested arrays for
        sequence models, etc.
    """
    raise NotImplementedError


def extract_player_features(player_frames: pd.DataFrame) -> dict:
    """Extract features from one player's tracking sequence within a play.

    Args:
        player_frames: DataFrame of tracking frames for a single player
            on a single play. Columns: frame_id, x, y, s, a, dir, o,
            plus metadata columns.

    Returns:
        Dictionary of per-player features (e.g. velocity profile,
        last known position, acceleration trend).
    """
    raise NotImplementedError


def extract_target(play_output: pd.DataFrame, nfl_id: int) -> np.ndarray:
    """Extract target (x, y) trajectory for one player from ground truth.

    Args:
        play_output: DataFrame with columns (game_id, play_id, nfl_id,
            frame_id, x, y) for one play.
        nfl_id: The player to extract targets for.

    Returns:
        Array of shape (num_frames_output, 2) with [x, y] per frame.
    """
    player = play_output[play_output["nfl_id"] == nfl_id].sort_values("frame_id")
    return player[["x", "y"]].values


def build_dataset(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
) -> tuple[list, list]:
    """Build feature/target pairs from raw data.

    Args:
        input_df: Input tracking data (multiple plays).
        output_df: Ground-truth output data (multiple plays).

    Returns:
        Tuple of (features_list, targets_list) where each element
        corresponds to one (play, player_to_predict) pair.
    """
    raise NotImplementedError
