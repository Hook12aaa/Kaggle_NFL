"""Feature engineering for NFL Big Data Bowl 2026 Prediction.

Transforms raw tracking data into model-ready features. Each play provides
pre-throw frames for ~9-14 players at 10Hz with (x, y, s, a, dir, o) plus
metadata (player info, ball landing coords, num_frames_output).

The model must predict future (x, y) positions for players marked
player_to_predict=True.
"""

import numpy as np
import pandas as pd


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


def compute_displacements(last_input_xy: np.ndarray, output_xy: np.ndarray) -> np.ndarray:
    """Compute per-frame (dx, dy) displacements from absolute positions.

    Args:
        last_input_xy: Array of shape (2,), the target player's last known (x, y).
        output_xy: Array of shape (num_frames, 2), ground truth absolute positions.

    Returns:
        Array of shape (num_frames, 2), per-frame displacements.
    """
    positions = np.vstack([last_input_xy.reshape(1, 2), output_xy])
    return np.diff(positions, axis=0)


def extract_play_features(play_input: pd.DataFrame) -> dict:
    """Extract features from a single play's input data.

    Args:
        play_input: DataFrame of pre-throw tracking data for one play.
            Contains all players (both to_predict and context players).

    Returns:
        Dictionary of features for this play. Structure is up to your
        model design, e.g. flat dict for tree models, nested arrays for
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
