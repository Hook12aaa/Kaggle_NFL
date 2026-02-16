"""Data loading and preprocessing for NFL Big Data Bowl 2026 Prediction.

Competition task: predict (x, y) positions of select players while the ball
is in the air, using pre-throw tracking data.

Data schema
-----------
Input columns:
    game_id, play_id, player_to_predict (bool), nfl_id, frame_id,
    play_direction, absolute_yardline_number, player_name, player_height,
    player_weight, player_birth_date, player_position, player_side,
    player_role (Passer | Targeted Receiver | Other Route Runner | Defensive Coverage),
    x, y, s, a, dir, o, num_frames_output, ball_land_x, ball_land_y

Output columns (ground truth / what we predict):
    game_id, play_id, nfl_id, frame_id, x, y

Key relationships:
    - Input has ~9-14 players per play, each with ~10-67 frames at 10Hz
    - Output has only players where player_to_predict=True (~1-8, mean ~3)
    - Output frame_id restarts at 1 (first frame after the throw)
    - num_frames_output tells you how many frames to predict (5-30, mean ~11)
"""

from pathlib import Path
from typing import Optional

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_DIR = DATA_DIR / "train"
WEEKS = list(range(1, 19))


def load_train_week(week: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a single week of training data.

    Args:
        week: Week number (1-18).

    Returns:
        Tuple of (input_df, output_df) where input_df has pre-throw tracking
        and output_df has ground-truth positions while ball is in the air.
    """
    inp = pd.read_csv(TRAIN_DIR / f"input_2023_w{week:02d}.csv")
    out = pd.read_csv(TRAIN_DIR / f"output_2023_w{week:02d}.csv")
    return inp, out


def load_train(weeks: Optional[list[int]] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load multiple weeks of training data.

    Args:
        weeks: List of week numbers to load. Defaults to all 18 weeks.

    Returns:
        Tuple of (input_df, output_df) concatenated across all requested weeks.
    """
    weeks = weeks or WEEKS
    inputs, outputs = [], []
    for w in weeks:
        inp, out = load_train_week(w)
        inputs.append(inp)
        outputs.append(out)
    return pd.concat(inputs, ignore_index=True), pd.concat(outputs, ignore_index=True)


def load_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load test data for submission.

    Returns:
        Tuple of (test_template, test_input) where test_template has the
        (id, game_id, play_id, nfl_id, frame_id) rows to predict, and
        test_input has the pre-throw tracking data for all players on
        each play.
    """
    template = pd.read_csv(DATA_DIR / "test.csv")
    test_input = pd.read_csv(DATA_DIR / "test_input.csv")
    return template, test_input


def iter_plays(input_df: pd.DataFrame, output_df: Optional[pd.DataFrame] = None):
    """Iterate over plays, yielding per-play DataFrames.

    Args:
        input_df: Input tracking data (may span multiple plays).
        output_df: Ground-truth output data. If None, yields (play_input, None).

    Yields:
        Tuple of (play_input_df, play_output_df) for each unique
        (game_id, play_id) combination.
    """
    for (game_id, play_id), play_inp in input_df.groupby(["game_id", "play_id"]):
        play_out = None
        if output_df is not None:
            mask = (output_df["game_id"] == game_id) & (output_df["play_id"] == play_id)
            play_out = output_df[mask]
        yield play_inp, play_out


def train_val_split(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    val_weeks: Optional[list[int]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train and validation sets by week.

    Uses week 17-18 as default validation since the test set evaluates on
    late-season games (Weeks 14-18 of 2025 season).

    Args:
        input_df: Full input DataFrame (must contain week info in game_id).
        output_df: Full output DataFrame.
        val_weeks: Week numbers to hold out for validation. Defaults to [17, 18].

    Returns:
        Tuple of (train_input, train_output, val_input, val_output).
    """
    val_weeks = val_weeks or [17, 18]

    # Week is encoded in the file origin; we need to recover it from game_id
    # Game IDs follow pattern: YYYYMMDD## — week can be derived from the
    # file they came from. If you loaded via load_train(), add a 'week' column
    # before calling this, or pass pre-filtered DataFrames.
    raise NotImplementedError(
        "Add a 'week' column to your DataFrames before calling this, "
        "or filter by week numbers in your training loop."
    )
