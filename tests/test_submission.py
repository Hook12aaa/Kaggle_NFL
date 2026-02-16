import torch
from src.submission import predict_play
from src.model import TrajectoryModel


def test_predict_play_returns_correct_shape():
    """predict_play should return DataFrame with x, y columns, correct row count."""
    from src.data import load_train_week

    inp, out = load_train_week(1)
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
