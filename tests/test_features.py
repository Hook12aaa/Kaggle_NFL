import numpy as np
from src.features import downsample_frames, encode_player_static, encode_play_metadata, compute_displacements


def test_downsample_exact_length():
    """If input has exactly N frames, return as-is."""
    frames = np.arange(40).reshape(20, 2).astype(np.float32)
    result = downsample_frames(frames, n=20)
    assert result.shape == (20, 2)
    np.testing.assert_array_equal(result, frames)


def test_downsample_longer():
    """If input has more than N frames, evenly sample N."""
    frames = np.arange(80).reshape(40, 2).astype(np.float32)
    result = downsample_frames(frames, n=20)
    assert result.shape == (20, 2)
    np.testing.assert_array_equal(result[0], frames[0])
    np.testing.assert_array_equal(result[-1], frames[-1])


def test_downsample_shorter():
    """If input has fewer than N frames, pad with last frame."""
    frames = np.arange(20).reshape(10, 2).astype(np.float32)
    result = downsample_frames(frames, n=20)
    assert result.shape == (20, 2)
    np.testing.assert_array_equal(result[0], frames[0])
    np.testing.assert_array_equal(result[-1], frames[-1])


def test_encode_player_static():
    """Encodes role (4), side (2), is_target (1) into a 7-dim vector."""
    vec = encode_player_static(
        player_role="Defensive Coverage",
        player_side="Defense",
        player_to_predict=True,
    )
    assert vec.shape == (7,)
    assert vec[3] == 1.0
    assert vec[0] == 0.0
    assert vec[5] == 1.0
    assert vec[6] == 1.0


def test_encode_play_metadata():
    """Encodes play-level metadata into a 5-dim vector."""
    vec = encode_play_metadata(
        ball_land_x=63.26,
        ball_land_y=-0.22,
        play_direction="right",
        absolute_yardline_number=42,
        num_frames_output=21,
    )
    assert vec.shape == (5,)
    np.testing.assert_almost_equal(vec[0], 63.26, decimal=2)
    np.testing.assert_almost_equal(vec[1], -0.22, decimal=2)
    assert vec[2] == 1.0
    assert vec[3] == 42.0
    assert vec[4] == 21.0


def test_compute_displacements():
    """Displacements are differences between consecutive positions."""
    last_input_xy = np.array([10.0, 20.0])
    output_xy = np.array([
        [11.0, 21.0],
        [13.0, 23.0],
        [16.0, 22.0],
    ])
    displacements = compute_displacements(last_input_xy, output_xy)
    assert displacements.shape == (3, 2)
    np.testing.assert_array_almost_equal(displacements[0], [1.0, 1.0])
    np.testing.assert_array_almost_equal(displacements[1], [2.0, 2.0])
    np.testing.assert_array_almost_equal(displacements[2], [3.0, -1.0])
