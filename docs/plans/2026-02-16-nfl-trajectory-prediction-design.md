# NFL Big Data Bowl 2026 — Trajectory Prediction Design

## Problem

Predict (x, y) positions of select players while the ball is in the air, using pre-throw tracking data. Code competition — model runs via gRPC inference server, one play at a time.

- Input: 6-14 players per play, 9-74 frames each at 10Hz with (x, y, s, a, dir, o) + metadata
- Output: (x, y) per frame for `player_to_predict=True` players (1-8 per play, Defensive Coverage + Targeted Receiver only)
- Output length varies per play (5-94 frames, median 10), given as `num_frames_output`
- Ball landing position given at inference (`ball_land_x`, `ball_land_y`)
- Training: 2023 season (18 weeks, ~4.9M input rows, ~563K output rows)
- Test: 143 plays, 5,837 rows to predict

## Approach: Encoder-Decoder LSTM with Displacement Prediction

### Data Pipeline

1. Downsample every player's input frames to fixed N=20 (evenly spaced). Each frame: `[x, y, s, a, dir, o]` (6 features).
2. Player tensor: `(num_players, 20, 6)`, padded to max 14 players with zeros + mask.
3. Per-player static features: `[player_role_encoded, player_side_encoded, player_to_predict]`.
4. Play-level metadata: `[ball_land_x, ball_land_y, play_direction_encoded, absolute_yardline_number, num_frames_output]`.
5. Target: per-frame displacements `(dx, dy)` from ground truth. Frame 1 = output_frame1 - last_input_frame. Subsequent = output_frameN - output_frameN-1.
6. One sample = one (play, target_player) pair. A play with 3 target players = 3 samples sharing encoded context.

### Model Architecture

**Encoder (shared weights):**
- Single LSTM processes each player's sequence `(20, 6 + static_dim)` independently.
- Hidden size: 64. Output: final hidden state per player → `(num_players, 64)`.

**Context Fusion:**
- Attention pooling over all player encodings, query = target player's encoding.
- Concatenate with play-level metadata → linear → fused context vector `(128,)`.

**Decoder (autoregressive):**
- LSTM decoder initialized with fused context as hidden state.
- Each step input: `[prev_dx, prev_dy, ball_land_x, ball_land_y, target_player_encoding]`. First step: `(0, 0)`.
- Output head: linear → `(dx, dy)` per step.
- Run for `num_frames_output` steps. Accumulate from last known (x, y).

**Size:** ~150K-300K parameters.

### Training

- Loss: MSE on displacements + auxiliary MSE on accumulated absolute positions.
- Teacher forcing with scheduled sampling in later epochs.
- One sample per (play, target_player) pair.
- AdamW, lr 1e-3, cosine decay, 30-50 epochs, early stopping.
- Validation: weeks 17-18 held out.
- Monitoring metric: mean Euclidean distance on absolute (x, y).

### Inference

- Encode all players → fuse context per target player → decode displacements → accumulate to absolute (x, y).
- Wrap in `make_predict_fn()` for Kaggle gRPC server.

### Compute

- M3 Max 48GB, PyTorch MPS backend.
