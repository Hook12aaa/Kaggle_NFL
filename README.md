# NFL Big Data Bowl 2026 - Predicting Player Movement

I came across the [NFL Big Data Bowl 2026 Prediction competition](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction) a bit late, but I couldn't resist giving it a shot. The idea of predicting where players will move *while the ball is in the air* — using only what happened before the throw — is such a fun problem. It sits right at the intersection of physics, human behavior, and sports strategy.

## The Problem

On every passing play in football, there's a moment between when the quarterback releases the ball and when it arrives. During that window, receivers are running routes, defenders are reacting, and everyone is adjusting. The competition asks: **can you predict those movements before they happen?**

You're given pre-throw tracking data for every player on the field — positions, speeds, accelerations, orientations, sampled at 10Hz — along with where the ball will land. Your job is to output the (x, y) coordinates of select players for every frame while the ball is in the air.

## My Approach: Encoder-Decoder LSTM with Displacement Prediction

Rather than predicting absolute positions directly, the model predicts **frame-by-frame displacements** (dx, dy) from each player's last known position. This is more physically grounded — players have momentum and inertia, so modeling the *change* in position makes the learning problem smoother.

The architecture has three stages:

**1. Shared Player Encoder** — A single LSTM processes each player's pre-throw tracking sequence (downsampled to 20 frames). Every player on the field goes through the same encoder, producing a 64-dimensional representation of their recent movement pattern.

**2. Attention-Based Context Fusion** — The target player's encoding acts as a query over all other player encodings. This lets the model learn which teammates and defenders matter most for predicting a given player's movement. The attended context gets concatenated with play-level metadata (ball landing position, yardline, play direction) and fused into a 128-dimensional context vector.

**3. Autoregressive Decoder** — An LSTM decoder, initialized with the fused context, generates one displacement (dx, dy) per timestep. At each step it receives the previous displacement, the ball's landing coordinates, and the target player's encoding. Displacements are accumulated from the last known position to produce absolute coordinates.

The full model is intentionally compact — **84,290 parameters**. This is a deliberate choice. Compared to many competition entries that reach into the millions of parameters, this lightweight architecture trains fast and generalizes well. It runs comfortably on a MacBook Pro M3 Max using PyTorch's MPS backend.

### Training Details

- **Data**: 2023 NFL season, weeks 1-16 for training, weeks 17-18 for validation
- **Loss**: MSE on predicted displacements, masked for variable-length output sequences
- **Teacher forcing**: Starts at 100% and linearly decays to 0% over 30 epochs, letting the model gradually learn to rely on its own predictions
- **Optimizer**: AdamW with cosine annealing, early stopping with patience of 7

## Results

| Metric | Value |
|--------|-------|
| Validation mean Euclidean distance | **0.937 yards** |
| Best displacement MSE (val) | 0.0379 |
| Training epochs | 25 (early stopped) |
| Model parameters | 84,290 |

Under a yard of average prediction error feels like a decent starting point, especially from a model this small.

## Project Structure

```
NFL/
├── src/
│   ├── data.py          # Data loading (per-week CSVs, train/val/test)
│   ├── features.py      # Feature engineering (downsampling, encoding, displacements)
│   ├── dataset.py       # PyTorch Dataset + collation with variable-length padding
│   ├── model.py         # TrajectoryModel (encoder + attention + decoder)
│   ├── train.py         # Training loop with masked loss
│   └── submission.py    # Inference + Kaggle gRPC server integration
├── tests/               # 15 tests covering features, dataset, model, training, inference
├── run_train.py         # End-to-end training script
├── run_eval.py          # Validation scoring + submission generation
└── docs/plans/          # Design docs and implementation plan
```

## Quick Start

```bash
# Download competition data
kaggle competitions download nfl-big-data-bowl-2026-prediction -p data/
unzip data/nfl-big-data-bowl-2026-prediction.zip -d data/

# Run tests
pytest tests/ -v

# Train
python run_train.py

# Evaluate + generate submission
python run_eval.py
```

## What I'd Try Next

- Adding a position loss (auxiliary MSE on accumulated absolute coordinates) to reduce drift on longer sequences
- Experimenting with the encoder — temporal convolutions might capture short-range patterns better than LSTMs
- Incorporating player physical attributes (height, weight, position) as additional conditioning
- Ensembling with a non-autoregressive model that predicts all frames at once
