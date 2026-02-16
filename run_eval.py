"""Evaluate model on validation set and generate submission."""

import pandas as pd
import torch

from src.data import load_train_week
from src.model import TrajectoryModel, evaluate
from src.submission import generate_submission, predict_play


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = TrajectoryModel()
    model.load_state_dict(torch.load("models/best_model.pt", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    print("Evaluating on weeks 17-18...")
    val_inp_17, val_out_17 = load_train_week(17)
    val_inp_18, val_out_18 = load_train_week(18)
    val_inp = pd.concat([val_inp_17, val_inp_18], ignore_index=True)
    val_out = pd.concat([val_out_17, val_out_18], ignore_index=True)

    all_preds = []
    for (game_id, play_id), play_inp in val_inp.groupby(["game_id", "play_id"]):
        pred = predict_play(model, play_inp, device=device)
        pred["game_id"] = game_id
        pred["play_id"] = play_id
        all_preds.append(pred)

    predicted = pd.concat(all_preds, ignore_index=True)
    score = evaluate(predicted, val_out)
    print(f"Mean Euclidean distance (val): {score:.4f} yards")

    print("\nGenerating submission...")
    submission = generate_submission(model, device=device)
    submission.to_csv("submissions/submission.csv", index=False)
    print(f"Submission saved: {len(submission)} rows")
    print(submission.head())


if __name__ == "__main__":
    main()
