"""End-to-end training script for NFL trajectory prediction."""

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import load_train_week
from src.dataset import NFLDataset, collate_plays
from src.model import TrajectoryModel
from src.train import train_one_epoch, validate


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading training data (weeks 1-16)...")
    train_inputs, train_outputs = [], []
    for w in range(1, 17):
        inp, out = load_train_week(w)
        train_inputs.append(inp)
        train_outputs.append(out)

    train_inp = pd.concat(train_inputs, ignore_index=True)
    train_out = pd.concat(train_outputs, ignore_index=True)
    del train_inputs, train_outputs

    print("Loading validation data (weeks 17-18)...")
    val_inp_17, val_out_17 = load_train_week(17)
    val_inp_18, val_out_18 = load_train_week(18)
    val_inp = pd.concat([val_inp_17, val_inp_18], ignore_index=True)
    val_out = pd.concat([val_out_17, val_out_18], ignore_index=True)

    print("Building datasets...")
    train_ds = NFLDataset(train_inp, train_out)
    val_ds = NFLDataset(val_inp, val_out)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_plays)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_plays)

    model = TrajectoryModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_val_loss = float("inf")
    patience = 7
    patience_counter = 0

    for epoch in range(50):
        tf_ratio = max(0.0, 1.0 - epoch / 30)

        train_loss = train_one_epoch(model, train_loader, optimizer, device, teacher_forcing_ratio=tf_ratio)
        val_loss = validate(model, val_loader, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={lr:.6f} | tf={tf_ratio:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "models/best_model.pt")
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print("Model saved to models/best_model.pt")


if __name__ == "__main__":
    main()
