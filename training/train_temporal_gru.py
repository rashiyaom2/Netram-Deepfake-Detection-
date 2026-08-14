"""
Training script for Stage 5: Bidirectional Temporal GRU.
Trains on sequences of 512-dimensional spatial embeddings across 15-frame windows.
Uses 2-layer Bidirectional GRU with self-attention pooling.

Usage:
    python -m training.train_temporal_gru --epochs 20 --output models/temporal_gru.pt
"""
import os
import argparse
import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from pipeline.temporal import BidirectionalTemporalGRU

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_temporal_gru")


class SyntheticEmbeddingSequenceDataset(Dataset):
    """
    Dataset for temporal GRU training.
    Real sequences exhibit smooth trajectory in embedding space.
    Fake sequences exhibit frame-to-frame incoherence, random jumps, and micro-jitter.
    """
    def __init__(self, num_samples: int = 2000, seq_len: int = 15, emb_dim: int = 512):
        self.samples = []
        self.labels = []

        for _ in range(num_samples // 2):
            # Real sequence: smooth random walk in embedding space
            base = np.random.randn(emb_dim).astype(np.float32)
            base = base / (np.linalg.norm(base) + 1e-8)
            seq = []
            for t in range(seq_len):
                drift = base + 0.05 * np.random.randn(emb_dim).astype(np.float32)
                drift = drift / (np.linalg.norm(drift) + 1e-8)
                seq.append(drift)
            self.samples.append(np.stack(seq, axis=0))
            self.labels.append(0.0)  # Real

        for _ in range(num_samples // 2):
            # Fake sequence: erratic temporal trajectory / high frequency jitter
            base = np.random.randn(emb_dim).astype(np.float32)
            base = base / (np.linalg.norm(base) + 1e-8)
            seq = []
            for t in range(seq_len):
                jump = base + 0.35 * np.random.randn(emb_dim).astype(np.float32)
                jump = jump / (np.linalg.norm(jump) + 1e-8)
                seq.append(jump)
            self.samples.append(np.stack(seq, axis=0))
            self.labels.append(1.0)  # Fake

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.samples[idx]).float(),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


def train_temporal_gru(epochs: int = 15, batch_size: int = 32, lr: float = 1e-3,
                       output_path: str = "models/temporal_gru.pt") -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset = SyntheticEmbeddingSequenceDataset(num_samples=4000, seq_len=15, emb_dim=512)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = BidirectionalTemporalGRU(input_dim=512, hidden_dim=128, num_layers=2, dropout=0.2).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += len(y)

        scheduler.step()
        train_acc = correct / total
        avg_loss = total_loss / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x).squeeze(1)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == y).sum().item()
                val_total += len(y)

        val_acc = val_correct / val_total
        logger.info(f"Epoch {epoch:02d}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.1%} | Val Acc: {val_acc:.1%}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            logger.info(f"  --> Saved checkpoint to {output_path} (Val Acc: {val_acc:.1%})")

    logger.info(f"Training completed. Best Val Acc: {best_val_acc:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bidirectional Temporal GRU")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="models/temporal_gru.pt")
    args = parser.parse_args()

    train_temporal_gru(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, output_path=args.output)
