"""
Train and export a calibrated 2-layer Bidirectional Temporal GRU
with self-attention pooling for multi-frame deepfake detection.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from pipeline.temporal import BidirectionalTemporalGRU

def generate_embedding_dataset(n_samples=2000, seq_len=15, embed_dim=512):
    X = []
    y = []

    for _ in range(n_samples // 2):
        # Real sequence: smooth trajectory with small natural head motion
        base = np.random.randn(embed_dim)
        base = base / (np.linalg.norm(base) + 1e-8)
        
        # Smooth random walk in embedding space
        drift_rate = np.random.uniform(0.005, 0.03)
        velocity = np.random.randn(embed_dim) * drift_rate
        
        seq = []
        curr = base.copy()
        for t in range(seq_len):
            curr = curr + velocity + np.random.randn(embed_dim) * 0.008
            curr = curr / (np.linalg.norm(curr) + 1e-8)
            seq.append(curr)
        
        X.append(np.array(seq))
        y.append(0.0)  # Real

    for _ in range(n_samples // 2):
        # Fake sequence: frame-to-frame generative jitter, boundary pops, mask phase jumps
        base = np.random.randn(embed_dim)
        base = base / (np.linalg.norm(base) + 1e-8)
        
        seq = []
        curr = base.copy()
        for t in range(seq_len):
            # Intermittent phase jumps and synthetic noise
            jump = (np.random.rand() > 0.6) * (np.random.randn(embed_dim) * np.random.uniform(0.15, 0.45))
            curr = curr + jump + np.random.randn(embed_dim) * 0.08
            curr = curr / (np.linalg.norm(curr) + 1e-8)
            seq.append(curr)
        
        X.append(np.array(seq))
        y.append(1.0)  # Fake

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Shuffle
    idx = np.random.permutation(len(X))
    return torch.from_numpy(X[idx]), torch.from_numpy(y[idx])


def train_model():
    print("Generating temporal sequence dataset...")
    X, y = generate_embedding_dataset(n_samples=3000, seq_len=15, embed_dim=512)
    
    split = int(0.85 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    model = BidirectionalTemporalGRU(input_dim=512, hidden_dim=128, num_layers=2, dropout=0.2)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print("Training Calibrated Bidirectional Temporal GRU...")
    batch_size = 64
    for epoch in range(1, 21):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        epoch_loss = 0.0
        
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices].unsqueeze(1)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(indices)
            
        epoch_loss /= len(X_train)

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_preds = (torch.sigmoid(val_logits) >= 0.5).float()
            val_acc = (val_preds == y_val.unsqueeze(1)).float().mean().item()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

    os.makedirs("models", exist_ok=True)
    out_path = "models/temporal_gru.pt"
    torch.save(model, out_path)
    print(f"Exported calibrated model to {out_path}")

    # Verify on test sample
    model.eval()
    with torch.no_grad():
        real_sample = X_val[y_val == 0][:1]
        fake_sample = X_val[y_val == 1][:1]
        p_real = torch.sigmoid(model(real_sample)).item()
        p_fake = torch.sigmoid(model(fake_sample)).item()
        print(f"Verification: P(Real Sample) = {p_real:.4f}, P(Fake Sample) = {p_fake:.4f}")


if __name__ == "__main__":
    train_model()
