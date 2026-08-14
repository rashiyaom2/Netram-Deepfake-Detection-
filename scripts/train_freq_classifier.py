"""
Self-supervised training for the FreqSpectrumCNN.

Generates "real" DCT spectra from clean synthetic face crops and "fake"
spectra with injected GAN/diffusion-like spectral anomalies, then trains
the lightweight CNN to discriminate between them.

Usage:
    python -m scripts.train_freq_classifier

Outputs:
    models/freq_classifier.pt
"""
import logging
import os
import sys

import numpy as np
from scipy.fft import dctn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    logger.error("PyTorch is required. Install with: pip install torch")
    sys.exit(1)

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.branches.freq_classifier import FreqSpectrumCNN


# ─── Synthetic Data Generator ───────────────────────────────────────────────

def _generate_real_face_crop(size: int = 224) -> np.ndarray:
    """
    Generate a synthetic "real" face crop with natural-looking texture:
    smooth skin-tone base + noise + natural high-frequency detail.
    """
    # Skin-tone base
    base = np.random.uniform(0.35, 0.75, (size, size)).astype(np.float64)
    # Smooth it to simulate skin
    from scipy.ndimage import gaussian_filter
    base = gaussian_filter(base, sigma=np.random.uniform(3.0, 8.0))
    # Natural high-frequency texture (pores, hair)
    texture = np.random.randn(size, size) * np.random.uniform(0.01, 0.04)
    # Natural gradient (lighting variation)
    y, x = np.mgrid[0:size, 0:size] / float(size)
    gradient = (x * np.random.uniform(-0.1, 0.1) +
                y * np.random.uniform(-0.1, 0.1))
    return np.clip(base + texture + gradient, 0.0, 1.0)


def _generate_fake_face_crop(size: int = 224) -> np.ndarray:
    """
    Generate a synthetic "fake" face crop that mimics spectral signatures
    of GAN/diffusion-generated images:
    - Over-smoothed mid-frequencies (diffusion models)
    - Periodic grid artifacts (GAN upsampling)
    - Missing or attenuated high-frequency detail
    - Unnaturally uniform texture
    """
    anomaly_type = np.random.choice(["gan_grid", "diffusion_smooth",
                                     "upscale_artifact", "uniform_texture"])

    if anomaly_type == "gan_grid":
        # GAN checkerboard artifact: periodic grid in frequency domain
        base = np.random.uniform(0.3, 0.7, (size, size)).astype(np.float64)
        from scipy.ndimage import gaussian_filter
        base = gaussian_filter(base, sigma=6.0)
        # Add periodic grid (GAN upsampling artifact)
        period = np.random.choice([2, 4, 8])
        y, x = np.mgrid[0:size, 0:size]
        grid = 0.03 * np.sin(2 * np.pi * x / period) * np.sin(2 * np.pi * y / period)
        return np.clip(base + grid, 0.0, 1.0)

    elif anomaly_type == "diffusion_smooth":
        # Diffusion model: overly smooth, missing high-frequency detail
        base = np.random.uniform(0.35, 0.7, (size, size)).astype(np.float64)
        from scipy.ndimage import gaussian_filter
        base = gaussian_filter(base, sigma=np.random.uniform(10.0, 20.0))
        # Very little high-freq noise
        texture = np.random.randn(size, size) * 0.002
        return np.clip(base + texture, 0.0, 1.0)

    elif anomaly_type == "upscale_artifact":
        # AI upscaling artifact: generate at low res, upscale
        from scipy.ndimage import gaussian_filter
        low_res = np.random.uniform(0.3, 0.7, (size // 4, size // 4))
        low_res = gaussian_filter(low_res, sigma=2.0)
        # Bilinear upscale introduces spectral aliasing
        from scipy.ndimage import zoom
        upscaled = zoom(low_res, 4.0, order=1)[:size, :size]
        return np.clip(upscaled, 0.0, 1.0)

    else:  # uniform_texture
        # Unnaturally uniform face texture (face swap blending)
        base = np.random.uniform(0.4, 0.6, (size, size)).astype(np.float64)
        from scipy.ndimage import gaussian_filter
        base = gaussian_filter(base, sigma=np.random.uniform(8.0, 15.0))
        # Slight noise but much less than real
        texture = np.random.randn(size, size) * 0.003
        return np.clip(base + texture, 0.0, 1.0)


def _compute_dct_input(gray_image: np.ndarray, target_size: int = 112) -> np.ndarray:
    """Compute DCT log-magnitude spectrum and resize to fixed dimensions."""
    dct = dctn(gray_image, type=2, norm="ortho")
    log_mag = np.log1p(np.abs(dct))
    # Normalize to [0, 1]
    vmin, vmax = log_mag.min(), log_mag.max()
    if vmax - vmin > 1e-8:
        log_mag = (log_mag - vmin) / (vmax - vmin)
    else:
        log_mag = np.zeros_like(log_mag)
    # Resize
    from scipy.ndimage import zoom
    h, w = log_mag.shape
    log_mag = zoom(log_mag, (target_size / h, target_size / w), order=1)
    return log_mag.astype(np.float32)


class SpectralDataset(Dataset):
    def __init__(self, num_samples: int = 5000, target_size: int = 112):
        self.num_samples = num_samples
        self.target_size = target_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        is_fake = idx >= self.num_samples // 2
        if is_fake:
            gray = _generate_fake_face_crop(224)
        else:
            gray = _generate_real_face_crop(224)

        dct_map = _compute_dct_input(gray, self.target_size)
        tensor = torch.from_numpy(dct_map).unsqueeze(0)  # (1, H, W)
        label = torch.tensor([1.0 if is_fake else 0.0], dtype=torch.float32)
        return tensor, label


# ─── Training Loop ──────────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    model = FreqSpectrumCNN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"FreqSpectrumCNN parameters: {total_params:,}")

    train_ds = SpectralDataset(num_samples=1600, target_size=112)
    val_ds = SpectralDataset(num_samples=400, target_size=112)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    os.makedirs("models", exist_ok=True)
    save_path = "models/freq_classifier.pt"

    for epoch in range(6):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_x.size(0)

        scheduler.step()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_x.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_loss = train_loss / train_total

        logger.info(
            f"Epoch {epoch + 1:02d}/06 — "
            f"Loss: {avg_loss:.4f} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            logger.info(f"✅ Checkpoint saved to {save_path} (val_acc={val_acc:.3f})")

    logger.info(f"🎉 Training complete. Final model saved at {save_path}")



if __name__ == "__main__":
    train()
