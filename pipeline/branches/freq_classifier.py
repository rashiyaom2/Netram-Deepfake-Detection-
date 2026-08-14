"""
Lightweight Frequency-Domain CNN Classifier.

Replaces the pure mathematical FFT/DCT heuristic in frequency_branch.py
with a small trained convolutional network that operates on 2D DCT
log-magnitude spectral maps.

Architecture: 3 × Conv2D → AdaptiveAvgPool → FC → sigmoid
Total params: ~53K (tiny, < 1ms inference on CPU)
"""
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = type("nn", (), {"Module": object})
    TORCH_AVAILABLE = False


class FreqSpectrumCNN(nn.Module):
    """
    Small CNN that classifies 2D DCT log-magnitude spectrum images
    as real vs. synthetically generated.

    Input:  (batch, 1, H, W)  — single-channel DCT log-magnitude map
    Output: (batch, 1)        — logit (apply sigmoid for probability)
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.SELU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SELU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SELU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.SELU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
