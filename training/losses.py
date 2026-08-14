"""
Training — Loss Functions (doc Training Strategy table).

Combined loss:
    L_total = L_CE + λ · L_Contrastive

Purpose:
  - Cross-entropy (L_CE): standard classification objective.
  - Contrastive loss (L_Contrastive): pulls real embeddings together and
    pushes fake embeddings apart in the 512-d spatial-branch embedding
    space. This makes the embeddings more discriminative for downstream
    temporal analysis and improves generalisation across generator types.
"""
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class ContrastiveLoss(nn.Module):
        """
        Supervised contrastive loss over embeddings.

        Given a batch of embeddings and binary labels (0=real, 1=fake):
          - Pairs with the same label are pulled together (positive pairs).
          - Pairs with different labels are pushed apart (negative pairs).

        Uses the margin-based formulation:
          L = y * d^2 + (1 - y) * max(0, margin - d)^2
        where d = ||e_i - e_j||_2, y = 1 if same class else 0.
        """

        def __init__(self, margin: float = 2.0):
            super().__init__()
            self.margin = margin

        def forward(self, embeddings: torch.Tensor,
                    labels: torch.Tensor) -> torch.Tensor:
            """
            Args:
                embeddings: (B, D) normalised embeddings
                labels: (B,) binary labels (0 or 1)
            Returns:
                scalar loss
            """
            # L2 normalise embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            batch_size = embeddings.size(0)

            if batch_size < 2:
                return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

            # Pairwise distances
            dists = torch.cdist(embeddings, embeddings, p=2)  # (B, B)

            # Same-class mask
            labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B) bool
            # Exclude diagonal (self-pairs)
            diag_mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)

            pos_mask = labels_eq & diag_mask
            neg_mask = ~labels_eq & diag_mask

            # Positive pairs: minimise distance
            pos_loss = (dists ** 2 * pos_mask.float()).sum()
            # Negative pairs: push apart (margin)
            neg_loss = (F.relu(self.margin - dists) ** 2 * neg_mask.float()).sum()

            n_pairs = pos_mask.float().sum() + neg_mask.float().sum() + 1e-8
            return (pos_loss + neg_loss) / n_pairs


    class CombinedLoss(nn.Module):
        """
        L_total = L_CE + λ · L_Contrastive

        Used for jointly training the spatial branch's classifier head
        and embedding representation.
        """

        def __init__(self, lambda_contrastive: float = 0.5,
                     contrastive_margin: float = 2.0,
                     label_smoothing: float = 0.05):
            super().__init__()
            self.lambda_contrastive = lambda_contrastive
            self.ce_loss = nn.BCEWithLogitsLoss(
                # Label smoothing for robustness
                # Applied manually since BCEWithLogitsLoss doesn't have built-in smoothing
            )
            self.contrastive_loss = ContrastiveLoss(margin=contrastive_margin)
            self.label_smoothing = label_smoothing

        def forward(self, logits: torch.Tensor, embeddings: torch.Tensor,
                    labels: torch.Tensor) -> dict:
            """
            Args:
                logits: (B, 1) raw classifier output
                embeddings: (B, D) spatial-branch embeddings
                labels: (B,) binary labels (0 or 1)
            Returns:
                dict with 'total', 'ce', 'contrastive' loss components
            """
            # Label smoothing
            smoothed_labels = labels.float() * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

            l_ce = self.ce_loss(logits.squeeze(-1), smoothed_labels)
            l_contrastive = self.contrastive_loss(embeddings, labels)
            l_total = l_ce + self.lambda_contrastive * l_contrastive

            return {
                "total": l_total,
                "ce": l_ce.detach(),
                "contrastive": l_contrastive.detach(),
            }
