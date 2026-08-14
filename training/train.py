"""
Training — Main training loop (doc Training Strategy table).

Implements:
  - Multi-dataset training with per-generator tracking
  - Heavy augmentation pipeline
  - Combined CE + Contrastive loss
  - Per-generator evaluation (not one blended accuracy number)
  - Adversarial robustness testing hook
  - Fusion head calibration (scikit-learn, for Stage 6)

Usage:
    python -m training.train --config training_config.yaml
"""
import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, ConcatDataset
    import timm
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from training.datasets import (
    SampleDescriptor, scan_faceforensics, scan_celeb_df, scan_generic_paired,
)
from training.augmentations import AugmentationPipeline
from training.losses import CombinedLoss


# --------------------------------------------------------------------------- #
# Per-generator evaluation (doc: "Per-generator evaluation (new)")
# --------------------------------------------------------------------------- #
def evaluate_per_generator(
    predictions: List[float],
    labels: List[int],
    generators: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Report accuracy broken out by generator type.

    Per the doc: "A blended 95% can hide 99% on old GAN fakes and 70%
    on current diffusion methods — the number that predicts real-world
    performance."
    """
    from collections import defaultdict

    gen_preds = defaultdict(list)
    gen_labels = defaultdict(list)

    for pred, label, gen in zip(predictions, labels, generators):
        gen_preds[gen].append(pred)
        gen_labels[gen].append(label)

    results = {}
    for gen in sorted(gen_preds.keys()):
        preds = np.array(gen_preds[gen])
        labs = np.array(gen_labels[gen])
        binary_preds = (preds >= 0.5).astype(int)
        accuracy = float(np.mean(binary_preds == labs))
        n_samples = len(labs)

        # Compute precision/recall if we have both classes
        tp = np.sum((binary_preds == 1) & (labs == 1))
        fp = np.sum((binary_preds == 1) & (labs == 0))
        fn = np.sum((binary_preds == 0) & (labs == 1))
        precision = float(tp / (tp + fp + 1e-8))
        recall = float(tp / (tp + fn + 1e-8))

        results[gen] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "n_samples": n_samples,
        }

    return results


# --------------------------------------------------------------------------- #
# Fusion head training (scikit-learn, for Stage 6)
# --------------------------------------------------------------------------- #
def train_fusion_head(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: str = "models/fusion_head.pkl",
    model_type: str = "logistic",
) -> None:
    """
    Train the Stage 6 fusion head on held-out labelled data.

    Features should be the 6-d vector:
      [p_spatial, p_freq, p_temporal, p_sync, jitter, pose_confidence]

    Args:
        features: (N, 6) array of fusion inputs
        labels: (N,) binary labels
        output_path: where to save the trained model
        model_type: "logistic" or "mlp"
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.calibration import CalibratedClassifierCV
    import joblib

    if model_type == "mlp":
        base_model = MLPClassifier(
            hidden_layer_sizes=(32, 16), max_iter=500, random_state=42,
            early_stopping=True, validation_fraction=0.15,
        )
    else:
        base_model = LogisticRegression(max_iter=1000, random_state=42)

    # Calibrate probabilities (doc §6: "calibrated, not fixed")
    model = CalibratedClassifierCV(base_model, cv=5, method="isotonic")
    model.fit(features, labels)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logger.info(f"Fusion head saved to {output_path}")


# --------------------------------------------------------------------------- #
# Main training loop (spatial branch as the primary model)
# --------------------------------------------------------------------------- #
if TORCH_AVAILABLE:
    class SpatialBranchTrainer:
        """
        Trains the EfficientNet-B4 spatial branch with combined loss.
        Tracks per-generator performance throughout training.
        """

        def __init__(self, model_name: str = "efficientnet_b4",
                     embedding_dim: int = 512,
                     lr: float = 1e-4,
                     lambda_contrastive: float = 0.5,
                     device: Optional[str] = None):
            self.device = torch.device(
                device or ("cuda" if torch.cuda.is_available() else "cpu")
            )

            # Create model with custom head
            self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
            # Get the feature dimension from the model
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 299, 299)
                feat_dim = self.model(dummy).shape[-1]

            self.embedding_head = nn.Linear(feat_dim, embedding_dim)
            self.classifier_head = nn.Linear(embedding_dim, 1)

            self.model.to(self.device)
            self.embedding_head.to(self.device)
            self.classifier_head.to(self.device)

            self.criterion = CombinedLoss(
                lambda_contrastive=lambda_contrastive
            )

            params = (
                list(self.model.parameters())
                + list(self.embedding_head.parameters())
                + list(self.classifier_head.parameters())
            )
            self.optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=50
            )

        def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
            self.model.train()
            self.embedding_head.train()
            self.classifier_head.train()

            total_loss = 0.0
            total_ce = 0.0
            total_contrastive = 0.0
            n_batches = 0

            for batch in dataloader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                # Forward
                features = self.model(images)
                embeddings = self.embedding_head(features)
                logits = self.classifier_head(embeddings)

                losses = self.criterion(logits, embeddings, labels)

                # Backward
                self.optimizer.zero_grad()
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters())
                    + list(self.embedding_head.parameters())
                    + list(self.classifier_head.parameters()),
                    max_norm=1.0,
                )
                self.optimizer.step()

                total_loss += losses["total"].item()
                total_ce += losses["ce"].item()
                total_contrastive += losses["contrastive"].item()
                n_batches += 1

            self.scheduler.step()

            return {
                "loss": total_loss / max(n_batches, 1),
                "ce_loss": total_ce / max(n_batches, 1),
                "contrastive_loss": total_contrastive / max(n_batches, 1),
            }

        @torch.no_grad()
        def evaluate(self, dataloader: DataLoader) -> Tuple[
            Dict[str, Dict[str, float]], float
        ]:
            """
            Per-generator evaluation.
            Returns (per_generator_metrics, overall_accuracy).
            """
            self.model.eval()
            self.embedding_head.eval()
            self.classifier_head.eval()

            all_preds = []
            all_labels = []
            all_generators = []

            for batch in dataloader:
                images = batch["image"].to(self.device)
                labels = batch["label"]
                generators = batch["generator"]

                features = self.model(images)
                embeddings = self.embedding_head(features)
                logits = self.classifier_head(embeddings)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

                all_preds.extend(probs.tolist())
                all_labels.extend(labels.numpy().astype(int).tolist())
                all_generators.extend(generators)

            per_gen = evaluate_per_generator(all_preds, all_labels, all_generators)
            overall_acc = float(
                np.mean(
                    (np.array(all_preds) >= 0.5).astype(int) == np.array(all_labels)
                )
            )
            return per_gen, overall_acc

        def save(self, path: str) -> None:
            """Save all model weights."""
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": self.model.state_dict(),
                "embedding_head": self.embedding_head.state_dict(),
                "classifier_head": self.classifier_head.state_dict(),
            }, path)
            logger.info(f"Model saved to {path}")

        def load(self, path: str) -> None:
            """Load saved model weights."""
            state = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state["model"], strict=False)
            self.embedding_head.load_state_dict(state["embedding_head"])
            self.classifier_head.load_state_dict(state["classifier_head"])


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Train deepfake detection models")
    parser.add_argument("--ff-root", type=str, help="FaceForensics++ dataset root")
    parser.add_argument("--celeb-root", type=str, help="Celeb-DF v2 dataset root")
    parser.add_argument("--extra-roots", type=str, nargs="*",
                        help="Additional dataset roots (generic paired layout)")
    parser.add_argument("--extra-generators", type=str, nargs="*",
                        help="Generator type names for extra roots")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--fusion-features", type=str,
                        help="Path to fusion feature CSV for training fusion head")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not TORCH_AVAILABLE:
        logger.error("PyTorch is required for training. Install with: pip install torch torchvision timm")
        return

    # Collect samples from all datasets
    all_samples: List[SampleDescriptor] = []

    if args.ff_root:
        logger.info(f"Scanning FaceForensics++ at {args.ff_root}")
        all_samples.extend(scan_faceforensics(args.ff_root))

    if args.celeb_root:
        logger.info(f"Scanning Celeb-DF v2 at {args.celeb_root}")
        all_samples.extend(scan_celeb_df(args.celeb_root))

    if args.extra_roots:
        generators = args.extra_generators or ["unknown"] * len(args.extra_roots)
        for root, gen in zip(args.extra_roots, generators):
            logger.info(f"Scanning {root} (generator: {gen})")
            all_samples.extend(scan_generic_paired(root, generator_type=gen))

    if not all_samples:
        logger.error("No samples found. Provide at least one dataset root.")
        return

    logger.info(f"Total samples: {len(all_samples)}")

    # Log per-generator distribution
    gen_counts = defaultdict(int)
    for s in all_samples:
        gen_counts[s.generator_type] += 1
    for gen, count in sorted(gen_counts.items()):
        logger.info(f"  {gen}: {count} samples")

    # Build dataset + augmentation
    from training.datasets import DeepfakeDataset
    aug = AugmentationPipeline()

    def transform_fn(img):
        img = aug(img)
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1)

    dataset = DeepfakeDataset(all_samples, transform=transform_fn)
    # 80/20 split
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    # Train
    trainer = SpatialBranchTrainer(lr=args.lr)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        metrics = trainer.train_epoch(train_loader)
        per_gen, val_acc = trainer.evaluate(val_loader)

        logger.info(
            f"Epoch {epoch}/{args.epochs} — "
            f"loss: {metrics['loss']:.4f}, "
            f"CE: {metrics['ce_loss']:.4f}, "
            f"contrastive: {metrics['contrastive_loss']:.4f}, "
            f"val_acc: {val_acc:.4f}"
        )
        for gen, gen_metrics in per_gen.items():
            logger.info(
                f"  {gen}: acc={gen_metrics['accuracy']:.4f}, "
                f"prec={gen_metrics['precision']:.4f}, "
                f"rec={gen_metrics['recall']:.4f} "
                f"(n={gen_metrics['n_samples']})"
            )

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = str(Path(args.output_dir) / "spatial_b4_best.pt")
            trainer.save(save_path)

    # Final save
    trainer.save(str(Path(args.output_dir) / "spatial_b4_final.pt"))

    # Save per-generator results
    results_path = Path(args.output_dir) / "eval_per_generator.json"
    per_gen_final, _ = trainer.evaluate(val_loader)
    with open(results_path, "w") as f:
        json.dump(per_gen_final, f, indent=2)
    logger.info(f"Per-generator results saved to {results_path}")


if __name__ == "__main__":
    main()
