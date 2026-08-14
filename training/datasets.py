"""
Training — Dataset loaders (doc Training Strategy table).

Supports multiple dataset directory layouts used by common deepfake
detection benchmarks:

  - FaceForensics++  : {root}/{manipulation_type}/{split}/{video_id}/{frame}.png
  - Celeb-DF v2      : {root}/{real_or_fake}/{video_id}/{frame}.png
  - WildDeepfake     : {root}/{split}/{real_or_fake}/{clip_id}/{frame}.png
  - Generic paired   : {root}/real/{id}.png + {root}/fake/{id}.png

All dataset classes produce (image_path, audio_path_or_None, label, generator_type)
tuples for per-generator evaluation (doc: "Per-generator evaluation (new)").
"""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    import cv2
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Sample descriptor (framework-agnostic)
# --------------------------------------------------------------------------- #
class SampleDescriptor:
    """Metadata for one training sample, independent of loading strategy."""

    __slots__ = ("image_path", "audio_path", "label", "generator_type", "group_id")

    def __init__(self, image_path: str, label: int,
                 generator_type: str = "unknown",
                 audio_path: Optional[str] = None,
                 group_id: Optional[str] = None):
        self.image_path = image_path
        self.audio_path = audio_path
        self.label = label  # 0 = real, 1 = fake
        self.generator_type = generator_type
        self.group_id = group_id or Path(image_path).parent.name


# --------------------------------------------------------------------------- #
# Dataset scanners — produce lists of SampleDescriptor
# --------------------------------------------------------------------------- #
def scan_faceforensics(root: str, manipulations: Optional[List[str]] = None,
                        split: str = "train") -> List[SampleDescriptor]:
    """
    FaceForensics++ layout:
      {root}/original_sequences/youtube/{split}/images/{video_id}/{frame}.png
      {root}/manipulated_sequences/{manipulation}/{split}/images/{video_id}/{frame}.png
    """
    root_path = Path(root)
    samples: List[SampleDescriptor] = []

    # Real samples
    real_dir = root_path / "original_sequences" / "youtube" / split / "images"
    if real_dir.exists():
        for img_file in sorted(real_dir.rglob("*.png")):
            samples.append(SampleDescriptor(str(img_file), label=0, generator_type="real"))
        for img_file in sorted(real_dir.rglob("*.jpg")):
            samples.append(SampleDescriptor(str(img_file), label=0, generator_type="real"))

    # Manipulated samples
    if manipulations is None:
        manipulations = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

    manip_base = root_path / "manipulated_sequences"
    for manip in manipulations:
        manip_dir = manip_base / manip / split / "images"
        if not manip_dir.exists():
            continue
        for img_file in sorted(manip_dir.rglob("*.png")):
            samples.append(SampleDescriptor(str(img_file), label=1, generator_type=manip))
        for img_file in sorted(manip_dir.rglob("*.jpg")):
            samples.append(SampleDescriptor(str(img_file), label=1, generator_type=manip))

    return samples


def scan_celeb_df(root: str) -> List[SampleDescriptor]:
    """
    Celeb-DF v2 layout:
      {root}/Celeb-real/{video_id}/{frame}.png
      {root}/Celeb-synthesis/{video_id}/{frame}.png
    """
    root_path = Path(root)
    samples: List[SampleDescriptor] = []

    real_dir = root_path / "Celeb-real"
    if real_dir.exists():
        for img_file in sorted(real_dir.rglob("*.png")):
            samples.append(SampleDescriptor(str(img_file), label=0, generator_type="real"))

    fake_dir = root_path / "Celeb-synthesis"
    if fake_dir.exists():
        for img_file in sorted(fake_dir.rglob("*.png")):
            samples.append(SampleDescriptor(str(img_file), label=1, generator_type="CelebDF"))

    return samples


def scan_generic_paired(root: str, generator_type: str = "unknown") -> List[SampleDescriptor]:
    """
    Generic layout:
      {root}/real/{file}
      {root}/fake/{file}
    Useful for custom datasets (diffusion, Gaussian-splatting, etc.).
    """
    root_path = Path(root)
    samples: List[SampleDescriptor] = []

    real_dir = root_path / "real"
    if real_dir.exists():
        for img_file in sorted(real_dir.rglob("*")):
            if img_file.suffix.lower() in (".png", ".jpg", ".jpeg"):
                samples.append(SampleDescriptor(str(img_file), label=0, generator_type="real"))

    fake_dir = root_path / "fake"
    if fake_dir.exists():
        for img_file in sorted(fake_dir.rglob("*")):
            if img_file.suffix.lower() in (".png", ".jpg", ".jpeg"):
                samples.append(SampleDescriptor(str(img_file), label=1, generator_type=generator_type))

    return samples


# --------------------------------------------------------------------------- #
# PyTorch Dataset wrapper
# --------------------------------------------------------------------------- #
if TORCH_AVAILABLE:
    class DeepfakeDataset(Dataset):
        """
        PyTorch Dataset that wraps a list of SampleDescriptors.

        Applies optional transforms (augmentations) to loaded images.
        Optionally loads paired audio for A/V sync branch training.
        """

        def __init__(self, samples: List[SampleDescriptor],
                     image_size: Tuple[int, int] = (299, 299),
                     transform=None,
                     load_audio: bool = False,
                     audio_sample_rate: int = 16000,
                     audio_window_seconds: float = 3.0):
            self.samples = samples
            self.image_size = image_size
            self.transform = transform
            self.load_audio = load_audio
            self.audio_sample_rate = audio_sample_rate
            self.audio_window_samples = int(audio_sample_rate * audio_window_seconds)

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int):
            sample = self.samples[idx]

            # Load image
            img = cv2.imread(sample.image_path)
            if img is None:
                # Return a black placeholder if image is missing
                img = np.zeros((*self.image_size, 3), dtype=np.uint8)
            img = cv2.resize(img, self.image_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.transform is not None:
                img = self.transform(img)
            else:
                img = img.astype(np.float32) / 255.0
                img = torch.from_numpy(img).permute(2, 0, 1)  # HWC → CHW

            label = torch.tensor(sample.label, dtype=torch.float32)

            result = {"image": img, "label": label, "generator": sample.generator_type}

            # Load audio if available
            if self.load_audio and sample.audio_path is not None:
                try:
                    import soundfile as sf
                    audio, sr = sf.read(sample.audio_path)
                    if audio.ndim > 1:
                        audio = audio[:, 0]  # mono
                    audio = audio.astype(np.float32)
                    # Truncate or pad to window size
                    if len(audio) > self.audio_window_samples:
                        audio = audio[:self.audio_window_samples]
                    else:
                        audio = np.pad(audio, (0, self.audio_window_samples - len(audio)))
                    result["audio"] = torch.from_numpy(audio)
                except Exception:
                    result["audio"] = torch.zeros(self.audio_window_samples)
            else:
                result["audio"] = torch.zeros(self.audio_window_samples)

            return result
