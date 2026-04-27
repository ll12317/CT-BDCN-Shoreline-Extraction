"""
Data loading utilities for shoreline extraction.

This file defines dataset loading and preprocessing functions used by
training and validation scripts in the CT-BDCN shoreline extraction project.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


# Default paths are defined relative to the repository root where this file is located.
BASE_DIR = Path(__file__).resolve().parent

DEFAULT_TRAIN_IMG_DIR = BASE_DIR / "datasets" / "train" / "images"
DEFAULT_TRAIN_MASK_DIR = BASE_DIR / "datasets" / "train" / "masks"
DEFAULT_VAL_IMG_DIR = BASE_DIR / "datasets" / "val" / "images"
DEFAULT_VAL_MASK_DIR = BASE_DIR / "datasets" / "val" / "masks"

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _list_images(image_dir: Path) -> List[str]:
    """Return sorted image file names from a directory."""
    files = [
        f.name
        for f in image_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS
    ]
    files.sort()

    if not files:
        raise RuntimeError(f"[MyDataset] No image files were found in: {image_dir}")

    return files


class MyDataset(Dataset):
    """
    Dataset for shoreline extraction.

    Each image should have a corresponding mask with the same base file name.
    For binary sea-land segmentation, masks are converted to tensors with shape [1, H, W].
    For multi-class segmentation, masks are converted to Long tensors with shape [H, W].
    """

    def __init__(
        self,
        image_dir: Optional[Union[str, Path]] = None,
        mask_dir: Optional[Union[str, Path]] = None,
        num_classes: int = 1,
        size: Optional[Tuple[int, int]] = None,
        augment: bool = False,
        mask_suffix: str = "",
        strict: bool = True,
        mode: str = "train",
    ):
        """
        Initialize the dataset.

        Args:
            image_dir: Directory containing input images. If None, the default path is used.
            mask_dir: Directory containing mask images. If None, the default path is used.
            num_classes: Number of segmentation classes. Use 1 for binary segmentation.
            size: Optional resized image size as (width, height).
            augment: Whether to apply random data augmentation.
            mask_suffix: Optional suffix added to the mask file name before the extension.
            strict: Whether to raise errors for missing or inconsistent files.
            mode: Dataset split used for default path selection. Supported values: "train" and "val".
        """
        if mode not in {"train", "val"}:
            raise RuntimeError(
                f"[MyDataset] Unsupported mode: {mode}. Please use 'train' or 'val', "
                "or manually provide image_dir and mask_dir."
            )

        if image_dir is None:
            image_dir = DEFAULT_TRAIN_IMG_DIR if mode == "train" else DEFAULT_VAL_IMG_DIR
        if mask_dir is None:
            mask_dir = DEFAULT_TRAIN_MASK_DIR if mode == "train" else DEFAULT_VAL_MASK_DIR

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.num_classes = num_classes
        self.size = size
        self.augment = augment
        self.mask_suffix = mask_suffix
        self.strict = strict

        if not self.image_dir.is_dir():
            raise RuntimeError(f"[MyDataset] image_dir does not exist or is not a directory: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise RuntimeError(f"[MyDataset] mask_dir does not exist or is not a directory: {self.mask_dir}")

        image_names = _list_images(self.image_dir)
        self.samples = []

        for name in image_names:
            stem = Path(name).stem
            mask_candidates = [
                self.mask_dir / f"{stem}{self.mask_suffix}{ext}"
                for ext in IMG_EXTENSIONS
            ]

            mask_path = None
            for candidate in mask_candidates:
                if candidate.is_file():
                    mask_path = candidate
                    break

            if mask_path is None:
                msg = f"[MyDataset] No corresponding mask was found for {stem}* in {self.mask_dir}"
                if self.strict:
                    raise RuntimeError(msg)
                print(msg + "; skipped.")
                continue

            self.samples.append((self.image_dir / name, mask_path))

        if not self.samples:
            raise RuntimeError(
                "[MyDataset] No valid image-mask pairs were found. "
                "Please check whether image and mask file names are consistent."
            )

        # Store original tile names without file extensions for prediction export.
        self.names = [Path(img_path).stem for (img_path, _) in self.samples]

        print(f"[MyDataset] Loaded {len(self.samples)} samples. mode={mode}, image_dir={self.image_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def tile_name(self, index: int) -> str:
        """Return the original tile name for a given sample index."""
        return self.names[index]

    def _load_pair(self, index: int):
        """Load an image-mask pair."""
        img_path, mask_path = self.samples[index]

        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        mask = Image.open(mask_path)
        if mask.mode not in ("L", "I", "P"):
            mask = mask.convert("L")

        if self.size is None and img.size != mask.size:
            msg = (
                f"[MyDataset] Image and mask sizes are inconsistent: {Path(img_path).name}, "
                f"image={img.size}, mask={mask.size}"
            )
            if self.strict:
                raise RuntimeError(msg)
            print(msg + "; the mask will be resized to match the image size.")
            mask = mask.resize(img.size, Image.NEAREST)

        return img, mask

    def _augment(self, img: Image.Image, mask: Image.Image):
        """Apply random spatial augmentation to an image-mask pair."""
        if random.random() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        if random.random() < 0.2:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        if random.random() < 0.3:
            k = random.choice([1, 2, 3])
            img = img.rotate(90 * k, expand=True)
            mask = mask.rotate(90 * k, expand=True)
        return img, mask

    def _resize(self, img: Image.Image, mask: Image.Image):
        """Resize an image-mask pair if a target size is specified."""
        if self.size is None:
            return img, mask

        width, height = self.size
        img = img.resize((width, height), Image.BILINEAR)
        mask = mask.resize((width, height), Image.NEAREST)
        return img, mask

    def __getitem__(self, index: int):
        img, mask = self._load_pair(index)

        if self.augment:
            img, mask = self._augment(img, mask)

        img, mask = self._resize(img, mask)

        img = TF.to_tensor(img)
        mask_np = np.array(mask)

        if self.num_classes == 1:
            mask_bin = (mask_np > 0).astype("float32")
            mask_tensor = torch.from_numpy(mask_bin).unsqueeze(0)  # [1, H, W]
        else:
            mask_long = mask_np.astype("int64")
            mask_tensor = torch.from_numpy(mask_long)  # [H, W]

        return img, mask_tensor
