from pathlib import Path
import json

import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import center_of_mass


class AbdominalCTDataset(Dataset):
    def __init__(self, image_paths: list, labels_dir: Path, json_path: Path, transform=None):
        self.image_paths = image_paths
        self.labels_dir = labels_dir
        self.transform = transform

        # Load partially annotated labels
        with open(json_path, "r") as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.labels_dir / img_path.name

        img_id = int(img_path.stem)

        # Load in RAM
        image = np.load(img_path)
        mask = np.load(label_path)

        # Expansion for Albumentation
        image_expanded = np.expand_dims(image, axis=-1)

        if self.transform:
            augmented = self.transform(image=image_expanded, mask=mask)
            image_tensor = augmented["image"]
            mask_tensor = augmented["mask"].long()
        else:
            raise ValueError(
                "Transformation required.")

        # Mask initialisation
        valid_mask = np.zeros(54, dtype=np.float32)

        # If the image is not labeled
        if img_id >= 800:
            valid_mask[:] = 1.0
        else:
            valid_labels = self.annotations[img_id]
            for label in valid_labels:
                # Security
                if 1 <= label <= 54:
                    valid_mask[label - 1] = 1.0

        return image_tensor, mask_tensor, torch.from_numpy(valid_mask)


class CascadeDataset(Dataset):
    def __init__(self, image_ids: list, images_dir: Path, fused_masks_dir: Path,
                 target_class: int, anchor_class: int, vector: tuple, std_dev: float,
                 transform=None, patch_size: int = 64, is_train: bool = True):
        self.image_ids = image_ids
        self.images_dir = images_dir
        self.fused_masks_dir = fused_masks_dir

        self.target_class = target_class
        self.anchor_class = anchor_class
        self.vector_y, self.vector_x = vector
        self.std_dev = std_dev

        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        self.is_train = is_train
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = self.images_dir / f"{img_id}.npy"
        mask_path = self.fused_masks_dir / f"{img_id}.npy"

        image = np.load(img_path)
        fused_mask = np.load(mask_path)

        anchor_binary = (fused_mask == self.anchor_class)

        if np.sum(anchor_binary) == 0:
            center_y, center_x = 128, 128
            print(f"[WARN] Ancre L{self.anchor_class} absente dans {img_id}")
        else:
            center_y, center_x = center_of_mass(anchor_binary)

        target_y = center_y + self.vector_y
        target_x = center_x + self.vector_x

        if self.is_train:
            noise_y = np.random.uniform(-self.std_dev, self.std_dev)
            noise_x = np.random.uniform(-self.std_dev, self.std_dev)
            target_y += noise_y
            target_x += noise_x

        ty, tx = int(round(target_y)), int(round(target_x))

        padded_img = np.pad(image, pad_width=self.half_patch,
                            mode="constant", constant_values=0)
        padded_mask = np.pad(
            fused_mask, pad_width=self.half_patch, mode="constant", constant_values=0)

        py = ty + self.half_patch
        px = tx + self.half_patch

        patch_img = padded_img[py - self.half_patch: py + self.half_patch,
                               px - self.half_patch: px + self.half_patch]
        patch_mask = padded_mask[py - self.half_patch: py + self.half_patch,
                                 px - self.half_patch: px + self.half_patch]

        binary_target_mask = (
            patch_mask == self.target_class).astype(np.float32)

        patch_img_exp = np.expand_dims(patch_img, axis=-1)

        if self.transform:
            augmented = self.transform(
                image=patch_img_exp, mask=binary_target_mask)
            img_tensor = augmented["image"]
            mask_tensor = augmented["mask"].float()
        else:
            raise ValueError("Transformation required.")

        return img_tensor, mask_tensor
