from pathlib import Path
import json

import torch
from torch.utils.data import Dataset
import numpy as np


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
