import os
import re
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def extract_number(filename: str) -> int:
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0


class TestDataset(Dataset):
    def __init__(self, image_dir, image_files, transform=None):
        self.image_dir = image_dir
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        if img_name.endswith('.npy'):
            image = np.load(img_path)
        else:
            image = np.array(Image.open(img_path).convert('L'))

        image_expanded = np.expand_dims(image, axis=-1)

        if self.transform:
            augmented = self.transform(image=image_expanded)
            image_tensor = augmented["image"]
        else:
            raise ValueError("A transformation function is needed.")

        return image_tensor, img_name


def apply_conditional_argmax(probs: torch.Tensor, thresholds_dict: dict) -> torch.Tensor:
    B, C, H, W = probs.shape
    device = probs.device

    thresholds_array = [thresholds_dict[str(i)] for i in range(1, C + 1)]
    t_tensor = torch.tensor(thresholds_array, device=device).view(1, C, 1, 1)

    valid_activations = probs > t_tensor
    filtered_probs = probs * valid_activations.float()

    max_probs, max_idx = torch.max(filtered_probs, dim=1)
    final_mask = torch.where(max_probs > 0, max_idx + 1, 0)

    return final_mask.to(torch.uint8)
