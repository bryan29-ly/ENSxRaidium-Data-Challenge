import os
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm


def get_sorted_image_paths(image_dir: Path):
    image_paths = list(image_dir.glob("*.png"))
    image_paths.sort(key=lambda p: int(p.stem))
    return image_paths


def extract_and_save_labels(csv_path: Path, output_dir: Path):
    df = pd.read_csv(csv_path, index_col=0, header=0)
    image_names = df.columns.tolist()

    masks_3d = df.T.values.reshape((-1, 256, 256)).astype(np.uint8)

    for idx, img_name in enumerate(tqdm(image_names, desc="Labels Extraction")):
        out_fname = Path(img_name).with_suffix(".npy").name
        np.save(output_dir / out_fname, masks_3d[idx])


def compute_dataset_statistics(image_dir: Path, label_dir: Path, num_annotated: int = 800):
    image_paths = get_sorted_image_paths(image_dir)[:num_annotated]
    all_pixels = []

    for img_path in tqdm(image_paths, desc="Pixels Extraction"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        mask_path = label_dir / img_path.with_suffix(".npy").name
        mask = np.load(mask_path)

        foreground_pixels = img[mask > 0]
        all_pixels.append(foreground_pixels)

    combined = np.concatenate(all_pixels)

    p05 = np.percentile(combined, 0.5)
    p995 = np.percentile(combined, 99.5)

    clipped = np.clip(combined, p05, p995)
    mean = np.mean(clipped)
    std = np.std(clipped)

    print(f"p05: {p05}, p995: {p995}, mean: {mean}, std: {std}")

    del combined, clipped, all_pixels
    return p05, p995, mean, std


def clip_and_save_images(input_dir: Path, output_dir: Path, p05: float, p995: float):
    image_paths = get_sorted_image_paths(input_dir)

    for img_path in tqdm(image_paths, desc="Clipping and saving"):
        img = cv2.imread(
            str(img_path), cv2.IMREAD_GRAYSCALE)

        img = np.clip(img, p05, p995).astype(np.uint8)

        out_fname = img_path.with_suffix(".npy").name
        np.save(output_dir / out_fname, img)
