import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from ens_data_challenge import config
from ens_data_challenge.models.unet import PlainConvUNet
from ens_data_challenge.inference.inference import extract_number, TestDataset, apply_conditional_argmax
from ens_data_challenge.data_processing.augmentations import get_validation_augmentations
from ens_data_challenge.data_processing.dataloader import get_val_dataloader

DATASET_MEAN = 94.301
DATASET_STD = 48.477
NUM_CLASSES = 54
FOLD = 2
BATCH_SIZE = 16
NUM_WORKERS = 4


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    input_dir = config.DATA_TEST_DIR
    output_csv = config.SUBMISSION_PATH
    fold_dir = config.EXPERIMENTS_DIR / f"run_01_phase1/fold_{FOLD}"
    checkpoint_path = fold_dir / "checkpoint_best.pth"
    thresholds_path = fold_dir / "thresholds.json"

    all_files = os.listdir(input_dir)
    image_files = sorted(
        [f for f in all_files if f.endswith(('.png', '.npy', '.jpg'))],
        key=extract_number
    )
    n_images = len(image_files)

    if n_images == 0:
        raise ValueError(f"Aucune image trouvée dans {input_dir}")

    with open(thresholds_path, "r") as f:
        thresholds_dict = json.load(f)

    # Init the model
    model = PlainConvUNet(in_channels=1, num_classes=54,
                          deepsupervision=True).to(device)

    checkpoint_path = fold_dir / "checkpoint_best.pth"
    print(f"Load the weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint['model_state_dict']

    model.load_state_dict(state_dict)
    model.eval()

    val_transforms = get_validation_augmentations(DATASET_MEAN, DATASET_STD)
    test_dataset = TestDataset(
        input_dir, image_files, transform=val_transforms)
    test_loader = get_val_dataloader(
        test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    first_image, _ = test_dataset[0]
    pixels_per_image = first_image.shape[1] * first_image.shape[2]

    all_predictions = np.zeros((n_images, pixels_per_image), dtype=np.uint8)

    current_idx = 0
    with torch.no_grad():
        for images, filenames in tqdm(test_loader, desc="Extraction et Résolution"):
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            pred_masks = apply_conditional_argmax(probs, thresholds_dict)
            pred_masks_np = pred_masks.cpu().numpy()

            batch_size = pred_masks_np.shape[0]
            for b in range(batch_size):
                all_predictions[current_idx] = pred_masks_np[b].flatten()
                current_idx += 1

    names_columns = [f"{extract_number(f)}.png" for f in image_files]
    names_rows = [f"Pixel {i}" for i in range(pixels_per_image)]

    df_output = pd.DataFrame(
        all_predictions.T, index=names_rows, columns=names_columns)
    df_output.to_csv(output_csv)


if __name__ == "__main__":
    main()
