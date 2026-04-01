import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import center_of_mass
from PIL import Image

from ens_data_challenge import config
from ens_data_challenge.models.unet import PlainConvUNet, ParametricUNet
from ens_data_challenge.inference.inference import extract_number, apply_conditional_argmax

# GLOBAL CONFIGURATION
P05 = 13.0
P995 = 213.0
DATASET_MEAN_GLOBAL = 94.301
DATASET_STD_GLOBAL = 48.477

NUM_CLASSES = 54
FOLD = 2
PATCH_SIZE = 64
HALF = PATCH_SIZE // 2

# Cascade models
CASCADE_CONFIG = {
    1: {"anchor": 3, "vector": (-18.5, 1.9), "mean": 124.766, "std": 18.845,
        "threshold": 0.5, "checkpoint": config.EXPERIMENTS_DIR / f"run_01_phase2/fold_{FOLD}/cascade_L1/checkpoint_best.pth"},
    2: {"anchor": 33, "vector": (5.6, -12.0), "mean": 132.546, "std": 16.706,
        "threshold": 0.5, "checkpoint": config.EXPERIMENTS_DIR / f"run_01_phase2/fold_{FOLD}/cascade_L2/checkpoint_best.pth"}
}


def load_cascade_model(checkpoint_path, device):
    if not checkpoint_path.exists():
        return None
    model = ParametricUNet(in_channels=1, num_classes=1,
                           deepsupervision=True).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available(
    ) else "mps" if torch.backends.mps.is_available() else "cpu")
    input_dir = config.DATA_TEST_DIR
    output_csv = config.SUBMISSION_PATH
    fold_dir = config.EXPERIMENTS_DIR / f"run_01_phase1/fold_{FOLD}"

    # 1. Global Model loading
    model_global = PlainConvUNet(
        in_channels=1, num_classes=NUM_CLASSES, deepsupervision=True).to(device)
    checkpoint_global = torch.load(
        fold_dir / "checkpoint_best.pth", map_location=device)
    model_global.load_state_dict(checkpoint_global['model_state_dict'])
    model_global.eval()

    # 2. Cascade models loading
    cascade_models = {tc: load_cascade_model(cfg["checkpoint"], device)
                      for tc, cfg in CASCADE_CONFIG.items()}
    cascade_models = {k: v for k, v in cascade_models.items() if v is not None}

    # 3. Hard thresholds
    thresholds_dict = {str(i): 0.5 for i in range(1, NUM_CLASSES + 1)}

    # 4. PNG files
    image_files = sorted([f for f in os.listdir(
        input_dir) if f.endswith('.png')], key=extract_number)
    if not image_files:
        raise ValueError(f"Aucun PNG trouvé dans {input_dir}")

    # Spatial dimensions
    first_img = np.array(Image.open(
        os.path.join(input_dir, image_files[0])).convert('L'))
    h, w = first_img.shape
    all_predictions = np.zeros((len(image_files), h * w), dtype=np.uint8)

    # 5. Inference Loop
    for idx, filename in enumerate(tqdm(image_files, desc="Inférence Cascade")):
        raw_img = np.array(Image.open(os.path.join(
            input_dir, filename)).convert('L')).astype(np.float32)
        raw_img = np.clip(raw_img, P05, P995)

        # A. Global Prediction
        img_norm = (raw_img - DATASET_MEAN_GLOBAL) / DATASET_STD_GLOBAL
        img_tensor = torch.from_numpy(
            img_norm).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            logits_global = model_global(img_tensor)
            probs_global = torch.sigmoid(logits_global)
            pred_mask = apply_conditional_argmax(
                probs_global, thresholds_dict).squeeze(0).cpu().numpy()

        # B. Cascade adjustment
        padded_raw_img = np.pad(
            raw_img, HALF, mode="constant", constant_values=0)
        padded_pred_mask = np.pad(
            pred_mask, HALF, mode="constant", constant_values=0)

        for target_class, model_local in cascade_models.items():
            cfg = CASCADE_CONFIG[target_class]
            anchor_mask = (padded_pred_mask == cfg["anchor"])

            if np.sum(anchor_mask) > 0:
                cy, cx = center_of_mass(anchor_mask)
                py = int(round(float(cy) + cfg["vector"][0]))
                px = int(round(float(cx) + cfg["vector"][1]))

                patch_raw = padded_raw_img[py-HALF:py+HALF, px-HALF:px+HALF]

                if patch_raw.shape == (PATCH_SIZE, PATCH_SIZE):
                    # Normalization for cascade models
                    patch_norm = (patch_raw - cfg["mean"]) / cfg["std"]
                    patch_tensor = torch.from_numpy(
                        patch_norm).float().unsqueeze(0).unsqueeze(0).to(device)

                    with torch.no_grad():
                        # One channel for binary
                        logits_local = model_local(patch_tensor)
                        probs_local = torch.sigmoid(
                            logits_local).squeeze().cpu().numpy()

                    binary_patch = (probs_local > cfg["threshold"])

                    if np.sum(binary_patch) > 0:
                        # Overwrite
                        padded_pred_mask[padded_pred_mask == target_class] = 0
                        roi = padded_pred_mask[py -
                                               HALF:py+HALF, px-HALF:px+HALF]
                        roi[binary_patch] = target_class
                        padded_pred_mask[py-HALF:py +
                                         HALF, px-HALF:px+HALF] = roi

        # Final mask
        final_mask = padded_pred_mask[HALF:-HALF, HALF:-HALF]
        all_predictions[idx] = final_mask.flatten()

    print("Final CSV Generation...")
    df = pd.DataFrame(all_predictions.T,
                      # 6. CSV Generation
                      index=[f"Pixel {i}" for i in range(h*w)],
                      columns=[f"{extract_number(f)}.png" for f in image_files])
    df.to_csv(output_csv)
    print(f"Submission saved : {output_csv}")


if __name__ == "__main__":
    main()
