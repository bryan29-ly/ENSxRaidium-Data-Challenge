import os
import json

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F


def find_best_thresholds(val_loader, model, device, save_dir, num_classes=54):
    model.eval()

    all_probs = []
    all_gts = []
    all_valid_masks = []

    print("1. Generate Probabilities on validation set ...")
    with torch.no_grad():
        for images, targets, valid_masks in tqdm(val_loader, desc="Inference"):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu()

            gt = F.one_hot(targets.long(), num_classes=num_classes + 1)
            gt = gt[..., 1:].permute(0, 3, 1, 2).float().cpu()

            all_probs.append(probs)
            all_gts.append(gt)
            all_valid_masks.append(valid_masks.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_gts = torch.cat(all_gts, dim=0)
    all_valid_masks = torch.cat(all_valid_masks, dim=0)

    thresholds_to_test = np.arange(0.1, 1.0, 0.01)
    best_thresholds = {}
    best_dices = {}

    print("2. Iterative optimisation per class ...")
    for c in range(num_classes):
        valid_images_for_c = all_valid_masks[:, c] == 1.0

        if not valid_images_for_c.any():
            best_thresholds[str(c + 1)] = 0.5
            best_dices[str(c + 1)] = 0.0
            continue

        probs_c = all_probs[valid_images_for_c, c, ...]
        gt_c = all_gts[valid_images_for_c, c, ...]

        best_t = 0.5
        best_dice = 0.0

        for t in thresholds_to_test:
            preds = (probs_c > t).float()

            inter = (preds * gt_c).sum().item()
            union = preds.sum().item() + gt_c.sum().item()

            if union == 0:
                dice = 1.0
            else:
                dice = 2.0 * inter / union

            if dice >= best_dice:
                best_dice = dice
                best_t = t

        best_thresholds[str(c + 1)] = round(float(best_t), 2)
        best_dices[str(c + 1)] = round(float(best_dice), 4)

    mean_dice = sum(best_dices.values()) / len(best_dices)
    print(f"\nGlobal Optimised Dice (Validation) : {mean_dice:.4f}")

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "thresholds.json")
    with open(out_path, "w") as f:
        json.dump(best_thresholds, f, indent=4)

    print(f"Thresholds saved in : {out_path}")
    return best_thresholds
