import os
import json

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ens_data_challenge import config
from ens_data_challenge.data_processing.dataset import CascadeDataset
from ens_data_challenge.data_processing.augmentations import get_patch_augmentations, get_validation_augmentations
from ens_data_challenge.models.unet import ParametricUNet
from ens_data_challenge.models.losses import BinarySegmentationLoss, BinaryDeepSupervisionWrapper
from ens_data_challenge.training.logger_utils import setup_logger, plot_training_curves


@torch.no_grad()
def compute_binary_dice(logits, target, epsilon=1e-5):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    target = target.unsqueeze(1).float()

    inter = (preds * target).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + epsilon) / (union + epsilon)
    return dice.mean().item()


def train_cascade():
    FOLD = 2
    TARGET_CLASS = 1
    ANCHOR_CLASS = 3
    VECTOR = (-18.5, 1.9)
    STD_DEV = 3.8

    BATCH_SIZE = 4
    NUM_EPOCHS = 300
    LR = 2e-4

    fused_masks_dir = config.DATA_PREPROCESSED_DIR / "fused_masks_l1_l2"
    save_dir = f"experiments/run_01_phase2/fold_{FOLD}/cascade_L{TARGET_CLASS}"
    os.makedirs(save_dir, exist_ok=True)

    logger = setup_logger(save_dir)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 1. SPLITS
    with open(config.DATA_PREPROCESSED_DIR / "splits.json", "r") as f:
        splits = json.load(f)

    def filter_exists(ids):
        valid = []
        for i in ids:
            p = fused_masks_dir / f"{i}.npy"
            if p.exists():
                mask = np.load(p)
                if TARGET_CLASS in mask:
                    valid.append(i)
        return valid

    train_ids = filter_exists(splits[str(FOLD)]["train"])
    val_ids = filter_exists(splits[str(FOLD)]["val"])

    logger.info(
        f"Train: {len(train_ids)} | Val: {len(val_ids)} pour L{TARGET_CLASS}")

    # 2. DATASETS
    train_ds = CascadeDataset(
        train_ids, config.IMAGES_PREPROCESSED_DIR, fused_masks_dir,
        TARGET_CLASS, ANCHOR_CLASS, VECTOR, STD_DEV,
        transform=get_patch_augmentations(124.766, 18.845), is_train=True
    )
    val_ds = CascadeDataset(
        val_ids, config.IMAGES_PREPROCESSED_DIR, fused_masks_dir,
        TARGET_CLASS, ANCHOR_CLASS, VECTOR, 0.0,
        transform=get_validation_augmentations(124.766, 18.845), is_train=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2)

    # 3. MODELE, LOSS, OPTIMIZER, SCHEDULER
    model = ParametricUNet(in_channels=1, num_classes=1,
                           deepsupervision=True).to(device)
    criterion = BinaryDeepSupervisionWrapper(BinarySegmentationLoss())
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)

    # CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    history = {"train_loss": [], "val_loss": [],
               "val_dice": [], "ema_dice": [], "lr": []}
    best_dice = 0.0
    ema_dice = 0.0
    ema_alpha = 0.9

    # 4. TRAINING LOOP
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Scheduler step per epoch
        scheduler.step()

        # Validation
        model.eval()
        v_loss, v_dice = 0.0, 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                v_loss += criterion(logits, masks).item()
                v_dice += compute_binary_dice(logits, masks)

        v_loss /= len(val_loader)
        v_dice /= len(val_loader)
        ema_dice = ema_alpha * ema_dice + \
            (1 - ema_alpha) * v_dice if epoch > 0 else v_dice

        # History
        history["train_loss"].append(epoch_loss / len(train_loader))
        history["val_loss"].append(v_loss)
        history["val_dice"].append(v_dice)
        history["ema_dice"].append(ema_dice)
        history["lr"].append(optimizer.param_groups[0]['lr'])

        plot_training_curves(history, save_dir)

        if ema_dice > best_dice:
            best_dice = ema_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice,
            }, os.path.join(save_dir, "checkpoint_best.pth"))

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_dice': best_dice,
        }, os.path.join(save_dir, "checkpoint_last.pth"))

        logger.info(
            f"Epoch [{epoch:04d}/{NUM_EPOCHS-1:04d}] | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"Train Loss: {epoch_loss/len(train_loader):.4f} | "
            f"Val Loss: {v_loss:.4f} | "
            f"Val Dice: {v_dice:.4f} | "
            f"EMA Dice: {ema_dice:.4f}"
        )


if __name__ == "__main__":
    train_cascade()
