import os
import random
from pathlib import Path
import json

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.data import WeightedRandomSampler

from ens_data_challenge import config
from ens_data_challenge.data_processing.dataloader import get_train_dataloader, get_val_dataloader
from ens_data_challenge.data_processing.dataset import AbdominalCTDataset
from ens_data_challenge.data_processing.augmentations import get_training_augmentations, get_validation_augmentations
from ens_data_challenge.models.unet import PlainConvUNet
from ens_data_challenge.models.losses import MarginalSegmentationLoss, DeepSupervisionWrapper
from ens_data_challenge.training.trainer import Trainer
from ens_data_challenge.training.logger_utils import setup_logger, plot_training_curves


def main():
    FOLD = 3

    BATCH_SIZE = 42
    ACCUMULATION_STEPS = 1
    NUM_WORKERS = 8
    NUM_EPOCHS = 500
    ITERS_PER_EPOCH = 250
    NUM_CLASSES = 54

    DATASET_MEAN = 94.301
    DATASET_STD = 48.477

    SAVE_DIR = f"experiments/run_01_phase1/fold_{FOLD}"
    os.makedirs(SAVE_DIR, exist_ok=True)

    logger = setup_logger(SAVE_DIR)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Device configured: {device.type}")
    logger.info(f"Start of the training for the FOLD {FOLD}")

    splits_path = config.DATA_PREPROCESSED_DIR / "splits.json"

    with open(splits_path, "r") as f:
        splits = json.load(f)

    train_ids = splits[str(FOLD)]["train"]
    val_ids = splits[str(FOLD)]["val"]

    logger.info(
        f"Training images : {len(train_ids)} | Validation Images : {len(val_ids)}")

    # Rebuilds the paths
    images_dir = config.IMAGES_PREPROCESSED_DIR
    train_paths = [images_dir / f"{img_id}.npy" for img_id in train_ids]
    val_paths = [images_dir / f"{img_id}.npy" for img_id in val_ids]

    train_transforms = get_training_augmentations(DATASET_MEAN, DATASET_STD)
    val_transforms = get_validation_augmentations(DATASET_MEAN, DATASET_STD)

    train_dataset = AbdominalCTDataset(
        image_paths=train_paths,
        labels_dir=config.LABELS_PREPROCESSED_DIR,
        json_path=config.LABELS_JSON_PATH,
        transform=train_transforms
    )

    val_dataset = AbdominalCTDataset(
        image_paths=val_paths,
        labels_dir=config.LABELS_PREPROCESSED_DIR,
        json_path=config.LABELS_JSON_PATH,
        transform=val_transforms
    )

    # ISQR WEIGHTS
    logger.info("Mask analysis for ISQR")
    class_doc_frequencies = {i: 0 for i in range(1, NUM_CLASSES + 1)}
    num_train_images = len(train_ids)

    images_to_classes = {}

    for img_id in train_ids:
        mask_path = config.LABELS_PREPROCESSED_DIR / f"{img_id}.npy"
        mask = np.load(mask_path)

        unique_classes = np.unique(mask)
        present_classes = [c for c in unique_classes if c > 0]

        images_to_classes[img_id] = present_classes

        for c in present_classes:
            class_doc_frequencies[c] += 1

    MAX_WEIGHT = 15
    class_weights = {}

    for c, freq in class_doc_frequencies.items():
        if freq > 0:
            weight = 1.0 / ((freq / num_train_images) ** 0.5)
            class_weights[c] = min(weight, MAX_WEIGHT)
        else:
            class_weights[c] = 1.0

    sample_weights = []
    for img_id in train_ids:
        present_classes = images_to_classes[img_id]
        if not present_classes:
            img_weight = 1.0
        else:
            # For each image, take the weight of the rarest class
            img_weight = max([class_weights[c] for c in present_classes])
        sample_weights.append(img_weight)

    train_sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = get_train_dataloader(
        train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, sampler=train_sampler)
    val_loader = get_val_dataloader(
        val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    model = PlainConvUNet(in_channels=1, num_classes=NUM_CLASSES).to(device)

    base_criterion = MarginalSegmentationLoss(
        w_dice=0.6, w_focal=0.4, gamma=2.0, alpha=0.25)
    criterion = DeepSupervisionWrapper(
        criterion=base_criterion, weights=[1.0, 0.5, 0.25])

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.99,
        weight_decay=3e-5,
        nesterov=True
    )

    total_iters = NUM_EPOCHS * ITERS_PER_EPOCH
    scheduler = PolynomialLR(optimizer, total_iters=total_iters, power=0.9)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=SAVE_DIR,
        logger=logger,
        num_epochs=NUM_EPOCHS,
        iters_per_epoch=ITERS_PER_EPOCH,
        num_classes=NUM_CLASSES,
        ema_alpha=0.2,
        accumulation_steps=ACCUMULATION_STEPS
    )

    logger.info(f"Start of the training for {NUM_EPOCHS} epochs.")

    for epoch in range(NUM_EPOCHS):
        trainer.current_epoch = epoch
        current_lr = trainer.optimizer.param_groups[0]['lr']

        train_loss = trainer.train_one_epoch()
        val_loss, val_dice, ema_dice, class_dices = trainer.validate()

        trainer.history['train_loss'].append(train_loss)
        trainer.history['val_loss'].append(val_loss)
        trainer.history['val_dice'].append(val_dice)
        trainer.history['ema_dice'].append(ema_dice)
        trainer.history['lr'].append(current_lr)

        plot_training_curves(trainer.history, trainer.save_dir)

        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict(),
            'scaler_state_dict': trainer.scaler.state_dict() if trainer.scaler else None,
            'best_ema_dice': trainer.best_ema_dice,
            'history': trainer.history
        }

        torch.save(checkpoint_state, os.path.join(
            trainer.save_dir, "checkpoint_latest.pth"))

        if ema_dice > trainer.best_ema_dice:
            trainer.best_ema_dice = ema_dice
            torch.save(checkpoint_state, os.path.join(
                trainer.save_dir, "checkpoint_best.pth"))
            new_best_flag = " (Yess NEW BEST SCORE hehe)"
        else:
            new_best_flag = ""

        log_summary = (
            f"Epoch [{epoch:04d}/{NUM_EPOCHS-1}] | LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | EMA Dice: {ema_dice:.4f}{new_best_flag}\n"
            f"Detail of {NUM_CLASSES} classes :\n"
        )

        class_details = ""
        for i in range(1, NUM_CLASSES + 1):
            class_details += f"L{i:02d}: {class_dices.get(i, 0.0):.4f} | "
            if i % 6 == 0:
                class_details += "\n"

        logger.info(log_summary + class_details)


if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(42)

    main()
