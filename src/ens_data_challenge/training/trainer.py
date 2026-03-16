import os
import contextlib

import torch
from tqdm import tqdm

from ens_data_challenge.models.losses import compute_partial_dice_raw


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            criterion: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler,
            device: torch.device,
            save_dir: str,
            logger,
            num_epochs: int = 400,
            iters_per_epoch: int = 250,
            num_classes: int = 54,
            val_every: int = 5,
            ema_alpha: float = 0.2,
            accumulation_steps: int = 1
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.logger = logger
        self.num_epochs = num_epochs
        self.iters_per_epoch = iters_per_epoch
        self.num_classes = num_classes
        self.val_every = val_every

        self.current_epoch = 0
        self.best_ema_dice = 0.0
        self.current_ema_dice = 0.0
        self.ema_alpha = ema_alpha
        self.accumulation_steps = accumulation_steps

        self.device_type = device.type
        self.use_amp = (self.device_type in ["cuda", "mps"])
        self.scaler = torch.amp.GradScaler(
            "cuda") if self.device_type == "cuda" else None

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_dice": [],
            "ema_dice": [],
            "lr": []
        }

        os.makedirs(self.save_dir, exist_ok=True)

    def train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        train_iter = iter(self.train_loader)

        pbar = tqdm(
            range(self.iters_per_epoch),
            desc=f"[Train] Epoch {self.current_epoch:04d}/{self.num_epochs}",
            leave=False,
            dynamic_ncols=True
        )

        self.optimizer.zero_grad()

        for _ in pbar:
            step_loss = 0.0

            for _ in range(self.accumulation_steps):
                try:
                    images, masks, valid_masks = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    images, masks, valid_masks = next(train_iter)

                images = images.to(self.device)
                masks = masks.to(self.device)
                valid_masks = valid_masks.to(self.device)

                if self.use_amp:
                    with torch.amp.autocast(self.device_type):
                        logits = self.model(images)
                        raw_loss = self.criterion(logits, masks, valid_masks)
                        loss = raw_loss / self.accumulation_steps

                    if self.scaler is not None:
                        # CUDA
                        self.scaler.scale(loss).backward()
                    else:
                        # MPS
                        loss.backward()
                else:
                    # CPU
                    logits = self.model(images)
                    raw_loss = self.criterion(logits, masks, valid_masks)
                    loss = raw_loss / self.accumulation_steps
                    loss.backward()

                # Restore the visual value of the loss for logging
                step_loss += raw_loss.item()

            # Update weights once accumulation is complete
            if self.use_amp and self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=12.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=12.0)
                self.optimizer.step()

            self.scheduler.step()
            self.optimizer.zero_grad()

            total_loss += step_loss
            pbar.set_postfix({"loss": f"{step_loss:.4f}"})

        return total_loss / self.iters_per_epoch

    @torch.no_grad()
    def validate(self) -> tuple[float, float, dict]:
        self.model.eval()
        total_loss = 0.0

        # Global accumulators for the validation dataset
        total_inter = torch.zeros(self.num_classes, device=self.device)
        total_union = torch.zeros(self.num_classes, device=self.device)
        total_valid = torch.zeros(self.num_classes, device=self.device)

        pbar = tqdm(
            self.val_loader,
            desc=f"[Val]   Epoch {self.current_epoch:04d}/{self.num_epochs}",
            leave=False,
            dynamic_ncols=True,
        )

        for images, masks, valid_masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            valid_masks = valid_masks.to(self.device)

            autocast_ctx = torch.amp.autocast(
                self.device_type) if self.use_amp else contextlib.nullcontext()

            with autocast_ctx:
                logits = self.model(images)
                loss = self.criterion(logits, masks, valid_masks)

            total_loss += loss.item()

            # Extraction of raw metrics
            metrics = compute_partial_dice_raw(
                logits, masks, valid_masks, threshold=0.5)
            total_inter += metrics["inter"]
            total_union += metrics["union"]
            total_valid += metrics["valid"]

        mean_loss = total_loss / max(len(self.val_loader), 1)

        dice_per_class = {}
        all_dices = []

        # Compute the global Dice at the end of the epoch
        for c in range(self.num_classes):
            if total_valid[c].item() > 0:
                u = total_union[c].item()
                i = total_inter[c].item()

                if u == 0:
                    # The organ is completely absent from the GT and the network has perfectly predicted 0 on the entire validation set
                    dice = 1.0
                else:
                    dice = (2.0 * i) / u

                dice_per_class[c + 1] = round(dice, 4)
                all_dices.append(dice)

        mean_dice = sum(all_dices) / len(all_dices) if all_dices else 0.0

        if self.current_epoch == 0:
            self.current_ema_dice = mean_dice
        else:
            self.current_ema_dice = (
                self.ema_alpha * mean_dice) + ((1.0 - self.ema_alpha) * self.current_ema_dice)

        return mean_loss, mean_dice, self.current_ema_dice, dice_per_class
