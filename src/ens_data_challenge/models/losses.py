import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginalDiceLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape
        probs = torch.sigmoid(logits)

        gt = F.one_hot(target.long(), num_classes=C + 1)
        gt = gt[..., 1:].permute(0, 3, 1, 2).float()

        # Apply validity mask (B, C) -> (B, C, 1, 1)
        mask_4d = valid_mask.unsqueeze(-1).unsqueeze(-1)
        probs_masked = probs * mask_4d
        gt_masked = gt * mask_4d

        # Aggregate over spatial dimensions AND batch dimension (Batch Dice)
        inter = (probs_masked * gt_masked).sum(dim=(0, 2, 3))
        union = probs_masked.sum(dim=(0, 2, 3)) + gt_masked.sum(dim=(0, 2, 3))

        dice = (2.0 * inter + self.epsilon) / (union + self.epsilon)

        # Identify classes present at least once in the batch
        classes_present_in_gt = gt_masked.sum(dim=(0, 2, 3)) > 0
        valid_classes = (valid_mask.sum(dim=0) > 0) & classes_present_in_gt

        if not valid_classes.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Strict mean over validated classes
        loss = 1.0 - dice[valid_classes].mean()
        return loss


class MarginalFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape

        gt = F.one_hot(target.long(), num_classes=C + 1)
        gt = gt[..., 1:].permute(0, 3, 1, 2).float()

        # Stable computation of Binary Cross Entropy
        bce = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")

        # Stabilized probability p_t (Sigmoid formula)
        p_t = torch.exp(-bce)

        # Asymmetric alpha (Standard)
        alpha_t = torch.where(gt == 1.0, self.alpha, 1.0 - self.alpha)

        # Final computation of the focal multiplier
        focal_loss = alpha_t * (1.0 - p_t) ** self.gamma * bce

        # Apply marginal mask (same)
        mask_4d = valid_mask.unsqueeze(-1).unsqueeze(-1)
        focal_masked = focal_loss * mask_4d

        n_valid_pixels = valid_mask.sum() * H * W

        if n_valid_pixels < 1:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return focal_masked.sum() / n_valid_pixels


class MarginalSegmentationLoss(nn.Module):
    def __init__(self, w_dice: float = 0.6, w_focal: float = 0.4, gamma: float = 2.0, alpha: float = 0.25, epsilon: float = 1e-5):
        super().__init__()
        self.dice = MarginalDiceLoss(epsilon=epsilon)
        self.focal = MarginalFocalLoss(gamma=gamma, alpha=alpha)
        self.w_dice = w_dice
        self.w_focal = w_focal

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return self.w_dice * self.dice(logits, target, valid_mask) + self.w_focal * self.focal(logits, target, valid_mask)


class DeepSupervisionWrapper(nn.Module):
    def __init__(self, criterion: nn.Module, weights: list = None):
        super().__init__()
        self.criterion = criterion
        self.weights = weights if weights is not None else [1.0, 0.5, 0.25]

    def forward(self, logits_list, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if not isinstance(logits_list, list):
            return self.criterion(logits_list, target, valid_mask)

        loss = 0.0
        for i, logits in enumerate(logits_list):
            if i >= len(self.weights):
                break

            # Downsampling
            if logits.shape[2:] != target.shape[1:]:
                target_down = F.interpolate(
                    target.unsqueeze(1).float(),
                    size=logits.shape[2:],
                    mode="nearest"
                ).squeeze(1).long()
            else:
                target_down = target

            loss += self.weights[i] * \
                self.criterion(logits, target_down, valid_mask)

        return loss


class BinarySegmentationLoss(nn.Module):
    def __init__(self, w_dice: float = 0.6, w_focal: float = 0.4, gamma: float = 2.0, alpha: float = 0.75, epsilon: float = 1e-5):
        super().__init__()
        self.w_dice = w_dice
        self.w_focal = w_focal
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.unsqueeze(1).float()
        probs = torch.sigmoid(logits)

        # 1. Binary Dice Loss (Réduction par image)
        inter = (probs * target).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2.0 * inter + self.epsilon) / (union + self.epsilon)
        dice_loss = 1.0 - dice.mean()

        # 2. Binary Focal Loss
        bce = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none")
        p_t = torch.exp(-bce)
        alpha_t = torch.where(target == 1.0, self.alpha, 1.0 - self.alpha)
        focal_loss = (alpha_t * (1.0 - p_t) ** self.gamma * bce).mean()

        return self.w_dice * dice_loss + self.w_focal * focal_loss


class BinaryDeepSupervisionWrapper(nn.Module):
    def __init__(self, criterion: nn.Module, weights: list = None):
        super().__init__()
        self.criterion = criterion
        self.weights = weights if weights is not None else [1.0, 0.5, 0.25]

    def forward(self, logits_list, target: torch.Tensor) -> torch.Tensor:
        if not isinstance(logits_list, list):
            return self.criterion(logits_list, target)

        loss = 0.0
        for i, logits in enumerate(logits_list):
            if i >= len(self.weights):
                break

            if logits.shape[2:] != target.shape[1:]:
                target_down = F.interpolate(
                    target.unsqueeze(1).float(),
                    size=logits.shape[2:],
                    mode="nearest"
                ).squeeze(1).float()
            else:
                target_down = target

            loss += self.weights[i] * self.criterion(logits, target_down)

        return loss


@torch.no_grad()
def compute_partial_dice_raw(logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, threshold: float = 0.5) -> dict:
    B, C, H, W = logits.shape

    # Background is removed here (gt[..., 1:])
    gt = F.one_hot(target.long(), num_classes=C + 1)
    gt = gt[..., 1:].permute(0, 3, 1, 2).float()

    # Binarization
    preds = (torch.sigmoid(logits) > threshold).float()

    # Intersections and unions per image and per class
    inter = (preds * gt).sum(dim=(-2, -1))
    union = preds.sum(dim=(-2, -1)) + gt.sum(dim=(-2, -1))

    # Apply validity mask (JSON)
    inter_masked = inter * valid_mask
    union_masked = union * valid_mask

    # Sum over the batch
    batch_inter = inter_masked.sum(dim=0)
    batch_union = union_masked.sum(dim=0)
    batch_valid = valid_mask.sum(dim=0)

    return {
        "inter": batch_inter,
        "union": batch_union,
        "valid": batch_valid
    }
