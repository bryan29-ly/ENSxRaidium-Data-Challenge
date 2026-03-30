import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Avant Elastic à 120 p=0.3 et dowscale 0.5-0.9


def get_training_augmentations(dataset_mean: float, dataset_std: float):
    return A.Compose([
        # 1. Distorsion
        A.Affine(
            scale=(0.85, 1.25),
            translate_percent=(-0.0625, 0.0625),
            rotate=(-15, 15),
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REPLICATE,
            p=0.6
        ),

        A.ElasticTransform(
            alpha=100,
            sigma=100 * 0.05,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REPLICATE,
            p=0.3
        ),

        # 2. Intensity
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),

        A.RandomBrightnessContrast(
            brightness_limit=0.10,
            contrast_limit=0.10,
            p=0.3
        ),
        A.RandomGamma(
            gamma_limit=(70, 150),
            p=0.3
        ),

        # 3. Normalisation (Z-score)
        A.Normalize(
            mean=(dataset_mean,),
            std=(dataset_std,),
            max_pixel_value=1.0,
            p=1.0
        ),
        ToTensorV2(p=1.0)
    ])


def get_validation_augmentations(dataset_mean: float, dataset_std: float):
    return A.Compose([
        A.Normalize(
            mean=(dataset_mean,),
            std=(dataset_std,),
            max_pixel_value=1.0,
            p=1.0
        ),
        ToTensorV2(p=1.0)
    ])


def get_patch_augmentations(dataset_mean: float, dataset_std: float):
    return A.Compose([
        A.Affine(
            translate_percent=(-0.1, 0.1),
            scale=(1.0, 1.0),
            rotate=(-10, 10),
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT, fill=0,
            p=0.8
        ),
        A.GaussNoise(std_range=(0.01, 0.03), p=0.4),
        A.GaussianBlur(blur_limit=(3, 3), p=0.2),
        A.RandomBrightnessContrast(
            brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.Normalize(
            mean=(dataset_mean,),
            std=(dataset_std,),
            max_pixel_value=1.0,
            p=1.0
        ),
        ToTensorV2(p=1.0)
    ])
