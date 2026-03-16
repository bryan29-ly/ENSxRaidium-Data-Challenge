import json

import torch

from ens_data_challenge.inference.find_thresholds import find_best_thresholds
from ens_data_challenge.models.unet import PlainConvUNet
from ens_data_challenge.data_processing.dataset import AbdominalCTDataset
from ens_data_challenge.data_processing.dataloader import get_val_dataloader
from ens_data_challenge.data_processing.augmentations import get_validation_augmentations
from ens_data_challenge import config


def main():

    DATASET_MEAN = 94.301
    DATASET_STD = 48.477
    FOLD = 4

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    fold_dir = config.EXPERIMENTS_DIR / f"run_01_phase1/fold_{FOLD}"
    splits_path = config.DATA_PREPROCESSED_DIR / "splits.json"

    # 1. Read the split file for FOLD 0
    with open(splits_path, "r") as f:
        splits = json.load(f)
    val_keys = splits[str(FOLD)]["val"]

    val_paths = [config.IMAGES_PREPROCESSED_DIR /
                 f"{img_id}.npy" for img_id in val_keys]

    # 2. Init the dataloader
    val_transforms = get_validation_augmentations(DATASET_MEAN, DATASET_STD)

    val_dataset = AbdominalCTDataset(
        image_paths=val_paths,
        labels_dir=config.LABELS_PREPROCESSED_DIR,
        json_path=config.LABELS_JSON_PATH,
        transform=val_transforms
    )

    val_loader = get_val_dataloader(
        val_dataset, batch_size=16, num_workers=4)

    # 3. Init the model
    model = PlainConvUNet(in_channels=1, num_classes=54,
                          deepsupervision=True).to(device)

    checkpoint_path = fold_dir / "checkpoint_best.pth"
    print(f"Load the weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 1. Extraction du dictionnaire de poids
    state_dict = checkpoint['model_state_dict']

    # 2. Remplacement dynamique des clés de l'ancienne architecture vers la nouvelle
    if 'segmentation_head.weight' in state_dict:
        state_dict['heads.0.weight'] = state_dict.pop(
            'segmentation_head.weight')
    if 'segmentation_head.bias' in state_dict:
        state_dict['heads.0.bias'] = state_dict.pop('segmentation_head.bias')

    # 3. Chargement avec strict=False (les têtes 1 et 2 seront initialisées aléatoirement
    # mais ne seront jamais appelées grâce à deepsupervision=False)
    model.load_state_dict(state_dict, strict=False)

    # Find the thresholds
    print(f"Launch the search for optimal thresholds for the fold {FOLD}...")
    find_best_thresholds(val_loader, model, device, save_dir=fold_dir)


if __name__ == "__main__":
    main()
