import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from ens_data_challenge import config


def create_stratified_splits(
    masks_dir: Path,
    output_json: Path,
    n_splits: int = 5,
    num_classes: int = 54,
    num_images: int = 800
):
    print(f"Analyse of {num_images} masks for stratification...")

    # 1. List the 800 annotated
    mask_paths = sorted(list(masks_dir.glob("*.npy")),
                        key=lambda p: int(p.stem))[:num_images]
    image_ids = [p.stem for p in mask_paths]

    # 2. Init the binary amsks
    y = np.zeros((num_images, num_classes), dtype=np.int8)

    # 3. Grount Truth
    for idx, path in enumerate(tqdm(mask_paths, desc="Extraction des labels GT")):
        mask = np.load(path)
        present_classes = np.unique(mask)

        for c in present_classes:
            # Background and borders
            if c > 0 and c <= num_classes:
                y[idx, c - 1] = 1

    # 4. Stratification checking
    mskf = MultilabelStratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=42)

    splits_dict = {}
    verification_passed = True

    print("\n--- Creation and checking of stratification ---")

    for fold_idx, (train_index, val_index) in enumerate(mskf.split(image_ids, y)):

        train_ids = [image_ids[i] for i in train_index]
        val_ids = [image_ids[i] for i in val_index]

        splits_dict[str(fold_idx)] = {
            "train": train_ids,
            "val": val_ids
        }

        train_presence = y[train_index].sum(axis=0)
        val_presence = y[val_index].sum(axis=0)

        missing_in_train = np.where(train_presence == 0)[0] + 1
        missing_in_val = np.where(val_presence == 0)[0] + 1

        if len(missing_in_train) > 0:
            print(
                f"Warning - Fold {fold_idx} : Missing classes from training phase -> {missing_in_train.tolist()}")
            verification_passed = False

        if len(missing_in_val) > 0:
            print(
                f"Warning - Fold {fold_idx} : Missing classing from training phase -> {missing_in_val.tolist()}")
            verification_passed = False

    # 5. JSON Saving
    with open(output_json, "w") as f:
        json.dump(splits_dict, f, indent=4)

    print(f"\nFile saved in : {output_json}")
    for k, v in splits_dict.items():
        print(f"Fold {k} - Train: {len(v['train'])} | Val: {len(v['val'])}")

    print("-" * 40)
    if verification_passed:
        print("Success. All classes are in Train and Val for all folds.")
    else:
        print("Fail.")


if __name__ == "__main__":
    mask_directory = config.LABELS_PREPROCESSED_DIR
    json_output = config.DATA_PREPROCESSED_DIR / "splits.json"
    create_stratified_splits(mask_directory, json_output)
