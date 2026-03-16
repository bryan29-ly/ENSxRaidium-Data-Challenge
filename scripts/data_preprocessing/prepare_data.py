from ens_data_challenge import config
from ens_data_challenge.data_processing.utils.preprocess_utils import compute_dataset_statistics, clip_and_save_images, extract_and_save_labels


def main():
    print("1. Segmentation mask from CSV...")
    extract_and_save_labels(config.LABELS_CSV_PATH,
                            config.LABELS_PREPROCESSED_DIR)

    print("2. Statistics of raw iamges...")
    p05, p995, mean, std = compute_dataset_statistics(
        config.DATA_TRAIN_DIR, config.LABELS_PREPROCESSED_DIR, num_annotated=800)

    print("3. Clipping and save .npy...")
    clip_and_save_images(
        config.DATA_TRAIN_DIR, config.IMAGES_PREPROCESSED_DIR, p05, p995)

    # Statistics to copy
    print(f"\n--- STATISTICS to copy in main training script ---")
    print(f"MEAN: {mean}")
    print(f"STD: {std}")
    print(f"--------------------------------------------------")


if __name__ == "__main__":
    main()
