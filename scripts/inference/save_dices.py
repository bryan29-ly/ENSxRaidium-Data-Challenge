import os
import re
import json
import numpy as np

from ens_data_challenge import config


def extract_oof_matrix():
    base_dir = config.EXPERIMENTS_DIR / "run_01_phase1"
    num_folds = 5
    num_classes = 54

    # Initialisation de la matrice [5, 54]
    oof_dice_matrix = np.zeros((num_folds, num_classes))
    json_matrix = {}

    # Regex pour extraire l'époque et les scores des classes
    epoch_pattern = re.compile(r"Epoch \[(\d+)/\d+\]")
    dice_pattern = re.compile(r"L(\d+):\s+([\d\.]+)")

    print("--- Extraction des matrices de compétence (OOF Dice) ---")

    for fold in range(num_folds):
        log_path = os.path.join(base_dir, f"fold_{fold}", "training.log")

        if not os.path.exists(log_path):
            print(f"Fichier introuvable : {log_path}")
            continue

        best_epoch = -1
        best_scores = np.zeros(num_classes)

        current_epoch = -1
        is_best_epoch = False

        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                # Détection d'une nouvelle ligne d'époque
                epoch_match = epoch_pattern.search(line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                    # Vérification si c'est un nouveau record
                    if "(Yess NEW BEST SCORE hehe)" in line:
                        is_best_epoch = True
                        best_epoch = current_epoch
                    else:
                        is_best_epoch = False

                # Si nous sommes en train de lire les lignes de détail du meilleur checkpoint
                if is_best_epoch:
                    class_matches = dice_pattern.findall(line)
                    for class_idx_str, score_str in class_matches:
                        # L01 correspond à l'index 0
                        c_idx = int(class_idx_str) - 1
                        score = float(score_str)
                        best_scores[c_idx] = score

        # Sauvegarde des scores du fold dans la matrice globale
        oof_dice_matrix[fold] = best_scores
        json_matrix[f"fold_{fold}"] = {
            f"L{i+1:02d}": best_scores[i] for i in range(num_classes)}

        print(
            f"Fold {fold} : Meilleurs poids récupérés à l'Epoch {best_epoch:04d}")

    # --- Sauvegardes ---
    out_npy = os.path.join(base_dir, "oof_dice_matrix.npy")
    out_json = os.path.join(base_dir, "oof_dice_matrix.json")

    np.save(out_npy, oof_dice_matrix)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(json_matrix, f, indent=4)

    print("\n--- Sauvegarde terminée ---")
    print(f"Matrice Numpy : {out_npy}")
    print(f"Fichier JSON  : {out_json}")


if __name__ == "__main__":
    extract_oof_matrix()
