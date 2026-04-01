# ENS × Raidium Data Challenge — 5th Place Solution

Semantic segmentation of 54 anatomical structures from 2D CT scans. Full methodology in [`Rapport_ENSxRaidium_DataChallenge_Bryan_Ly.pdf`](./Rapport_ENSxRaidium_DataChallenge_Bryan_Ly.pdf).

## Results

| Model | Dice Score |
|---|---|
| Custom U-Net + Anchor-Based Cascade | **0.5569** |

## Setup

**Option A — `uv` (Recommended)**
```bash
uv sync
source .venv/bin/activate
```

**Option B — `pip`**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Data Preparation

Place the raw dataset as follows:
```
data/raw/
├── train-images/
├── annotated_labels.json
└── label_Hnl61pT.csv
```
```bash
python scripts/data_preprocessing/prepare_data.py
python src/ens_data_challenge/data_processing/make_splits.py

```

## Training
```bash
# Phase 1 - Global U-Net
python scripts/train.py

# Phase 2 - Anchor-Based Cascade
python scripts/train_cascade.py
```

## Inference
```bash
# Phase 1 - Global U-Net Inference
python scripts/run_inference_no_thresh.py

# Phase 2 - Anchor-Based Cascade Inference
python scripts/run_inference_cascade_final.py
```
