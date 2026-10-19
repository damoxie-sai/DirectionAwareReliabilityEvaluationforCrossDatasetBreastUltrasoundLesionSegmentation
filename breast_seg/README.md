# breast_seg: Cross-Dataset Breast Ultrasound Segmentation

This directory contains the experiment code for comparing U-Net, U-Net++, and SegFormer on BUSI and BUS-UCLM. The study evaluates cross-dataset generalization and whether uncertainty signals can identify failed segmentation cases.

---

## 0. Environment

```bash
python -m venv .venv
source .venv/bin/activate
cd breast_seg
pip install -r requirements.txt
python -m src.config
```

## 1. Obtain and place data

- BUSI: place the extracted `benign/`, `malignant/`, and `normal/` directories under `data/BUSI/`.
- BUS-UCLM: place `images/`, `masks/`, and the metadata CSV under `data/BUS-UCLM/`.
- Obtain both datasets from their official source records and follow their terms. Do not commit raw data to this repository.

BUS-UCLM contains approximately 38 patients; patient-level aggregation is required where patient identifiers are available.

## 2. Prepare the manifest

```bash
python -m src.prepare_data      # 生成 data/manifest.csv + data/processed/
```

## 3. Use the fixed splits

```bash
python -m src.make_splits       # 生成 splits/expA_BUSI_train.json 等
```

## 4. Train

```bash
python -m src.train --split splits/expA_BUSI_train.json --model unet    --seed 42
python -m src.train --split splits/expA_BUSI_train.json --model unetpp  --seed 42
python -m src.train --split splits/expA_BUSI_train.json --model segformer --seed 42
```

## 5. Evaluate internal and external performance

```bash
python -m src.eval --split splits/expA_BUSI_train.json --model unet --seed 42
# The command reports internal Dice, external Dice, and the generalization gap.
```

## 6. Failure detection and risk-coverage

```bash
python -m src.uncertainty --split splits/expA_BUSI_train.json --model unet --seed 42 --which external --T 20
# The command reports failure-detection AUROC and risk-coverage outputs.
```

## 7. Symmetric experiment B

Replace the split with `splits/expB_UCLM_train.json` to run the reverse transfer direction.

## 8. Multiple seeds and ensemble

Run seeds 42, 43, and 44 separately and average the probability maps for the deep ensemble.

---

## Suggested order

Start with one U-Net configuration and verify the manifest, split, training, and evaluation outputs before expanding to other models, seeds, uncertainty signals, and ensemble analysis.

## Directory layout
```
breast_seg/
├─ data/         原始+整理后数据、manifest.csv
├─ splits/       患者级划分（可复现的关键）
├─ checkpoints/  模型权重
├─ results/      逐图指标、不确定性、risk-coverage
└─ src/          config/utils/prepare_data/make_splits/dataset/models/metrics/train/eval/uncertainty
```

## Scope

This is a public-data methodological evaluation of generalization, reliability, and failure risk. It does not claim clinical diagnosis, replacement of clinicians, or reduction of biopsy procedures. The manuscript was prepared with reference to CLAIM and STARD-AI principles where applicable.
