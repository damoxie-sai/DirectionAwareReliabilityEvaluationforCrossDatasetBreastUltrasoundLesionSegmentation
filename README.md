# Direction-Aware Reliability Evaluation for Cross-Dataset Breast Ultrasound Lesion Segmentation

This repository contains the code and reproducibility materials for the Paper4 methodological study:

> Direction-aware evaluation of cross-dataset generalization and failure detection in breast ultrasound lesion segmentation.

The study evaluates U-Net, U-Net++, and SegFormer in both BUSI-to-BUS-UCLM and BUS-UCLM-to-BUSI transfer directions. It reports external failure behavior, predictive entropy, approximate test-time MC-dropout, deep-ensemble disagreement, and selective prediction.

This is a public-data methodological evaluation. It does not establish diagnostic performance, clinical safety, or replacement of clinical experts.

## Repository Contents

- `breast_seg/src/`: data preparation, training, evaluation, uncertainty, ensemble, table, and figure-generation code.
- `breast_seg/splits/`: fixed experiment split definitions.
- `breast_seg/figures/`: generated figure outputs included for reference.
- `breast_seg/requirements.txt`: Python package requirements.
- `manuscript/tables/`: aggregated tables used in the manuscript.
- `release/figures/`: submission-named PDF and PNG figures.

Raw data, model checkpoints, and per-case result files are not included in this public repository.

## Data Access

Users must obtain the datasets from their official sources and follow the applicable terms of use. Do not upload the raw datasets to this repository.

### BUSI

The Breast Ultrasound Images dataset (BUSI) is described in:

Al-Dhabyani et al., “Dataset of breast ultrasound images,” *Data in Brief*, 2020.

DOI: https://doi.org/10.1016/j.dib.2019.104863

The exact download source and license for the local BUSI copy should be verified from the actual source used. The dataset is not redistributed here.

### BUS-UCLM

The Breast Ultrasound Lesion Segmentation dataset from the University of Castilla-La Mancha (BUS-UCLM) is described at:

- Article: https://doi.org/10.1038/s41597-025-04562-3
- Data record: https://doi.org/10.17632/7fvgj4jsp7.3
- Official dataset utilities: https://github.com/noeliavallez/BUS-UCLM-Dataset

The BUS-UCLM dataset is distributed under CC BY 4.0. This dataset license is distinct from the license of the associated Scientific Data article.

## Environment

The original experiments were completed on Apple Silicon and included an environment-specific SegFormer CPU fallback. The code also supports CPU execution. Users should create a clean environment appropriate for their operating system and record package versions.

```bash
cd breast_seg
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The historical shell scripts refer to local checkpoint and result paths that are intentionally not part of this public release. Treat them as experiment orchestration references; adapt them after obtaining the data and running the required stages.

## Reproduction Outline

1. Obtain BUSI and BUS-UCLM from the official sources above.
2. Place the datasets under `breast_seg/data/` according to `breast_seg/README.md`.
3. Run data preparation and verify the manifest.
4. Use the committed split JSON files rather than silently generating new splits.
5. Train or evaluate the requested model and seed.
6. Generate per-case results locally.
7. Generate tables and figures from the local result files.

The committed split files and scripts are intended to make the protocol explicit; raw data, checkpoints, and local result files remain outside this repository.

## Citation

The manuscript citation and final author information will be added when the internal review version is finalized.

## License

The license for original code in this repository is pending author confirmation. Dataset licenses remain separate and are governed by their respective source records.
