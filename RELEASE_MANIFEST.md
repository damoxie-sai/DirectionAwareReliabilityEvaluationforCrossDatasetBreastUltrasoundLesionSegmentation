# Public Repository Staging Manifest

Target repository:

`https://github.com/damoxie-sai/DirectionAwareReliabilityEvaluationforCrossDatasetBreastUltrasoundLesionSegmentation`

## Included

- Root `README.md` with study scope, data sources, non-redistribution statement, and reproduction outline.
- Root `.gitignore` excluding data, checkpoints, results, environments, archives, and secrets.
- `breast_seg/src/` with experiment and analysis code.
- `breast_seg/splits/` with the two fixed experiment split files.
- `breast_seg/README.md` with local experiment instructions.
- `breast_seg/requirements.txt`.
- Experiment orchestration shell scripts as historical protocol references.
- Aggregated manuscript tables.
- Generated PDF and PNG figures for reference and manuscript support.
- `LICENSE_DECISION_REQUIRED.md`, which must be resolved before the repository is considered a final code release.

## Excluded

- Raw BUSI and BUS-UCLM data.
- Data archives and ZIP files.
- `breast_seg/data/`.
- `breast_seg/checkpoints/` and all model weights.
- `breast_seg/results/` and per-case result files.
- `.venv/` and other environments.
- Internal review notes, unfinished author metadata, and local project state.

## Pre-Push Checks

- Confirm repository visibility is Public.
- Select and add a license for original Paper4 code, or intentionally leave the code license pending for internal review.
- Review `git diff --cached --name-only` before pushing.
- Confirm no raw data, checkpoint, secret, or local path is staged.
- Open the public repository in an incognito browser after push.
