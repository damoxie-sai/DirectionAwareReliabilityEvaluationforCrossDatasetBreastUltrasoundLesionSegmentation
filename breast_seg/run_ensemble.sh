#!/bin/bash
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
for split in splits/expA_BUSI_train.json splits/expB_UCLM_train.json; do
  for model in unet unetpp segformer; do
    echo ""; echo "########## $split | $model ##########"
    $PY -m src.ensemble --split "$split" --model "$model"
  done
done
echo ""; echo "########## Ensemble 全部完成 ##########"
