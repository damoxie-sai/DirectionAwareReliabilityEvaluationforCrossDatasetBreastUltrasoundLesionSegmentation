#!/bin/bash
# SegFormer 全流程(expA · seed42)。因 MPS 反向不兼容,segformer 自动走 CPU(见 config.device_for)。
# 训练约 1.5 小时。日志:logs/segformer.log
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
SPLIT=splits/expA_BUSI_train.json
say(){ echo ""; echo "==================== $* ===================="; }

say "训练 segformer (CPU, 约1.5h)"
$PY -m src.train       --split "$SPLIT" --model segformer --seed 42
if [ ! -f checkpoints/expA_BUSI_train_segformer_seed42.pt ]; then
  say "训练失败:权重未生成,终止"; exit 1
fi
say "评估 segformer"
$PY -m src.eval        --split "$SPLIT" --model segformer --seed 42
say "失败检测 segformer"
$PY -m src.uncertainty --split "$SPLIT" --model segformer --seed 42 --which external --T 20
say "SegFormer 全流程完成"
