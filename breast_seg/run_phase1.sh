#!/bin/bash
# Phase 1 编排:expA · seed42 · unetpp 与 segformer 全流程串行执行
# 说明见 TRAINING_PLAN.md。日志在 logs/phase1.log。
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
SPLIT=splits/expA_BUSI_train.json
LOG=logs/phase1.log
mkdir -p logs

say(){ echo ""; echo "==================== $* ===================="; }

# 等待已在后台运行的 unetpp 训练进程结束(如果还在跑)
wait_pid=${1:-}
if [ -n "$wait_pid" ]; then
  say "等待现有 unetpp 训练进程 PID=$wait_pid 结束"
  while ps -p "$wait_pid" >/dev/null 2>&1; do sleep 20; done
  say "unetpp 训练已结束"
fi

# 若 unetpp 权重不存在(说明前面训练失败),则补训
if [ ! -f checkpoints/expA_BUSI_train_unetpp_seed42.pt ]; then
  say "unetpp 权重缺失,重新训练"
  $PY -m src.train --split "$SPLIT" --model unetpp --seed 42
fi

say "评估 unetpp"
$PY -m src.eval        --split "$SPLIT" --model unetpp --seed 42
say "失败检测 unetpp"
$PY -m src.uncertainty --split "$SPLIT" --model unetpp --seed 42 --which external --T 20

say "训练 segformer"
$PY -m src.train       --split "$SPLIT" --model segformer --seed 42
say "评估 segformer"
$PY -m src.eval        --split "$SPLIT" --model segformer --seed 42
say "失败检测 segformer"
$PY -m src.uncertainty --split "$SPLIT" --model segformer --seed 42 --which external --T 20

say "Phase 1 全部完成"
