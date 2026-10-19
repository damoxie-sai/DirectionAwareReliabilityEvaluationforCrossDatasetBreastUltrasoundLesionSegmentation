#!/bin/bash
# Phase 3:对称实验 expB(BUS-UCLM 训练 → BUSI 外部)。三模型 × seed{42,43,44} 全流程。
# 与 expA 对称,验证泛化差距是否双向。幂等/断点续跑。
# 建议 caffeinate 防睡眠:
#   nohup caffeinate -is ./run_phase3.sh > logs/phase3.log 2>&1 &
# unet/unetpp 走 MPS(快),segformer 自动走 CPU(~1.4h)。
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
SPLIT=splits/expB_UCLM_train.json
say(){ echo ""; echo "==================== $* ===================="; }

run_one(){
  local model=$1 seed=$2
  local base="expB_UCLM_train_${model}_seed${seed}"
  say "[$model seed$seed] 开始"

  if [ -f "checkpoints/${base}.pt" ]; then
    echo ">> 已有权重,跳过训练"
  else
    say "[$model seed$seed] 训练"
    $PY -m src.train --split "$SPLIT" --model "$model" --seed "$seed"
  fi
  if [ ! -f "checkpoints/${base}.pt" ]; then
    echo "!! 训练失败:权重未生成,跳过该组后续步骤"; return 1
  fi

  if [ -f "results/${base}_external.csv" ]; then
    echo ">> 已有评估结果,跳过评估"
  else
    say "[$model seed$seed] 评估"
    $PY -m src.eval --split "$SPLIT" --model "$model" --seed "$seed"
  fi

  if [ -f "results/${base}_external_uncertainty.csv" ]; then
    echo ">> 已有失败检测结果,跳过"
  else
    say "[$model seed$seed] 失败检测"
    $PY -m src.uncertainty --split "$SPLIT" --model "$model" --seed "$seed" --which external --T 20
  fi
  say "[$model seed$seed] 完成"
}

# 先快后慢:MPS(unet/unetpp)→ CPU(segformer)
for model in unet unetpp segformer; do
  for seed in 42 43 44; do
    run_one "$model" "$seed" || true
  done
done

say "Phase 3 全部结束(请核验 results/ 下 expB 的 18 个 csv 是否齐全)"
