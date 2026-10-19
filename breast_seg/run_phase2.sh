#!/bin/bash
# Phase 2:多种子稳健性。三模型 × seed{43,44} 全流程(训练→评估→失败检测)。
# 幂等/断点续跑:已有权重则跳过训练,已有结果 csv 则跳过评估/失败检测。
# 建议用 caffeinate 包裹以防睡眠:
#   nohup caffeinate -is ./run_phase2.sh > logs/phase2.log 2>&1 &
# 说明见 TRAINING_PLAN.md。unet/unetpp 走 MPS(快),segformer 自动走 CPU(慢~1.5h)。
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
SPLIT=splits/expA_BUSI_train.json
say(){ echo ""; echo "==================== $* ===================="; }

run_one(){
  local model=$1 seed=$2
  local base="expA_BUSI_train_${model}_seed${seed}"
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

# 先跑快的(MPS: unet/unetpp),再跑慢的(CPU: segformer),这样早点看到结果
for model in unet unetpp segformer; do
  for seed in 43 44; do
    run_one "$model" "$seed" || true   # 单组失败不影响其它组
  done
done

say "Phase 2 全部结束(请核验 results/ 下 12 个 csv 是否齐全)"
