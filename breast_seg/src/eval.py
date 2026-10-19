# -*- coding: utf-8 -*-
"""
eval.py —— 在某个 split 的 test_internal 与 test_external 上评估模型

用法:
  python -m src.eval --split splits/expA_BUSI_train.json --model unet --seed 42
产出:
  results/<split>_<model>_seed<seed>_internal.csv   (逐图指标)
  results/<split>_<model>_seed<seed>_external.csv
  并在屏幕打印"内部 vs 外部"的 Dice 患者级 95%CI —— 这就是"泛化差距"核心结果
"""
import argparse
import csv
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config as C
from .dataset import BreastSegDataset
from .models import build_model
from .metrics import all_metrics
from .utils import bootstrap_ci_by_patient


@torch.no_grad()
def infer_split(model, image_list, device):
    ds = BreastSegDataset(image_list, train=False)
    ld = DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=False, num_workers=C.NUM_WORKERS)
    rows = []
    for b in ld:
        prob = torch.sigmoid(model(b["image"].to(device))).cpu().numpy()
        gt = b["mask"].numpy()
        for k in range(prob.shape[0]):
            m = all_metrics(prob[k, 0], gt[k, 0])
            m.update({"id": b["id"][k], "patient_id": b["patient_id"][k],
                      "label": b["label"][k]})
            rows.append(m)
    return rows


def summarize(rows, tag):
    dices = [r["dice"] for r in rows]
    pats = [r["patient_id"] for r in rows]
    pt, lo, hi = bootstrap_ci_by_patient(dices, pats)
    hd_valid = [r["hd95"] for r in rows if not np.isnan(r["hd95"])]
    print(f"  [{tag}] 图数={len(rows)} 患者数={len(set(pats))} | "
          f"Dice={pt:.3f} (95%CI {lo:.3f}-{hi:.3f}) | "
          f"HD95可算={len(hd_valid)}/{len(rows)}")
    return pt, lo, hi


def save_csv(rows, path):
    if not rows:
        return
    keys = ["id", "patient_id", "label", "dice", "iou", "precision", "recall", "hd95", "asd"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--model", default="unet")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    split = json.loads((C.ROOT / args.split).read_text())
    split_name = (C.ROOT / args.split).stem
    device = C.device_for(args.model)
    ckpt = C.CKPT_DIR / f"{split_name}_{args.model}_seed{args.seed}.pt"
    state = torch.load(ckpt, map_location=device)
    model = build_model(args.model).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    print(f"评估 {ckpt.name}")
    internal = infer_split(model, split["test_internal"], device)
    external = infer_split(model, split["test_external"], device)
    pi = summarize(internal, "内部测试")
    pe = summarize(external, "外部测试")
    print(f"  >>> 泛化差距(内部-外部 Dice)= {pi[0]-pe[0]:.3f}  <<<  (本研究核心数字)")

    base = f"{split_name}_{args.model}_seed{args.seed}"
    save_csv(internal, C.RESULT_DIR / f"{base}_internal.csv")
    save_csv(external, C.RESULT_DIR / f"{base}_external.csv")
    print("逐图指标已保存到 results/")


if __name__ == "__main__":
    main()
