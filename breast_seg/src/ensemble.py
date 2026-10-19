# -*- coding: utf-8 -*-
"""
ensemble.py —— Deep Ensemble(多种子集成)+ 集成分歧失败检测

思路:
  同一模型结构、三个不同随机种子(42/43/44)训练出三套权重。
  - 集成预测 = 三个成员概率图的平均,再阈值。通常比单模型更准、更稳。
  - 不确定性(集成分歧)= 三个成员在每个像素上的方差,对全图取平均 → 每图一个标量。
    成员们越"不一致",越可能是难/失败样本。
  - 另报集成均值的预测熵作对照。
  失败检测:分歧/熵越高是否越对应低 Dice?用 AUROC 衡量。免训练(直接读已有权重)。

用法:
  python -m src.ensemble --split splits/expA_BUSI_train.json --model unet --seeds 42 43 44
产出(both internal & external):
  results/<split>_<model>_ensemble_{internal,external}.csv
  results/<split>_<model>_ensemble_external_riskcoverage.csv
  屏幕打印:集成内部/外部 Dice(患者级 95%CI)、泛化差距、失败检测 AUROC(分歧 vs 熵)
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
from .metrics import all_metrics, dice as dice_metric
from .utils import bootstrap_ci_by_patient
from .uncertainty import auroc, risk_coverage


@torch.no_grad()
def ensemble_infer(models, image_list, device):
    """对一批图跑集成,返回逐图记录(含指标 + 分歧 + 熵)。"""
    ds = BreastSegDataset(image_list, train=False)
    ld = DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=False, num_workers=C.NUM_WORKERS)
    rows = []
    for b in ld:
        x = b["image"].to(device)
        member_probs = []
        for m in models:
            member_probs.append(torch.sigmoid(m(x)).cpu().numpy())   # [B,1,H,W]
        member_probs = np.stack(member_probs, 0)          # [M,B,1,H,W]
        ens = member_probs.mean(0)                        # 集成均值概率 [B,1,H,W]
        disagree = member_probs.var(0)                    # 逐像素成员方差 [B,1,H,W]
        ep = np.clip(ens, 1e-6, 1 - 1e-6)
        entropy_map = -(ep * np.log(ep) + (1 - ep) * np.log(1 - ep))
        gt = b["mask"].numpy()
        for k in range(ens.shape[0]):
            met = all_metrics(ens[k, 0], gt[k, 0])
            met.update({
                "id": b["id"][k], "patient_id": b["patient_id"][k], "label": b["label"][k],
                "disagreement": float(disagree[k, 0].mean()),
                "entropy": float(entropy_map[k, 0].mean()),
            })
            rows.append(met)
    return rows


def summarize(rows, tag):
    dices = [r["dice"] for r in rows]
    pats = [r["patient_id"] for r in rows]
    pt, lo, hi = bootstrap_ci_by_patient(dices, pats)
    hd_valid = [r["hd95"] for r in rows if not np.isnan(r["hd95"])]
    print(f"  [{tag}] 图数={len(rows)} 患者数={len(set(pats))} | "
          f"集成 Dice={pt:.3f} (95%CI {lo:.3f}-{hi:.3f}) | HD95可算={len(hd_valid)}/{len(rows)}")
    return pt


def save_csv(rows, path):
    keys = ["id", "patient_id", "label", "dice", "iou", "precision", "recall",
            "hd95", "asd", "disagreement", "entropy"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--model", default="unet")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--fail_dice", type=float, default=0.5)
    args = ap.parse_args()

    split = json.loads((C.ROOT / args.split).read_text())
    split_name = (C.ROOT / args.split).stem
    device = C.device_for(args.model)

    # 加载三个种子成员
    models = []
    for s in args.seeds:
        ckpt = C.CKPT_DIR / f"{split_name}_{args.model}_seed{s}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"缺少权重:{ckpt}(先训练该种子)")
        m = build_model(args.model).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
        m.eval()
        models.append(m)
    print(f"Deep Ensemble | {args.model} | 成员种子={args.seeds} | 设备={device}")

    internal = ensemble_infer(models, split["test_internal"], device)
    external = ensemble_infer(models, split["test_external"], device)
    pi = summarize(internal, "内部测试")
    pe = summarize(external, "外部测试")
    print(f"  >>> 集成泛化差距(内部-外部 Dice)= {pi-pe:.3f} <<<")

    # 失败检测:用外部集
    dices = np.array([r["dice"] for r in external])
    disc = np.array([r["disagreement"] for r in external])
    ent = np.array([r["entropy"] for r in external])
    fail = (dices < args.fail_dice).astype(int)
    au_disc = auroc(disc, fail)
    au_ent = auroc(ent, fail)
    print(f"  外部失败率(<{args.fail_dice})={fail.mean():.2%}")
    print(f"  失败检测 AUROC:  集成分歧={au_disc:.3f}   集成熵={au_ent:.3f}")

    best_name, best_unc, best_au = (("集成分歧", disc, au_disc) if au_disc >= au_ent
                                    else ("集成熵", ent, au_ent))
    cov, md = risk_coverage(dices, best_unc)
    print(f"  用『{best_name}』(AUROC={best_au:.3f})做拒绝:")
    for c in (0.5, 0.7, 0.8, 0.9):
        i = min(range(len(cov)), key=lambda j: abs(cov[j] - c))
        print(f"    保留最确定 {cov[i]*100:.0f}% -> 平均Dice {md[i]:.3f}  (全体={dices.mean():.3f})")

    base = f"{split_name}_{args.model}_ensemble"
    save_csv(internal, C.RESULT_DIR / f"{base}_internal.csv")
    save_csv(external, C.RESULT_DIR / f"{base}_external.csv")
    with open(C.RESULT_DIR / f"{base}_external_riskcoverage.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["coverage", "mean_dice", "signal"])
        for c, m in zip(cov, md):
            w.writerow([f"{c:.4f}", f"{m:.4f}", best_name])
    print("已保存集成结果到 results/")


if __name__ == "__main__":
    main()
