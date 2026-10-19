# -*- coding: utf-8 -*-
"""
uncertainty.py —— MC Dropout 不确定性 + 失败检测 + Risk-Coverage 曲线

思路(呼应笔记第六部分):
  同一张图,推理时打开 Dropout,跑 T 次,得到 T 张概率图。
  - 预测掩膜 = T 次平均后阈值
  - 不确定性 = T 次预测的"波动"(逐像素方差的平均)
  失败检测:不确定性越高,是否越对应低 Dice?用 AUROC 衡量。
  Risk-Coverage:按不确定性从低到高保留病例,看平均 Dice 如何变化。

用法:
  python -m src.uncertainty --split splits/expA_BUSI_train.json --model unet --seed 42 --which external --T 20
"""
import argparse
import csv
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

import torch.nn.functional as F
from . import config as C
from .dataset import BreastSegDataset
from .models import build_model
from .metrics import dice as dice_metric


# ---------- 正确版 MC Dropout:把 dropout 注入"解码器多通道特征图" ----------
# 旧实现把 Dropout2d 加在 1 通道输出上 → 整张图被丢,方差无意义、方向还反了。
# 这里用 forward hook 在 backbone.decoder 的输出(如 16 通道特征)上做 Dropout2d,
# 推理时临时注入、不改权重、无需重训。
def _find_decoder(model):
    bb = getattr(model, "backbone", model)
    return getattr(bb, "decoder", None)


@torch.no_grad()
def mc_predict(model, image, device, T=20, p=0.2):
    """
    返回 (mean_prob, mc_uncertainty, entropy):
      mean_prob     : T 次 dropout 前向的平均概率图
      mc_uncertainty: T 次预测的逐像素方差的均值(每图一个标量;越大越不确定)
      entropy       : 单次干净前向的平均像素预测熵(每图一个标量;另一种不确定性信号)
    """
    x = image.to(device)
    # 1) 单次干净前向 → 预测熵(不加 dropout)
    clean = torch.sigmoid(model(x)).clamp(1e-6, 1 - 1e-6)
    ent = -(clean * clean.log() + (1 - clean) * (1 - clean).log())     # 逐像素熵
    entropy = ent.reshape(ent.shape[0], -1).mean(1).cpu().numpy()

    # 2) 在解码器特征图上注入 dropout,跑 T 次 → MC 方差
    dec = _find_decoder(model)
    probs = []
    if dec is not None:
        def hook(module, inp, out):
            return F.dropout2d(out, p=p, training=True)
        h = dec.register_forward_hook(hook)
        try:
            for _ in range(T):
                probs.append(torch.sigmoid(model(x)).cpu().numpy())
        finally:
            h.remove()
        probs = np.stack(probs, 0)                 # [T,B,1,H,W]
        mean_p = probs.mean(0)
        mc_unc = probs.var(0).reshape(probs.shape[1], -1).mean(1)
    else:
        # 找不到 decoder(如某些结构)→ 退回用熵,MC 方差置 0
        mean_p = clean.cpu().numpy()
        mc_unc = np.zeros(entropy.shape, dtype=float)
    return mean_p, mc_unc, entropy


def auroc(scores, labels):
    """labels: 1=失败(低Dice),0=正常。scores: 不确定性(越大越可能失败)。"""
    scores = np.asarray(scores); labels = np.asarray(labels)
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # 概率:随机一对(pos,neg)中 pos 分更高
    wins = 0.0
    for s in pos:
        wins += (neg < s).sum() + 0.5 * (neg == s).sum()
    return float(wins / (len(pos) * len(neg)))


def auroc_ci(scores, labels, patient_ids, n_boot=2000, alpha=0.05, seed=0):
    """AUROC 的患者级聚类 bootstrap 95%CI。返回 (point, lo, hi)。
    labels: 1=失败,0=正常;scores: 不确定性(越大越可能失败);patient_ids: 每例所属患者。"""
    scores = np.asarray(scores); labels = np.asarray(labels)
    pids = np.asarray(patient_ids)
    uniq = np.unique(pids)
    idx_by_pat = {p: np.where(pids == p)[0] for p in uniq}
    rng = np.random.RandomState(seed)
    point = auroc(scores, labels)
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_pat[p] for p in samp])
        a = auroc(scores[idx], labels[idx])
        if not np.isnan(a):
            boots.append(a)
    if not boots:
        return point, float("nan"), float("nan")
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return point, lo, hi


def risk_coverage(dices, uncs):
    """按不确定性升序保留病例,返回 (coverage[], mean_dice[])。"""
    order = np.argsort(uncs)            # 最确定的排前面
    d_sorted = np.asarray(dices)[order]
    cov, md = [], []
    for k in range(1, len(d_sorted) + 1):
        cov.append(k / len(d_sorted))
        md.append(float(np.mean(d_sorted[:k])))
    return cov, md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--model", default="unet")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--which", default="external", choices=["internal", "external"])
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--fail_dice", type=float, default=0.5, help="Dice 低于此值视为失败")
    args = ap.parse_args()

    split = json.loads((C.ROOT / args.split).read_text())
    split_name = (C.ROOT / args.split).stem
    device = C.device_for(args.model)
    ckpt = C.CKPT_DIR / f"{split_name}_{args.model}_seed{args.seed}.pt"
    model = build_model(args.model).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    model.eval()                        # 全网 eval;dropout 由 hook 在推理时按需注入

    image_list = split["test_" + args.which]
    ds = BreastSegDataset(image_list, train=False)
    ld = DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=False, num_workers=C.NUM_WORKERS)

    dices, mc_uncs, ents, ids = [], [], [], []
    for b in ld:
        mean_p, mc_unc, ent = mc_predict(model, b["image"], device, T=args.T, p=C.DROPOUT_P)
        gt = b["mask"].numpy()
        for k in range(mean_p.shape[0]):
            dices.append(dice_metric(mean_p[k, 0], gt[k, 0]))
            mc_uncs.append(float(mc_unc[k]))
            ents.append(float(ent[k]))
            ids.append(b["id"][k])

    dices = np.array(dices); mc_uncs = np.array(mc_uncs); ents = np.array(ents)
    fail = (dices < args.fail_dice).astype(int)

    # 两种不确定性信号各自的失败检测能力
    au_mc = auroc(mc_uncs, fail)
    au_ent = auroc(ents, fail)
    print(f"[{args.which}] 图数={len(dices)} 平均Dice={dices.mean():.3f} "
          f"失败率(<{args.fail_dice})={fail.mean():.2%}")
    print(f"失败检测 AUROC:  MC-Dropout方差={au_mc:.3f}   预测熵={au_ent:.3f}  (0.5=瞎猜,越高越好)")

    # 用较好的那个信号画 risk-coverage
    best_name, best_unc, best_au = ("MC-Dropout方差", mc_uncs, au_mc) if (au_mc >= au_ent) else ("预测熵", ents, au_ent)
    cov, md = risk_coverage(dices, best_unc)
    print(f"用『{best_name}』(AUROC={best_au:.3f})做拒绝:")
    for c in (0.5, 0.7, 0.8, 0.9):
        i = min(range(len(cov)), key=lambda j: abs(cov[j] - c))
        print(f"  保留最确定的 {cov[i]*100:.0f}% 病例 -> 平均Dice {md[i]:.3f}  (全体={dices.mean():.3f})")

    base = f"{split_name}_{args.model}_seed{args.seed}_{args.which}"
    with open(C.RESULT_DIR / f"{base}_uncertainty.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "dice", "mc_uncertainty", "entropy", "fail"])
        for i in range(len(dices)):
            w.writerow([ids[i], f"{dices[i]:.4f}", f"{mc_uncs[i]:.6f}", f"{ents[i]:.6f}", fail[i]])
    with open(C.RESULT_DIR / f"{base}_riskcoverage.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["coverage", "mean_dice", "signal"])
        for c, m in zip(cov, md):
            w.writerow([f"{c:.4f}", f"{m:.4f}", best_name])
    print("已保存不确定性与 risk-coverage 到 results/")


if __name__ == "__main__":
    main()
