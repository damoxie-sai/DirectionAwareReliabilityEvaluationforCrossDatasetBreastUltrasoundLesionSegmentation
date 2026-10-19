# -*- coding: utf-8 -*-
"""
train.py —— 训练一个分割模型(单个模型 / 单个种子)

用法示例:
  python -m src.train --split splits/expA_BUSI_train.json --model unet --seed 42
产出:
  checkpoints/<split名>_<model>_seed<seed>.pt   (验证 Dice 最好的权重)
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config as C
from .utils import set_seed
from .dataset import BreastSegDataset
from .models import build_model
from .metrics import dice as dice_metric


# ---------- 损失:BCE + Dice(区域级,应对小病灶类别不平衡) ----------
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target, eps=1e-6):
        bce = self.bce(logits, target)
        prob = torch.sigmoid(logits)
        num = 2 * (prob * target).sum(dim=(1, 2, 3))
        den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
        dice_loss = 1 - (num / den).mean()
        return bce + dice_loss


def lr_at(epoch, base):
    """warmup + 余弦衰减(简化版)。"""
    if epoch < C.WARMUP_EPOCHS:
        return base * (epoch + 1) / C.WARMUP_EPOCHS
    prog = (epoch - C.WARMUP_EPOCHS) / max(1, C.EPOCHS - C.WARMUP_EPOCHS)
    return base * 0.5 * (1 + np.cos(np.pi * prog))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ds = []
    for b in loader:
        logits = model(b["image"].to(device))
        prob = torch.sigmoid(logits).cpu().numpy()
        gt = b["mask"].numpy()
        for k in range(prob.shape[0]):
            ds.append(dice_metric(prob[k, 0], gt[k, 0]))
    return float(np.mean(ds)) if ds else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--model", default="unet", choices=["unet", "unetpp", "segformer"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    args = ap.parse_args()

    set_seed(args.seed)
    device = C.device_for(args.model)
    print("设备:", device, "| 模型:", args.model, "| 种子:", args.seed)

    split = json.loads((C.ROOT / args.split).read_text() if not args.split.startswith("/")
                       else open(args.split).read())
    train_ds = BreastSegDataset(split["train"], train=True)
    val_ds = BreastSegDataset(split["val"], train=False)
    train_ld = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
                          num_workers=C.NUM_WORKERS)
    val_ld = DataLoader(val_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                        num_workers=C.NUM_WORKERS)

    model = build_model(args.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    crit = BCEDiceLoss()

    split_name = C.ROOT.joinpath(args.split).stem if not args.split.startswith("/") else "split"
    ckpt = C.CKPT_DIR / f"{split_name}_{args.model}_seed{args.seed}.pt"
    best = -1.0

    for ep in range(args.epochs):
        for g in opt.param_groups:
            g["lr"] = lr_at(ep, C.LR)
        model.train()
        losses = []
        for b in train_ld:
            opt.zero_grad()
            logits = model(b["image"].to(device))
            loss = crit(logits, b["mask"].to(device))
            loss.backward()
            opt.step()
            losses.append(loss.item())
        vdice = evaluate(model, val_ld, device)
        print(f"epoch {ep+1:3d}/{args.epochs} | loss {np.mean(losses):.4f} | val Dice {vdice:.4f}")
        if vdice > best:
            best = vdice
            torch.save({"model": args.model, "state_dict": model.state_dict(),
                        "val_dice": best}, ckpt)
    print(f"训练完成,最佳 val Dice={best:.4f},权重保存到 {ckpt}")


if __name__ == "__main__":
    main()
