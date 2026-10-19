# -*- coding: utf-8 -*-
"""
make_qualitative.py —— Fig6 定性分割对比图(发表级)

对每个迁移方向(expA 外部=BUS-UCLM;expB 外部=BUSI),自动挑代表性外部病例:
  - 一个"成功"病例(三模型 seed42 平均 Dice 最高)
  - 一个"失败"病例(平均 Dice 低,含空/漏预测,体现 confident-empty 失败模式)
列 = Input | Ground truth | U-Net | U-Net++ | SegFormer;
每个模型格叠加 预测(红)+ 金标准轮廓(绿)+ 标注 Dice。

输出:figures/fig6_qualitative.pdf / .png
用法:python -m src.make_qualitative
"""
import csv
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from . import config as C
from .models import build_model
from .metrics import dice as dice_metric

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 330, "savefig.bbox": "tight",
})

MODELS = ["unet", "unetpp", "segformer"]
MLABEL = {"unet": "U-Net", "unetpp": "U-Net++", "segformer": "SegFormer"}
EXPS = {"expA": ("expA_BUSI_train", "Train BUSI \u2192 Test BUS-UCLM"),
        "expB": ("expB_UCLM_train", "Train BUS-UCLM \u2192 Test BUSI")}
GT_COLOR = "#00B050"     # 金标准=绿
PR_COLOR = "#E8000B"     # 预测=红


def load_csv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def manifest_map():
    m = {}
    with open(C.MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m[r["image"]] = r
    return m


def _gt_area(row):
    msk = np.array(Image.open(C.ROOT / row["mask"]).convert("L"), dtype=np.float32)
    return int((msk > 127).sum())


def pick_cases(exp, man):
    """返回 [(id, tag), ...]:一个 success + 一个 failure。均要求金标准有真实病灶(面积>=300px)。"""
    stem = EXPS[exp][0]
    dd = defaultdict(dict)
    for m in MODELS:
        for r in load_csv(C.RESULT_DIR / f"{stem}_{m}_seed42_external.csv"):
            dd[r["id"]][m] = float(r["dice"])
    ids = [i for i in dd if len(dd[i]) == 3]
    meandice = {i: np.mean(list(dd[i].values())) for i in ids}
    MIN_AREA = 300

    def has_lesion(i):
        return _gt_area(man[i]) >= MIN_AREA

    # success:病灶非平凡 + 三模型都不错(最小 dice 最高)
    succ_sorted = sorted(ids, key=lambda i: min(dd[i].values()), reverse=True)
    success = next((i for i in succ_sorted if has_lesion(i)), succ_sorted[0])
    # failure:病灶非平凡但至少一个模型近空(confident empty),平均 dice 低
    fail_sorted = sorted(ids, key=lambda i: meandice[i])
    failure = next((i for i in fail_sorted
                    if has_lesion(i) and min(dd[i].values()) < 0.1), None)
    if failure is None:
        failure = next((i for i in fail_sorted if has_lesion(i)), fail_sorted[0])
    return [(success, "typical success"), (failure, "failure / empty prediction")], dd


@torch.no_grad()
def load_img_mask(rel, row):
    img = np.array(Image.open(C.ROOT / rel).convert("L"), dtype=np.float32)
    msk = np.array(Image.open(C.ROOT / row["mask"]).convert("L"), dtype=np.float32)
    msk = (msk > 127).astype(np.float32)
    return img, msk


def to_input(img):
    x = img / 255.0
    x = (x - 0.5) / 0.5
    return torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float()


def main():
    man = manifest_map()
    # 预加载每个模型每方向的权重
    fig, axes = plt.subplots(4, 5, figsize=(13, 10.6))
    row_specs = []  # (exp, id, tag)
    for exp in ["expA", "expB"]:
        cases, _ = pick_cases(exp, man)
        for cid, tag in cases:
            row_specs.append((exp, cid, tag))

    # 缓存模型
    net_cache = {}
    for ri, (exp, cid, tag) in enumerate(row_specs):
        stem = EXPS[exp][0]
        row = man[cid]
        img, gt = load_img_mask(cid, row)
        x = to_input(img)
        preds = {}
        for m in MODELS:
            key = (stem, m)
            if key not in net_cache:
                dev = C.device_for(m)
                net = build_model(m).to(dev)
                net.load_state_dict(torch.load(C.CKPT_DIR / f"{stem}_{m}_seed42.pt",
                                               map_location=dev)["state_dict"])
                net.eval()
                net_cache[key] = (net, dev)
            net, dev = net_cache[key]
            with torch.no_grad():
                pr = torch.sigmoid(net(x.to(dev))).cpu().numpy()[0, 0]
            preds[m] = (pr > 0.5).astype(np.float32)

        cols = ["Input", "Ground truth"] + [MLABEL[m] for m in MODELS]
        # col0 input
        ax = axes[ri, 0]; ax.imshow(img, cmap="gray"); ax.set_ylabel(
            f"{EXPS[exp][1].split('Test ')[-1]}\n({tag})", fontsize=9.5)
        # col1 GT
        ax = axes[ri, 1]; ax.imshow(img, cmap="gray")
        ax.imshow(np.ma.masked_where(gt < 0.5, gt), cmap=_solid(GT_COLOR), alpha=0.45)
        # col2-4 preds
        for ci, m in enumerate(MODELS):
            ax = axes[ri, 2 + ci]
            ax.imshow(img, cmap="gray")
            pr = preds[m]
            ax.imshow(np.ma.masked_where(pr < 0.5, pr), cmap=_solid(PR_COLOR), alpha=0.45)
            ax.contour(gt, levels=[0.5], colors=GT_COLOR, linewidths=1.3)
            d = dice_metric(pr, gt)
            ax.text(0.5, -0.02, f"Dice = {d:.2f}", transform=ax.transAxes, ha="center",
                    va="top", fontsize=10, fontweight="bold",
                    color="#111" if d >= 0.5 else PR_COLOR)
        for ci in range(5):
            axes[ri, ci].set_xticks([]); axes[ri, ci].set_yticks([])
            if ri == 0:
                axes[ri, ci].set_title(cols[ci], fontsize=12, fontweight="bold")

    legend = [Line2D([0], [0], color=GT_COLOR, lw=3, label="Ground truth"),
              Line2D([0], [0], color=PR_COLOR, lw=3, label="Prediction")]
    fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Qualitative cross-dataset segmentation: typical successes and failure modes",
                 fontsize=13.5, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0.02, 1, 0.99])
    for ext in ("pdf", "png"):
        fig.savefig(C.ROOT / "figures" / f"fig6_qualitative.{ext}")
    plt.close(fig)
    print("已保存 figures/fig6_qualitative.pdf / .png")
    print("选用病例:")
    for exp, cid, tag in row_specs:
        print(f"  [{exp}] {tag}: {cid}")


def _solid(hexcolor):
    from matplotlib.colors import ListedColormap
    return ListedColormap([hexcolor])


if __name__ == "__main__":
    main()
