# -*- coding: utf-8 -*-
"""
prepare_data.py —— 把 BUSI 和 BUS-UCLM 整理成"统一清单" manifest.csv

零基础说明:
不同数据集下载后目录结构不一样。本脚本负责扫描它们、合并多张掩膜、
统一成一张表(manifest.csv),列 = [dataset, image, mask, patient_id, label]。
后续所有脚本都只认这张表,不再关心原始目录长什么样。

★ 使用前请先手动下载并解压:
   - BUSI  -> data/BUSI/        (含 benign/ malignant/ normal/ 子目录)
   - BUS-UCLM -> data/BUS-UCLM/ (含 images/ masks/ 和一个 metadata csv)
运行:
   python -m src.prepare_data
如果某个数据集的目录结构和下面假设不一致,按提示改 PATTERN 即可。
"""
import csv
import re
from pathlib import Path
import numpy as np
from PIL import Image

from . import config as C


def _merge_masks(mask_paths, save_path, size=None):
    """把同一张图的多个掩膜合并成一张(逻辑或),并二值化保存。"""
    merged = None
    for mp in mask_paths:
        m = np.array(Image.open(mp).convert("L"))
        m = (m > 127).astype(np.uint8)
        merged = m if merged is None else np.maximum(merged, m)
    if merged is None:
        return False
    img = Image.fromarray((merged * 255).astype(np.uint8))
    if size:
        img = img.resize((size, size), Image.NEAREST)
    img.save(save_path)
    return True


def _save_image(src, save_path, size=None):
    img = Image.open(src).convert("L")
    if size:
        img = img.resize((size, size), Image.BILINEAR)
    img.save(save_path)


def scan_busi(rows):
    """
    BUSI 结构假设:
      data/BUSI/benign/benign (1).png
      data/BUSI/benign/benign (1)_mask.png   (可能有 _mask_1.png 多张)
    BUSI 没有患者ID -> 每张图当作独立"伪患者"(并在手册中说明其局限)。
    """
    if not C.RAW_BUSI.exists():
        print("[跳过] 未找到 BUSI 目录:", C.RAW_BUSI)
        return
    out_img = C.PROC_DIR / "BUSI" / "images"
    out_msk = C.PROC_DIR / "BUSI" / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_msk.mkdir(parents=True, exist_ok=True)
    n = 0
    for label in ("benign", "malignant", "normal"):
        d = C.RAW_BUSI / label
        if not d.exists():
            continue
        # 找出所有"非 mask"的原图
        imgs = [p for p in d.glob("*.png") if "_mask" not in p.stem]
        for img in imgs:
            stem = img.stem                       # 例如 "benign (1)"
            # ★ 用"前缀筛选"找掩膜,避免 glob 通配符/转义匹配不到带空格括号的文件名。
            #   掩膜名形如 "benign (1)_mask.png" 或多张 "benign (1)_mask_1.png"。
            prefix = stem + "_mask"
            masks = sorted(p for p in d.glob("*.png") if p.stem.startswith(prefix))
            key = f"BUSI_{label}_{stem}".replace(" ", "_")
            img_out = out_img / f"{key}.png"
            msk_out = out_msk / f"{key}.png"
            _save_image(img, img_out, C.IMG_SIZE)
            if masks:
                _merge_masks(masks, msk_out, C.IMG_SIZE)
            else:
                # normal 常无病灶 -> 全黑掩膜
                Image.fromarray(np.zeros((C.IMG_SIZE, C.IMG_SIZE), np.uint8)).save(msk_out)
            rows.append({
                "dataset": "BUSI",
                "image": str(img_out.relative_to(C.ROOT)),
                "mask": str(msk_out.relative_to(C.ROOT)),
                "patient_id": f"BUSI_{key}",     # ★ 伪患者:BUSI 无真实患者ID
                "label": label,
            })
            n += 1
    print(f"[BUSI] 整理 {n} 张")


def _rgb_mask_to_binary(mask_path, save_path, size=None):
    """
    BUS-UCLM 掩膜是彩色 RGB:红(255,0,0)=恶性病灶, 绿(0,255,0)=良性病灶, 黑=背景。
    本函数把"红∪绿(任何有病灶的像素)"设为前景=1(白255),其余=0(黑),存为二值掩膜。
    返回:(是否有病灶, 病灶颜色标签)  颜色标签∈{'malignant','benign','both','none'}
    """
    a = np.array(Image.open(mask_path).convert("RGB"))
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    red = (r > 127) & (g < 80) & (b < 80)     # 恶性
    green = (g > 127) & (r < 80) & (b < 80)   # 良性
    fg = red | green
    binary = (fg.astype(np.uint8)) * 255
    img = Image.fromarray(binary)
    if size:
        img = img.resize((size, size), Image.NEAREST)
    img.save(save_path)
    has_red, has_green = bool(red.any()), bool(green.any())
    if has_red and has_green:
        color_label = "both"
    elif has_red:
        color_label = "malignant"
    elif has_green:
        color_label = "benign"
    else:
        color_label = "none"
    return fg.any(), color_label


def scan_uclm(rows):
    """
    BUS-UCLM 真实结构(核对自官方 Mendeley Data + Scientific Data 2025;2025-07 核实):
      data/BUS-UCLM/images/XXXX_YYY.png   原图(PNG)
      data/BUS-UCLM/masks/XXXX_YYY.png    彩色RGB掩膜(红=恶性/绿=良性/黑=背景)
      data/BUS-UCLM/INFO.csv              分号(;)分隔,列: Image;Resolution;Label;Doppler;Marks;Combined
    要点:
      · 患者ID = 文件名前4个字母(XXXX);683张 / 约38名患者
      · 掩膜需从彩色转二值(见 _rgb_mask_to_binary)
      · 良恶性标签优先取 INFO.csv 的 Label 列;缺失时用掩膜颜色兜底
    """
    if not C.RAW_UCLM.exists():
        print("[跳过] 未找到 BUS-UCLM 目录:", C.RAW_UCLM)
        return
    out_img = C.PROC_DIR / "UCLM" / "images"
    out_msk = C.PROC_DIR / "UCLM" / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_msk.mkdir(parents=True, exist_ok=True)

    # 读 INFO.csv(分号分隔);建立 文件名stem -> Label 的映射
    label_map = {}
    info_path = C.RAW_UCLM / "INFO.csv"
    if info_path.exists():
        with open(info_path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f, delimiter=";"):
                low = {k.lower().strip(): (v or "").strip() for k, v in r.items()}
                name = low.get("image", "")
                lab = low.get("label", "").lower() or "unknown"
                if name:
                    label_map[Path(name).stem] = lab
        print(f"[UCLM] 读到 INFO.csv,{len(label_map)} 条标签")
    else:
        print("[UCLM][提醒] 未找到 INFO.csv,良恶性标签将改用掩膜颜色推断。")

    img_dir = C.RAW_UCLM / "images"
    msk_dir = C.RAW_UCLM / "masks"
    imgs = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    n = 0
    patients = set()
    for img in imgs:
        stem = img.stem                       # 例如 ALWI_000
        pid = stem.split("_")[0]              # ★ 患者ID = 前4字母(下划线前)
        patients.add(pid)
        key = f"UCLM_{stem}".replace(" ", "_")
        img_out = out_img / f"{key}.png"
        msk_out = out_msk / f"{key}.png"
        _save_image(img, img_out, C.IMG_SIZE)

        # 掩膜:彩色RGB -> 二值
        mpath = msk_dir / f"{stem}.png"
        color_label = "none"
        if mpath.exists():
            _, color_label = _rgb_mask_to_binary(mpath, msk_out, C.IMG_SIZE)
        else:
            Image.fromarray(np.zeros((C.IMG_SIZE, C.IMG_SIZE), np.uint8)).save(msk_out)

        # 标签:优先 INFO.csv 的 Label;缺失时用掩膜颜色兜底
        label = label_map.get(stem)
        if not label or label == "unknown":
            label = {"malignant": "malignant", "benign": "benign",
                     "both": "malignant", "none": "normal"}.get(color_label, "unknown")

        rows.append({
            "dataset": "UCLM",
            "image": str(img_out.relative_to(C.ROOT)),
            "mask": str(msk_out.relative_to(C.ROOT)),
            "patient_id": f"UCLM_{pid}",      # ★ 患者级(38人),不是每图一个
            "label": label,
        })
        n += 1
    print(f"[UCLM] 整理 {n} 张,患者 {len(patients)} 人(应≈38)")


def main():
    rows = []
    scan_busi(rows)
    scan_uclm(rows)
    if not rows:
        print("没有任何数据被整理。请确认已把 BUSI / BUS-UCLM 解压到 data/ 下。")
        return
    with open(C.MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "image", "mask", "patient_id", "label"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n清单已写出: {C.MANIFEST}  共 {len(rows)} 行")
    # 简单统计
    from collections import Counter
    print("按数据集:", Counter(r["dataset"] for r in rows))
    print("按标签:", Counter(r["label"] for r in rows))
    print("患者数:",
          {ds: len(set(r["patient_id"] for r in rows if r["dataset"] == ds))
           for ds in set(r["dataset"] for r in rows)})


if __name__ == "__main__":
    main()
