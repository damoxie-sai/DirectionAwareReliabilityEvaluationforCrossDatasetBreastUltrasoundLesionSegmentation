# -*- coding: utf-8 -*-
"""
dataset.py —— PyTorch 数据集:按"图片路径清单"读图+掩膜,做预处理与增强
"""
import csv
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from . import config as C

try:
    import albumentations as A
    HAS_ALB = True
except Exception:
    HAS_ALB = False


def _index_manifest():
    """建立 image路径 -> 行 的索引,方便按 split 里的路径取 mask/patient/label。"""
    idx = {}
    with open(C.MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            idx[r["image"]] = r
    return idx


def get_train_aug():
    if not HAS_ALB:
        return None
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
                           border_mode=0, p=0.5),
        A.RandomBrightnessContrast(p=0.3),   # 模拟不同设备的亮度/对比度差异
    ])


class BreastSegDataset(Dataset):
    def __init__(self, image_list, train=False):
        self.items = image_list                 # split 里的 image 相对路径列表
        self.map = _index_manifest()
        self.train = train
        self.aug = get_train_aug() if train else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel = self.items[i]
        row = self.map[rel]
        img = np.array(Image.open(C.ROOT / rel).convert("L"), dtype=np.float32)
        msk = np.array(Image.open(C.ROOT / row["mask"]).convert("L"), dtype=np.float32)
        msk = (msk > 127).astype(np.float32)

        if self.aug is not None:
            out = self.aug(image=img, mask=msk)
            img, msk = out["image"], out["mask"]

        img = img / 255.0
        img = (img - 0.5) / 0.5                  # 归一化到 [-1,1]
        img = torch.from_numpy(img).unsqueeze(0).float()   # [1,H,W]
        msk = torch.from_numpy(msk).unsqueeze(0).float()   # [1,H,W]
        return {"image": img, "mask": msk, "id": rel,
                "patient_id": row["patient_id"], "label": row["label"]}
