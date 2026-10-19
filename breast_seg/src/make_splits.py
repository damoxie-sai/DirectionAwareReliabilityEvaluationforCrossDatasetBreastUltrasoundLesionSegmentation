# -*- coding: utf-8 -*-
"""
make_splits.py —— 生成"患者级"划分清单(固定下来,保证可复现)

产出两套实验的划分文件到 splits/:
  实验A: BUSI 内部划分(train/val/test) + BUS-UCLM 作为纯外部测试
  实验B: UCLM 内部划分(患者级) + BUSI 作为纯外部测试
运行:
  python -m src.make_splits
"""
import csv
import json
from collections import defaultdict

from . import config as C
from .utils import patient_level_split


def load_manifest():
    rows = []
    with open(C.MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


def build(rows, internal_ds, external_ds, seed=42):
    """internal_ds 做患者级 train/val/test;external_ds 整体作为外部测试。"""
    internal = [r for r in rows if r["dataset"] == internal_ds]
    external = [r for r in rows if r["dataset"] == external_ds]
    pids = [r["patient_id"] for r in internal]
    tr, va, te = patient_level_split(pids, seed=seed)

    def pick(rs, pset):
        return [r["image"] for r in rs_filter(rs, pset)]

    def rs_filter(rs, pset):
        return [r for r in rs if r["patient_id"] in pset]

    split = {
        "internal_dataset": internal_ds,
        "external_dataset": external_ds,
        "seed": seed,
        "train": [r["image"] for r in internal if r["patient_id"] in tr],
        "val":   [r["image"] for r in internal if r["patient_id"] in va],
        "test_internal": [r["image"] for r in internal if r["patient_id"] in te],
        "test_external": [r["image"] for r in external],
        "counts": {
            "train_patients": len(tr), "val_patients": len(va), "test_patients": len(te),
            "train_imgs": sum(r["patient_id"] in tr for r in internal),
            "val_imgs": sum(r["patient_id"] in va for r in internal),
            "test_internal_imgs": sum(r["patient_id"] in te for r in internal),
            "test_external_imgs": len(external),
        },
    }
    return split


def main():
    rows = load_manifest()
    datasets = sorted(set(r["dataset"] for r in rows))
    print("清单里的数据集:", datasets)

    if "BUSI" in datasets and "UCLM" in datasets:
        a = build(rows, "BUSI", "UCLM")
        b = build(rows, "UCLM", "BUSI")
        (C.SPLIT_DIR / "expA_BUSI_train.json").write_text(
            json.dumps(a, ensure_ascii=False, indent=2))
        (C.SPLIT_DIR / "expB_UCLM_train.json").write_text(
            json.dumps(b, ensure_ascii=False, indent=2))
        print("实验A(BUSI训练):", a["counts"])
        print("实验B(UCLM训练):", b["counts"])
    else:
        print("[提醒] 需要同时有 BUSI 和 UCLM 才能做跨数据集实验;先补数据。")


if __name__ == "__main__":
    main()
