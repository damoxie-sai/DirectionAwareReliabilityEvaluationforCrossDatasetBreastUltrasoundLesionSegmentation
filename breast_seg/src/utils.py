# -*- coding: utf-8 -*-
"""
utils.py —— 通用工具:随机种子、患者级划分、Bootstrap 置信区间
"""
import random
import numpy as np
import torch


def set_seed(seed: int):
    """固定随机种子,保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def patient_level_split(patient_ids, ratios=(0.7, 0.15, 0.15), seed=42):
    """
    按"患者"划分 train/val/test —— 防数据泄漏的关键!
    输入:每张图对应的 patient_id 列表(可重复)
    输出:三个集合,里面是"患者id",保证同一患者只出现在一个集合
    """
    uniq = sorted(set(patient_ids))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    train = set(uniq[:n_train])
    val = set(uniq[n_train:n_train + n_val])
    test = set(uniq[n_train + n_val:])
    return train, val, test


def bootstrap_ci_by_patient(values, patient_ids, n_boot=2000, alpha=0.05, seed=0):
    """
    患者级聚类 Bootstrap 求 95% 置信区间。
    values: 每张图的指标(如 Dice) ; patient_ids: 每张图对应患者
    做法:每次重采样"患者",取该患者所有图的均值,再对所有患者求均值。
    """
    values = np.asarray(values, dtype=float)
    pats = np.asarray(patient_ids)
    uniq = np.unique(pats)
    # 预先把每个患者的图索引存好
    idx_by_pat = {p: np.where(pats == p)[0] for p in uniq}
    rng = np.random.RandomState(seed)
    boot_means = []
    for _ in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        per_pat = [np.nanmean(values[idx_by_pat[p]]) for p in sampled]
        boot_means.append(np.nanmean(per_pat))
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    point = np.nanmean([np.nanmean(values[idx_by_pat[p]]) for p in uniq])
    return float(point), float(lo), float(hi)
