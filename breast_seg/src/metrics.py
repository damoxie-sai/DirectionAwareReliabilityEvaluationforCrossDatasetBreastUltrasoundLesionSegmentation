# -*- coding: utf-8 -*-
"""
metrics.py —— 分割评价指标:Dice / IoU / HD95 / ASD,以及"空掩膜"边界情况处理
"""
import numpy as np

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


def _bin(x, thr=0.5):
    return (np.asarray(x) > thr).astype(np.uint8)


def dice(pred, gt, thr=0.5, eps=1e-7):
    """
    空掩膜约定:
      两者都空 -> 1.0(都判"无病灶",正确)
      一空一非空 -> 0.0
    """
    p, g = _bin(pred, thr), _bin(gt, 0.5)
    ps, gs = p.sum(), g.sum()
    if ps == 0 and gs == 0:
        return 1.0
    inter = np.logical_and(p, g).sum()
    return float(2 * inter / (ps + gs + eps))


def iou(pred, gt, thr=0.5, eps=1e-7):
    p, g = _bin(pred, thr), _bin(gt, 0.5)
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(inter / (union + eps))


def precision_recall(pred, gt, thr=0.5, eps=1e-7):
    p, g = _bin(pred, thr), _bin(gt, 0.5)
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, 1 - g).sum()
    fn = np.logical_and(1 - p, g).sum()
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    return float(prec), float(rec)


def _surface_distances(a, b):
    """a,b 为二值图;返回 a 表面到 b 的距离数组。"""
    a, b = a.astype(bool), b.astype(bool)
    a_surf = a ^ binary_erosion(a)
    dt_b = distance_transform_edt(~b)
    return dt_b[a_surf]


def hd95_asd(pred, gt, thr=0.5):
    """
    返回 (HD95, ASD)。任一为空 -> 返回 (nan, nan)(需在汇总时单独统计这类失败)。
    需要 scipy。
    """
    if not HAS_SCIPY:
        return float("nan"), float("nan")
    p, g = _bin(pred, thr).astype(bool), _bin(gt, 0.5).astype(bool)
    if p.sum() == 0 or g.sum() == 0:
        return float("nan"), float("nan")
    d_pg = _surface_distances(p, g)
    d_gp = _surface_distances(g, p)
    all_d = np.concatenate([d_pg, d_gp])
    hd95 = float(np.percentile(all_d, 95))
    asd = float(all_d.mean())
    return hd95, asd


def all_metrics(pred, gt, thr=0.5):
    d = dice(pred, gt, thr)
    j = iou(pred, gt, thr)
    pr, rc = precision_recall(pred, gt, thr)
    hd, asd = hd95_asd(pred, gt, thr)
    return {"dice": d, "iou": j, "precision": pr, "recall": rc,
            "hd95": hd, "asd": asd}
