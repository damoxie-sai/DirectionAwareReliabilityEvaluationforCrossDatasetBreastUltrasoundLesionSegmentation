# -*- coding: utf-8 -*-
"""
make_tables.py —— 生成论文三张主表(数值实时读自 results/*.csv)

  Table 1  泛化主表:内/外 Dice(患者级均值 [95%CI])+ 泛化差距 [95%CI]
  Table 2  失败画像:外部失败率(Dice<0.5)、Dice=0 例数、HD95 可算比例
  Table 3  失败检测:AUROC [95%CI] × 信号(单·熵 / 单·MC方差 / 集成·分歧 / 集成·熵)

输出:manuscript/tables/table{1,2,3}.csv  + 屏幕 markdown 预览。
所有 CI = 患者级聚类 bootstrap(n=2000)。
用法:python -m src.make_tables
"""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np

from . import config as C
from .uncertainty import auroc, auroc_ci

MODELS = ["unet", "unetpp", "segformer"]
MLABEL = {"unet": "U-Net", "unetpp": "U-Net++", "segformer": "SegFormer"}
EXPS = {"expA": ("expA_BUSI_train", "BUSI\u2192BUS-UCLM"),
        "expB": ("expB_UCLM_train", "BUS-UCLM\u2192BUSI")}
SEEDS = [42, 43, 44]
NBOOT = 2000

OUT = C.ROOT.parent / "manuscript" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def load_csv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rows_seed(exp, model, seed, which):
    return load_csv(C.RESULT_DIR / f"{EXPS[exp][0]}_{model}_seed{seed}_{which}.csv")


def per_patient_seedavg(exp, model, which):
    """每患者 Dice = 先跨 3 种子对同一图求平均,再按患者聚合。返回 {patient: [dice...]}。"""
    # id -> 每种子 dice
    acc = defaultdict(list)
    pat_of = {}
    for s in SEEDS:
        for r in rows_seed(exp, model, s, which):
            acc[r["id"]].append(float(r["dice"]))
            pat_of[r["id"]] = r["patient_id"]
    byp = defaultdict(list)
    for iid, ds in acc.items():
        byp[pat_of[iid]].append(float(np.mean(ds)))
    return byp


def boot_mean_ci(byp, n=NBOOT, seed=0):
    """患者级 bootstrap:每患者取其内部均值,重采样患者。返回 (point, lo, hi)。"""
    pats = list(byp.keys())
    per_pat = np.array([np.mean(byp[p]) for p in pats])
    rng = np.random.RandomState(seed)
    point = float(per_pat.mean())
    boots = [per_pat[rng.randint(0, len(per_pat), len(per_pat))].mean() for _ in range(n)]
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def boot_gap_ci(byp_int, byp_ext, n=NBOOT, seed=1):
    pi = np.array([np.mean(v) for v in byp_int.values()])
    pe = np.array([np.mean(v) for v in byp_ext.values()])
    rng = np.random.RandomState(seed)
    point = float(pi.mean() - pe.mean())
    boots = [pi[rng.randint(0, len(pi), len(pi))].mean() - pe[rng.randint(0, len(pe), len(pe))].mean()
             for _ in range(n)]
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def seed_sd(exp, model, which):
    def patlevel(rows):
        byp = defaultdict(list)
        for r in rows:
            byp[r["patient_id"]].append(float(r["dice"]))
        return np.mean([np.mean(v) for v in byp.values()])
    vals = [patlevel(rows_seed(exp, model, s, which)) for s in SEEDS]
    return float(np.std(vals, ddof=1))


def fmt(p, lo, hi):
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


# ---------------- Table 1 ----------------
def table1():
    hdr = ["Experiment", "Model", "Internal Dice [95% CI]", "External Dice [95% CI]",
           "Generalization gap [95% CI]", "seed s.d. (int/ext)"]
    rows = []
    for exp in ["expA", "expB"]:
        for m in MODELS:
            bi = per_patient_seedavg(exp, m, "internal")
            be = per_patient_seedavg(exp, m, "external")
            i = boot_mean_ci(bi); e = boot_mean_ci(be); g = boot_gap_ci(bi, be)
            rows.append([EXPS[exp][1], MLABEL[m], fmt(*i), fmt(*e), fmt(*g),
                         f"{seed_sd(exp, m, 'internal'):.3f}/{seed_sd(exp, m, 'external'):.3f}"])
    return hdr, rows


# ---------------- Table 2 ----------------
def table2():
    hdr = ["Experiment", "Model", "N (ext)", "Failure rate Dice<0.5 (mean\u00b1sd)",
           "Empty/zero-Dice (seed42)", "HD95 computable (seed42)"]
    rows = []
    for exp in ["expA", "expB"]:
        for m in MODELS:
            frates = []
            for s in SEEDS:
                d = np.array([float(r["dice"]) for r in rows_seed(exp, m, s, "external")])
                frates.append(np.mean(d < 0.5))
            r42 = rows_seed(exp, m, 42, "external")
            n = len(r42)
            zero = sum(1 for r in r42 if float(r["dice"]) == 0.0)
            hdok = sum(1 for r in r42 if r["hd95"] not in ("", "nan"))
            rows.append([EXPS[exp][1], MLABEL[m], n,
                         f"{np.mean(frates)*100:.1f}\u00b1{np.std(frates, ddof=1)*100:.1f}%",
                         f"{zero} ({zero/n*100:.0f}%)", f"{hdok}/{n} ({hdok/n*100:.0f}%)"])
    return hdr, rows


# ---------------- Table 3 ----------------
def _id2pat(exp, model):
    return {r["id"]: r["patient_id"] for r in rows_seed(exp, model, 42, "external")}


def _auroc_single(exp, model, col):
    rows = load_csv(C.RESULT_DIR / f"{EXPS[exp][0]}_{model}_seed42_external_uncertainty.csv")
    id2p = _id2pat(exp, model)
    sc = np.array([float(r[col]) for r in rows])
    fail = np.array([int(r["fail"]) for r in rows])
    pids = np.array([id2p[r["id"]] for r in rows])
    return auroc_ci(sc, fail, pids, n_boot=NBOOT)


def _auroc_ensemble(exp, model, col):
    rows = load_csv(C.RESULT_DIR / f"{EXPS[exp][0]}_{model}_ensemble_external.csv")
    sc = np.array([float(r[col]) for r in rows])
    fail = (np.array([float(r["dice"]) for r in rows]) < 0.5).astype(int)
    pids = np.array([r["patient_id"] for r in rows])
    return auroc_ci(sc, fail, pids, n_boot=NBOOT)


def table3():
    hdr = ["Experiment", "Model", "Single\u00b7Entropy", "Single\u00b7MC-variance",
           "Ensemble\u00b7Disagreement", "Ensemble\u00b7Entropy"]
    rows = []
    for exp in ["expA", "expB"]:
        for m in MODELS:
            se = _auroc_single(exp, m, "entropy")
            sm = _auroc_single(exp, m, "mc_uncertainty")
            ed = _auroc_ensemble(exp, m, "disagreement")
            ee = _auroc_ensemble(exp, m, "entropy")
            rows.append([EXPS[exp][1], MLABEL[m], fmt(*se), fmt(*sm), fmt(*ed), fmt(*ee)])
    return hdr, rows


def save_and_print(name, hdr, rows, caption):
    with open(OUT / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
    print(f"\n### {name}  ({caption})")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")


def main():
    print(f"生成论文主表(患者级 bootstrap 95%CI, n={NBOOT})...")
    save_and_print("table1_generalization", *table1(),
                   caption="Cross-dataset generalization (patient-level Dice)")
    save_and_print("table2_failure", *table2(),
                   caption="External failure characterization")
    save_and_print("table3_detection", *table3(),
                   caption="Failure-detection AUROC by signal")
    print(f"\n已保存到 {OUT}/")


if __name__ == "__main__":
    main()
