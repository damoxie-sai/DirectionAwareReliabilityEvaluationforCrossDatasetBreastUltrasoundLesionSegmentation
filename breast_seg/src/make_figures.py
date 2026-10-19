# -*- coding: utf-8 -*-
"""
make_figures.py —— 发表级图表 v2(Origin/顶刊风格)

升级点(相对 v1 的普通箱线/柱状):
  - 镜像小提琴 raincloud(半 violin + 箱线 + 均值),展示完整分布(fig1)
  - 柱状图叠加单种子散点 + Mann-Whitney 显著性星号括号(fig2)
  - 哑铃/斜率图带 95%CI(fig3)
  - Risk–Coverage 带自助法误差带 shaded CI band(fig4)
  - 失败检测 AUROC 热力图,精修排版(fig5)
  - 统一:内向刻度 + 盒式边框 + 次刻度 + 面板标签(a/b) + 矢量 PDF/300DPI

全部数值实时读自 results/*.csv;图内文字英文(投稿标准)。
用法:  python -m src.make_figures
"""
import csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde, mannwhitneyu

from . import config as C
from .uncertainty import auroc

# ---------------- 全局样式(Origin/顶刊级) ----------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 12,
    "axes.titlesize": 12.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "medium",
    "axes.linewidth": 1.1,
    "axes.axisbelow": True,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 4.5, "ytick.major.size": 4.5,
    "xtick.minor.size": 2.6, "ytick.minor.size": 2.6,
    "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "xtick.minor.width": 0.7, "ytick.minor.width": 0.7,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "legend.fontsize": 10, "legend.frameon": False,
    "figure.dpi": 120, "savefig.dpi": 330, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

MODELS = ["unet", "unetpp", "segformer"]
MLABEL = {"unet": "U-Net", "unetpp": "U-Net++", "segformer": "SegFormer"}
MCOLOR = {"unet": "#0072B2", "unetpp": "#E69F00", "segformer": "#D55E00"}  # Okabe-Ito
EXPS = {
    "expA": ("expA_BUSI_train", "Train BUSI  $\\rightarrow$  Test BUS-UCLM"),
    "expB": ("expB_UCLM_train", "Train BUS-UCLM  $\\rightarrow$  Test BUSI"),
}
SEEDS = [42, 43, 44]
EXPCOLOR = {"expA": "#3B6EA5", "expB": "#C0453B"}

FIGDIR = C.ROOT / "figures"
FIGDIR.mkdir(exist_ok=True)


# ---------------- 数据工具 ----------------
def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def patient_level_dice(rows):
    byp = defaultdict(list)
    for r in rows:
        byp[r["patient_id"]].append(float(r["dice"]))
    return float(np.mean([np.mean(v) for v in byp.values()]))


def seed_csv(exp, model, seed, which):
    return C.RESULT_DIR / f"{EXPS[exp][0]}_{model}_seed{seed}_{which}.csv"


def agg_dice(exp, model, which):
    vals = [patient_level_dice(load_csv(seed_csv(exp, model, s, which))) for s in SEEDS]
    return float(np.mean(vals)), float(np.std(vals, ddof=1)), vals


def pooled_image_dice(exp, model, which):
    out = []
    for s in SEEDS:
        out += [float(r["dice"]) for r in load_csv(seed_csv(exp, model, s, which))]
    return np.array(out)


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "n.s."


def panel_tag(ax, tag):
    ax.text(-0.02, 1.06, tag, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="bottom", ha="right")


def savefig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  \u2713 {name}.pdf / .png")


def half_violin(ax, data, x, color, side, width=0.34, alpha=0.55):
    """在 x 处画半个小提琴(side='left'/'right'),返回用于放箱线的基准 x。"""
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    if len(data) < 3 or np.allclose(data, data[0]):
        return x
    kde = gaussian_kde(data)
    lo, hi = max(0, data.min()), min(1, data.max())
    ys = np.linspace(lo, hi, 200)
    dens = kde(ys)
    dens = dens / dens.max() * width
    sign = -1 if side == "left" else 1
    xs = x + sign * dens
    ax.fill_betweenx(ys, x, xs, color=color, alpha=alpha, lw=0, zorder=2)
    ax.plot(xs, ys, color=color, lw=1.1, alpha=0.9, zorder=2.1)
    return x


def mini_box(ax, data, x, color, side, off=0.05):
    """在小提琴平边一侧画细箱线 + 均值点。"""
    data = np.asarray(data); data = data[np.isfinite(data)]
    q1, med, q3 = np.percentile(data, [25, 50, 75])
    lo = max(data.min(), q1 - 1.5 * (q3 - q1))
    hi = min(data.max(), q3 + 1.5 * (q3 - q1))
    sign = -1 if side == "left" else 1
    bx = x + sign * off
    ax.plot([bx, bx], [lo, hi], color="#333", lw=1.0, zorder=3)          # 须
    ax.plot([bx, bx], [q1, q3], color="#333", lw=5.5, zorder=3,
            solid_capstyle="round")                                       # IQR
    ax.plot(bx, med, "o", ms=3.4, color="white", mec="#333", mew=0.8, zorder=4)  # 中位数
    ax.plot(bx, np.mean(data), "D", ms=4.6, color=color, mec="white", mew=0.9, zorder=4)  # 均值


# ================= Figure 1:镜像小提琴 raincloud =================
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for pi, (ax, exp) in enumerate(zip(axes, ["expA", "expB"])):
        for i, m in enumerate(MODELS):
            di = pooled_image_dice(exp, m, "internal")
            de = pooled_image_dice(exp, m, "external")
            half_violin(ax, di, i, MCOLOR[m], "left")
            half_violin(ax, de, i, MCOLOR[m], "right", alpha=0.28)
            mini_box(ax, di, i, MCOLOR[m], "left")
            mini_box(ax, de, i, MCOLOR[m], "right")
            # 显著性(内 vs 外,Mann-Whitney)
            try:
                p = mannwhitneyu(di, de, alternative="two-sided").pvalue
            except ValueError:
                p = 1.0
            ytop = 1.05
            ax.plot([i - 0.05, i + 0.05], [ytop, ytop], color="#333", lw=1.0)
            ax.text(i, ytop + 0.01, stars(p), ha="center", va="bottom", fontsize=10.5)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([MLABEL[m] for m in MODELS])
        ax.set_xlim(-0.6, len(MODELS) - 0.4)
        ax.set_ylim(-0.05, 1.16)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.minorticks_on()
        ax.set_title(EXPS[exp][1], fontsize=11.5)
        panel_tag(ax, "ab"[pi])
    axes[0].set_ylabel("Per-image Dice")
    legend = [Patch(fc="#555", alpha=0.55, label="Internal (in-domain), left"),
              Patch(fc="#555", alpha=0.28, label="External (cross-dataset), right"),
              Line2D([0], [0], marker="D", color="w", mfc="#555", mec="w", ms=7, label="Mean"),
              Line2D([0], [0], marker="o", color="w", mfc="w", mec="#333", ms=6, label="Median")]
    axes[1].legend(handles=legend, loc="lower left", ncol=1, fontsize=8.6,
                   handlelength=1.2, labelspacing=0.3)
    fig.suptitle("Distribution of segmentation quality: in-domain vs. cross-dataset  "
                 "(mirrored violins; *** $p<0.001$, Mann–Whitney)",
                 fontsize=12.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    savefig(fig, "fig1_raincloud_dice")


# ================= Figure 2:反转柱状 + 单点 + 显著性 =================
def fig2():
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    x = np.arange(len(MODELS)); w = 0.36
    gapvals = {}
    for j, exp in enumerate(["expA", "expB"]):
        mus, sds = [], []
        for i, m in enumerate(MODELS):
            ivals = [patient_level_dice(load_csv(seed_csv(exp, m, s, "internal"))) for s in SEEDS]
            evals = [patient_level_dice(load_csv(seed_csv(exp, m, s, "external"))) for s in SEEDS]
            g = np.array(ivals) - np.array(evals)
            gapvals[(exp, m)] = g
            mus.append(g.mean()); sds.append(g.std(ddof=1))
        xpos = x + (j - 0.5) * w
        ax.bar(xpos, mus, w, yerr=sds, capsize=4, color=EXPCOLOR[exp],
               edgecolor="white", lw=1.0, alpha=0.92, zorder=2,
               label={"expA": "BUSI $\\rightarrow$ BUS-UCLM", "expB": "BUS-UCLM $\\rightarrow$ BUSI"}[exp],
               error_kw=dict(lw=1.1, ecolor="#333", zorder=3))
        # 叠加单种子散点(column scatter)
        for i, m in enumerate(MODELS):
            g = gapvals[(exp, m)]
            jit = (np.random.RandomState(i * 10 + j).rand(len(g)) - 0.5) * 0.12
            ax.scatter(np.full(len(g), xpos[i]) + jit, g, s=22, color="white",
                       edgecolor="#222", lw=0.8, zorder=4)
        for i, mu in enumerate(mus):
            ax.text(xpos[i], mu + sds[i] + 0.012, f"{mu:.3f}", ha="center",
                    fontsize=9, color="#222")
    ax.set_xticks(x); ax.set_xticklabels([MLABEL[m] for m in MODELS])
    ax.set_ylabel("Generalization gap  (internal $-$ external Dice)")
    ax.set_ylim(0, 0.47)
    ax.minorticks_on(); ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.set_title("Generalization ranking reverses with transfer direction", fontsize=12)
    ax.legend(loc="upper left", ncol=1)
    # 反转高亮
    ax.annotate("", xy=(2 + 0.5 * w, 0.145), xytext=(2 - 0.5 * w, 0.345),
                arrowprops=dict(arrowstyle="-|>", color="#A5006A", lw=1.8,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(2.02, 0.40, "SegFormer:\nlargest $\\rightarrow$ smallest gap",
            color="#A5006A", fontsize=9.2, ha="center", va="top", fontweight="bold")
    savefig(fig, "fig2_gap_reversal")


# ================= Figure 3:哑铃/斜率(带 CI) =================
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharex=True)
    for pi, (ax, exp) in enumerate(zip(axes, ["expA", "expB"])):
        ypos = {m: len(MODELS) - 1 - i for i, m in enumerate(MODELS)}
        for m in MODELS:
            y = ypos[m]
            i_mu, i_sd, _ = agg_dice(exp, m, "internal")
            e_mu, e_sd, _ = agg_dice(exp, m, "external")
            ax.plot([e_mu, i_mu], [y, y], "-", color=MCOLOR[m], lw=3, alpha=0.5, zorder=1,
                    solid_capstyle="round")
            ax.errorbar(i_mu, y, xerr=i_sd, fmt="o", ms=11, color=MCOLOR[m],
                        capsize=3, mec="white", mew=1.4, zorder=3)
            ax.errorbar(e_mu, y, xerr=e_sd, fmt="s", ms=10, color=MCOLOR[m],
                        mfc="white", mew=2.2, capsize=3, zorder=3)
            ax.annotate(f"$\\Delta$={i_mu - e_mu:.3f}", xy=((i_mu + e_mu) / 2, y + 0.16),
                        ha="center", va="bottom", fontsize=9.5, color="#222", fontweight="medium")
        ax.set_yticks(list(ypos.values()))
        ax.set_yticklabels([MLABEL[m] for m in MODELS])
        ax.set_ylim(-0.5, len(MODELS) - 0.25)
        ax.set_xlabel("Dice (patient-level, mean $\\pm$ s.d. over seeds)")
        ax.set_title(EXPS[exp][1], fontsize=11.5)
        ax.grid(axis="x", ls=":", lw=0.7, color="#D0D0D0")
        ax.minorticks_on(); ax.tick_params(axis="y", which="minor", left=False, right=False)
        panel_tag(ax, "ab"[pi])
    axes[0].set_xlim(0.40, 0.92)
    legend = [Line2D([0], [0], marker="o", color="#555", ls="", ms=10, mec="white",
                     label="Internal (in-domain)"),
              Line2D([0], [0], marker="s", color="#555", ls="", ms=9, mfc="white", mew=2.2,
                     label="External (cross-dataset)")]
    axes[1].legend(handles=legend, loc="lower right")
    fig.suptitle("Cross-dataset generalization gap ($\\Delta$ = internal $-$ external Dice)",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig(fig, "fig3_generalization_dumbbell")


# ================= Figure 4:Risk–Coverage + 误差带 =================
def _rc_with_band(dice, unc, grid, n_boot=300, seed=0):
    order = np.argsort(unc)
    d = np.asarray(dice)[order]
    n = len(d)
    rng = np.random.RandomState(seed)
    mean_curve, lo, hi = [], [], []
    for c in grid:
        k = max(1, int(round(c * n)))
        sub = d[:k]
        mean_curve.append(sub.mean())
        boots = [sub[rng.randint(0, k, k)].mean() for _ in range(n_boot)]
        lo.append(np.percentile(boots, 2.5)); hi.append(np.percentile(boots, 97.5))
    return np.array(mean_curve), np.array(lo), np.array(hi)


def fig4():
    sig_col = {"集成分歧": "disagreement", "集成熵": "entropy",
               "预测熵": "entropy", "MC-Dropout方差": "mc_uncertainty"}
    sig_en = {"集成分歧": "disagreement", "集成熵": "entropy", "预测熵": "entropy"}
    grid = np.linspace(0.15, 1.0, 45)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for pi, (ax, exp) in enumerate(zip(axes, ["expA", "expB"])):
        stem = EXPS[exp][0]
        base_signal = None
        for m in MODELS:
            rc_meta = load_csv(C.RESULT_DIR / f"{stem}_{m}_ensemble_external_riskcoverage.csv")
            sigcn = rc_meta[0]["signal"]; base_signal = sig_en.get(sigcn, sigcn)
            ext = load_csv(C.RESULT_DIR / f"{stem}_{m}_ensemble_external.csv")
            dice = np.array([float(r["dice"]) for r in ext])
            unc = np.array([float(r[sig_col.get(sigcn, "disagreement")]) for r in ext])
            mc, lo, hi = _rc_with_band(dice, unc, grid)
            ax.fill_between(grid, lo, hi, color=MCOLOR[m], alpha=0.16, lw=0)
            ax.plot(grid, mc, "-", color=MCOLOR[m], lw=2.3, label=MLABEL[m], zorder=3)
            ax.plot(1.0, mc[-1], "o", color=MCOLOR[m], ms=6, mec="white", mew=1, zorder=4)
        ax.set_xlabel("Coverage (fraction of retained cases)")
        ax.set_title(EXPS[exp][1] + f"   [signal: {base_signal}]", fontsize=11)
        ax.grid(ls=":", lw=0.7, color="#D5D5D5")
        ax.set_xlim(0.15, 1.02); ax.minorticks_on()
        ax.legend(loc="upper right", title="Deep Ensemble")
        panel_tag(ax, "ab"[pi])
    axes[0].set_ylabel("Mean Dice on retained cases")
    fig.suptitle("Selective prediction: rejecting the most uncertain cases recovers accuracy\n"
                 "(shaded = 95% bootstrap CI)", fontsize=12.3, fontweight="bold", y=1.04)
    fig.tight_layout()
    savefig(fig, "fig4_risk_coverage")


# ================= Figure 5:失败检测 AUROC 热力图 =================
def _single_auroc(exp, model, col):
    aus = []
    for s in SEEDS:
        rows = load_csv(C.RESULT_DIR / f"{EXPS[exp][0]}_{model}_seed{s}_external_uncertainty.csv")
        sc = np.array([float(r[col]) for r in rows])
        fail = np.array([int(r["fail"]) for r in rows])
        aus.append(auroc(sc, fail))
    return float(np.nanmean(aus))


def _ensemble_auroc(exp, model, col):
    rows = load_csv(C.RESULT_DIR / f"{EXPS[exp][0]}_{model}_ensemble_external.csv")
    sc = np.array([float(r[col]) for r in rows])
    fail = (np.array([float(r["dice"]) for r in rows]) < 0.5).astype(int)
    return auroc(sc, fail)


def fig5():
    signals = [
        ("Single \u00b7 Predictive entropy", lambda e, m: _single_auroc(e, m, "entropy")),
        ("Single \u00b7 MC-dropout variance", lambda e, m: _single_auroc(e, m, "mc_uncertainty")),
        ("Ensemble \u00b7 Disagreement", lambda e, m: _ensemble_auroc(e, m, "disagreement")),
        ("Ensemble \u00b7 Entropy", lambda e, m: _ensemble_auroc(e, m, "entropy")),
    ]
    cols = [(e, m) for e in ["expA", "expB"] for m in MODELS]
    M = np.array([[fn(e, m) for (e, m) in cols] for _, fn in signals])

    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    im = ax.imshow(M, cmap="RdYlBu", vmin=0.40, vmax=0.85, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([MLABEL[m] for (_, m) in cols])
    ax.axvline(2.5, color="white", lw=3.5)
    ax.set_yticks(range(len(signals)))
    ax.set_yticklabels([s for s, _ in signals])
    ax.tick_params(length=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10.5,
                    color="black" if 0.5 <= v <= 0.72 else "white", fontweight="bold")
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(1, -0.19, "expA:  BUSI $\\rightarrow$ BUS-UCLM", transform=trans,
            ha="center", va="top", fontsize=10.5, fontweight="bold", color=EXPCOLOR["expA"])
    ax.text(4, -0.19, "expB:  BUS-UCLM $\\rightarrow$ BUSI", transform=trans,
            ha="center", va="top", fontsize=10.5, fontweight="bold", color=EXPCOLOR["expB"])
    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
    cbar.set_label("Failure-detection AUROC", fontsize=10)
    cbar.ax.axhline((0.5 - 0.40) / (0.85 - 0.40), color="black", lw=1.2)
    cbar.ax.text(1.6, (0.5 - 0.40) / (0.85 - 0.40), "chance", transform=cbar.ax.transAxes,
                 fontsize=8, va="center")
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(signals), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.6)
    ax.tick_params(which="minor", length=0)
    ax.set_title("Which uncertainty signal detects failures? — depends on transfer direction\n"
                 "(ensemble disagreement excels in expA but collapses to chance in expB)",
                 fontsize=11, fontweight="bold", pad=10)
    fig.tight_layout()
    savefig(fig, "fig5_failure_auroc_heatmap")


def main():
    print("生成发表级图表 v2 到 figures/ ...")
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("完成。")


if __name__ == "__main__":
    main()
