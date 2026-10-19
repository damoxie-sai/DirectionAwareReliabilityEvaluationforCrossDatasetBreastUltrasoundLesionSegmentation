# -*- coding: utf-8 -*-
"""
config.py —— 全局配置(路径、超参数、设备)
零基础说明:所有"可调的东西"都集中放这里,别的脚本从这里 import。
"""
from pathlib import Path
import torch

# ---------- 路径 ----------
ROOT = Path(__file__).resolve().parent.parent          # breast_seg/
DATA_DIR = ROOT / "data"                               # 放原始/整理后的数据
RAW_BUSI = DATA_DIR / "BUSI"                           # 你下载解压后的 BUSI 目录
RAW_UCLM = DATA_DIR / "BUS-UCLM"                       # 你下载解压后的 BUS-UCLM 目录
PROC_DIR = DATA_DIR / "processed"                      # 统一整理后的图/掩膜
MANIFEST = DATA_DIR / "manifest.csv"                   # 统一清单(核心)
SPLIT_DIR = ROOT / "splits"                            # 患者级划分清单
CKPT_DIR = ROOT / "checkpoints"                        # 模型权重
RESULT_DIR = ROOT / "results"                          # 指标/图/表

for d in (DATA_DIR, PROC_DIR, SPLIT_DIR, CKPT_DIR, RESULT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------- 图像 ----------
IMG_SIZE = 256                                         # 统一缩放到 256x256(Mac 够用)

# ---------- 训练超参(先用这套跑通,后面再调) ----------
BATCH_SIZE = 8
EPOCHS = 60
LR = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 3
NUM_WORKERS = 2                                        # Mac 上别太大
SEEDS = [42, 43, 44]                                   # 多种子:防"运气好",也用于 Deep Ensemble

# ---------- 模型 ----------
# 可选:"unet" / "unetpp" / "segformer"
ENCODER = "resnet34"                                   # U-Net 系列的骨干(轻量、Mac 友好)
DROPOUT_P = 0.2                                        # MC Dropout 用

# ---------- 设备(自动选 Mac 的 MPS / 或 CUDA / 或 CPU) ----------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()


def device_for(model_name: str):
    """按模型选设备。
    SegFormer(Transformer)在 Mac 的 MPS 后端反向传播有兼容 bug
    (transpose 后 .view 触发 'view size not compatible' 错误),
    因此 segformer 自动回落到 CPU;U-Net/U-Net++ 仍用 MPS 全速。
    这是环境限制,非本研究方法问题;论文实现细节需如实说明。
    """
    if model_name and model_name.lower() == "segformer" and DEVICE.type == "mps":
        return torch.device("cpu")
    return DEVICE

if __name__ == "__main__":
    print("设备:", DEVICE)
    print("数据目录:", DATA_DIR)
    print("清单文件:", MANIFEST)
