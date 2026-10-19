# -*- coding: utf-8 -*-
"""
models.py —— 统一构建三类分割模型,并支持 MC Dropout

用 segmentation-models-pytorch(smp)统一接口:
  - U-Net         (CNN 经典基线)
  - U-Net++       (CNN 改进)
  - SegFormer     (Transformer 类;若当前 smp 版本不支持则给出提示)
所有模型:输入单通道灰度图,输出 1 通道 logits(病灶概率的未归一化值)。
额外:在分割头前插入 Dropout2d,用于 MC Dropout 估计不确定性。
"""
import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except Exception:
    HAS_SMP = False

from . import config as C


class SegWithDropout(nn.Module):
    """在骨干模型输出前加一层 Dropout2d,便于 MC Dropout。"""
    def __init__(self, backbone, p=0.2):
        super().__init__()
        self.backbone = backbone
        self.drop = nn.Dropout2d(p=p)

    def forward(self, x):
        feat = self.backbone(x)      # smp 模型直接输出 logits [B,1,H,W]
        return self.drop(feat)


def build_model(name: str):
    """name in {'unet','unetpp','segformer'}"""
    if not HAS_SMP:
        raise RuntimeError("请先安装 segmentation-models-pytorch:pip install segmentation-models-pytorch")
    name = name.lower()
    common = dict(encoder_name=C.ENCODER, encoder_weights="imagenet",
                  in_channels=1, classes=1)
    if name == "unet":
        net = smp.Unet(**common)
    elif name == "unetpp":
        net = smp.UnetPlusPlus(**common)
    elif name == "segformer":
        try:
            # 新版 smp 支持 Segformer;骨干可用 mit_b0(轻量)
            net = smp.Segformer(encoder_name="mit_b0", encoder_weights="imagenet",
                                in_channels=1, classes=1)
        except Exception as e:
            raise RuntimeError(
                "当前 smp 版本可能不支持 Segformer。可升级 smp,或先只跑 unet/unetpp。原始错误:%s" % e)
    else:
        raise ValueError("未知模型:%s" % name)
    return SegWithDropout(net, p=C.DROPOUT_P)


def enable_mc_dropout(model):
    """推理时把所有 Dropout 层设为 train 模式(其余保持 eval)——MC Dropout 的关键。"""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()
