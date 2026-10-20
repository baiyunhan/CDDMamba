"""Leakage-free 1024 px training protocol for ProposedGlobalLocalNetV2."""

import os
import torch
from torch.utils.data import DataLoader

from geoseg.datasets.transform import (
    ColorJitter,
    Compose,
    RandomHorizontalFlip,
    RandomRotate90,
    RandomScale,
    RandomVerticalFlip,
    Resize,
    SmartCropV1,
)
from geoseg.datasets.vaihingen_dataset import CLASSES
from geoseg.datasets.vaihingen_strict import build_strict_vaihingen_datasets
from geoseg.losses.proposed_loss import ProposedJointLoss
from geoseg.models.proposed_method import ProposedGlobalLocalNetV2


max_epoch = 100
ignore_index = len(CLASSES)
train_batch_size = 4
val_batch_size = 8
lr = 6e-4
backbone_lr = 6e-5
weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "proposed_method_v2"
weights_path = f"model_weights/vaihingen/{weights_name}"
test_weights_name = weights_name
log_name = f"vaihingen/{weights_name}"
monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
resume_ckpt_path = None
gpus = "auto"
# Match the effective batch size used by the strongest repository protocol.
accumulate_grad_batches = 2

train_aug = Compose([
    RandomScale(scale_list=[0.75, 1.0, 1.25], mode="value"),
    SmartCropV1(crop_size=1024, max_ratio=0.75, ignore_index=ignore_index, nopad=False),
    RandomHorizontalFlip(prob=0.5),
    RandomVerticalFlip(prob=0.5),
    RandomRotate90(prob=0.5),
    ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
])
val_aug = Compose([Resize(size=(1024, 1024))])

net = ProposedGlobalLocalNetV2(num_classes=num_classes, pretrained=True)
loss = ProposedJointLoss(
    num_classes=num_classes,
    ignore_index=ignore_index,
    dice_weight=1.0,
    flow_weight=0.1,
)
use_aux_loss = False

data_root = os.environ.get("VAIHINGEN_ROOT", "data/vaihingen")
train_dataset, val_dataset, test_dataset = build_strict_vaihingen_datasets(
    data_root=data_root,
    train_transform=train_aug,
    val_transform=val_aug,
    mosaic_ratio=0.25,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    num_workers=4,
    pin_memory=True,
    shuffle=True,
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=val_batch_size,
    num_workers=4,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
)

backbone_params = []
other_params = []
for name, parameter in net.named_parameters():
    if "backbone" in name:
        backbone_params.append(parameter)
    else:
        other_params.append(parameter)

optimizer = torch.optim.AdamW(
    [
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": other_params, "lr": lr, "weight_decay": weight_decay},
    ],
    lr=lr,
    weight_decay=weight_decay,
)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)
