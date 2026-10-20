"""Locked 100-epoch V3-Detail run, initialized anew from backbone weights."""

import torch

from config.vaihingen.proposed_method_v3_detail_screen import *


max_epoch = 100
weights_name = "proposed_method_v3_detail_formal100"
weights_path = f"model_weights/vaihingen/{weights_name}"
test_weights_name = weights_name
log_name = f"vaihingen/{weights_name}"
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)
