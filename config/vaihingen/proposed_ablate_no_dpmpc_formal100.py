"""Formal 100-epoch V3 core ablation without DP-MPC."""

import torch

from config.vaihingen.proposed_ablate_no_dpmpc import *


max_epoch = 100
weights_name = "proposed_ablate_no_dpmpc_formal100"
weights_path = f"model_weights/vaihingen/{weights_name}"
test_weights_name = weights_name
log_name = f"vaihingen/{weights_name}"
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)
