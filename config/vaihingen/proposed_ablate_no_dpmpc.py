"""V3-Detail core ablation without DP-MPC."""

from config.vaihingen.proposed_method_v2 import *
from geoseg.models.proposed_method import ProposedCoreAblationNet

max_epoch = 30
weights_name = "proposed_ablate_no_dpmpc"
weights_path = f"model_weights/vaihingen/{weights_name}"
test_weights_name = weights_name
log_name = f"vaihingen/{weights_name}"
net = ProposedCoreAblationNet(num_classes=num_classes, pretrained=True, use_dp_mpc=False)
loss = ProposedJointLoss(num_classes, ignore_index, dice_weight=1.0, flow_weight=0.1)

backbone_params, other_params = [], []
for name, parameter in net.named_parameters():
    (backbone_params if "backbone" in name else other_params).append(parameter)
optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
    {"params": other_params, "lr": lr, "weight_decay": weight_decay},
], lr=lr, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)
