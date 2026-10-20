"""V3-Detail ablation: keep detail recovery and disable auxiliary supervision."""

from config.vaihingen.proposed_method_v3_screen import *


weights_name = "proposed_method_v3_detail_screen"
weights_path = f"model_weights/vaihingen/{weights_name}"
test_weights_name = weights_name
log_name = f"vaihingen/{weights_name}"

loss = ProposedJointLoss(
    num_classes=num_classes,
    ignore_index=ignore_index,
    dice_weight=1.0,
    flow_weight=0.1,
    aux_weight=0.0,
)
