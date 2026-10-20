"""Potsdam 100-epoch V3 ablation without DP-MPC."""

from config.vaihingen.proposed_ablate_no_dpmpc_formal100 import *
from config.potsdam.strict_formal_common import apply_potsdam_protocol

apply_potsdam_protocol(globals(), "proposed_ablate_no_dpmpc_potsdam_formal100")
