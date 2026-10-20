"""CDFM internal ablation: fixed random channel grouping."""

from config.potsdam.proposed_method_v3_detail_potsdam_formal100 import *
from config.potsdam.proposed_internal_ablation_common import apply_internal_variant

apply_internal_variant(
    globals(),
    "proposed_internal_cdfm_random_potsdam_formal100",
    cdfm_grouping_mode="random",
    cdfm_grouping_seed=42,
)
