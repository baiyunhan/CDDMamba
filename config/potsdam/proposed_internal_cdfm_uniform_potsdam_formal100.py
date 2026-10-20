"""CDFM internal ablation: clarity ranking with uniform perturbation."""

from config.potsdam.proposed_method_v3_detail_potsdam_formal100 import *
from config.potsdam.proposed_internal_ablation_common import apply_internal_variant

apply_internal_variant(
    globals(),
    "proposed_internal_cdfm_uniform_potsdam_formal100",
    cdfm_grouping_mode="clarity",
    cdfm_lambda_clear=0.30,
    cdfm_lambda_ambiguous=0.30,
)
