"""DGR-CSMamba internal ablation: row-major raster scanning."""

from config.potsdam.proposed_method_v3_detail_potsdam_formal100 import *
from config.potsdam.proposed_internal_ablation_common import apply_internal_variant

apply_internal_variant(
    globals(),
    "proposed_internal_dgr_raster_potsdam_formal100",
    dgr_scan_mode="raster",
)
