"""Potsdam 100-epoch V3 ablation without CDFM."""

from config.vaihingen.proposed_ablate_no_cdfm_formal100 import *
from config.potsdam.strict_formal_common import apply_potsdam_protocol

apply_potsdam_protocol(globals(), "proposed_ablate_no_cdfm_potsdam_formal100")
