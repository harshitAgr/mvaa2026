from __future__ import annotations

import numpy as np
from scipy import ndimage

from postprocess_t3_largestcc import cc_keep_by_area_frac


BOUNDARY_RADIUS_PX = 16.0
MIN_CC_FRAC = 0.10
MIN_TOTAL_FG_FRAC = 0.005


def postprocess(mask: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    cleaned, _, _ = cc_keep_by_area_frac(binary, min_area_frac=MIN_CC_FRAC)
    if cleaned.size and float(cleaned.sum()) / float(cleaned.size) < MIN_TOTAL_FG_FRAC:
        cleaned = np.zeros_like(cleaned, dtype=np.uint8)
    return cleaned.astype(np.uint8, copy=False)


def spatial_refinement(base_mask: np.ndarray, proposal_mask: np.ndarray) -> np.ndarray:
    base = np.asarray(base_mask, dtype=bool)
    proposal = np.asarray(proposal_mask, dtype=bool)
    if base.shape != proposal.shape:
        raise ValueError(f"mask shape mismatch: {base.shape} vs {proposal.shape}")
    if not base.any():
        return np.zeros_like(base, dtype=np.uint8)
    protected_core = ndimage.distance_transform_edt(base) > BOUNDARY_RADIUS_PX
    outward_support = base | (ndimage.distance_transform_edt(~base) <= BOUNDARY_RADIUS_PX)
    candidate = postprocess(protected_core | (proposal & outward_support))
    return candidate if candidate.any() else base.astype(np.uint8)
