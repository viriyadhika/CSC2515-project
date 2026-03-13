#!/usr/bin/env python3
from __future__ import annotations

"""
Thin re-export layer for paper reproduction scripts.

These symbols are imported from `common.lib` to keep a single source
of truth for shared ECG utilities.
"""

from common.lib import (
    AAMI_MAP,
    EXCLUDED_RECORDS,
    add_em_noise,
    baseline_remove_and_lowpass,
    load_electrode_motion_noise,
)

__all__ = [
    "AAMI_MAP",
    "EXCLUDED_RECORDS",
    "add_em_noise",
    "baseline_remove_and_lowpass",
    "load_electrode_motion_noise",
]

