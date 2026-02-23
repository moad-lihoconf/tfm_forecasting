"""DynSCM prior components (phase 1-3)."""

from .config import DynSCMConfig
from .graph import DynSCMGraphSample, sample_regime_graphs
from .stability import (
    DynSCMStabilitySample,
    build_companion_matrix,
    companion_spectral_radius,
    guardc_rescale_lag_block,
    project_after_drift,
    project_to_column_budgets,
    sample_stable_coefficients,
)

__all__ = [
    "DynSCMConfig",
    "DynSCMGraphSample",
    "DynSCMStabilitySample",
    "build_companion_matrix",
    "companion_spectral_radius",
    "guardc_rescale_lag_block",
    "project_after_drift",
    "project_to_column_budgets",
    "sample_regime_graphs",
    "sample_stable_coefficients",
]
