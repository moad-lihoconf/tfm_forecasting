"""DynSCM prior components (phase 1-4)."""

from .config import DynSCMConfig
from .graph import DynSCMGraphSample, sample_regime_graphs
from .mechanisms import (
    DynSCMMechanismSample,
    evaluate_contemporaneous_effect,
    evaluate_lagged_mechanism,
    sample_regime_mechanisms,
)
from .stability import (
    DynSCMStabilitySample,
    build_companion_matrix,
    companion_spectral_radius,
    project_after_drift,
    project_to_column_budgets,
    rescale_lag_block_to_spectral_radius,
    sample_stable_coefficients,
)

__all__ = [
    "DynSCMConfig",
    "DynSCMGraphSample",
    "DynSCMMechanismSample",
    "DynSCMStabilitySample",
    "build_companion_matrix",
    "companion_spectral_radius",
    "evaluate_contemporaneous_effect",
    "evaluate_lagged_mechanism",
    "project_after_drift",
    "project_to_column_budgets",
    "rescale_lag_block_to_spectral_radius",
    "sample_regime_mechanisms",
    "sample_regime_graphs",
    "sample_stable_coefficients",
]
