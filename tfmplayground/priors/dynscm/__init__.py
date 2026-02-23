"""DynSCM prior: dynamic structural causal model sampling and simulation."""

from .config import DynSCMConfig
from .graph import DynSCMGraphSample, sample_regime_graphs
from .mechanisms import (
    DynSCMMechanismSample,
    evaluate_contemporaneous_effect,
    evaluate_lagged_mechanism,
    sample_regime_mechanisms,
)
from .simulate import (
    DynSCMSimulationSample,
    sample_innovations,
    sample_regime_path,
    simulate_dynscm_series,
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
    "DynSCMSimulationSample",
    "DynSCMStabilitySample",
    "build_companion_matrix",
    "companion_spectral_radius",
    "evaluate_contemporaneous_effect",
    "evaluate_lagged_mechanism",
    "project_after_drift",
    "project_to_column_budgets",
    "rescale_lag_block_to_spectral_radius",
    "sample_innovations",
    "sample_regime_path",
    "sample_regime_mechanisms",
    "sample_regime_graphs",
    "sample_stable_coefficients",
    "simulate_dynscm_series",
]
