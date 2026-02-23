"""DynSCM prior: dynamic structural causal model sampling and simulation."""

from .config import DynSCMConfig
from .features import (
    build_forecasting_table,
    extract_feature_block,
    sample_origins_and_horizons,
)
from .graph import DynSCMGraphSample, sample_regime_graphs
from .mechanisms import (
    DynSCMMechanismSample,
    evaluate_contemporaneous_effect,
    evaluate_lagged_mechanism,
    sample_regime_mechanisms,
)
from .missingness import sample_observation_mask
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
    "build_forecasting_table",
    "build_companion_matrix",
    "companion_spectral_radius",
    "evaluate_contemporaneous_effect",
    "evaluate_lagged_mechanism",
    "extract_feature_block",
    "project_after_drift",
    "project_to_column_budgets",
    "rescale_lag_block_to_spectral_radius",
    "sample_observation_mask",
    "sample_innovations",
    "sample_origins_and_horizons",
    "sample_regime_path",
    "sample_regime_mechanisms",
    "sample_regime_graphs",
    "sample_stable_coefficients",
    "simulate_dynscm_series",
]
