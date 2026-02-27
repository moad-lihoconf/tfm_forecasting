"""Priors Python module for data prior configurations."""

from .audit import audit_prior_dump, integrity_errors
from .dataloader import (
    DynSCMPriorDataLoader,
    PriorDataLoader,
    PriorDumpDataLoader,
    TabICLPriorDataLoader,
    TICLPriorDataLoader,
)
from .dynscm import (
    DynSCMConfig,
    dynscm_family_id_mappings,
    make_get_batch_dynscm,
    sample_dynscm_variant_cfg,
)
from .utils import build_tabpfn_prior, build_ticl_prior

try:
    from .dataloader import TabPFNPriorDataLoader  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - legacy optional integration.
    TabPFNPriorDataLoader = None

__version__ = "0.0.1"
__all__ = [
    "DynSCMConfig",
    "DynSCMPriorDataLoader",
    "PriorDataLoader",
    "PriorDumpDataLoader",
    "TabICLPriorDataLoader",
    "TICLPriorDataLoader",
    "audit_prior_dump",
    "build_ticl_prior",
    "build_tabpfn_prior",
    "integrity_errors",
    "make_get_batch_dynscm",
    "sample_dynscm_variant_cfg",
    "dynscm_family_id_mappings",
]

if TabPFNPriorDataLoader is not None:
    __all__.append("TabPFNPriorDataLoader")
