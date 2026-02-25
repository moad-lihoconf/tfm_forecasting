"""Priors Python module for data prior configurations."""

from .dataloader import (
    DynSCMPriorDataLoader,
    PriorDataLoader,
    PriorDumpDataLoader,
    TabICLPriorDataLoader,
    TICLPriorDataLoader,
)
from .dynscm import DynSCMConfig, make_get_batch_dynscm
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
    "build_ticl_prior",
    "build_tabpfn_prior",
    "make_get_batch_dynscm",
]

if TabPFNPriorDataLoader is not None:
    __all__.append("TabPFNPriorDataLoader")
