"""Registries for legacy callables and typed baseline adapters."""

from methods import diagGMM as diag_gmm_method
from methods import MuDo_nll as nll_method
from methods import VaDE_linear as vade_linear_method
from methods import SP as sp_method
from methods import umni as umni_method
from methods.base import LegacyBaselineAdapter
from methods.bang import BANGBaseline
from methods.dcd import DCDBaseline
from methods.lingam import LiNGAMBaseline
from methods.rcd import RCDBaseline


METHOD_REGISTRY = {
    "MuDo-nll": nll_method.estimate,
    "SP": sp_method.estimate,
    "UMNI": umni_method.estimate,
    "VaDE-linear": vade_linear_method.estimate,
    "diagGMM": diag_gmm_method.estimate,
}

BASELINE_REGISTRY = {
    name: LegacyBaselineAdapter(name=name, estimator=estimator)
    for name, estimator in METHOD_REGISTRY.items()
}
BASELINE_REGISTRY["LiNGAM"] = LiNGAMBaseline()
BASELINE_REGISTRY["BANG"] = BANGBaseline()
BASELINE_REGISTRY["DCD"] = DCDBaseline()
BASELINE_REGISTRY["RCD"] = RCDBaseline()
