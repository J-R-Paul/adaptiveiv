from importlib.metadata import PackageNotFoundError, version

from .estimators import fit_fully_interacted_2sls, fit_pooled_2sls
from .formula import parse_iv_formula
from .model import AdaptiveIV, InferenceSupport
from .results import AdaptiveIVResults
from .simulation import (
    paper_group_strengths,
    simulate_paper_dgp,
    simulate_paper_section4_dgp,
)
from .validation import (
    estimate_methods_once,
    summarize_simulation_results,
    validate_simulation_summary,
)

try:
    __version__ = version("adaptiveiv")
except PackageNotFoundError:  # pragma: no cover - only for unusual source-tree use.
    __version__ = "0+unknown"

__all__ = [
    "AdaptiveIV",
    "AdaptiveIVResults",
    "InferenceSupport",
    "__version__",
    "fit_fully_interacted_2sls",
    "fit_pooled_2sls",
    "paper_group_strengths",
    "parse_iv_formula",
    "simulate_paper_dgp",
    "simulate_paper_section4_dgp",
    "estimate_methods_once",
    "summarize_simulation_results",
    "validate_simulation_summary",
]
