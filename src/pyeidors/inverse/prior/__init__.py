"""Prior builders for reconstruction-matrix workflows."""

from .laplace import (
    graph_curvature_prior,
    graph_difference_operator,
    graph_laplacian,
    graph_ltl,
    graph_ltl_prior,
)
from .rtr import (
    RtRPrior,
    as_rtr_prior,
    load_rtr_prior_artifact,
    write_rtr_prior_artifact,
)
from .tv_irls import (
    TVIRLSResult,
    solve_tv_irls_batch,
    solve_tv_irls_frame,
    tv_irls_objective,
    tv_irls_prior_from_state,
)

__all__ = [
    "RtRPrior",
    "TVIRLSResult",
    "as_rtr_prior",
    "graph_curvature_prior",
    "graph_difference_operator",
    "graph_laplacian",
    "graph_ltl",
    "graph_ltl_prior",
    "load_rtr_prior_artifact",
    "solve_tv_irls_batch",
    "solve_tv_irls_frame",
    "tv_irls_objective",
    "tv_irls_prior_from_state",
    "write_rtr_prior_artifact",
]
