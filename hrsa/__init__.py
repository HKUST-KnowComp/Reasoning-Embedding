"""
Metrics package for model representation comparison.

This package provides a unified, object-oriented interface for computing
various similarity metrics between model representations.
"""

from hrsa.base import BaseEvaluationMetric, ActivationBundle, PairActivations
from hrsa.config import DatasetConfig, ProbeDatasetConfig
from hrsa.dimension_wise_correlation import DimensionWiseCorrelationMetric
from hrsa.knn_overlap import KNNNeighborOverlapMetric
from hrsa.linear_cka import LinearCKAMetric
from hrsa.orthogonal_procrustes import OrthogonalProcrustesMetric
from hrsa.cross_model_linear_probe import CrossModelLinearProbeMetric

# Import from the file with double dots
# Metric registry for dynamic dispatch
METRIC_REGISTRY = {
    "linear_cka": LinearCKAMetric,
    "procrustes": OrthogonalProcrustesMetric,
    "correlation": DimensionWiseCorrelationMetric,
    "knn_overlap": KNNNeighborOverlapMetric,
    "linear_probe": CrossModelLinearProbeMetric,
}

__all__ = [
    # Base classes
    "BaseEvaluationMetric",
    "ActivationBundle",
    "PairActivations",
    # Config classes
    "DatasetConfig",
    "ProbeDatasetConfig",
    # Metric classes
    "LinearCKAMetric",
    "OrthogonalProcrustesMetric",
    "DimensionWiseCorrelationMetric",
    "KNNNeighborOverlapMetric",
    "CrossModelLinearProbeMetric",
    # Registry
    "METRIC_REGISTRY",
]
