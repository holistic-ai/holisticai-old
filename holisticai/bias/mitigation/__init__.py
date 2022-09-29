"""
The :mod:`holisticai.bias.mitigation` module includes preprocessing, inprocessing and postprocessing bias mitigation algorithms.
"""

# inprocessing algorithm classes
from .inprocessing import (
    AdversarialDebiasing,
    ExponentiatedGradientReduction,
    GridSearchReduction,
    MetaFairClassifier,
    PrejudiceRemover,
)

# postprocessing algorithm classes
from .postprocessing import (
    CalibratedEqualizedOdds,
    EqualizedOdds,
    RejectOptionClassification,
)

# preprocessing algorithm classes
from .preprocessing import LearningFairRepresentation, Reweighing

# all
__all__ = [
    "Reweighing",
    "LearningFairRepresentation",
    "ExponentiatedGradientReduction",
    "GridSearchReduction",
    "CalibratedEqualizedOdds",
    "EqualizedOdds",
    "RejectOptionClassification",
    "PrejudiceRemover",
    "MetaFairClassifier",
    "AdversarialDebiasing",
]
