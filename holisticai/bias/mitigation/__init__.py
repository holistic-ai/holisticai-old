"""
The :mod:`holisticai.bias.mitigation` module includes preprocessing, inprocessing and postprocessing bias mitigation algorithms.
"""

# inprocessing algorithm classes
from .inprocessing import (
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
]

import importlib

torch_spec = importlib.util.find_spec("torch")
if torch_spec is not None:
    from .inprocessing import AdversarialDebiasing
__all__ += ["AdversarialDebiasing"]
