"""
The :mod:`holisticai.bias.mitigation` module includes preprocessing, inprocessing and postprocessing bias mitigation algorithms.
"""

# inprocessing algorithm classes
from .inprocessing import (
    ExponentiatedGradientReduction,
    FairKCenterClustering,
    FairKmedianClustering,
    FairletClustering,
    GridSearchReduction,
    MetaFairClassifier,
    PrejudiceRemover,
    VariationalFairClustering,
)

# postprocessing algorithm classes
from .postprocessing import (
    CalibratedEqualizedOdds,
    EqualizedOdds,
    LPDebiaserBinary,
    LPDebiaserMulticlass,
    MLDebiaser,
    PluginEstimationAndCalibration,
    RejectOptionClassification,
    WassersteinBarycenter,
)

# preprocessing algorithm classes
from .preprocessing import (
    CorrelationRemover,
    FairletClusteringPreprocessing,
    LearningFairRepresentation,
    Reweighing,
)

# all
__all__ = [
    "CorrelationRemover",
    "Reweighing",
    "LearningFairRepresentation",
    "ExponentiatedGradientReduction",
    "GridSearchReduction",
    "CalibratedEqualizedOdds",
    "EqualizedOdds",
    "RejectOptionClassification",
    "WassersteinBarycenter",
    "MLDebiaser",
    "LPDebiaserBinary",
    "LPDebiaserMulticlass",
    "PluginEstimationAndCalibration",
    "PrejudiceRemover",
    "MetaFairClassifier",
    "VariationalFairClustering",
    "FairletClustering",
    "FairletClusteringPreprocessing",
    "FairKCenterClustering",
    "FairKmedianClustering",
]

import importlib

torch_spec = importlib.util.find_spec("torch")
if torch_spec is not None:
    from .inprocessing import AdversarialDebiasing
__all__ += ["AdversarialDebiasing"]
