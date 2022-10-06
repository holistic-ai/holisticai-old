# imports
from .exponentiated_gradient.transformer import ExponentiatedGradientReduction
from .grid_search.transformer import GridSearchReduction
from .meta_fair_classifier.transformer import MetaFairClassifier
from .prejudice_remover.transformer import PrejudiceRemover

__all__ = [
    "ExponentiatedGradientReduction",
    "GridSearchReduction",
    "PrejudiceRemover",
    "MetaFairClassifier",
]


import importlib

torch_spec = importlib.util.find_spec("torch")
if torch_spec is not None:
    from .adversarial_debiasing.torch.transformer import AdversarialDebiasing

__all__ += ["AdversarialDebiasing"]
