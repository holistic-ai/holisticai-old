from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from holisticai.utils.transformers.bias import BMInprocessing as BMImp

from ..commons import _constraints as con
from ._grid_generator import GridGenerator
from .algorithm import GridSearchAlgorithm


class GridSearchReduction(BaseEstimator, ClassifierMixin, BMImp):
    """
    Grid search technique can be used for fair classification or fair regression.
    - For classification it reduces fair classification to a sequence of cost-sensitive classification problems,
    returning the deterministic classifier with the lowest empirical error subject to fair classification constraints among the
    candidates searched.
    - For regression it uses the same priniciple to return a deterministic regressor with the lowest empirical error subject to the
    constraint of bounded group loss.

    References:
        Agarwal, Alekh, et al. "A reductions approach to fair classification."
        International Conference on Machine Learning. PMLR, 2018.

        Agarwal, Alekh, Miroslav Dudík, and Zhiwei Steven Wu.
        "Fair regression: Quantitative definitions and reduction-based algorithms."
        International Conference on Machine Learning. PMLR, 2019.
    """

    CONSTRAINTS = [
        "DemographicParity",
        "EqualizedOdds",
        "TruePositiveRateParity",
        "FalsePositiveRateParity",
        "ErrorRateParity",
    ]

    def __init__(
        self,
        constraints: str = "EqualizedOdds",
        constraint_weight: Optional[float] = 0.5,
        grid_size: Optional[int] = 10,
        grid_limit: Optional[float] = 2.0,
        verbose: Optional[int] = 0.0,
    ):
        """
        Init Grid Search Reduction Transformer

        Parameters
        ----------

        constraints : string
            The disparity constraints expressed as string:
                - "DemographicParity",
                - "EqualizedOdds",
                - "TruePositiveRateParity",
                - "FalsePositiveRateParity",
                - "ErrorRateParity"

        constraint_weight : float
            Specifies the relative weight put on the constraint violation when selecting the
            best model. The weight placed on the error rate will be :code:`1-constraint_weight`

        grid_size : int
            The number of Lagrange multipliers to generate in the grid

        grid_limit : float
            The largest Lagrange multiplier to generate. The grid will contain
            values distributed between :code:`-grid_limit` and :code:`grid_limit`
            by default

        verbose : int
            If >0, will show progress percentage.

        """
        self.constraints = constraints
        self.constraint_weight = constraint_weight
        self.grid_size = grid_size
        self.grid_limit = grid_limit
        self.verbose = verbose

    def transform_estimator(self, estimator):
        self.estimator = estimator
        return self

    def fit(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
    ):
        """

        Fit model using Grid Search Reduction.

        Parameters
        ----------
        X : matrix-like
            Input matrix
        y_true : array-like
            Target vector
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)

        Returns
        -------
            Self
        """

        params = self._load_data(X=X, y_true=y_true, group_a=group_a, group_b=group_b)
        group_a = params["group_a"]
        group_b = params["group_b"]
        X = params["X"]
        y_true = params["y_true"]

        # Support onlyt binary classification with labels 1 and 0
        assert set(np.unique(y_true)) == {
            0,
            1,
        }, "Grid Search only supports binary classification with labels 0 and 1"

        sensitive_features = np.stack([group_a, group_b], axis=1)

        self.estimator_ = clone(self.estimator)

        moments = {
            "DemographicParity": con.DemographicParity,
            "EqualizedOdds": con.EqualizedOdds,
            "TruePositiveRateParity": con.TruePositiveRateParity,
            "FalsePositiveRateParity": con.FalsePositiveRateParity,
            "ErrorRateParity": con.ErrorRateParity,
        }
        self.moment_ = moments[self.constraints]()

        self.generator_ = GridGenerator(
            grid_size=self.grid_size, grid_limit=self.grid_limit
        )

        self.model_ = GridSearchAlgorithm(
            constraints=self.moment_,
            estimator=self.estimator_,
            generator=self.generator_,
            constraint_weight=self.constraint_weight,
            verbose=self.verbose,
        )

        self.model_.fit(X, y_true, sensitive_features=sensitive_features)

        return self

    def predict(self, X):
        """
        Prediction

        Description
        ----------
        Predict output for the given samples.

        Parameters
        ----------
        X : matrix-like
            Input Matrix

        Returns
        -------
        numpy.ndarray: Predicted output
        """
        return self.model_.predict(X)

    def predict_proba(self, X):
        """
        Probability Prediction

        Description
        ----------
        Probability estimate for the given samples.

        Parameters
        ----------
        X : matrix-like
            Input Matrix

        Returns
        -------
        numpy.ndarray
            probability output
        """
        return self.model_.predict_proba(X)
