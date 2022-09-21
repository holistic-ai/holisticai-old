from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from holisticai.utils.transformers.bias import BMInprocessing as BMImp

from ..commons import _constraints as con
from .algorithm import ExponentiatedGradientAlgorithm


class ExponentiatedGradientReduction(BaseEstimator, ClassifierMixin, BMImp):
    """
    Exponentiated gradient reduction is an in-processing technique that reduces
    fair classification to a sequence of cost-sensitive classification problems,
    returning a randomized classifier with the lowest empirical error subject to
    fair classification constraints.

    References
    ----------
        Agarwal, Alekh, et al. "A reductions approach to fair classification."
        International Conference on Machine Learning. PMLR, 2018.
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
        eps: Optional[float] = 0.01,
        max_iter: Optional[int] = 50,
        nu: Optional[float] = None,
        eta0: Optional[float] = 2.0,
        verbose: Optional[int] = 0,
    ):

        """
        Parameters
        ----------

        estimator: An estimator implementing methods
            ``fit(X, y, sample_weight)`` and ``predict(X)``, where ``X`` is
            the matrix of features, ``y`` is the vector of labels, and
            ``sample_weight`` is a vector of weights; labels ``y`` and
            predictions returned by ``predict(X)`` are either 0 or 1 -- e.g.
            scikit-learn classifiers.

        constraints (str or BaseMoment): If string, keyword
            denoting the :class:`BaseMoment` object
            defining the disparity constraints:
            [
                "DemographicParity",
                "EqualizedOdds",
                "TruePositiveRateParity",
                "FalsePositiveRateParity",
                "ErrorRateParity",
            ]

        eps: Allowed fairness constraint violation; the solution is
            guaranteed to have the error within ``2*best_gap`` of the best
            error under constraint eps; the constraint violation is at most
            ``2*(eps+best_gap)``.

        T: Maximum number of iterations.

        nu: Convergence threshold for the duality gap, corresponding to a
            conservative automatic setting based on the statistical
            uncertainty in measuring classification error.

        eta_mul: Initial setting of the learning rate.

        drop_prot_attr: Boolean flag indicating whether to drop protected
            attributes from training data.

        verbose : int
            If >0, will show progress percentage.
        """

        self.constraints = constraints
        self.eps = eps
        self.max_iter = max_iter
        self.nu = nu
        self.eta0 = eta0
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
        Fit Exponentiated Gradient Reduction

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

        self.model_ = ExponentiatedGradientAlgorithm(
            self.estimator_,
            self.moment_,
            eps=self.eps,
            max_iter=self.max_iter,
            nu=self.nu,
            eta0=self.eta0,
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
            Input matrix.

        Returns
        -------
        numpy.ndarray: Predicted output
        """
        return self.model_.predict(X).ravel()

    def predict_proba(self, X):
        """
        Predict Probabilities

        Description
        ----------
        Probability estimate for the given samples.

        Parameters
        ----------
        X : matrix-like
            Input matrix.

        Returns
        -------
        numpy.ndarray: probability output
        """
        return self.model_._pmf_predict(X)
