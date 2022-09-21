from typing import Optional

import numpy as np

from holisticai.utils.transformers.bias import BMPreprocessing as BMPre


class Reweighing(BMPre):
    """
    Reweighing preprocessing weights the examples in each group-label combination to ensure fairness before
    classification.

    References
    ----------
        Kamiran, Faisal, and Toon Calders. "Data preprocessing techniques for classification
        without discrimination." Knowledge and information systems 33.1 (2012): 1-33.
    """

    def __init__(self):
        self.w_p_fav = 1.0
        self.w_p_unfav = 1.0
        self.w_up_fav = 1.0
        self.w_up_unfav = 1.0

    def fit(
        self,
        y_true: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ):
        """
        Fit.

        Description
        ----------
        Access fitted sample_weight param with self.estimator_params["sample_weight"].

        Parameters
        ----------
        y_true : array-like
            Target vector
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)
        sample_weight (optional) : array-like
            Samples weights vector

        Returns
        -------
        Self
        """

        params = self._load_data(
            y_true=y_true, sample_weight=sample_weight, group_a=group_a, group_b=group_b
        )
        y_true = params["y_true"]
        sample_weight = params["sample_weight"]
        group_a = params["group_a"]
        group_b = params["group_b"]

        fav_labels, unfav_labels = self.map_favorable_unfavorable(y_true)
        f = self.compute_frequencies(fav_labels, unfav_labels, sample_weight)

        fav_labels_a, unfav_labels_a = self.map_favorable_unfavorable(y_true, group_a)
        fa = self.compute_frequencies(
            fav_labels_a, unfav_labels_a, sample_weight, group_a
        )

        fav_labels_b, unfav_labels_b = self.map_favorable_unfavorable(y_true, group_b)
        fb = self.compute_frequencies(
            fav_labels_b, unfav_labels_b, sample_weight, group_b
        )

        # reweighing weights
        self.w_a_fav = (f["F"] * fa["N"]) / (f["N"] * fa["F"])
        self.w_a_unfav = (f["U"] * fa["N"]) / (f["N"] * fa["U"])
        self.w_b_fav = (f["F"] * fb["N"]) / (f["N"] * fb["F"])
        self.w_b_unfav = (f["U"] * fb["N"]) / (f["N"] * fb["U"])

        # apply reweighing
        sample_weight[fav_labels_a] = sample_weight[fav_labels_a] * self.w_a_fav
        sample_weight[unfav_labels_a] = sample_weight[unfav_labels_a] * self.w_a_unfav
        sample_weight[fav_labels_b] = sample_weight[fav_labels_b] * self.w_b_fav
        sample_weight[unfav_labels_b] = sample_weight[unfav_labels_b] * self.w_b_unfav

        self.update_estimator_param("sample_weight", sample_weight)
        return self

    def transform(self, X: np.ndarray):
        """passthrough"""
        return X

    def fit_transform(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ):
        """
        Fit transform.

        Description
        ----------
        Access fitted sample_weight param with self.estimator_params["sample_weight"].
        The transform returns the same object inputed.

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
        sample_weight (optional) : array-like
            Samples weights vector

        Returns
        -------
            X
        """
        return self.fit(y_true, group_a, group_b, sample_weight).transform(X)

    def map_favorable_unfavorable(self, y, group=None):
        """
        match favorable and unfavorable labels
        """

        fav_labels = y == 1
        unfav_labels = y == 0

        if group is not None:
            fav_labels = np.logical_and(fav_labels, group == 1)
            unfav_labels = np.logical_and(unfav_labels, group == 1)

        return fav_labels, unfav_labels

    def compute_frequencies(self, fav_labels, unfav_labels, sample_weight, group=None):
        """
        compute frequencies about favorable and unfavorable labels
        """
        n = (
            np.sum(sample_weight, dtype=np.float64)
            if group is None
            else np.sum(sample_weight[group == 1], dtype=np.float64)
        )
        n_fav = np.sum(sample_weight[fav_labels], dtype=np.float64)
        n_unfav = np.sum(sample_weight[unfav_labels], dtype=np.float64)

        return {"N": n, "F": n_fav, "U": n_unfav}
