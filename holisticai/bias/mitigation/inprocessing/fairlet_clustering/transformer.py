from typing import Optional, Union

import numpy as np
from sklearn.base import BaseEstimator

from holisticai.bias.mitigation.commons.fairlet_clustering.clustering._kcenters import (
    KCenters,
)
from holisticai.bias.mitigation.commons.fairlet_clustering.clustering._kmedoids import (
    KMedoids,
)
from holisticai.bias.mitigation.commons.fairlet_clustering.decomposition._scalable import (
    ScalableFairletDecomposition,
)
from holisticai.bias.mitigation.commons.fairlet_clustering.decomposition._vanilla import (
    VanillaFairletDecomposition,
)
from holisticai.bias.mitigation.inprocessing.fairlet_clustering.algorithm import (
    FairletClusteringAlgorithm,
)
from holisticai.utils.transformers.bias import BMInprocessing as BMImp

DECOMPOSITION_CATALOG = {
    "Scalable": ScalableFairletDecomposition,
    "Vanilla": VanillaFairletDecomposition,
}
CLUSTERING_CATALOG = {"KCenters": KCenters, "KMedoids": KMedoids}


class FairletClustering(BaseEstimator, BMImp):
    """
    Strategies based in Fairlet clustering  Fair Clustering helps you to find clusters with specified proportions
    of different demographic groups pertaining to a sensitive attribute of the dataset
    (group_a and group_b) for any well-known clustering method such as K-means, K-median
    or Spectral clustering (Normalized cut).


    References
    ----------
        Ziko, Imtiaz Masud, et al. "Variational fair clustering." Proceedings of the AAAI
        Conference on Artificial Intelligence. Vol. 35. No. 12. 2021.
    """

    def __init__(
        self,
        n_clusters: Optional[int],
        decomposition: Union["str", "DecompositionMixin"] = "Vanilla",
        clustering_model: Optional["str"] = "KCenters",
        p: Optional[str] = 1,
        q: Optional[float] = 3,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
            n_clusters : int
                The number of clusters to form as well as the number of centroids to generate.

            decomposition : str
                Fairlet decomposition strategy, available: Vanilla, Scalable

            clustering_model : str
                specified lambda parameter

            p : int
                fairlet decomposition parameter for Vanilla and Scalable strategy

            q : int
                fairlet decomposition parameter for Vanilla and Scalable strategy

            seed : int
                Random seed.
        """
        if decomposition in ["Scalable", "Vanilla"]:
            self.decomposition = DECOMPOSITION_CATALOG[decomposition](p=p, q=q)
        self.clustering_model = CLUSTERING_CATALOG[clustering_model](
            n_clusters=n_clusters
        )

        # Constant parameters
        self.algorithm = FairletClusteringAlgorithm(
            decomposition=self.decomposition, clustering_model=self.clustering_model
        )
        self.p = p
        self.q = q
        self.n_clusters = n_clusters
        self.seed = seed

    def fit(
        self,
        X: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
    ):
        """
        Fit the model

        Description
        -----------
        Learn a fair cluster.

        Parameters
        ----------

        X : numpy array
            input matrix

        group_a : numpy array
            binary mask vector

        group_b : numpy array
            binary mask vector

        Returns
        -------
        the same object
        """
        params = self._load_data(X=X, group_a=group_a, group_b=group_b)
        X = params["X"]
        group_a = params["group_a"].astype("int32")
        group_b = params["group_b"].astype("int32")
        np.random.seed(self.seed)
        self.algorithm.fit(X, group_a=group_a, group_b=group_b)
        return self

    @property
    def cluster_centers_(self):
        return self.algorithm.cluster_centers_

    @property
    def labels_(self):
        return self.algorithm.labels

    def predict(self, X: np.ndarray):
        """
        Prediction

        Description
        ----------
        Predict cluster for the given samples.

        Parameters
        ----------
        X : pandas.DataFrame or numpy array
            Test samples.

        Returns
        -------

        numpy.ndarray: Predicted output per sample.
        """
        params = self._load_data(X=X)
        X = params["X"]
        return self.algorithm.predict(X)

    def fit_predict(self, X: np.ndarray, group_a: np.ndarray, group_b: np.ndarray):
        """
        Prediction

        Description
        ----------
        Fit and Predict the cluster for the given samples.

        Parameters
        ----------
        X : pandas.DataFrame or numpy array
            Test samples.

        group_a : numpy array
            binary mask vector

        group_b : numpy array
            binary mask vector


        Returns
        -------

        numpy.ndarray: Predicted cluster per sample.
        """
        self.fit(X, group_a, group_b)
        return self.labels_
