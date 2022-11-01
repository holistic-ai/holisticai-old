from sklearn.metrics.pairwise import (
    pairwise_distances,
    pairwise_distances_argmin,
)
import numpy as np
from holisticai.bias.mitigation.commons.fairlet_clustering._utils import distance

class FairletClusteringAlgorithm:
    def __init__(self, decomposition, clustering_model):
        self.decomposition = decomposition
        self.clustering_model = clustering_model
        
    def fit(self, X, group_a, group_b, decompose=None):
        if decompose is not None:
            fairlets, fairlet_centers, fairlet_costs = decompose
        else:
            fairlets, fairlet_centers, fairlet_costs = self.decomposition.fit_transform(X, group_a, group_b)
        
        self.clustering_model.fit([X[i] for i in fairlet_centers])
        mapping = self.clustering_model.assign()

        self.labels = np.zeros(len(X), dtype='int32')
        for fairlet_id, final_cluster in mapping:
            self.labels[fairlets[fairlet_id]] = int(fairlet_centers[final_cluster])

        self.centers = [fairlet_centers[i] for i in self.clustering_model.centers]
        self.cluster_centers_ = np.array([X[c] for c in self.centers])
        self.cost = max([min([distance(X[j], i) for j in self.centers]) for i in X])
        self.X  = X
        
    def predict(self, X):
        """
        Assigning every point in the dataset to the closest center.

        Returns:
        mapping (list) : tuples of the form (point, center)
        """
        fairlets_midxs = pairwise_distances_argmin(X, Y=self.X)
        return self.labels[fairlets_midxs]
    
"""

from .algorithm_utils._fair_tree_utils import build_quadtree, FairletDescomposition
from .algorithm_utils._fast_k_medoids import KMedoids
from .algorithm_utils._logger import MFLogger
import numpy as np
from sklearn.metrics.pairwise import (
    pairwise_distances,
    pairwise_distances_argmin,
)

def euclidiean_distance_func(data1, data2):
    return np.linalg.norm(data1 - data2)
        
class ScalableFairClusteringAlgorithm:
    def __init__(self, K, p, q, distance_func=euclidiean_distance_func, verbose=False):
        self.K = K
        self.p = p
        self.q = q
        self.epsilon = 0.0001
        self.fairlets = []
        self.fairlet_centers = []
        self.faird= FairletDescomposition()
        self.distance_func = distance_func
        self.verbose = verbose
        self.logger = MFLogger(total_iterations=6, verbose=verbose)
 
    def fit(self, X, p_attr, seed):
        np.random.seed(seed)
        
        self.logger.update(step=1, status="Constructing tree...")
        root = build_quadtree(X)
        
        self.logger.update(step=2, status="Doing fair clustering...")
        self.tree_fairlet_cost = self.faird.tree_fairlet_decomposition(self.p, self.q, root, X, p_attr)
        
        self.logger.update(step=3, status=f"Fairlet decomposition cost: {self.tree_fairlet_cost:.2f}")

        self.logger.update(step=4, status="Doing k-median clustering on fairlet centers...")
        fairlet_center_idx = [X[index] for index in self.faird.fairlet_centers]
        fairlet_center_pt = np.array([np.array(xi) for xi in fairlet_center_idx])
        
        model = KMedoids(n_clusters=self.K, random_state=seed)
        model.fit(fairlet_center_pt)
        np_midx = (np.array(model.centers)).flatten()
        
        centroids = [self.faird.fairlet_centers[index] for index in np_midx]
        self.cluster_centers_ = [fairlet_center_pt[index] for index in np_midx]
        
        self.kmedian_cost = self.faird.fair_kmedian_cost(centroids, X)
        self.logger.update(step=5, status=f"k-Median cost: {self.kmedian_cost}")
        
        self.fairlet_center_pt = fairlet_center_pt
        self.fairlets_member = np.array(model.members)

        self.logger.update(step=6, status="Saving Parameters")
        labels = np.array([-1]*len(X))
        for i,fairlets in enumerate(self.faird.fairlets):
            labels[fairlets] = self.fairlets_member[i] 
            #for xid in fairlets:
            #    labels[xid] = self.fairlets_member[i]
                
        self.labels = np.array(labels)
        self.labels_points = X

    def predict(self, X):
        
        fairlets_midxs = pairwise_distances_argmin(X, Y=self.labels_points)

        return self.labels[fairlets_midxs]
        
"""