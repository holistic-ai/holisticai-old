import numpy as np
from sklearn.base import BaseEstimator
from holisticai.utils.transformers.bias import BMInprocessing as BMImp
from .algorithm import FairScoreClassifierAlgorithm


def get_indexes_from_names(df, names):
    indexes = []
    for item in names:
        indexes.append(df.columns.get_loc(item))
    return indexes


def remove_inconcsistency(x, y):
    """
    Remove the inconsistencies

    @x : The dataset features (np.array)
    @y : The dataset labels (np.array)
    
    return the dataset withtout the inconsistencies
    """
    x = x.tolist()
    y = y.tolist()

    new_x = []
    new_y = []

    for i in range(len(x)):
        target = get_max_y(x[i], x, y)
        if y[i][target] == 1:
            new_x.append(x[i])
            new_y.append(y[i])

    return np.array(new_x), np.array(new_y)


def get_max_y(cur_x, x, y):
    y = [y[i] for i, x in enumerate(x) if x == cur_x]

    counts = [0 for i in range(len(y[0]))]

    for i in range(len(y)):
        y[i] = y[i].index(1)

    for i in range(len(y)):
        counts[y[i]] += 1

    return counts.index(max(counts))

class FairScoreClassifier(BaseEstimator, BMImp):
    """
    Generates a classification model that integrates fairness constraints for multiclass classification. This algorithm 
    returns a matrix of lambda coefficients that scores a given input vector. The higher the score, the higher the probability 
    of the input vector to be classified as the majority class.

    References:
        Julien Rouzot, Julien Ferry, Marie-José Huguet. Learning Optimal Fair Scoring Systems for Multi-
        Class Classification. ICTAI 2022 - The 34th IEEE International Conference on Tools with Artificial
        Intelligence, Oct 2022, Virtual, United States. ￿
    """
    def __init__(
        self,
        objectives: dict,
        constraints: dict = {},
        lambda_bound: int = 9,
        time_limit: int = 100,
    ):
        """
        Init FairScoreClassifier object

        Parameters
        ----------
        objectives : dict
            The weighted objectives list to be optimized.

        constraints : dict
            The constraints list to be used in the optimization. The keys are the constraints names and the values are the bounds.

        lambda_bound : int
            Lower and upper bound for the scoring system cofficients.
        """
        self.objectives = objectives
        self.constraints = constraints
        self.lambda_bound = lambda_bound
        self.time_limit = time_limit

    def fit(self, X, y, protected_groups, protected_labels=[]):
        """
        Fit model using Grid Search Algorithm.

        Parameters
        ----------

        X : matrix-like
            input matrix

        y : numpy array
            target vector

        protected_groups : list
            The sensitive groups.

        protected_labels : list
            The senstive labels.

        sensitive_features : numpy array
            Matrix where each columns is a sensitive feature e.g. [col_1=group_a, col_2=group_b]

        Returns
        -------
        the same object
        """
        X, y, fairness_groups, fairness_labels = self.format_dataframe(
            X, y, protected_groups, protected_labels
        )
        self.model_ = FairScoreClassifierAlgorithm(
            self.objectives,
            fairness_groups,
            fairness_labels,
            self.constraints,
            self.lambda_bound,
            self.time_limit
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        X_ = X.copy()
        X_.insert(0, "starts with", np.ones(len(X_.index)))
        X_ = X_.to_numpy()
        return self.model_.predict(X_)

    def transform_estimator(self, estimator):
        self.estimator = estimator
        return self

    def format_dataframe(self, X_df, y_df, protected_groups, protected_labels):
        x_df = X_df.copy()
        x_df.insert(0, "starts with", np.ones(len(x_df.index)))
        sgroup_indexes = get_indexes_from_names(x_df, protected_groups)
        slabels_indexes = get_indexes_from_names(y_df, protected_labels)
        
        x = x_df.to_numpy()
        y = y_df.to_numpy()
        # optional
        x, y = remove_inconcsistency(x, y)
        return x, y, sgroup_indexes, slabels_indexes
