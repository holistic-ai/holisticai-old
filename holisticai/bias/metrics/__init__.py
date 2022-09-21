"""
The :mod:`holisticai.bias.metrics` module includes classification, regression, multiclass, recommender and clustering bias metrics
"""

# Classification
from ._classification import (
    abroca,
    accuracy_diff,
    average_odds_diff,
    classification_bias_metrics,
    cohen_d,
    disparate_impact,
    equal_opportunity_diff,
    false_negative_rate_diff,
    false_positive_rate_diff,
    four_fifths,
    statistical_parity,
    true_negative_rate_diff,
)

# Clustering
from ._clustering import (
    cluster_balance,
    cluster_dist_entropy,
    cluster_dist_kl,
    cluster_dist_l1,
    clustering_bias_metrics,
    min_cluster_ratio,
    silhouette_diff,
    social_fairness_ratio,
)

# Multiclass
from ._multiclass import (
    accuracy_matrix,
    confusion_matrix,
    confusion_tensor,
    frequency_matrix,
    multiclass_average_odds,
    multiclass_bias_metrics,
    multiclass_equality_of_opp,
    multiclass_statistical_parity,
    multiclass_true_rates,
    precision_matrix,
    recall_matrix,
)

# Recommender
from ._recommender import (
    aggregate_diversity,
    avg_f1_ratio,
    avg_precision_ratio,
    avg_recall_ratio,
    avg_recommendation_popularity,
    exposure_entropy,
    exposure_kl,
    exposure_l1,
    gini_index,
    mad_score,
    recommender_bias_metrics,
    recommender_mae_ratio,
    recommender_rmse_ratio,
)

# Regression
from ._regression import (
    avg_score_diff,
    correlation_diff,
    disparate_impact_regression,
    mae_ratio,
    max_statistical_parity,
    no_disparate_impact_level,
    regression_bias_metrics,
    rmse_ratio,
    statistical_parity_auc,
    statistical_parity_regression,
    zscore_diff,
)

# All bias functions and classes
__all__ = [
    "statistical_parity",
    "disparate_impact",
    "four_fifths",
    "cohen_d",
    "accuracy_diff",
    "false_negative_rate_diff",
    "true_negative_rate_diff",
    "abroca",
    "statistical_parity_regression",
    "disparate_impact_regression",
    "no_disparate_impact_level",
    "avg_score_diff",
    "zscore_diff",
    "max_statistical_parity",
    "statistical_parity_auc",
    "correlation_diff",
    "rmse_ratio",
    "mae_ratio",
    "classification_bias_metrics",
    "regression_bias_metrics",
    "aggregate_diversity",
    "gini_index",
    "avg_recommendation_popularity",
    "exposure_l1",
    "exposure_kl",
    "mad_score",
    "exposure_entropy",
    "cluster_balance",
    "min_cluster_ratio",
    "cluster_dist_l1",
    "cluster_dist_kl",
    "cluster_dist_entropy",
    "social_fairness_ratio",
    "avg_precision_ratio",
    "avg_recall_ratio",
    "avg_f1_ratio",
    "recommender_rmse_ratio",
    "recommender_mae_ratio",
    "equal_opportunity_diff",
    "false_positive_rate_diff",
    "average_odds_diff",
    "silhouette_diff",
    "recommender_bias_metrics",
    "clustering_bias_metrics",
    "confusion_matrix",
    "frequency_matrix",
    "confusion_tensor",
    "accuracy_matrix",
    "precision_matrix",
    "recall_matrix",
    "multiclass_equality_of_opp",
    "multiclass_average_odds",
    "multiclass_true_rates",
    "multiclass_statistical_parity",
    "multiclass_bias_metrics",
]
