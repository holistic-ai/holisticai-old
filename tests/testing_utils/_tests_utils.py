import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from holisticai.bias import metrics
from holisticai.datasets import load_adult


@pytest.fixture
def data_info():
    dataset = load_adult()
    df = pd.concat([dataset["data"], dataset["target"]], axis=1)
    df = df.sample(n=600)

    protected_variables = ["sex", "race"]
    output_variable = ["class"]
    favorable_label = 1
    unfavorable_label = 0

    y = df[output_variable].replace(
        {">50K": favorable_label, "<=50K": unfavorable_label}
    )
    x = pd.get_dummies(df.drop(protected_variables + output_variable, axis=1))

    group = ["sex"]
    group_a = df[group] == "Female"
    group_a = np.array(group_a).ravel()
    group_b = df[group] == "Male"
    group_b = np.array(group_b).ravel()
    y = np.array(y).ravel()
    data = [x, y, group_a, group_b]

    dataset = train_test_split(*data, test_size=0.2, shuffle=True)
    train_data = dataset[::2]
    test_data = dataset[1::2]
    return train_data, test_data


def fit(model, data_info):
    train_data, test_data = data_info
    X, y, group_a, group_b = train_data

    fit_params = {"bm__group_a": group_a, "bm__group_b": group_b}
    model.fit(X, y, **fit_params)
    return model


class MetricsHelper:
    @staticmethod
    def false_negative_rate_difference(group_a, group_b, y_pred, y_true):
        tnra, fpra, fnra, tpra = confusion_matrix(
            y_true[group_a == 1], y_pred[group_a == 1], normalize="true"
        ).ravel()
        tnrb, fprb, fnrb, tprb = confusion_matrix(
            y_true[group_b == 1], y_pred[group_b == 1], normalize="true"
        ).ravel()
        return fnra - fnrb

    @staticmethod
    def true_positive_rate_difference(group_a, group_b, y_pred, y_true):
        tnra, fpra, fnra, tpra = confusion_matrix(
            y_true[group_a == 1], y_pred[group_a == 1], normalize="true"
        ).ravel()
        tnrb, fprb, fnrb, tprb = confusion_matrix(
            y_true[group_b == 1], y_pred[group_b == 1], normalize="true"
        ).ravel()
        return tprb - tpra


def evaluate_pipeline(pipeline, data_info, metric_names, thresholds):
    train_data, test_data = data_info
    X, y, group_a, group_b = train_data
    predict_params = {"bm__group_a": group_a, "bm__group_b": group_b}
    y_pred = pipeline.predict(X, **predict_params)

    for metric_name, threshold in zip(metric_names, thresholds):
        if metric_name == "False Negative Rate difference":
            assert (
                MetricsHelper.false_negative_rate_difference(
                    group_a, group_b, y_pred, y
                )
                < threshold
            )
        elif metric_name == "False Positive Rate difference":
            assert (
                metrics.false_positive_rate_diff(group_a, group_b, y_pred, y)
                < threshold
            )
        elif metric_name == "Statistical parity difference":
            assert metrics.statistical_parity(group_a, group_b, y_pred) < threshold
        elif metric_name == "Average odds difference":
            assert metrics.average_odds_diff(group_a, group_b, y_pred, y) < threshold
        elif metric_name == "Equal opportunity difference":
            assert (
                MetricsHelper.true_positive_rate_difference(group_a, group_b, y_pred, y)
                < threshold
            )
        else:
            raise Exception(f"Unknown metric {metric_name}")
