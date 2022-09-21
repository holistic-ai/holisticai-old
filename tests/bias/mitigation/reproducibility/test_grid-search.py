import os
import sys

sys.path.append(os.getcwd())

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from holisticai.bias.metrics import classification_bias_metrics
from holisticai.bias.mitigation import GridSearchReduction
from holisticai.pipeline import Pipeline
from tests.testing_utils._tests_data_utils import (
    check_results,
    load_preprocessed_adult_v2,
)

seed = 42
train_data, test_data = load_preprocessed_adult_v2()


def running_without_pipeline():
    X, y, group_a, group_b = train_data

    scaler = StandardScaler()
    Xt = scaler.fit_transform(X)

    model = LogisticRegression()
    inprocessing_model = GridSearchReduction(
        constraints="ErrorRateParity", grid_size=20
    ).transform_estimator(model)

    fit_params = {"group_a": group_a, "group_b": group_b}
    inprocessing_model.fit(Xt, y, **fit_params)

    # Test
    X, y, group_a, group_b = test_data
    Xt = scaler.transform(X)

    y_pred = inprocessing_model.predict(Xt)

    df = classification_bias_metrics(
        group_b.to_numpy().ravel(),
        group_a.to_numpy().ravel(),
        y_pred,
        y.to_numpy().ravel(),
        metric_type="both",
    )
    return df


def running_with_pipeline():
    model = LogisticRegression()
    inprocessing_model = GridSearchReduction(
        constraints="ErrorRateParity", grid_size=20
    ).transform_estimator(model)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("bm_inprocessing", inprocessing_model),
        ]
    )

    X, y, group_a, group_b = train_data
    fit_params = {"bm__group_a": group_a, "bm__group_b": group_b}

    pipeline.fit(X, y, **fit_params)

    X, y, group_a, group_b = test_data
    predict_params = {
        "bm__group_a": group_a,
        "bm__group_b": group_b,
    }
    y_pred = pipeline.predict(X, **predict_params)
    df = classification_bias_metrics(
        group_b.to_numpy().ravel(),
        group_a.to_numpy().ravel(),
        y_pred,
        y.to_numpy().ravel(),
        metric_type="both",
    )
    return df


def test_reproducibility_with_and_without_pipeline():
    df1 = running_without_pipeline()
    df2 = running_with_pipeline()
    check_results(df1, df2)


test_reproducibility_with_and_without_pipeline()
