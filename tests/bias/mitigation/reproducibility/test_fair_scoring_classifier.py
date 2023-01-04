import numpy as np
import pandas as pd
from holisticai.datasets import load_student
from holisticai.bias.mitigation import FairScoreClassifier
from sklearn.model_selection import train_test_split
from holisticai.bias.metrics import multiclass_bias_metrics

def ohot_encoding(df):
    new_df = pd.DataFrame()
    for col in df.columns:
        if df[col].dtype == object:
            if df[col].nunique() == 2:
                unique_values = df[col].unique()
                tmp = df[col].map({unique_values[0]:0, unique_values[1]:1})
            elif df[col].nunique() > 2 and df[col].nunique() <= 5:
                tmp = pd.get_dummies(df[col], prefix=col)
            elif df[col].nunique() > 5:
                continue
            new_df = pd.concat([new_df, tmp], axis=1)
        else:
            if df[col].nunique() > 5:
                continue
            elif df[col].nunique() > 2 and df[col].nunique() <= 5:
                tmp = pd.get_dummies(df[col], prefix=col)
                new_df = pd.concat([new_df, tmp], axis=1)
            else:
                new_df = pd.concat([new_df, df[col]], axis=1)
    return new_df

def test_fair_scoring_classifier():
    df = load_student()['frame']

    # Make data multiclass by slicing into 4 buckets
    y = df['G3'].to_numpy()
    buckets = np.array([8, 11, 14])
    y_cat = (y.reshape(-1, 1) > buckets.reshape(1, -1)).sum(axis=1)
    df['target'] = y_cat

    # map dictionary
    grade_dict = {0:'very-low', 1:'low', 2:'high',3:'very-high'}
    df['target'] = df['target'].map(grade_dict)

    # drop the other grade columns
    df = df.drop(columns=['G1','G2','G3'])

    df = ohot_encoding(df)

    labels_name = list(df.iloc[:,-4:].columns)

    train, test = train_test_split(df, test_size=0.4, random_state=42)

    X_train = train.drop(columns=labels_name)
    X_test = test.drop(columns=labels_name)
    y_train = train[labels_name]
    y_test = test[labels_name]

    objectives = "ba"
    constraints = {}
    protected_attributes = ["sex"]
    protected_labels = ["target_very-high"]

    numUsers, _ = X_test.shape
    numLabels = len(labels_name)

    model = FairScoreClassifier(objectives, constraints)

    model.fit(X_train, y_train, protected_attributes, protected_labels)

    assert model.predict(X_test).shape == (numUsers, numLabels)
