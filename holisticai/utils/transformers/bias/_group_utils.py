import pandas as pd


class SensitiveGroups:
    def merge_columns(self, feature_columns):
        return pd.DataFrame(feature_columns).apply(
            lambda row: ",".join([str(r) for r in row.values]), axis=1
        )

    def fit(self, sensitive_features):
        self.tags = pd.DataFrame()
        group_ids = self.merge_columns(sensitive_features)
        _, unique_group_id = pd.factorize(group_ids)
        self.group2num = {g: i for i, g in enumerate(unique_group_id)}
        return self

    def transform(self, sensitive_features, convert_numeric=False):
        group_ids = self.merge_columns(sensitive_features)
        if convert_numeric:
            return group_ids.apply(lambda x: self.group2num[x])
        return group_ids

    def fit_transform(self, sensitive_features, convert_numeric=False):
        return self.fit(sensitive_features).transform(
            sensitive_features, convert_numeric=convert_numeric
        )
