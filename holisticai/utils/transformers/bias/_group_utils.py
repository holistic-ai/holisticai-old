import pandas as pd


class GroupUtils:
    def merge_columns(self, feature_columns):
        return pd.DataFrame(feature_columns).apply(
            lambda row: ",".join([str(r) for r in row.values]), axis=1
        )

    def create_groups(self, sensitive_features, convert_numeric=False):
        self.tags = pd.DataFrame()
        group_ids = self.merge_columns(sensitive_features)
        group_num, unique_group_id = pd.factorize(group_ids)
        self.group2num = {g: i for i, g in enumerate(unique_group_id)}
        return group_num if convert_numeric else group_ids
