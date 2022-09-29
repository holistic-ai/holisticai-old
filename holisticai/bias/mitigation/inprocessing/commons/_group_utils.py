import numpy as np
import pandas as pd

from ._conventions import _ALL, _EVENT, _GROUP_ID, _GROUP_NUM, _LABEL


def format_data(y=None):
    new_y = pd.Series(np.array(y).reshape(-1))
    return {"y": new_y}


class BaseConstraint:
    params = ["X", "y", "sensitive_features"]

    def save_params(self, *args):
        for name, value in zip(self.params, args):
            setattr(self, name, value)

    def merge_columns(self, feature_columns):
        return pd.DataFrame(feature_columns).apply(
            lambda row: ",".join([str(r) for r in row.values]), axis=1
        )

    def load_data(self, X, y, sensitive_features):
        self.tags = pd.DataFrame()
        self.tags[_LABEL] = y
        self.tags[_GROUP_ID] = self.merge_columns(sensitive_features)
        self.tags[_GROUP_NUM], unique_group_id = pd.factorize(self.tags[_GROUP_ID])
        self.group2num = {g: i for i, g in enumerate(unique_group_id)}
        self.save_params(X, y, sensitive_features)


class ClassificationConstraint(BaseConstraint):
    def load_data(self, X, y, sensitive_features, event):
        super().load_data(X, y, sensitive_features)
        self._build_event_variables(event)

    def get_index_format(self, event_ids, group_values):
        index = (
            pd.DataFrame(
                [{_EVENT: e, _GROUP_ID: g} for e in event_ids for g in group_values]
            )
            .set_index([_EVENT, _GROUP_ID])
            .index
        )
        return index

    def _build_event_variables(self, event):
        """
        This method:
        - adds a column `event` to the `tags` field.
        - fill in the information about the basis
        """
        self.tags[_EVENT] = event

        # Events
        self.event_ids = np.sort(self.tags[_EVENT].dropna().unique())
        self.event_prob = self.tags[_EVENT].dropna().value_counts() / len(self.tags)

        # Groups and Events
        self.group_values = np.sort(self.tags[_GROUP_ID].unique())
        self.group_event_prob = (
            self.tags.dropna(subset=[_EVENT]).groupby([_EVENT, _GROUP_ID]).count()
            / len(self.tags)
        ).iloc[:, 0]

        self.index = self.get_index_format(self.event_ids, self.group_values)


class Moment(ClassificationConstraint):
    def load_data(self, X, y, sensitive_features):
        params = format_data(y=y)
        y = params["y"]
        base_event = pd.Series(data=_ALL, index=y.index)
        super().load_data(X, y, sensitive_features, base_event)


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
