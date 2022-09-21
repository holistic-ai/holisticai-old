from typing import Optional

import numpy as np
import pandas as pd

from ._conventions import (
    ALL,
    EVENT,
    GROUP_ID,
    LABEL,
    LOWER_BOUND_DIFF,
    PRED,
    UPPER_BOUND_DIFF,
)
from ._moments_utils import BaseMoment, get_index_format
from ._objectives import ErrorRate


def format_data(y=None):
    new_y = pd.Series(np.array(y).reshape(-1))
    return {"y": new_y}


class ClassificationConstraint(BaseMoment):
    """Extend Base Moment for problem that can be expressed as weighted classification error."""

    def __init__(self, ratio_bound: Optional[float] = 1.0):
        """
        Initialize with the ratio value.

        Parameters
        ----------
        ratio_bound : float
            The constraints' ratio bound for constraints that are expressed as
            ratios. The specified value needs to be in (0,1].
        """

        self.eps = 0.001
        self.ratio = ratio_bound

    def load_data(
        self,
        X,
        y,
        sensitive_features: pd.Series,
        event: pd.Series,
        utilities: np.ndarray = None,
    ):
        """
        Description
        -----------
        Load the specified data into this object.

        The `utilities` is a 2-d array which corresponds to g(X,A,Y,h(X)) [agarwal2018reductions]
        The `utilities` defaults to h(X), i.e. [0, 1] for each X_i.

        The first column is G^0 and the second is G^1.
        Assumes binary classification with labels 0/1.

        .. math::
            utilities = [g(X,A,Y,h(X)=0), g(X,A,Y,h(X)=1)]
        """
        if utilities is None:
            utilities = np.vstack(
                [
                    np.zeros(y.shape, dtype=np.float64),
                    np.ones(y.shape, dtype=np.float64),
                ]
            ).T
        self.utilities = utilities
        super().load_data(X, y, sensitive_features)
        self._build_event_variables(event)

    def _build_event_variables(self, event):
        """
        This method:
        - adds a column `event` to the `tags` field.
        - fill in the information about the basis
        """
        self.tags[EVENT] = event

        # Events
        self.event_ids = np.sort(self.tags[EVENT].dropna().unique())
        self.event_prob = self.tags[EVENT].dropna().value_counts() / len(self.tags)

        # Groups and Events
        self.group_values = np.sort(self.tags[GROUP_ID].unique())
        self.group_event_prob = (
            self.tags.dropna(subset=[EVENT]).groupby([EVENT, GROUP_ID]).count()
            / len(self.tags)
        ).iloc[:, 0]

        self.index = get_index_format(self.event_ids, self.group_values)

    def signed_weights(self, lambda_vec):
        """
        Compute the signed weights.

        Description
        -----------

        Uses the equations for :math:`C_i^0` and :math:`C_i^1` as defined
        in Section 3.2 of :footcite:t:`agarwal2018reductions`
        in the 'best response of the Q-player' subsection to compute the
        signed weights to be applied to the data by the next call to the underlying
        estimator.

        Parameters
        ----------
        lambda_vec : :class:`pandas:pandas.Series`
            The vector of Lagrange multipliers indexed by `index`
        """

        lambda_event = (lambda_vec["+"] - self.ratio * lambda_vec["-"]).sum(
            level=EVENT
        ) / self.event_prob

        lambda_group_event = (
            self.ratio * lambda_vec["+"] - lambda_vec["-"]
        ) / self.group_event_prob

        adjust = lambda_event - lambda_group_event

        def get_signed_weight(row):
            if pd.isna(row[EVENT]):
                return 0
            else:
                return adjust[row[EVENT], row[GROUP_ID]]

        signed_weights = self.tags.apply(get_signed_weight, axis=1)
        utility_diff = self.utilities[:, 1] - self.utilities[:, 0]
        signed_weights = utility_diff.T * signed_weights
        return signed_weights

    def default_objective(self):
        return ErrorRate()

    def gamma(self, predictor):
        """Calculate the degree to which constraints are currently violated by the predictor."""

        utility_diff = self.utilities[:, 1] - self.utilities[:, 0]
        predictions = np.squeeze(predictor(self.X))
        pred = utility_diff.T * predictions + self.utilities[:, 0]

        return self._gamma_signed(pred)

    def _gamma_signed(self, pred):
        self.tags[PRED] = pred
        expect_event = self.tags.groupby(EVENT).mean()
        expect_group_event = self.tags.groupby([EVENT, GROUP_ID]).mean()
        expect_group_event[UPPER_BOUND_DIFF] = (
            self.ratio * expect_group_event[PRED] - expect_event[PRED]
        )
        expect_group_event[LOWER_BOUND_DIFF] = (
            -expect_group_event[PRED] + self.ratio * expect_event[PRED]
        )
        gamma_signed = pd.concat(
            [
                expect_group_event[UPPER_BOUND_DIFF],
                expect_group_event[LOWER_BOUND_DIFF],
            ],
            keys=["+", "-"],
            names=["signed", EVENT, GROUP_ID],
        )
        return gamma_signed


class DemographicParity(ClassificationConstraint):
    """
    Extend ClassificationConstraint for demographic parity constraint.

    A classifier :math:`h(X)` satisfies demographic parity if

      P[h(X) = 1 | A = a] = P[h(X) = 1] \; \forall a

    - base_event : defines a single event, `ALL`.
    - event_prob : will record only the probability 1.
    """

    short_name = "DemographicParity"

    def load_data(self, X, y, sensitive_features):
        """Load the specified data into the object."""
        params = format_data(y=y)
        y = params["y"]
        base_event = pd.Series(data=ALL, index=y.index)
        super().load_data(X, y, sensitive_features, base_event)


class EqualizedOdds(ClassificationConstraint):
    """
    Extend ClassificationConstraint for Equalized Odds constraint.

    A classifier :math:`h(X)` satisfies equalized odds if
    .. math::
       P[h(X) = 1 | A = a, Y = y] = P[h(X) = 1 | Y = y] \; \forall a, y

    - base_event: defines the event corresponding to unique value in the `Y` array.
    - event_prob: will record the fraction of the samples corresponding to each unique value in
    the `Y` array.
    """

    short_name = "EqualizedOdds"

    def load_data(self, X, y, sensitive_features):
        """Load the specified data into the object."""
        params = format_data(y=y)
        y = params["y"]
        base_event = y.apply(lambda v: LABEL + "=" + str(v))
        super().load_data(X, y, sensitive_features, base_event)


class TruePositiveRateParity(ClassificationConstraint):
    """
    Extend ClassificationConstraint for true positive rate parity (equal opportunity) constraint.

    A classifier :math:`h(X)` satisfies true positive rate parity if
    .. math::
       P[h(X) = 1 | A = a, Y = 1] = P[h(X) = 1 | Y = 1] \; \forall a

    - base_event: defines the event corresponding to Y = 1 .
    - event_prob: will record the fraction of the samples corresponding to `Y = 1` in the `Y` array.
    """

    short_name = "TruePositiveRateParity"

    def load_data(self, X, y, sensitive_features):
        """Load the specified data into the object."""
        params = format_data(y=y)
        y = params["y"]
        base_event = y.apply(lambda l: f"{LABEL}={l}").where(y == 1)
        super().load_data(X, y, sensitive_features, base_event)


class FalsePositiveRateParity(ClassificationConstraint):
    """
    Extend ClassificationConstraint for false positive rate parity constraint.

    A classifier :math:`h(X)` satisfies false positive rate parity if
    .. math::
       P[h(X) = 1 | A = a, Y = 0] = P[h(X) = 1 | Y = 0] \; \forall a

    - base_event: defines the event corresponding to Y = 0 .
    - event_prob: will record the fraction of the samples corresponding to `Y = 0` in the `Y` array.
    """

    short_name = "FalsePositiveRateParity"

    def load_data(self, X, y, sensitive_features):
        """Load the specified data into the object."""
        params = format_data(y=y)
        y = params["y"]
        base_event = y.apply(lambda v: LABEL + "=" + str(v)).where(y == 0)
        super().load_data(X, y, sensitive_features, base_event)


class ErrorRateParity(ClassificationConstraint):
    """
    Extend ClassificationConstraint for error rate parity constraint.

    A classifier :math:`h(X)` satisfies error rate parity if
    .. math::
       P[h(X) \ne Y | A = a] = P[h(X) \ne Y] \; \forall a

    - base_event : defines a single event, `ALL`.
    - event_prob : will record only the probability 1.
    - In this scenario, g = abs(h(x)-y), rather than g = h(x)
    """

    short_name = "ErrorRateParity"

    def load_data(self, X, y, sensitive_features):
        """Load the specified data into the object."""
        params = format_data(y=y)
        y = params["y"]
        utilities = np.vstack([y, 1 - y]).T
        base_event = pd.Series(data=ALL, index=y.index)
        super().load_data(X, y, sensitive_features, base_event, utilities=utilities)
