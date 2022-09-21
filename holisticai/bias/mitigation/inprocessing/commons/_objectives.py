import numpy as np
import pandas as pd

from ._conventions import ALL, LABEL
from ._moments_utils import BaseMoment


class ErrorRate(BaseMoment):
    """
    Extend BaseMoment for error rate objective.
    A classifier :math:`h(X)` has the misclassification error equal to
    .. math::
      P[h(X) \ne Y]
    """

    def load_data(self, X, y, sensitive_features):
        super().load_data(X, y, sensitive_features)
        self.index = [ALL]

    def signed_weights(self):
        """Return the signed weights."""
        return 2 * self.tags[LABEL] - 1

    def gamma(self, predictor):
        """Return the gamma values for the given predictor."""
        pred = predictor(self.X)

        if isinstance(pred, np.ndarray):
            pred = np.squeeze(pred)

        error = pd.Series(data=(self.tags[LABEL] - pred).abs().mean(), index=self.index)
        return error
