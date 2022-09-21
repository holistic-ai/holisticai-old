from typing import Optional, Union

import numpy as np

from holisticai.utils._validation import _check_valid_y_proba
from holisticai.utils.transformers._transformer_base import BMTransformerBase


class BMPostprocessing(BMTransformerBase):
    """
    Base Post Processing transformer
    """

    BM_NAME = "Postprocessing"

    def _load_data(self, **kargs):
        """Save postprocessing atributes and convert data to standard format parameters."""

        params = {}

        if "y_true" in kargs:
            y_true = np.array(kargs.get("y_true")).ravel()
            params.update({"y_true": y_true})

        if "y_pred" in kargs:
            y_pred = np.array(kargs.get("y_pred")).ravel()
            params.update({"y_pred": y_pred})

        if "y_proba" in kargs:
            y_proba = np.array(kargs.get("y_proba"))

            _check_valid_y_proba(y_proba=y_proba)

            params.update({"y_proba": y_proba})
            favorable_index = 1
            y_score = np.array(y_proba[:, favorable_index]).ravel()
            params.update({"y_score": y_score})

        params_to_numpy_format = ["group_a", "group_b", "y_score"]
        for param_name in params_to_numpy_format:
            if param_name in kargs:
                params.update({param_name: self._to_numpy(kargs, param_name)})

        if "X" in kargs:
            params.update({"X": self._to_numpy(kargs, "X", ravel=False)})

        if ("sample_weight" in kargs) and (not kargs["sample_weight"] is None):
            params.update({"sample_weight": self._to_numpy(kargs, "sample_weight")})

        elif "y_true" in locals():
            params.update({"sample_weight": np.ones_like(y_true).astype(np.float64)})

        return params
