from typing import Optional

import numpy as np
import pandas as pd

from ..commons._conventions import *


class GridGenerator:
    """
    Create a grid using a constraint
    """

    def __init__(
        self,
        grid_size: Optional[int] = 5,
        grid_limit: Optional[int] = 2.0,
        neg_values: Optional[bool] = True,
    ):
        """
        Initialize Grid Generator

        Parameters
        ----------
        grid_size: int
            number of columns to be generated in the grid.

        grid_limit : float
            range of the values in the grid generated.

        neg_values: bool
            ensures if we want to include negative values in the grid or not.
            If True, the range is doubled.
        """
        self.grid_size = grid_size
        self.grid_limit = grid_limit
        self.neg_values = neg_values

    def generate_grid(self, constraint):
        # Get pos and neg basis from constraint
        basis = self._get_basis(constraint)
        # Generate lambda vectors for each event
        coefs = self._generate_coefs(nb_events=len(constraint.event_ids))
        # Convert the grid of basis coefficients into a grid of lambda vectors
        grid = basis["+"].dot(coefs["+"]) + basis["-"].dot(coefs["-"])
        return grid

    def _build_grid(self, nb_events):
        """Create an integer grid"""
        grid_size = self.grid_size + 1 if self.grid_size % 2 == 0 else self.grid_size
        max_value = int(np.ceil(grid_size ** (1 / nb_events)))
        min_value = 0
        if self.neg_values:
            max_value = (max_value - 1 + 2 - 1) // 2
            min_value = -max_value
        xs = [np.arange(min_value, max_value + 1) for _ in range(nb_events)]
        xs = np.meshgrid(*xs)
        xs = np.stack([x.reshape(-1) for x in xs], axis=1)
        xs = xs * self.grid_limit / max_value
        return xs

    def _generate_coefs(self, nb_events):
        np_grid_values = self._build_grid(nb_events=nb_events)
        grid_values = pd.DataFrame(np_grid_values[: self.grid_size]).T
        pos_grid_values = grid_values.copy()
        neg_grid_values = -grid_values.copy()
        pos_grid_values[grid_values < 0] = 0
        neg_grid_values[grid_values < 0] = 0
        lambda_vector = {"+": pos_grid_values, "-": neg_grid_values}
        return lambda_vector

    def _get_basis(self, constraint):
        pos_basis = pd.DataFrame()
        neg_basis = pd.DataFrame()

        zero_vec = pd.Series(0.0, constraint.index)
        i = 0
        for event_val in constraint.event_ids:
            for group in constraint.group_values[:-1]:
                pos_basis[i] = zero_vec
                neg_basis[i] = zero_vec
                pos_basis[i]["+", event_val, group] = 1
                neg_basis[i]["-", event_val, group] = 1
                i += 1
        return {"+": pos_basis, "-": neg_basis}
