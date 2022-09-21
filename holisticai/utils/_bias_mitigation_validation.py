import numpy as np


def check_valid_y_proba(y_proba: np.ndarray):
    atol = 1e-3
    y_shape = np.shape(y_proba)
    assert len(y_shape) == 2, f"""y_proba must be 2d tensor, found: {y_shape}"""

    sum_all_probs = np.sum(y_proba, axis=1)
    assert np.all(
        np.isclose(sum_all_probs, 1, atol=atol)
    ), f"""probability must sum to 1 across the possible classes, found: {sum_all_probs}"""

    correct_proba_values = np.all(y_proba <= 1) and np.all(y_proba >= 0)
    assert (
        correct_proba_values
    ), f"""probability values must be in the interval [0,1], found: {y_proba}"""
