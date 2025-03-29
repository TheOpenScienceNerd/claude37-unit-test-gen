"""
Example functions to pass to Claude 3.7 Sonnet

Please note that all functions have

1. Type hints
2. Docstrings
3. Validation for input parameters.

"""


import numpy as np
import warnings

import numpy.typing as npt
from typing import Tuple


def validate_inputs(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns ground truth and predictions values as numpy arrays.

    Parameters:
    --------
    y_true -- array-like
        actual observations from time series
    y_pred -- array-like
        the predictions

    Returns:
    -------
    Tuple(np.array np.array)

    Raises:
    ------
    ValueError
        If inputs cannot be converted to arrays or have different lengths
    TypeError
        If inputs are not array-like
    """
    if not hasattr(y_true, "__iter__") or not hasattr(y_pred, "__iter__"):
        raise TypeError("Inputs must be iterable (array-like) objects")

    try:
        y_true_arr = np.asarray(y_true).flatten()
        y_pred_arr = np.asarray(y_pred).flatten()
    except ValueError as e:
        raise ValueError(f"Inputs cannot be converted to arrays: {str(e)}")

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(
            f"Input arrays must have the same length. Got {len(y_true_arr)} and {len(y_pred_arr)}"
        )

    if len(y_true_arr) == 0:
        raise ValueError("Input arrays cannot be empty")

    # Check if arrays contain numeric data
    if not np.issubdtype(y_true_arr.dtype, np.number) or not np.issubdtype(
        y_pred_arr.dtype, np.number
    ):
        raise ValueError("Input arrays must contain numeric values")

    return y_true_arr, y_pred_arr


def mean_absolute_percentage_error(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike
) -> float:
    """
    Mean Absolute Percentage Error (MAPE).

    MAPE is a relative error measure of forecast accuracy.

    Limitations of MAPE ->

    1. When the ground true value is close to zero MAPE is inflated.
    2. MAPE is not symmetric. MAPE produces smaller forecast
       errors when underforecasting.
    3. MAPE cannot be calculated when actual values contain zeros.

    Parameters:
    --------
    y_true: array-like
        actual observations from time series
    y_pred: arraylike
        the predictions to evaluate

    Returns:
    -------
    float,
        scalar value representing the MAPE (0-100)

    Raises:
    ------
    ValueError
        If y_true contains zeros or non-numeric values
        If inputs have different lengths or are empty
    TypeError
        If inputs are not array-like
    """
    y_true_arr, y_pred_arr = validate_inputs(y_true, y_pred)

    # Check for zeros in y_true which would cause division by zero
    if np.any(y_true_arr == 0):
        raise ValueError(
            "MAPE cannot be calculated when actual values (y_true) contain zeros"
        )

    # Optional: Check for very small values that might cause numerical instability
    small_values = np.abs(y_true_arr) < 1e-10
    if np.any(small_values):
        warnings.warn(
            "Some values in y_true are very close to zero (<1e-10), which may lead to inflated MAPE values",
            UserWarning,
        )

    return np.mean(np.abs((y_true_arr - y_pred_arr) / y_true_arr)) * 100


def mean_absolute_error(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """
    Mean Absolute Error (MAE)

    Parameters:
    --------
    y_true -- array-like
        actual observations from time series
    y_pred -- arraylike
        the predictions to evaluate

    Returns:
    -------
    float,
        scalar value representing the MAE

    Raises:
    ------
    ValueError
        If inputs cannot be converted to numeric arrays
    """
    y_true_arr, y_pred_arr = validate_inputs(y_true, y_pred)
    return np.mean(np.abs((y_true_arr - y_pred_arr)))
