"""
Example functions to pass to Claude 3.7 Sonnet

Please note that all functions have

1. Type hints
2. Docstrings
3. Validation for input parameters.

"""

import numpy as np
import warnings
import pandas as pd
import numpy.typing as npt


def _normalize_array_input(arr: npt.ArrayLike | int | float, name: str) -> np.ndarray:
    """
    Normalize various input types to a 1D numpy array with validation.
    
    Parameters:
    ----------
    arr : array-like or scalar
        The input to normalize (DataFrame, Series, array, list, or scalar)
    name : str
        Name of the input for error messages
        
    Returns:
    -------
    np.ndarray
        Flattened 1D numpy array of float values
        
    Raises:
    ------
    TypeError
        If input cannot be converted to numeric array
    ValueError
        If input contains non-numeric values
    """
    # Check for multi-dimensional inputs and warn before processing
    if isinstance(arr, pd.DataFrame) and arr.shape[1] > 1:
        warnings.warn(
            f"Multi-dimensional DataFrame provided for {name} with shape {arr.shape}. "
            "Only the flattened values will be used, which may not be what you intended.",
            UserWarning
        )
    
    # Check for numpy arrays that are multi-dimensional
    if isinstance(arr, np.ndarray) and arr.ndim > 1:
        warnings.warn(
            f"Multi-dimensional array provided for {name} with shape {arr.shape}. "
            "Only the flattened values will be used, which may not be what you intended.",
            UserWarning
        )
    
    # Handle pandas DataFrame, Series, or scalar values
    if isinstance(arr, (pd.DataFrame, pd.Series)):
        arr = arr.to_numpy()
    elif not hasattr(arr, "__iter__") or isinstance(arr, (int, float)):
        # Handle scalar values
        arr = np.asarray([arr], dtype=float)
 
    # String checking
    if isinstance(arr, (str, bytes)):
        raise TypeError(f"String inputs are not supported: {arr}")

    try:
        arr_processed = np.asarray(arr, dtype=float).flatten()
    except TypeError as e:
        raise TypeError(f"Cannot convert {name} to numeric array: {arr} - {str(e)}") from e
    except ValueError as e:
        raise ValueError(f"Failed to convert {name} to float array. Input contains non-numeric values: {str(e)}") from e
        
    return arr_processed


def _validate_inputs(
    y_true: npt.ArrayLike | int | float,
    y_pred: npt.ArrayLike | int | float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns ground truth and predictions values as numpy arrays with enhanced validation.

    Parameters:
    --------
    y_true : array-like or scalar
        actual observations from time series
    y_pred : array-like or scalar
        the predictions

    Returns:
    -------
    Tuple(np.ndarray, np.ndarray)
        Validated and processed arrays

    Raises:
    ------
    ValueError
        If inputs cannot be converted to arrays, have different lengths,
        are empty, contain non-numeric values, or contain NaN/inf values
    TypeError
        If inputs cannot be converted to numeric types
    """
    # Process both arrays using the helper function
    y_true_arr = _normalize_array_input(y_true, "y_true")
    y_pred_arr = _normalize_array_input(y_pred, "y_pred")

    # Check array lengths
    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(
            f"Input arrays must have the same length. Got {len(y_true_arr)} and {len(y_pred_arr)}"
        )

    if len(y_true_arr) == 0:
        raise ValueError("Input arrays cannot be empty")

    # Check for NaN and infinity values
    if np.isnan(y_true_arr).any() or np.isnan(y_pred_arr).any():
        raise ValueError("Input arrays contain NaN values")
    
    if np.isinf(y_true_arr).any() or np.isinf(y_pred_arr).any():
        raise ValueError("Input arrays contain infinity values")
    
    # Check for zero arrays - that might indicate errors
    if np.all(y_true_arr == 0):
        warnings.warn(
            "All values in y_true are zero, which may cause issues in percentage-based metrics",
            UserWarning
        )
    
    if np.all(y_pred_arr == 0):
        warnings.warn(
            "All values in y_pred are zero, which may indicate a problem with the prediction model",
            UserWarning
        )
        
    # Check for very large differences that might indicate errors
    if np.max(np.abs(y_true_arr - y_pred_arr)) > 1e6:
        warnings.warn(
            "Very large differences detected between true and predicted values",
            UserWarning
        )

    return y_true_arr, y_pred_arr



def mean_absolute_error(
        y_true: npt.ArrayLike | int | float, 
        y_pred: npt.ArrayLike | int | float
) -> float:
    """
    Mean Absolute Error (MAE)

    Parameters:
    --------
    y_true -- array-like or int or float
        actual observations from time series
    y_pred -- array-like or int or float
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
    y_true_arr, y_pred_arr = _validate_inputs(y_true, y_pred)
    return np.mean(np.abs((y_true_arr - y_pred_arr)))
