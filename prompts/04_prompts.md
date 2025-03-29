```python
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
```