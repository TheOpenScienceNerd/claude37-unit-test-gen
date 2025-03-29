# with one shot prompt engineering. Use prompt 1 first to setup.

The code is from a python module called `metrics`. Note that the code performs a calculation.  Ensure that the calculations you include in unit tests are correct.  An example unit test is: 

```python
# example unit test
import pytest
import numpy as np
import pandas as pd

import metrics as m

@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        # Basic test cases with lists
        ([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], 0.0),
        ([1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], 6.0),
        ([103, 130, 132, 124, 124, 108], [129, 111, 122, 129, 110, 141], 17.833333),
        
        # Different array-like data types
        (np.array([1, 2, 3]), np.array([4, 5, 6]), 3.0),
        (pd.Series([1, 2, 3]), [4, 5, 6], 3.0),
        ([1, 2, 3], pd.Series([4, 5, 6]), 3.0),
        (pd.DataFrame([1, 2, 3]), pd.DataFrame([4, 5, 6]), 3.0),
        (pd.DataFrame([1, 2, 3]), pd.Series([4, 5, 6]), 3.0),
        (pd.DataFrame([1, 2, 3]), np.array([4, 5, 6]), 3.0),
        
        # Float values
        ([1.5, 2.5, 3.5], [1.0, 2.0, 3.0], 0.5),
        
        # Mixed types
        ([1, 2, 3], [1.5, 2.5, 3.5], 0.5),
        
        # 2D arrays (should be flattened)
        (np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]), 4.0),
    ]
)
def test_mean_absolute_error(y_true, y_pred, expected):
    '''
    Test mean absolute error calculation with various input types
    '''
    error = m.mean_absolute_error(y_true, y_pred)
    assert pytest.approx(expected) == error
```

```python

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

```    
