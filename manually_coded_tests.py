"""
An example test coded by me. This can be used with
one-shot prompt engineering with Claude 3.7
"""

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