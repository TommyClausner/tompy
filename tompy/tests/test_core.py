import numpy as np
from numpy.testing import assert_array_equal
import pytest

from tompy.core import argsort_like


def test_argsort_like():
    # confirm sorting
    a = np.random.randint(0, 100, 42)
    b = a[np.random.permutation(len(a))]

    assert_array_equal(a[argsort_like(a, b)], b)

    # confirm raising value error if different number of elements
    with pytest.raises(ValueError):
        argsort_like(a, b[1:])

    # confirm raising value error if different elements
    b[0] = -1  # impossible value
    with pytest.raises(ValueError):
        argsort_like(a, b)
