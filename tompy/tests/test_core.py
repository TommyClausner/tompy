import numpy as np
from numpy.testing import assert_array_equal
import pytest

from tompy.core import argsort_like, rankdata
from tompy.stats import CustomPDF


def test_custompdf():
    n = 10000
    np.random.seed(42)
    arbitrary_dist = np.random.randn(n)/2 + (np.random.randn(n) + 1)/2
    pdf = CustomPDF(arbitrary_dist)
    np.random.seed(42)
    vals0 = np.sort(pdf(n))
    np.random.seed(42)
    vals1 = np.sort(pdf.draw(n))

    assert_array_equal(vals0, vals1)

    spread = np.std(arbitrary_dist)
    assert (np.std((np.sort(arbitrary_dist) - vals0))/spread) < 0.05  # 95% acc


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


def test_rankdata():
    # check vectors
    a = [0.5, 0.2, 4, -3]
    target = [2, 1, 3, 0]
    assert_array_equal(rankdata(a), target)

    # check matrices without axis definition
    a = [[0.5, 0.2, 4, -3], [2, 6, 1, 5]]
    target = [[2, 1, 3, 0], [1, 3, 0, 2]]
    assert_array_equal(rankdata(a), target)

    # check matrices with axis=1 (same as above)
    assert_array_equal(rankdata(a, axis=1), target)

    # check matrices with axis=0
    a = [[0.5, 0.2, 4, -3], [2, 6, 1, 5], [-3, 5, 2, 0]]
    target = [[1, 0, 2, 0], [2, 2, 0, 2], [0, 1, 1, 1]]
    assert_array_equal(rankdata(a, axis=0), target)

    # check matrices with axis=None
    target = [4,  3,  8,  0,  6, 11,  5,  9,  1, 10,  7,  2]
    assert_array_equal(rankdata(a, axis=None), target)
