import numpy as np
from numpy.testing import assert_array_equal

from tompy import CustomPDF


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
