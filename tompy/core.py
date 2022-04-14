import numpy as np


def argsort_like(a, b):
    """Returns indices sorting 1D array a like 1D array b. Arrays a and b must
    contain exactly the same elements.

    Uses `argsort
    <https://numpy.org/doc/stable/reference/generated/numpy.argsort.html>`_
    provided by numpy.

    :param array_like a:
    :param array_like b:
    :return:
        index_array ndarray, int
    """
    if not (a.size == b.size):
        raise ValueError('arrays a and b must contain the same number of '
                         'elements.')

    if not (np.sort(a) == np.sort(b)).all():
        raise ValueError('arrays a and b must contain the same elements.')

    return np.argsort(a)[np.argsort(np.argsort(b))]
