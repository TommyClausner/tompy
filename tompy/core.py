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


def cart2pol(x, y):
    """Converts cartesian to polar coordinates.

    :param x:
    :param y:
    :return:
        (r, theta)
    """
    return np.hypot(x, y), np.arctan2(y, x)


def cummean(a, axis=None):
    """Cumulative average along a given axis (similar to cumsum).

    :param array_like a:
        The input data.
    :param int | None axis:
        The axis over which to perform the operation
    :return:
        Cumulative average along axis.
    """
    b = np.asarray(a)
    return np.cumsum(b, axis=axis)/np.cumsum(np.ones(b.shape), axis=axis)


def pol2cart(r, th):
    """Converts polar to cartesian coordinates.

    :param r:
        Radius.
    :param th:
        Theta in radians.
    :return:
        (x, y)
    """
    return r * np.cos(th), r * np.sin(th)


def rankdata(a, axis=-1, direction='ascend'):
    """Ranks data and assigns ascending or descending values to each data point.

    Uses `argsort
    <https://numpy.org/doc/stable/reference/generated/numpy.argsort.html>`_
    provided by numpy.

    :param array_like a:
        The data to be ranked.
    :param int | None axis:
        Axis along which to operate.
    :param str direction:
        Direction of sorting (Default is 'ascend'). Can be 'ascend' or
        'descend'.
    :return:
        ndarray
    """
    b = np.argsort(a, axis=axis).argsort(axis=axis)
    if direction == 'ascend':
        return b
    else:
        return np.abs(b - b.max())
