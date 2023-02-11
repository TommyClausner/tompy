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


def median_smooth(d, kernel_size=5, use_less_memory=False):
    """Median smooth for 1D data

    :param array_like d:
        The data.
    :param int kernel_size:
        Size of the smoothing kernel. Must be odd or will be raised by 1.
    :param bool use_less_memory:
        Whether to use less system memory (roughly 10x slower).
    :return:
        Smoothed data.
    """
    # make kernel size odd
    if (kernel_size % 2) == 0:
        kernel_size += 1

    # determine padding
    padding = (kernel_size - 1)//2

    if not isinstance(d, list):
        d = d.tolist()

    # add NaN padding to use with nanmedian (Padding will not be included in
    # median calculation)
    d = [np.nan] * padding + d + [np.nan] * padding

    # accumulate data in matrix. Smoothing in one go. Basically the matrix is
    # replicated with a shift of 1 and a window size of kernel_size. After
    # computing the median across the kernel sized dimension, the matrix
    # collapses into the result
    if use_less_memory:  # normal loop and smooth
        tmp = []
        for ind in range(padding, len(d) - padding):
            tmp.append(np.nanmedian(d[(ind - padding):(ind + padding + 1)]))
        return np.array(tmp)
    else:  # build up matrix and compute median in one go
        tmp = [d[ind:-(kernel_size - ind - 1)]
               for ind in range(kernel_size - 1)]

        tmp.append(d[(kernel_size - 1):])
        return np.nanmedian(np.array(tmp), axis=0)


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


def submesh(vertices, faces, inds, conn=0):
    """Recomputes mesh faces (and vertices) based on a vertex sub-selection.

    :param array-like vertices:
        Vertex coordinates.
    :param array-like faces:
        Vertex indices that form faces.
    :param array-like inds:
        Indices of vertices that form the sub-selected mesh.
    :param int conn:
        Connectivity. Minimum number of points that need to be connected to
        the selected vertex in order to consider the respective adjacent face
        for the sub-selection.
    :return:
        Selected vertices and corresponding faces.
    """

    # determine faces that are connected with more than "conn" vertex points
    # to the sub-selected mesh.
    sel_inds_faces = np.isin(faces, inds).sum(axis=1) > conn

    # raw selection
    faces_sel = faces[sel_inds_faces, :]
    verts_sel = np.unique(faces_sel.flatten())

    # map old vertex indices to new list of vertices (and corresponding
    # indices)
    verts_map = {old: new for new, old in enumerate(verts_sel)}

    # apply selection to vertex list
    new_verts = vertices[verts_sel, :]

    # compute new faces (with updated indices)
    new_faces = [list(map(verts_map.get, f)) for f in faces_sel]

    # convert to array if input was not list
    if not isinstance(faces, list):
        new_faces = np.asarray(new_faces)

    return new_verts, new_faces


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
