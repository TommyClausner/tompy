import numpy as np


def xy2heatmap(x, y, s=None, include_zeros=False):
    """Generate heatmap data from x and y coordinates.

    Divides data into s bins and returns an array, where each grid point
    corresponds to the number of data points defined by coordinates x and y,
    that lie within a certain grid point. Note that this is somewhat similar
    to a 2D histogram.

    :param array_like x:
        x coordinate of the data
    :param array_like y:
        y coordinate of the data
    :param array_like | int s:
        Single value or 2 values, indicating the shape of the output matrix.
    :param bool include_zeros:
        If False (default), all zeros in the output array will be replaced
        with np.NaN. This can be helpful when plotting the data, as np.NaN
        is not displayed by e.g. plt.imshow().
    :return:
        plot_mat ndarray, int
    """

    # default grid size
    if s is None:
        s = [10] * 2

    # ensure s is a list with 2 values for sizes [x, y]
    if isinstance(s, (int, float)):
        s = [int(s)] * 2

    # compensating for the odd case where the user provides a single valued list
    if len(s) == 1:
        s *= 2

    # ensure correct type for computations below
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    plot_mat = np.zeros(s)

    # shift to 0 origin
    trans_xy = np.min([np.min(x), np.min(y)])
    x -= trans_xy
    y -= trans_xy

    # scale between 0 and s
    scale_xy = np.max([np.max(x), np.max(y)])
    x /= scale_xy / (s[1] - 1)
    y /= scale_xy / (s[0] - 1)

    # transform coordinates into indices
    x = x.round().astype(int)
    y = y.round().astype(int)

    # count occurrences per grid point and return unique indices
    plot_vals_and_inds = np.unique([x, y], axis=1, return_counts=True)

    # fill counts into output matrix
    plot_mat[plot_vals_and_inds[0][1],
             plot_vals_and_inds[0][0]] = plot_vals_and_inds[1]

    # replace zeros with NaN if desired
    if not include_zeros:
        plot_mat[plot_mat == 0] = np.NaN

    return plot_mat
