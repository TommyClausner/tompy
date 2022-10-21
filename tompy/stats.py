import sys

import numpy as np
from scipy.interpolate import interp1d


class CustomProbDensFct:
    """Draw random values from interpolated histogram data

        Input data will be transformed into histogram data given the
        corresponding bins. The histogram will be interpolated according to
        the selected method. Using 'CustomProbDensFct.gen_value()',
        a new value given the underlying probability distribution that has
        been inferred from the histogram data will be returned.

    Parameters
    ----------
    d : array_like
        The data on which the estimation is estimate.
    bins : int | str
        Either ‘auto’ or a integer indicating the number of bins for the
        histogram that is used for probability density estimation.
    method : str
        Which interpolation method to use. Must be of ‘linear’, ‘nearest’,
        ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’,
        ‘next’.

    Attributes
    ----------
    bins : array_like
        Histogram bins.
    counts : array_like
        Histogram counts.
    distribution : array_like
        Distribution from which is drawn.
    x : array_like
        x values of the discrete distribution
    y : array_like
        y values of the discrete distribution
    method : str
        Which interpolation method to use. Must be of ‘linear’, ‘nearest’,
        ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’,
        ‘next’.
    """
    def __init__(self, d, bins='auto', method='linear'):
        self.method = method + ''
        self.counts, self.bins = np.histogram(d, bins=bins,
                                              density=True)
        cum_counts = np.cumsum(self.counts)
        bin_widths = (self.bins[1:] - self.bins[:-1])
        self.x = cum_counts * bin_widths
        self.y = self.bins[1:]
        self.distribution = interp1d(self.x, self.y, kind=method)

    def _get_seed(self):
        return np.random.uniform(self.x[0], self.x[-1])

    def draw(self, n_values=1):
        """Return random sampled value

        :param int n_values:
            How many values to draw.
        """
        return self.distribution([self._get_seed() for _ in range(n_values)])

    def get_distribution(self):
        """Return interpolated distribution"""
        return self.distribution

    def get_counts(self):
        """Return histogram counts"""
        return self.counts

    def get_bins(self):
        """Return histogram bins"""
        return self.bins

    def get_xy(self):
        """Return x and y value as tuple"""
        return self.x, self.y


def arostest(data, normalize_ranks=True, method='lstsq', shape=None,
             permutations=10000, alpha=0.05, tail=None, shape_dim=1,
             unique_permutations=None):
    """Performs permutation aros test.

    The permutation autoregressive rank order similarity (aros) test in it's
    core functions like any other permutation test. A test statistic is
    computed and compared to a permutation distribution. Depending on whether
    shape was set to None, the null hypothesis varies (see below).

    The test statistic:
        1) The data is averaged over observations.
        2) The resulting average is transformed into a rank order. E.g. if
        the average over observations turns out to be [0.1, 0.3, -0.3],
        the rank order will be [2, 3, 1]. If normalized, it will be [0.5,
        0.75, 0.25].
        3) Depending on the method, the rank order shape will be related to
        each observation's data. Note, that the rank order is ordinal data,
        whereas the original data is numerical. In the default case method is
        set to 'lstsq', which means that for each observation Ax = b is
        solved for x in a least square way (see: `lstsq
        <https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html>`_
        provided by numpy).
        4) The coefficient obtained for each subject is averaged to obtain
        the test statistic for the dataset.

    The permutation distribution:
        1) In each permutation the columns are shuffled randomly for each
        observation separately.
        2) Each value of the permutation distribution is obtained by the same
        computation as used for the test statistic.

    The test:
        1) The p value is obtained by comparing the test statistic to the
        permutation distribution.
        2) The threshold is obtained by computing the respective percentile
        of the permutation distribution depending on the alpha level set.
        3) if the p value is smaller than the alpha level, H0 is rejected.

    H0 if shape is None:
        1) The rank order shape obtained from the data is as likely as any
        any other shape that could have been obtained from the same data if
        the condition order was shuffled.
        2) The column order does not cary meaningful information about the
        distribution of values.
        3) If rejected, the ordinal order between conditions for each
        observation matters.

    H0 if shape was predefined (e.g. by the hypothesis or previous research):
        1) The shape does not explain the data better than it would explain
        any other arrangement of the data.
        2) The column order does not cary meaningful information with respect to
        the tested shape.
        3) If rejected, the ordinal order between conditions for each
        observation can be explained better by the provided shape, than it
        would if the order of conditions would be meaningless.

    What a significant result tells you:
        1) The ordinal structure between the averages of different conditions is
        meaningful. E.g. the means between 3 conditions are distributed such
        that condition3 < condition1 < condition2
        2) The values between conditions are not exchangeable

    What a significant result does NOT tell you:
        1) The difference between conditions (i.e. absolute value differences)
        2) Whether the difference between conditions is meaningful (e.g.
        effect size)

    :param array_like data:
        The data as 2D array. If the data is arranged such, that the rows
        represent the examples and the columns the different groups,
        shape_dim must be set to 1. In the reverse case shape_dim must be set to
        0.
    :param bool normalize_ranks:
        Whether to normalize ranks between 0 and 1 (Default is true). Note
        that the normalization is performed such, that 0 and 1 themselves are
        not part of the rank values. However, the smallest rank will have a
        similar difference to 0 as the highest rank value to 1 and each
        neighboring ranks to each other.
    :param str method:
        Method to obtain test statistic (Default is 'lstsq'). Possible values
        are 'lstsq' (least square solution of ax = b for x), 'corr' (Pearson
        correlation), 'r2' (explained variance), 'cossim' (cosine similarity),
        'sse' (sum of squared errors).
    :param array_like | None shape:
        Predefined shape (Default is None). If None, the shape will be
        obtained from the average data. This affects the computation of the
        permutation distribution as well. If shape is not None, then the
        provided shape will be always used to test against. None means that
        during each permutation the shape is computed from the average after
        shuffling the data.
    :param int permutations:
        The number of permutations to create the permutation distribution.
    :param float alpha:
        The alpha level.
    :param int | None tail:
        The tail to test against in the permutation distribution (Default is
        None). If None, the tail will be chosen according to the test
        statistic. If the test statistic is < 0 tail will be set to -1 and if
        the test statistic is > 0 tail will be set to 1.
    :param int shape_dim:
        The dimension which represents the different groups.
    :param bool | None unique_permutations:
        Whether to use unique permutations (Default is None). If None,
        the probability of non-unique permutations will be assesed and
        compared to the data. If low enough uniqueness will not be forced to
        save computation time.
    :return:
        Dictionary with entries for multiple test related values:
        'distribution': the permutation distribution
        'test_stat': the test statistic of the data
        'tail': the test tail
        'thresh': the threshold necessary to reach significance
        'p-value': p value of the test statistic
        'alpha': tail adjusted alpha
        'h': whether to reject H0
        'shape': the rank order shape (normalized if set)
        'normalized': whether the shape was normalized
        'method': method to compute the test statistic
    """

    def make_permutations(r, n, p, unique):
        """Computes unique sets of group permutations. Each arrangement of
        rank orders has the same chance of being selected. Since each
        observation receives a set of ranks, it needs to be ensured that
        possible combinations are unique.

        :param int r:
            Number of unique ranks
        :param int n:
            Number of observations.
        :param int p:
            Number of permutations
        :param bool | None unique:
            Whether to use a unique set of permutations. This is done
            automatically if the probability of not obtaining all unique sets is
            lower than 1 / p / 20.
        :return:
            array
        """
        # to save time the probability that given the size of the data a
        # non-unique permutation will occur will be calculated and compared
        # to 1 / (p * 20). If the probability of a non-unique permutation is
        # lower than that, then it will not be checked whether the
        # permutations are actually unique.
        prob_repeat = 1 - (1 - 1 / n ** np.prod(range(1, r + 1))) ** p
        if unique is None:
            if prob_repeat > (1 / p / 20):
                unique = True
            else:
                unique = False

        if unique:
            sets = []
            while len(sets) < p:
                # randomly select n sets from all combinations
                a = [np.random.permutation(r).tolist() for _ in range(n)]

                if a not in sets:  # only allow unique sets
                    sets.append(a)

                    # status
                    if len(sets) % 500 == 0:
                        sys.stdout.write(
                            "\rPreparing permutation %d of %d" % (len(sets), p))
                        sys.stdout.flush()
            print('\ndone.')
        else:
            # way faster, but not guaranteed unique
            sets = [[np.random.permutation(r).tolist() for _ in range(n)]
                    for _ in range(p)]
        # convert to array and transpose for convenience
        return np.asarray(sets).T

    def make_rankshape(d, n):
        """Convert data into rank order shape.

        :param array_like d:
            The data.
        :param bool n:
            Whether to normalize rank shape (Default is true).
        :return:
        """

        ad = np.mean(d, axis=1)

        # make lowest rank be 1
        s = (ad.argsort().argsort() + 1).astype(float)

        # normalize between 0 and 1 excluding 0 and 1
        if n:
            s /= len(s) + 1

        # ensure dimensionality for later computation
        return s

    def get_stat(d, s, m):
        """Compute test statistic.

        :param array_like d:
            The data.
        :param array_like s:
            The shape.
        :param str m:
            The method to use. E.g. 'lstsq'
        :return:
        """
        if m is 'lstsq':
            stat = np.zeros(d.shape[1])

            for ind, c in enumerate(d.T):  # iterate through observations
                # Solves Ax = B for x. Thereby: x = inv(a'a)a'b. Since a is a
                # vector, inv(a'a)a' can be written as 1/(a'a) * a',
                # which corresponds to a'/(a'a)b. We can rewrite this as the
                # ratio between the dot product of a and b and the dot
                # product of a with itself. This boils down to the ratio
                # between the magnitudes of a and b times the cosine of angle
                # they spread apart from each other.
                stat[ind] = np.dot(c, s) / np.dot(c, c)

            return np.mean(stat)

        if m is 'cossim':
            stat = np.zeros(d.shape[1])

            for ind, c in enumerate(d.T):  # iterate through observations
                # computes cosine similarity between two vectors
                stat[ind] = np.dot(c, s) / np.linalg.norm(c) / np.linalg.norm(s)

            return np.mean(stat)

        elif (m is 'corr') or (m is 'r2'):
            stat = np.zeros(d.shape[1])
            s = s.T

            for ind, c in enumerate(d.T):
                c_2d = np.atleast_2d(c)  # 1D column to 2D array
                stat[ind] = np.corrcoef(c_2d, s)[0, 1]

            # square correlation for explained variance
            if m is 'r2':
                stat **= 2

            return np.mean(stat)
        elif m is 'sse':
            return np.sum((d.T - s) ** 2)
        else:
            raise ValueError('unknown method')

    def get_distribution(d, n, perms, m, s, u):
        """Compute permutation distribution.

        :param array_like d:
            The un-shuffled data.
        :param bool n:
            Whether to normalize rank shape (Default is true).
        :param int perms:
            Number of permutations.
        :param str m:
            The method to use. E.g. 'lstsq'
        :param array_like s:
            The shape.
        :param bool | None u:
            Whether to force unique permutations.
        :return:
        """
        # permutations
        perm_inds = make_permutations(*d.shape, perms, u)

        di = np.zeros((perm_inds.shape[2], 1))

        # inds for subjects to pair with permutation inds
        col_inds = (np.arange(d.shape[1]) +
                    np.zeros((d.shape[1], d.shape[0])).T).astype(int)

        for perm in range(perm_inds.shape[2]):
            if (perm + 1) % 500 == 0:
                sys.stdout.write(
                    "\rPerforming permutation %d of %d" % (perm + 1,
                                                           perm_inds.shape[2]))
                sys.stdout.flush()

            tmp_d = d[perm_inds[:, :, perm], col_inds]

            # if shape is not predefined use shape obtained from average
            if s is None:
                tmp_s = make_rankshape(tmp_d, n)
            else:
                tmp_s = s
            di[perm] = get_stat(tmp_d, tmp_s, m)
        print('\ndone.')
        return di

    def get_p(ts, di, t):
        """Compute p value.

        :param array_like ts:
            The test statistic.
        :param array_like di:
            The permutation distribution.
        :param int t:
            Tail of the distribution.
        :return:
        """
        if t > 0:
            return np.mean(di > ts)
        else:
            return np.mean(di < ts)

    def get_thresh(di, a, t):
        """Compute threshold where significance is reached, given alpha.

        :param array_like di:
            The permutation distribution.
        :param float a:
            The alpha level.
        :param int t:
            The test tail (-1, 0 or 1). Default is 1.
        :return:
        """
        if t == 1:
            return np.percentile(di, (1 - a) * 100)
        elif t == -1:
            return np.percentile(di, a * 100)
        else:
            return np.percentile(di, [a * 100, (1 - a) * 100])

    # bring data array in correct orientation
    ref_data = data.T if shape_dim == 1 else data + 0

    # make permutation distribution
    distribution = get_distribution(ref_data, normalize_ranks, permutations,
                                    method, shape, unique_permutations)

    # use predefined shape or avg shape must be after permutation
    # distribution to keep shape=None and thus evoke the function to use the
    # shape from average
    if shape is None:
        shape = make_rankshape(ref_data, normalize_ranks)

    # obtain test statistic
    test_stat = get_stat(ref_data, shape, method)

    # set tail according to test statistic
    if tail is None:
        tail = 1 if np.mean(distribution) < test_stat else -1

    if method is 'sse':
        tail = -1

    # adjust alpha in case tail is 0
    if tail == 0:
        alpha /= 2

    # obtain p value
    pval = get_p(test_stat, distribution, tail)

    # make results
    stats = {
        'distribution': distribution,
        'permutations': permutations,
        'eta': test_stat,
        'tail': tail,
        'thresh': get_thresh(distribution, alpha, tail),
        'p': pval,
        'alpha': alpha,
        'h': pval < alpha,
        'shape': shape,
        'normalized': normalize_ranks,
        'method': method}

    return stats
