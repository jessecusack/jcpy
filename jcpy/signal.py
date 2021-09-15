"Signal processing module."


from itertools import groupby as _groupby

import numpy as np


def _gap_sizes_1D(y):
    z = []
    for a, b in _groupby(np.isnan(y).astype(int), lambda x: x == 0):
        if a:
            z.extend(list(b))
        else:  # Where the value is one, replace 1 with the number of sequential 1's
            l = len(list(b))
            z.extend([l] * l)

    return np.asarray(z)


def gap_sizes(y, axis=0):
    """Find the size of NaN gaps along a given axis.

    Gaps are assigned an integer number denoting their size, e.g.

    [1, NaN, 3, 4, NaN, NaN, 7, 8] -> [0, 1, 0, 0, 2, 2, 0, 0]

    Parameters
    ----------
        y : array_like
            Input data.
        axis : int
            Axis along which to operate.
    Returns
    -------
        gap_sizes : numpy.ndarray
            Array of gaps, labelled by size, with the same shape as y.
    """

    return np.apply_along_axis(_gap_sizes_1D, axis, np.asarray(y))
