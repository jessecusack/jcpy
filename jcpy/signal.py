"Signal processing module."


from itertools import groupby as _groupby

import numpy as _np
import scipy.signal as _sig


def _gap_sizes_1D(y):
    z = []
    for a, b in _groupby(_np.isnan(y).astype(int), lambda x: x == 0):
        if a:
            z.extend(list(b))
        else:  # Where the value is one, replace 1 with the number of sequential 1's
            l = len(list(b))
            z.extend([l] * l)

    return _np.asarray(z)


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
        gap_sizes : ndarray
            Array of gaps, labelled by size, with the same shape as y.
    """

    return _np.apply_along_axis(_gap_sizes_1D, axis, _np.asarray(y))


def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""
    # Stole this off stack exchange...
    # https://stackoverflow.com/a/4495197
    # Find the indicies of changes in "condition"
    d = _np.diff(condition)
    (idx,) = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = _np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = _np.r_[idx, condition.size]

    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx


def _butter(cutoff, fs=1.0, btype="low", order=4):
    """Return Butterworth filter coefficients. See scipy.signal.butter for a
    more thorough documentation.

    Parameters
    ----------
    cutoff : array_like
        Cutoff frequency, e.g. roughly speaking, the frequency at which the
        filter acts. Units should be same as for fs paramter.
    fs : float, optional
        Sampling frequency of signal. Units should be same as for cutoff
        parameter. Default is 1.0.
    btype : {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}, optional
        Default is 'low'.
    order : int, optional
        Default is 4. The order of the Butterworth filter.

    Returns
    -------
    sos : ndarray
        Filter coefficients.

    """
    cutoff = _np.asarray(cutoff)
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    sos = _sig.butter(order, normal_cutoff, btype=btype, analog=False, output="sos")
    return sos


def butter_filter(y, cutoff, fs=1.0, btype="low", order=4, **kwargs):
    """Apply Butterworth filter to data using scipy.signal.sosfiltfilt.

    Parameters
    ----------
    y : array_like
        The data to be filtered. Must be evenly sampled.
    cutoff : array_like
        Cutoff frequency, e.g. roughly speaking, the frequency at which the
        filter acts. Units should be same as for fs paramter.
    fs : float, optional
        Sampling frequency of signal. Units should be same as for cutoff
        parameter. Default is 1.0.
    btype : {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}, optional
        Default is 'low'.
    order : int, optional
        Default is 4. The order of the Butterworth filter.
    kwargs : optional
        Additional key word arguments passed to sosfiltfilt.

    Returns
    -------
    y_filt : ndarray
        The filtered data.

    """
    sos = _butter(cutoff, fs=fs, btype=btype, order=order)
    y_filt = _sig.sosfiltfilt(sos, _np.asarray(y), **kwargs)
    return y_filt


def nan_butter_filter(
    y, cutoff, fs=1.0, axis=1, btype="low", order=4, dic=20, **kwargs
):
    """Apply Butterworth filter to data using scipy.signal.sosfiltfilt
    along the given axis. It can skip over some NaN regions.

    Parameters
    ----------
    y : array_like
        The data to be filtered. Must be evenly sampled.
    cutoff : array_like
        Cutoff frequency, e.g. roughly speaking, the frequency at which the
        filter acts. Units should be same as for fs paramter.
    fs : float, optional
        Sampling frequency of signal. Units should be same as for cutoff
        parameter. Default is 1.0.
    axis : int, optional
        Axis along which to perform operation, default is 1.
    btype : {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}, optional
        Default is 'low'.
    order : int, optional
        Default is 4. The order of the Butterworth filter.
    dic : int, optional
        Smallest contiguous region size, in number of data points, over which
        to perform the filtering. Default is 20.
    kwargs : optional
        Additional key word arguments passed to sosfiltfilt.

    Returns
    -------
    y_filt : ndarray
        The filtered data.

    """
    # TODO: determine dic from fs and cutoff.
    # TODO: Have a second option of filling NaN by interpolation before filtering.
    def discontinuous_filter(x, cutoff, fs, btype, order, dic, **kwargs):
        nans = _np.isnan(x)
        if nans.any():
            x_filt = _np.full_like(x, _np.nan)
            idxs = contiguous_regions(~nans)
            di = idxs[:, 1] - idxs[:, 0]
            iidxs = _np.argwhere(di > dic)
            for j in iidxs[:, 0]:

                sl = slice(*idxs[j, :])
                x_filt[sl] = butter_filter(x[sl], cutoff, fs, btype, **kwargs)
        else:
            x_filt = butter_filter(x, cutoff, fs, btype, **kwargs)

        return x_filt

    y_filt = _np.apply_along_axis(
        discontinuous_filter,
        axis,
        _np.asarray(y),
        cutoff,
        fs,
        btype,
        order,
        dic,
        **kwargs
    )

    return y_filt


def nan_butter_filter_renan(
    y, cutoff, fs=1.0, axis=1, btype="low", order=4, dic=20, **kwargs
):
    """See documentation for nan_butter_filter. This function adds the NaN
    values back after filtering."""
    # TODO: Make a decorator for this
    nans = _np.isnan(y)
    y_filt = nan_butter_filter(y, cutoff, fs, axis, btype, order, dic, **kwargs)
    y_filt[nans] = _np.nan
    return y_filt


def _interp_nans_1D(x, y, left=None, right=None):
    nans = _np.isnan(y)
    yi = y.copy()
    yi[nans] = _np.interp(x[nans], x[~nans], y[~nans], left, right)
    return yi


def interp_nans(x, y, y_fill=None, left=None, right=None, axis=0):
    """Fill NaNs in data by linear interpolation.

    Parameters
    ----------
    x : array_like of floats
        The x-coordinates of the interpolated values. Only 1 or 2 dimensional
        arrays are supported.
    y : array_like of floats
        The data that will be interpolated. Only 1 or 2 dimensional arrays are
        supported.
    y_fill : float, optional
        Fill value for rows or columns that are all NaN. Default is leave NaN.
    left : optional float or complex corresponding to fp
        Value to return for `x < xp[0]`, default is `fp[0]`.
    right : optional float or complex corresponding to fp
        Value to return for `x > xp[-1]`, default is `fp[-1]`.
    axis : [-1, 0, 1], int
        The axis along which to perform the interpolation. Default is 0.

    Returns
    -------
    yi : ndarray
        The interpolated array. Same size as y
    """
    if axis not in [-1, 0, 1]:
        raise ValueError("The axis may be only -1, 0 or 1.")

    ndimy = _np.ndim(y)
    ndimx = _np.ndim(x)

    if ndimy > 2 or ndimx > 2:
        raise ValueError("Only 1 or 2 dimensional arrays are supported.")

    if ndimx > ndimy:
        raise ValueError("The x input cannot have more dimensions than y.")

    # TODO: Simplify this using apply_along_axis (I can't quite figure out how though)
    if ndimy == 1:
        yi = _interp_nans_1D(x, y, left, right)
    if ndimy == 2:
        yi = _np.swapaxes(y.copy(), axis, 0)
        _, nc = yi.shape

        for j in range(nc):
            nanr = _np.isnan(yi[:, j])
            if not nanr.any():
                continue
            if nanr.all() and y_fill is not None:
                yi[:, j] = y_fill
                continue
            if ndimx == 2:
                yi[:, j] = _interp_nans_1D(x[:, j], yi[:, j], left, right)
            else:
                yi[:, j] = _interp_nans_1D(x, yi[:, j], left, right)

        yi = _np.swapaxes(yi, 0, axis)

    return yi
