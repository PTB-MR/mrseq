"""Helper functions for the creation of sequences."""

from typing import Literal

import numpy as np


def round_to_raster(value: float, raster_time: float, method: Literal['floor', 'round', 'ceil'] = 'round') -> float:
    """Round a value to the given raster time using the defined method.

    Parameters
    ----------
    value
        Value to be rounded.
    raster_time
        Raster time, e.g. gradient, rf or ADC raster time.
    method
        Rounding method. Options: "floor", "round", "ceil".

    Returns
    -------
    rounded_value
        Rounded value.
    """
    if method == 'floor':
        return raster_time * np.floor(value / raster_time)
    elif method == 'round':
        return raster_time * np.round(value / raster_time)
    elif method == 'ceil':
        return raster_time * np.ceil(value / raster_time)
    else:
        raise ValueError(f'Unknown rounding method: {method}. Expected: "floor", "round" or "ceil".')


def find_gx_flat_time_on_adc_raster(
    n_readout, adc_dwell_time, grad_raster_time, adc_raster_time, max_m=10000, tol=1e-9
):
    """Return flat time of readout gradient on gradient raster with adc dwell time on adc raster.

    For a given number of readout points n_readout we have:

    gx_flat_time = n_readout * adc_dwell_time

    In the following we try to find a pair of gx_flat_time and adc_dwell_time which full-fills the above
    equation and the conditions that gx_flat_time is an integer multiple of the gradient raster time:

    gx_flat_time = n_gx * grad_raster_time

    and that adc_dwell_time is an integer multiple of the adc raster time:

    adc_dwell_time = n_adc * adc_raster_time

    Parameters
    ----------
    n_readout
        Number of readout samples
    adc_dwell_time
        Ideal adc dwell time, does not have to be on adc raster
    grad_raster_time
        Gradient raster time
    adc_raster_time
        Adc raster time
    max_m
        Highest integer multiple to look for. max_m * adc_raster_time gives largest possible adc_dwell_time
    tol
        Tolerance of how close values have to be to an integer

    Returns
    -------
    gx_flat_time
        gx_flat_time on gradient raster
    adc_dwell_time
        Adc dwell time matching gx_flat_time / n_readout and on adc raster
    """
    raster_time_ratio = (n_readout * adc_raster_time) / grad_raster_time
    start_m = int(np.floor(adc_dwell_time / adc_raster_time))
    # We look for smaller adc_dwell_times
    adc_dwell_time_smaller = None
    for m in np.arange(start_m, 1, -1):
        k = m * raster_time_ratio
        if np.isclose(k, np.round(k), atol=tol):  # Check if k is "close enough" to an integer
            adc_dwell_time_smaller = m * adc_raster_time
            break
    adc_dwell_time_larger = None
    for m in range(start_m, max_m):
        k = m * raster_time_ratio
        if np.isclose(k, np.round(k), atol=tol):  # Check if k is "close enough" to an integer
            adc_dwell_time_larger = m * adc_raster_time
            break
    if adc_dwell_time_larger is None and adc_dwell_time_smaller is None:
        raise ValueError('No adc_dwell_time found within search range.')

    # Select value which is closer to original adc_dwell_time
    if adc_dwell_time_smaller is None:
        adc_dwell_time = adc_dwell_time_larger
    elif adc_dwell_time_larger is None:
        adc_dwell_time = adc_dwell_time_smaller
    else:
        if np.abs(adc_dwell_time - adc_dwell_time_smaller) < np.abs(adc_dwell_time - adc_dwell_time_larger):
            adc_dwell_time = adc_dwell_time_smaller
        else:
            adc_dwell_time = adc_dwell_time_larger

    return adc_dwell_time * n_readout, adc_dwell_time
