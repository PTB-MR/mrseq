"""Helper function to get interpolate gradient waveforms."""

import numpy as np
import pypulseq as pp


def get_interp_waveform_for_gx_gy(seq: pp.Sequence, dt: np.ndarray | None = None, scale: float = 1.0):
    """Interpolate gradient waveforms for the x and y axes.

    Parameters
    ----------
    seq
        The PyPulseq sequence object containing gradient waveforms.
    dt
        Desired time points for interpolation. If None, a default time array is generated.
    scale
        Scaling factor for the gradient waveforms. Default is 1.

    Returns
    -------
    gx_waveform_intp
        Interpolated gradient waveform for the x-axis.
    gy_waveform_intp
        Interpolated gradient waveform for the y-axis.
    dt
        Time points corresponding to the interpolated waveforms.
    """
    w = seq.waveforms_and_times()
    gx_waveform = w[0][0][1] * scale
    gx_waveform_time = w[0][0][0]

    gy_waveform = w[0][1][1] * scale
    gy_waveform_time = w[0][1][0]

    if dt is None:
        dt = np.arange(
            min(gx_waveform_time[0], gy_waveform_time[0]), max(gx_waveform_time[-1], gy_waveform_time[-1]), step=1e-7
        )
    gx_waveform_intp = np.interp(dt, gx_waveform_time, gx_waveform)
    gy_waveform_intp = np.interp(dt, gy_waveform_time, gy_waveform)

    return gx_waveform_intp, gy_waveform_intp, dt
