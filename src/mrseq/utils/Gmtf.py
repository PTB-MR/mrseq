"""Gradient Modulation Transfer Function (GMTF) estimation and correction."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pypulseq as pp
from mrpro.data import KData
from mrpro.data.traj_calculators import KTrajectoryCartesian
from pypulseq import eps
from scipy.interpolate import PPoly


def build_input_triangles(
    rise_times: Sequence[float],
    slew_rate: float,
    g_delay: float,
    enumerate_coeff: Sequence[float],
    dwell_time: float,
    n_samples: int,
) -> np.ndarray:
    """Build ideal triangular input waveforms.

    Parameters
    ----------
    rise_times
        Rise times of the triangular waveforms.
    slew_rate
        Slew rate used to construct the ramps.
    g_delay
        Gradient delay.
    enumerate_coeff
        Coefficients used to enumerate/scale the triangular waveforms.
    dwell_time
        Time between successive samples.
    n_samples
        Number of samples per waveform.

    Returns
    -------
    Array of shape ``(n_rise, n_samples)`` [T/m].
    """
    triangles = np.zeros((len(rise_times), n_samples))
    for i, rise_time in enumerate(rise_times):
        n_rise = round(rise_time / dwell_time)
        n_pre = round(g_delay / dwell_time)
        amplitude = enumerate_coeff[0] * slew_rate * rise_time

        slope_up = np.linspace(0.0, amplitude, n_rise + 1)
        slope_down = np.linspace(amplitude, 0.0, n_rise + 1)

        triangles[i, n_pre : n_pre + n_rise + 1] = slope_up
        triangles[i, n_pre + n_rise + 1 : n_pre + 2 * n_rise + 1] = slope_down[1:]

    return triangles


def _apply_gmtf_to_gradient_waveform(
    gradient_waveform: list[np.ndarray], gmtf: np.ndarray, frequency: np.ndarray
) -> list[np.ndarray]:
    """
    Apply Gmtf correction to a gradient waveform and return corrected gradient waveform.

    Parameters
    ----------
    gradient_waveform
        Input gradient waveform
    gmtf
        Complex-valued array containing the GMTF for the x-axis, y-axis and z-axis, of shape (3, N).
    frequency
        1D real-valued array of frequency values (Hz), of length N, corresponding to the frequency axis of  `gmtf`

    Raises
    ------
    ValueError
        If `gmtf` and `freq` do not have the same length.

    Returns
    -------
        Corrected gradient waveform
    """
    if gmtf.shape[-1] != frequency.shape[0]:
        raise ValueError(
            f'gmtf and freq must have the same shape along the last dimension but, got: {gmtf.shape[-1]} ',
            f'and {frequency.shape[0]}.',
        )

    gradient_waveform_corrected = []
    for grad_idx, grad in enumerate(gradient_waveform):
        if len(grad[1]):
            pad_factor = 2

            input_gradient_time = grad[0]
            input_gradient = grad[1]

            dt = 1 / frequency[-1] / 2

            # Build a uniform time grid covering the original (non-uniform) signal
            n = int(np.ceil((input_gradient_time[-1] - input_gradient_time[0]) / dt)) + 1
            uniform_time = input_gradient_time[0] + dt * np.arange(n)

            # Resample signal onto the uniform grid
            signal_uniform = np.interp(uniform_time, input_gradient_time, input_gradient, left=0, right=0)

            n_padded = n * pad_factor

            # FFT of the zero-padded signal
            signal_fft = np.fft.fft(signal_uniform, n=n_padded)
            signal_freq = np.fft.fftfreq(n_padded, d=dt)

            # Sort gmtf by frequency so interpolation x-values are ascending
            sort_idx = np.argsort(frequency)
            gmtf_frequency_sorted = frequency[sort_idx]
            gmtf_sorted = gmtf[grad_idx, sort_idx]

            # Interpolate GMTF (real and imaginary parts) onto the padded signal's frequency grid
            gmtf_interp_real = np.interp(signal_freq, gmtf_frequency_sorted, gmtf_sorted.real, left=0, right=0)
            gmtf_interp_imag = np.interp(signal_freq, gmtf_frequency_sorted, gmtf_sorted.imag, left=0, right=0)
            gmtf_interp = gmtf_interp_real + 1j * gmtf_interp_imag

            # Multiply in frequency domain (= convolution in time domain)
            output_fft = signal_fft * gmtf_interp

            # Back to time domain, crop to original signal length
            grad_corrected, grad_corrected_time = np.real(np.fft.ifft(output_fft))[..., :n], uniform_time[:n]

            gradient_waveform_corrected.append(np.stack((grad_corrected_time, grad_corrected)))
        else:
            gradient_waveform_corrected.append(grad)

    return gradient_waveform_corrected


def apply_gmtf_to_sequence(seq_file_or_object: str | Path | pp.Sequence, gmtf: 'Gmtf') -> list[np.ndarray]:
    """
    Apply Gmtf correction to all gradient waveforms in sequence and return corrected gradient waveform.

    Parameters
    ----------
    seq_file_or_object
        Path to the Pulseq (.seq) file or pulseq sequence object describing the nominal gradient waveforms that were
        played out during the measurement.
    gmtf
        Gmtf object

    Returns
    -------
        Corrected gradient waveform
    """
    if isinstance(seq_file_or_object, pp.Sequence):
        seq = seq_file_or_object
    else:
        seq = pp.Sequence()
        seq.read(str(seq_file_or_object))

    nominal_gradient_waveform = seq.waveforms()

    return _apply_gmtf_to_gradient_waveform(nominal_gradient_waveform, gmtf.gmtf, gmtf.frequency)


def convert_waveforms_to_ppoly(gw_data: list[np.ndarray]) -> Sequence[PPoly | None]:
    """
    Convert gradient waveforms into piecewise polynomial (PPoly) representations.

    Each waveform is padded with tiny zero-amplitude segments just before its start and after its end (to ensure clean
    extrapolation to zero outside the defined time range), and then converted into a linear piecewise polynomial
    using the time points as breakpoints.

    Parameters
    ----------
    gw_data
        List of gradient waveforms, one per gradient channel. Each waveform is expected to have shape (2, M): the first
        row containing time points (s) and the second row containing the corresponding gradient amplitudes.

    Returns
    -------
    List of piecewise polynomial representations, one per gradient channel, in the same order as `gw_data`.

    Raises
    ------
    ValueError
        If any waveform contains non-finite values (NaN or Inf).
    """
    n_grad_channels = len(gw_data)
    eps = 1e-12

    gw_pp: list[PPoly | None] = []
    for grad_idx in range(n_grad_channels):
        gw = gw_data[grad_idx]

        if len(gw[0]):
            if not np.all(np.isfinite(gw)):
                raise ValueError(f'Gradient channel {grad_idx}: not all elements of the generated waveform are finite.')

            # Pad with near-zero-amplitude points just before/after the waveform
            # so that extrapolation outside the defined range goes cleanly to zero.
            gw_time = gw[0].astype(np.float64)
            pre_pad = np.array([[gw_time[0] - 2 * eps, gw_time[0] - eps], [0, 0]])
            post_pad = np.array([[gw_time[-1] + eps, gw_time[-1] + 2 * eps], [0, 0]])
            gw = np.hstack((pre_pad, gw, post_pad))

            # Avoid signed-zero artifacts (-0.0) in the amplitude row
            gw[1][gw[1] == -0.0] = 0.0

            time = gw[0].astype(np.float64)
            amplitude = gw[1].astype(np.float64)
            slopes = np.diff(amplitude) / np.diff(time)

            gw_pp.append(PPoly(np.stack((slopes, amplitude[:-1])), time, extrapolate=True))
        else:
            gw_pp.append(None)

    return gw_pp


def calc_kspace_from_grad_waveforms(
    gw_pp: Sequence[PPoly | None], seq: pp.Sequence
) -> tuple[np.ndarray, np.ndarray, list[float], list[float], np.ndarray]:
    """Calculate k-space trajectory from gradient waveforms.

    This function is mainly a copy of pypulseq.Sequence.calculate_kspace but allows to calculate the k-space
    trajectory from a separately provided list of piecewise polynomial representations.

    Parameters
    ----------
    gw_pp
        Sequence of piecewise polynomial representations, one per gradient channel, in the same order as `gw_data`.
    seq
        Pulseq sequence object.

    Returns
    -------
    k_traj_adc
        K-space trajectory sampled at `t_adc` timepoints.
    k_traj
        K-space trajectory of the entire pulse sequence.
    t_excitation
        Excitation timepoints.
    t_refocusing
        Refocusing timepoints.
    t_adc
        Sampling timepoints.

    """
    # get timings from sequence
    total_duration = sum(seq.block_durations.values())
    t_excitation, _fp_excitation, t_refocusing, _ = seq.rf_times()
    t_adc, _ = seq.adc_times()

    ng = len(gw_pp)

    # Integrate waveforms as PPs to produce gradient moments
    gm_pp: list[PPoly | None] = []
    tc = []
    for i in range(ng):
        gw_i = gw_pp[i]
        if gw_i is None:
            gm_pp.append(None)
            continue

        gm_i = gw_i.antiderivative()
        gm_pp.append(gm_i)
        tc.append(gm_i.x)
        # "Sample" ramps for display purposes.  Otherwise piecewise-linear display (plot) fails
        ii = np.flatnonzero(np.abs(gm_i.c[0, :]) > 1e-7 * seq.system.max_slew)

        # Do nothing if there are no ramps
        if ii.shape[0] == 0:
            continue

        starts = np.floor((gm_i.x[ii] + eps) / seq.grad_raster_time).astype(np.int64)
        ends = np.ceil((gm_i.x[ii + 1] - eps) / seq.grad_raster_time).astype(np.int64)

        # Create all ranges starts[0]:ends[0], starts[1]:ends[1], etc.
        lengths = ends - starts + 1
        inds = np.ones((lengths).sum())
        # Calculate output index where each range will start
        start_inds = np.cumsum(np.concatenate(([0], lengths[:-1])))
        # Create element-wise differences that will cumsum into
        # the final indices: [starts[0], 1, 1, starts[1]-starts[0]-lengths[0]+1, 1, etc.]
        inds[start_inds] = np.concatenate(([starts[0]], np.diff(starts) - lengths[:-1] + 1))

        tc.append(np.cumsum(inds) * seq.grad_raster_time)
    tc_arr = np.concatenate(tc) if tc else np.array([])

    t_acc = 1e-10  # Temporal accuracy
    t_acc_inv = 1 / t_acc
    # tc = self.__flatten_jagged_arr(tc)
    t_ktraj = t_acc * np.unique(
        np.round(
            t_acc_inv
            * np.array(
                [
                    *tc_arr,
                    0,
                    *np.asarray(t_excitation) - 2 * seq.rf_raster_time,
                    *np.asarray(t_excitation) - seq.rf_raster_time,
                    *t_excitation,
                    *np.asarray(t_refocusing) - seq.rf_raster_time,
                    *t_refocusing,
                    *t_adc,
                    total_duration,
                ]
            )
        )
    )

    i_excitation = np.searchsorted(t_ktraj, t_acc * np.round(t_acc_inv * np.asarray(t_excitation)))
    i_refocusing = np.searchsorted(t_ktraj, t_acc * np.round(t_acc_inv * np.asarray(t_refocusing)))
    i_adc = np.searchsorted(t_ktraj, t_acc * np.round(t_acc_inv * np.asarray(t_adc)))

    i_periods = np.unique([0, *i_excitation, *i_refocusing, len(t_ktraj) - 1])
    if len(i_excitation) > 0:
        ii_next_excitation = 0
    else:
        ii_next_excitation = -1
    if len(i_refocusing) > 0:
        ii_next_refocusing = 0
    else:
        ii_next_refocusing = -1

    k_traj = np.zeros((ng, len(t_ktraj)))
    for n in range(ng):
        gm_n = gm_pp[n]
        if gm_n is None:
            continue

        it = np.where(
            np.logical_and(
                t_ktraj >= t_acc * round(t_acc_inv * gm_n.x[0]),
                t_ktraj <= t_acc * round(t_acc_inv * gm_n.x[-1]),
            )
        )[0]
        k_traj[n, it] = gm_n(t_ktraj[it])
        if t_ktraj[it[-1]] < t_ktraj[-1]:
            k_traj[n, it[-1] + 1 :] = k_traj[i, it[-1]]

    # Convert gradient moments to k-space positions
    dk = -k_traj[:, 0]
    for i in range(len(i_periods) - 1):
        i_period = i_periods[i]
        i_period_end = i_periods[i + 1]
        if ii_next_excitation >= 0 and i_excitation[ii_next_excitation] == i_period:
            if abs(t_ktraj[i_period] - t_excitation[ii_next_excitation]) > t_acc:
                raise Warning(
                    f'abs(t_ktraj[i_period]-t_excitation[ii_next_excitation]) < {t_acc} failed for ',
                    f'ii_next_excitation={ii_next_excitation} ',
                    f'error={t_ktraj[i_period] - t_excitation[ii_next_excitation]}',
                )
            dk = -k_traj[:, i_period]
            if i_period > 0:
                # Use nans to mark the excitation points since they interrupt the plots
                k_traj[:, i_period - 1] = np.nan
            # -1 on len(i_excitation) for 0-based indexing
            ii_next_excitation = min(len(i_excitation) - 1, ii_next_excitation + 1)
        elif ii_next_refocusing >= 0 and i_refocusing[ii_next_refocusing] == i_period:
            # dk = -k_traj[:, i_period]
            dk = -2 * k_traj[:, i_period] - dk
            # -1 on len(i_excitation) for 0-based indexing
            ii_next_refocusing = min(len(i_refocusing) - 1, ii_next_refocusing + 1)

        k_traj[:, i_period:i_period_end] = k_traj[:, i_period:i_period_end] + dk[:, None]

    k_traj[:, i_period_end] = k_traj[:, i_period_end] + dk
    k_traj_adc = k_traj[:, i_adc]

    return k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc


def estimate_gmtf(
    grad_input: np.ndarray,
    grad_output_mean: np.ndarray,
) -> np.ndarray:
    """Least-squares GMTF estimate over all rise times.

    Parameters
    ----------
    grad_input:
        Ideal waveforms, shape ``(n_rise, n_samples)``.
    grad_output_mean:
        Measured waveforms, shape ``(n_axes, n_rise, n_samples-1)``.

    Returns
    -------
    Complex tensor of shape ``(n_axes, n_freq)``.
    """
    n_fft_in = 2 * (grad_input.shape[-1])
    n_fft_out = 2 * grad_output_mean.shape[-1]

    in_spec = np.fft.fftshift(np.fft.fft(grad_input, n=n_fft_in, axis=-1), axes=-1)
    out_spec = np.fft.fftshift(np.fft.fft(grad_output_mean, n=n_fft_out, axis=-1), axes=-1)

    # Least-squares: sum_rt conj(H_in) * H_out  /  sum_rt |H_in|^2
    numerator = np.sum(in_spec.conj() * out_spec, axis=-2)  # sum over rise dim
    denominator = np.sum(np.abs(in_spec) ** 2, axis=-2)
    return numerator / denominator


class Gmtf:
    """
    Gradient Modulation Transfer Function (GMTF) container and utilities.

    The GMTF describes the frequency-domain response of the gradient system for each physical axis (x, y, z), and is
    related to the Gradient Impulse Response Function (GIRF) via GMTF = FT(GIRF).

    Attributes
    ----------
    gmtf
        Complex-valued array containing the concatenated GMTF for the x, y, and z axes (in that order),
        with shape (3, N), where N is the number of frequency samples.
    frequency
        1D array of frequency values (Hz) corresponding to the samples in `gmtf`, of length N.
    grad_input
        Input gradient triangular waveforms with shape `(n_rise_times n_adc_samples)` (read only)
    grad_output
        Measured output gradient waveforms with shape `(n_rise_times n_adc_samples)` (read only)
    """

    __slots__ = ('_grad_input', '_grad_output', 'frequency', 'gmtf')

    def __init__(self, gmtf_x: np.ndarray, gmtf_y: np.ndarray, gmtf_z: np.ndarray, frequency: np.ndarray) -> None:
        """
        Initialize the GMTF container.

        Parameters
        ----------
        gmtf_x
            Complex-valued 1D array containing the GMTF for the x-axis, of length N.
        gmtf_y
            Complex-valued 1D array containing the GMTF for the y-axis, of length N.
        gmtf_z
            Complex-valued 1D array containing the GMTF for the z-axis, of length N.
        frequency
            1D real-valued array of frequency values (Hz), of length N, corresponding to the frequency axis of
            `gmtf_x`, `gmtf_y`, and`gmtf_z`.

        Raises
        ------
        ValueError
            If `gmtf_x`, `gmtf_y`, `gmtf_z`, and `freq` are not all 1D arrays of the same length.

        Notes
        -----
        Sets `self.gmtf` as the concatenation (stacking) of `gmtf_x`, `gmtf_y`, and `gmtf_z`, in that order,
        along a new leading dimension, resulting in shape (3, N).
        """
        lengths = {
            'gmtf_x': gmtf_z.shape[0],
            'gmtf_y': gmtf_y.shape[0],
            'gmtf_z': gmtf_x.shape[0],
            'freq': frequency.shape[0],
        }
        if len(set(lengths.values())) > 1:
            raise ValueError(f'gmtf_x, gmtf_y, gmtf_z, and freq must all have the same length, got: {lengths}.')

        self.gmtf = np.stack((gmtf_x, gmtf_y, gmtf_z))
        self.frequency = frequency
        self._grad_input: np.ndarray | None = None
        self._grad_output: np.ndarray | None = None

    @property
    def grad_input(self):
        """Nominal gradient triangles."""
        return self._grad_input

    @property
    def grad_output(self):
        """Measured gradient waveforms."""
        return self._grad_output

    @classmethod
    def compute_gmtf(cls, mrd_file: str | Path, seq_file: str | Path) -> 'Gmtf':
        """
        Compute the GMTF from a measured MRD dataset and the corresponding Pulseq sequence file used to acquire it.

        Parameters
        ----------
        mrd_file
            Path to the measured raw data file in MRD (.mrd/.h5) format, containing the measured gradient response
            (e.g. field-camera or phantom-based measurement) for each axis.
        seq_file
            Path to the Pulseq (.seq) file describing the nominal gradient waveforms that were played out during the
            measurement.

        Returns
        -------
        Gmtf
            A new GMTF instance containing the computed gradient modulation transfer function and associated frequency
            vector for the x, y, and z axes.
        """
        # Load k-space data, sort into correct dimensions and combine to single coil
        kdata = KData.from_file(mrd_file, trajectory=KTrajectoryCartesian())
        idx = np.lexsort(
            (
                kdata.header.acq_info.idx.phase.squeeze(),
                kdata.header.acq_info.idx.repetition.squeeze(),
                kdata.header.acq_info.idx.average.squeeze(),
            )
        )
        kdata_sorted = kdata[idx.tolist()]
        kdata_sorted = kdata_sorted.rearrange(
            '(avg rep ph) ... -> avg rep ph ...',
            rep=int(kdata.header.acq_info.idx.repetition.max()) + 1,
            avg=int(kdata.header.acq_info.idx.average.max()) + 1,
            ph=int(kdata.header.acq_info.idx.phase.max()) + 1,
        )
        kdata_single_coil = kdata_sorted.compress_coils(n_compressed_coils=1).data.squeeze().numpy()

        # Get additional information from sequence
        sequence = pp.Sequence()
        sequence.read(seq_file)

        dwell_time = sequence.get_definition('DwellTime')
        slice_pos = sequence.get_definition('SlicePos')
        gamma = sequence.system.gamma * 2 * np.pi
        rise_times = sequence.get_definition('RiseTimes')
        slew_rate = sequence.get_definition('SlewRate') / sequence.system.gamma
        g_delay = sequence.get_definition('GradientPreEmphasisDelay')
        g_amplitude_coeff = sequence.get_definition('GradAmplitudeCoeff')

        # Unwrape phase
        phase_difference = np.angle(kdata_single_coil[:, 0, ...]) - np.angle(kdata_single_coil[:, 1, ...])
        phase_difference = np.unwrap(np.unwrap(phase_difference))
        phase_mean = phase_difference.mean(axis=0)

        # Calculate gradient from phase
        scale = slice_pos * gamma * dwell_time
        grad_output_mean = np.diff(phase_mean, axis=-1) / scale
        # Shift by one to account for shift due to difference calculation
        grad_output_mean = np.roll(grad_output_mean, 1, axis=-1)

        # Calculate ideal triangles
        grad_input = build_input_triangles(
            rise_times, slew_rate, g_delay, g_amplitude_coeff, dwell_time, grad_output_mean.shape[-1]
        )

        # Check for any sign flips
        corr = np.sum(grad_input * grad_output_mean)
        grad_output_mean = grad_output_mean if corr >= 0 else grad_output_mean * -1

        gmtf = estimate_gmtf(grad_input, grad_output_mean)
        frequency = (1.0 / dwell_time) * np.arange(-gmtf.shape[-1] // 2, gmtf.shape[-1] // 2) / (gmtf.shape[-1])
        gmtf_obj = Gmtf(gmtf_x=gmtf[0, :], gmtf_y=gmtf[1, :], gmtf_z=gmtf[2, :], frequency=frequency)
        gmtf_obj._grad_output = grad_output_mean
        gmtf_obj._grad_input = grad_input
        return gmtf_obj

    def plot(
        self,
        frequency_lim: tuple[float, float] | None = None,
        amplitude_lim: tuple[float, float] | None = None,
        phase_lim: tuple[float, float] | None = None,
    ) -> None:
        """
        Plot the magnitude and phase of the GMTF for each gradient axis.

        Parameters
        ----------
        frequency_lim
            (min, max) frequency range (Hz) to display on the x-axis of both subplots. If None, the full frequency
            range is shown.
        amplitude_lim
            (min, max) limits for the amplitude/magnitude subplot y-axis. If None, limits are chosen automatically.
        phase_lim
            (min, max) limits for the phase subplot y-axis (in radians). If None, limits are chosen automatically.

        """
        if frequency_lim is None:
            frequency_lim = (0.0, 10.0)
        if amplitude_lim is None:
            amplitude_lim = (0.8, 1.025)
        if phase_lim is None:
            phase_lim = (-0.1, np.pi / 4)

        # Plot magnitude and phase of the GMTF for all axes
        grad_axes = ['x', 'y', 'z']

        half = len(self.frequency) // 2

        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(10, 5))
        for k, label in enumerate(grad_axes):
            ax1.plot(self.frequency[half:] * 1e-3, abs(self.gmtf[k, half:]), label=label)
            ax2.plot(self.frequency[half:] * 1e-3, np.unwrap(np.angle(self.gmtf[k, half:])), label=label)

        ax1.set(xlabel='Frequency (kHz)', title='GMTF Magnitude', xlim=frequency_lim, ylim=amplitude_lim)
        ax2.set(xlabel='Frequency (kHz)', title='GMTF Phase', xlim=frequency_lim, ylim=phase_lim)

        for ax in (ax1, ax2):
            ax.legend()
            ax.grid(visible=True, which='major', color='#666666')
            ax.minorticks_on()
            ax.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

        fig.tight_layout()
        plt.show()

    def correct_gradients(
        self, seq_file_or_object: str | Path | pp.Sequence, freq_threshold: float | None = None
    ) -> np.ndarray:
        """
        Apply GMTF-based correction to the gradient waveforms in a Pulseq sequence file, and output new trajectory.

        Parameters
        ----------
        seq_file
            Path to the input Pulseq (.seq) file whose gradient waveform (x, y, z) should be corrected.
        freq_threshold
            Frequency (Hz) above which the GMTF-based correction is not applied (e.g. to avoid amplifying noise or
            unreliable GMTF estimates at high frequencies).

        Returns
        -------
            Corrected trajectory.
        """
        if isinstance(seq_file_or_object, pp.Sequence):
            seq = seq_file_or_object
        else:
            seq = pp.Sequence()
            seq.read(str(seq_file_or_object))

        if freq_threshold:
            print('Not implemented')
        gradient_waveform_corrected = apply_gmtf_to_sequence(seq, self)
        gradient_waveform_pp_corrected = convert_waveforms_to_ppoly(gradient_waveform_corrected)

        k_traj_adc, _k_traj, _t_excitation, _t_refocusing, _t_adc = calc_kspace_from_grad_waveforms(
            gradient_waveform_pp_corrected, seq
        )

        return k_traj_adc
