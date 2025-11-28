"""Basic functionality for trajectory calculation."""

from typing import Literal

import numpy as np
import pypulseq as pp

from mrseq.utils import find_gx_flat_time_on_adc_raster
from mrseq.utils import round_to_raster
from mrseq.utils import sys_defaults
from mrseq.utils import variable_density_spiral_trajectory


def cartesian_phase_encoding(
    n_phase_encoding: int,
    acceleration: int = 1,
    n_fully_sampled_center: int = 0,
    sampling_order: Literal['linear', 'low_high', 'high_low', 'random'] = 'linear',
    n_phase_encoding_per_shot: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate Cartesian sampling trajectory.

    Parameters
    ----------
    n_phase_encoding
        number of phase encoding points before undersampling
    acceleration
        undersampling factor
    n_fully_sampled_center
        number of phsae encoding points in the fully sampled center. This will reduce the overall undersampling factor.
    sampling_order
        order how phase encoding points are sampled
    n_phase_encoding_per_shot
        used to ensure that all phase encoding points can be acquired in an integer number of shots. If None, this
        parameter is ignored, i.e. equal to n_phase_encoding_per_shot = 1
    """
    if sampling_order == 'random':
        # Linear order of a fully sampled kpe dimension. Undersampling is done later.
        kpe = np.arange(0, n_phase_encoding)
    else:
        # Always include k-space center and more points on the negative side of k-space
        kpe_pos = np.arange(0, n_phase_encoding // 2, acceleration)
        kpe_neg = -np.arange(acceleration, n_phase_encoding // 2 + 1, acceleration)
        kpe = np.concatenate((kpe_neg, kpe_pos), axis=0)

    # Ensure fully sampled center
    kpe_fully_sampled_center = np.arange(
        -n_fully_sampled_center // 2, -n_fully_sampled_center // 2 + n_fully_sampled_center
    )
    kpe = np.unique(np.concatenate((kpe, kpe_fully_sampled_center)))

    # Always acquire more to ensure desired resolution
    if n_phase_encoding_per_shot and sampling_order != 'random':
        kpe_extended = np.arange(-n_phase_encoding, n_phase_encoding)
        kpe_extended = kpe_extended[np.argsort(np.abs(kpe_extended), kind='stable')]
        idx = 0
        while np.mod(len(kpe), n_phase_encoding_per_shot) > 0:
            kpe = np.unique(np.concatenate((kpe, (kpe_extended[idx],))))
            idx += 1

    # Different temporal orders of phase encoding points
    if sampling_order == 'random':
        perm = np.random.permutation(kpe)
        npe = len(perm) // acceleration
        if n_phase_encoding_per_shot:
            npe += n_phase_encoding_per_shot - np.mod(npe, n_phase_encoding_per_shot)
        kpe = kpe[perm[:npe]]
    elif sampling_order == 'linear':
        kpe = np.sort(kpe)
    elif sampling_order == 'low_high':
        sort_idx = np.argsort(np.abs(kpe), kind='stable')
        kpe = kpe[sort_idx]
    elif sampling_order == 'high_low':
        sort_idx = np.argsort(-np.abs(kpe), kind='stable')
        kpe = kpe[sort_idx]
    else:
        raise ValueError(f'sampling order {sampling_order} not supported.')

    return kpe, kpe_fully_sampled_center


class MultiEchoAcquisition:
    """
    Multi-echo gradient echo acquisition.

    Attributes
    ----------
    system
        PyPulseq system limits object.
    n_readout_post_echo
        Number of readout points after echo.
    n_readout_pre_echo
        Number of readout points before echo.
    n_readout_with_partial_echo
        Total number of readout points with partial echo.
    te_delay
        Additional delay after readout gradient gx to achieve desired delta echo time.
    adc
        ADC event object.
    gx
        Readout gradient object.
    gx_pre
        Pre-winder gradient object.
    gx_post
        Re-winder gradient object.
    gx_between
        Gradient between echoes.
    """

    def __init__(
        self,
        system: pp.Opts | None = None,
        delta_te: float | None = None,
        fov: float = 0.256,
        n_readout: int = 128,
        readout_oversampling: float = 2.0,
        partial_echo_factor: float = 0.7,
        gx_flat_time: float = 2.0e-3,
        gx_pre_duration: float = 0.8e-3,
    ):
        """
        Initialize the MultiEchoAcquisition class and compute all required attributes.

        Parameters
        ----------
        system
            PyPulseq system limits object.
        delta_te
            Desired echo spacing (in seconds). Minimum echo spacing is used if set to None.
        fov
            Field of view in x direction (in meters).
        n_readout
            Number of frequency encoding steps.
        readout_oversampling
            Readout oversampling factor.
        partial_echo_factor
            Partial echo factor.
        gx_flat_time
            Flat time of the readout gradient.
        gx_pre_duration
            Duration of readout pre-winder gradient.
        """
        # set system to default if not provided
        self._system = sys_defaults if system is None else system

        delta_k = 1 / (fov * readout_oversampling)
        self._n_readout_post_echo = int(n_readout * readout_oversampling / 2 - 1)
        self._n_readout_post_echo += np.mod(self._n_readout_post_echo + 1, 2)  # make odd
        self._n_readout_pre_echo = int(
            (n_readout * partial_echo_factor * readout_oversampling) - self._n_readout_post_echo - 1
        )
        self._n_readout_pre_echo += np.mod(self._n_readout_pre_echo, 2)  # make even

        self._n_readout_with_partial_echo = self._n_readout_pre_echo + 1 + self._n_readout_post_echo
        gx_flat_area = self._n_readout_with_partial_echo * delta_k

        # adc dwell time has to be on adc raster and gx flat time on gradient raster
        self._gx_flat_time, _ = find_gx_flat_time_on_adc_raster(
            self._n_readout_with_partial_echo,
            gx_flat_time / self._n_readout_with_partial_echo,
            self._system.grad_raster_time,
            self._system.adc_raster_time,
        )

        self._gx = pp.make_trapezoid(
            channel='x', flat_area=gx_flat_area, flat_time=self._gx_flat_time, system=self._system
        )

        self._adc = pp.make_adc(
            num_samples=self._n_readout_with_partial_echo,
            duration=self._gx.flat_time,
            delay=self._gx.rise_time,
            system=self._system,
        )

        self._gx_pre = pp.make_trapezoid(
            channel='x',
            area=-(self._gx.amplitude * self._gx.rise_time / 2 + delta_k * (self._n_readout_pre_echo + 0.5)),
            duration=gx_pre_duration * partial_echo_factor,
            system=self._system,
        )
        self._gx_post = pp.make_trapezoid(
            channel='x',
            area=-(self._gx.amplitude * self._gx.fall_time / 2 + delta_k * (self._n_readout_post_echo + 0.5)),
            duration=gx_pre_duration,
            system=self._system,
        )

        self._gx_between = pp.make_trapezoid(
            channel='x',
            area=self._gx_pre.area - self._gx_post.area,
            duration=gx_pre_duration,
            system=self._system,
        )

        min_delta_te = pp.calc_duration(self._gx) + pp.calc_duration(self._gx_between)
        if delta_te is None:
            self._te_delay = 0.0
        else:
            self._te_delay = round_to_raster(delta_te - min_delta_te, self._system.block_duration_raster)
            if not self._te_delay >= 0:
                raise ValueError(
                    f'Delta TE must be larger than {min_delta_te * 1000:.2f} ms. '
                    f'Current value is {delta_te * 1000:.2f} ms.'
                )

    def add_to_seq(self, seq: pp.Sequence, n_echoes: int) -> tuple[pp.Sequence, list[float]]:
        """Add all gradients and adc to sequence.

        Parameters
        ----------
        seq
            PyPulseq Sequence object.
        n_echoes
            Number of echoes

        Returns
        -------
        seq
            PyPulseq Sequence object.
        time_to_echoes
            Time from beginning of sequence to echoes.
        """
        # readout pre-winder
        seq.add_block(self._gx_pre)

        # add readout gradients and ADCs
        seq, time_to_echoes = self.add_to_seq_without_pre_post_gradient(seq, n_echoes)

        # readout re-winder
        seq.add_block(self._gx_post)

        return seq, time_to_echoes

    def add_to_seq_without_pre_post_gradient(self, seq: pp.Sequence, n_echoes: int) -> tuple[pp.Sequence, list[float]]:
        """Add readout gradients without pre- and re-winder gradients.

        Often the pre- and re-winder gradients are played out at the same time as phase encoding gradients or spoiler
        gradients.

        Parameters
        ----------
        seq
            PyPulseq Sequence object.
        n_echoes
            Number of echoes

        Returns
        -------
        seq
            PyPulseq Sequence object.
        time_to_echoes
            Time from beginning of sequence to echoes.
        """
        # add readout gradient and ADC
        time_to_echoes = []
        for echo_ in range(n_echoes):
            start_of_current_gx = sum(seq.block_durations.values())
            gx_sign = (-1) ** echo_
            labels = []
            labels.append(pp.make_label(type='SET', label='REV', value=gx_sign == -1))
            labels.append(pp.make_label(label='REV', type='SET', value=gx_sign == -1))
            labels.append(pp.make_label(label='ECO', type='SET', value=echo_))
            seq.add_block(pp.scale_grad(self._gx, gx_sign), self._adc, *labels)
            time_to_echoes.append(
                start_of_current_gx + self._adc.delay + self._n_readout_pre_echo * self._adc.dwell + self._adc.dwell / 2
            )
            start_of_current_gx = sum(seq.block_durations.values())
            if echo_ < n_echoes - 1:
                if self._te_delay > 0:
                    seq.add_block(pp.make_delay(self._te_delay))
                seq.add_block(pp.scale_grad(self._gx_between, -gx_sign))

        return seq, time_to_echoes


def undersampled_variable_density_spiral(
    system: pp.Opts, n_readout: int, fov: float, undersampling_factor: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float, float]:
    """Create undersampled variable density spiral.

    The distribution of the k-space points of a spiral trajectory are restricted by the maximum gradient amplitude and
    slew rate. This makes an analytic solution for a given undersampling factor challenging. Here we use an iterative
    approach in order to achieve a variable density spiral with a certain number of readout samplings and undersampling
    factor.

    During the iterative search, the undersampling for the edge of k-space is increased. If this is not enough, then we
    also start to increase the undersampling in the k-space center. The field-of-view varies linearly bewtween the
    k-space center and k-space edge.

    If the undersampling factor is to high, it might not be possible to find a suitable solution.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    n_readout
        Number of readout points per spiral.
    fov
        Field of view (in meters).
    undersampling_factor
        Undersampling factor of spiral trajectory

    Returns
    -------
    tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float, float)
        - k-space trajectory (traj)
        - Gradient waveform (grad)
        - Slew rate (slew)
        - Time points for the trajectory (timing)
        - Radius values (radius)
        - Angular positions (theta)
        - Number of spiral arms (n_spirals)
        - Scaling of the field-of-view in the k-space center
        - Scaling of the field-of-view in the k-space edge

    """
    # calculate single spiral trajectory
    n_k0 = np.inf
    fov_scaling_center = 1.0
    fov_scaling_edge = 1.0
    n_spirals = int(np.round(n_readout / undersampling_factor))
    while n_k0 > n_readout:
        fov_coefficients = [fov * fov_scaling_center, -fov * (1 - fov_scaling_edge)]

        try:
            traj, grad, slew, timing, radius, theta = variable_density_spiral_trajectory(
                system=system,
                sampling_period=system.grad_raster_time,
                n_interleaves=n_spirals,
                fov_coefficients=fov_coefficients,
                max_kspace_radius=0.5 / fov * n_readout,
            )
            n_k0 = len(grad)
            fov_scaling_edge *= 0.95
        except ValueError:
            # It is not possible to achieve the desired undersampling factor with the given system limits while keeping
            # the full field-of-view in the k-space center. Reduce the field-of-view and try again.
            n_k0 = np.inf
            fov_scaling_center *= 0.95
            fov_scaling_edge = fov_scaling_center

        if fov_scaling_center < 0.1:
            raise ValueError('Cannot find a suitable trajectory.')

    return traj, grad, slew, timing, radius, theta, n_spirals, fov_coefficients[0] / fov, fov_coefficients[1] / fov + 1


def spiral_acquisition(
    system: pp.Opts,
    n_readout: int,
    fov: float,
    undersampling_factor: float,
    readout_oversampling: Literal[1, 2, 4],
    n_spirals: int | None,
    max_pre_duration: float,
    spiral_type: Literal['out', 'in-out'],
):
    """Generate a spiral acquisition sequence.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    n_readout
        Number of readout points per spiral.
    fov
        Field of view (in meters).
    undersampling_factor
        Undersampling factor.
    readout_oversampling
        Oversampling factor for the readout trajectory.
    n_spirals
        Number of spirals to generate. If set to None, this value will be set based on the undersampling factor.
    max_pre_duration : float
        Maximum duration for pre-winder gradients (in seconds).
    spiral_type
        Type of spiral acquisition. 'out' for outward spirals, 'in-out' for spirals turning in and then out.

    Returns
    -------
    gx_combined
        List of combined gradient objects for the x-channel.
    gy_combined
        List of combined gradient objects for the y-channel.
    adc
        PyPulseq ADC object for the acquisition.
    trajectory
        K-space trajectory.
    time_to_echo
        Time to echo from beginning of gradients (in seconds).
    """
    # calculate single spiral trajectory
    traj, grad, _s, _timing, _r, _theta, n_spirals_undersampling, fov_scaling_center, fov_scaling_edge = (
        undersampled_variable_density_spiral(system, n_readout, fov, undersampling_factor)
    )
    n_spirals = n_spirals_undersampling if n_spirals is None else n_spirals
    print(
        f'Target undersampling: {undersampling_factor} - ',
        f'achieved undersampling: {n_readout**2 / (len(traj) * n_spirals_undersampling):.2f}',
        f'FOV: {fov * fov_scaling_center:.3f} (k-sapce center) - {fov * fov_scaling_edge:.3f} (k-space edge)',
    )

    delta_angle = 2 * np.pi / n_spirals
    n_samples_to_echo = 0.5
    if spiral_type == 'in-out':
        n_samples_to_echo = len(grad)
        grad = np.concatenate((-np.asarray(grad * np.exp(1j * np.pi))[::-1], grad))
        traj = np.concatenate((np.asarray(traj * np.exp(1j * np.pi))[::-1], traj))
        delta_angle = delta_angle / 2

    # calculate ADC
    n_readout_with_oversampling = len(grad) * readout_oversampling
    adc_dwell_time = system.grad_raster_time / readout_oversampling
    adc = pp.make_adc(
        num_samples=n_readout_with_oversampling, dwell=adc_dwell_time, system=system, delay=system.adc_dead_time
    )
    traj = np.interp(
        np.linspace(0.5 / readout_oversampling, len(grad) - 0.5 / readout_oversampling, n_readout_with_oversampling),
        np.linspace(0.5, len(grad) - 0.5, len(grad)),
        traj,
    )

    print(f'Receiver bandwidth: {int(1.0 / (adc_dwell_time * n_readout_with_oversampling))} Hz/pixel')

    # Create gradient values and trajectory for different spirals
    grad_list = [grad * np.exp(1j * delta_angle * idx) for idx in np.arange(n_spirals)]
    traj_list = [traj * np.exp(1j * delta_angle * idx) for idx in np.arange(n_spirals)]

    # Create gradient objects
    gx = [pp.make_arbitrary_grad(channel='x', waveform=g.real, delay=adc.delay, system=system) for g in grad_list]
    gy = [pp.make_arbitrary_grad(channel='y', waveform=g.imag, delay=adc.delay, system=system) for g in grad_list]

    # Calculate pre- and re-winder gradients
    gx_rew, gx_pre, gy_rew, gy_pre = [], [], [], []
    for gx_, gy_ in zip(gx, gy, strict=True):
        gx_rew.append(
            pp.make_extended_trapezoid_area(
                area=-gx_.area if spiral_type == 'out' else -gx_.area / 2,
                channel='x',
                grad_start=gx_.last,
                grad_end=0,
                system=system,
                convert_to_arbitrary=True,
            )[0]
        )
        gy_rew.append(
            pp.make_extended_trapezoid_area(
                area=-gy_.area if spiral_type == 'out' else -gy_.area / 2,
                channel='y',
                grad_start=gy_.last,
                grad_end=0,
                system=system,
                convert_to_arbitrary=True,
            )[0]
        )

        if spiral_type == 'in-out':
            gx_pre.append(
                pp.make_extended_trapezoid_area(
                    area=-gx_.area / 2,
                    channel='x',
                    grad_start=0,
                    grad_end=gx_.first,
                    system=system,
                    convert_to_arbitrary=True,
                )[0]
            )

            gy_pre.append(
                pp.make_extended_trapezoid_area(
                    area=-gy_.area / 2,
                    channel='y',
                    grad_start=0,
                    grad_end=gy_.first,
                    system=system,
                    convert_to_arbitrary=True,
                )[0]
            )
        else:
            gx_pre.append(None)
            gy_pre.append(None)

    if spiral_type == 'in-out':
        adc.delay = max_pre_duration

        for i in range(len(gx_pre)):
            gy_pre[i].delay = max_pre_duration - gy_pre[i].shape_dur
            gx_pre[i].delay = max_pre_duration - gx_pre[i].shape_dur
    else:
        max_pre_duration = 0.0

    def combine_gradients(*grad_objects, channel):
        grad_list = [grad for grad in grad_objects if grad is not None]  # Remove None
        waveform_combined = np.concatenate([grad.waveform for grad in grad_list])

        return pp.make_arbitrary_grad(
            channel=channel,
            waveform=waveform_combined,
            first=0,
            delay=grad_list[0].delay,
            last=0,
            system=system,
        )

    gx_combined = [
        combine_gradients(gx_pre, gx_in_out, gx_rew, channel='x')
        for gx_pre, gx_in_out, gx_rew in zip(gx_pre, gx, gx_rew, strict=True)
    ]
    gy_combined = [
        combine_gradients(gy_pre, gy_in_out, gy_rew, channel='y')
        for gy_pre, gy_in_out, gy_rew in zip(gy_pre, gy, gy_rew, strict=True)
    ]

    # times -1 to match pulseq trajectory calculation
    trajectory = -np.stack((np.asarray(traj_list).real, np.asarray(traj_list).imag), axis=-1)

    time_to_echo = max_pre_duration + n_samples_to_echo * readout_oversampling * adc.dwell

    return gx_combined, gy_combined, adc, trajectory, time_to_echo
