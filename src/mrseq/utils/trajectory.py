"""Basic functionality for trajectory calculation."""

from typing import Literal

import numpy as np
import pypulseq as pp

from mrseq.utils import variable_density_spiral_trajectory


def undersampled_variable_density_spiral(system: pp.Opts, n_readout: int, fov: float, undersampling_factor: float):
    """Create undersampled variable density spiral.

    The distribution of the k-space points of a spiral trajectory are restricted by the maximum gradient amplitude and
    slew rate. This makes an analytic solution for a given undersampling factor challenging. Here we use an iterative
    approach in order to achieve a variable density spiral with a certain number of readout samplings and undersampling
    factor.

    During the iterative search the undersampling for the edge of k-space is increased. If this is not enough, then we
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
    tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float)
        - k-space trajectory (k)
        - Gradient waveform (g)
        - Slew rate (s)
        - Time points for the trajectory (time)
        - Radius values (r)
        - Angular positions (theta)
        - Number of spiral arms
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
            traj, grad, s, timing, r, theta = variable_density_spiral_trajectory(
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
            # print(e)
            n_k0 = np.inf
            fov_scaling_center *= 0.95
            fov_scaling_edge = fov_scaling_center

        if fov_scaling_center < 0.1:
            raise ValueError('Cannot find a suitable trajectory.')

    return traj, grad, s, timing, r, theta, n_spirals, fov_coefficients[0] / fov, fov_coefficients[1] / fov + 1


def spiral_acquisition(
    system: pp.Opts,
    n_readout: int,
    fov: float,
    undersampling_factor: float,
    readout_oversampling: Literal[1, 2, 4],
    n_spirals: int | None,
    max_pre_duration: float,
    spiral_type=Literal['out', 'in-out'],
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
    grad = [grad * np.exp(1j * delta_angle * idx) for idx in np.arange(n_spirals)]
    traj = [traj * np.exp(1j * delta_angle * idx) for idx in np.arange(n_spirals)]

    # Create gradient objects
    gx = [pp.make_arbitrary_grad(channel='x', waveform=g.real, delay=adc.delay, system=system) for g in grad]
    gy = [pp.make_arbitrary_grad(channel='y', waveform=g.imag, delay=adc.delay, system=system) for g in grad]

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
        grad_objects = [grad for grad in grad_objects if grad]  # Remove None
        waveform_combined = np.concatenate([grad.waveform for grad in grad_objects])

        return pp.make_arbitrary_grad(
            channel=channel,
            waveform=waveform_combined,
            first=0,
            delay=grad_objects[0].delay,
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

    # -1 to match pulseq trajectory calculation
    trajectory = -np.stack((np.asarray(traj).real, np.asarray(traj).imag), axis=-1)

    time_to_echo = max_pre_duration + n_samples_to_echo * readout_oversampling * adc.dwell

    return gx_combined, gy_combined, adc, trajectory, time_to_echo
