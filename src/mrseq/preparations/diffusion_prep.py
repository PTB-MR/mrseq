"""Diffusion preparation block."""

from typing import Literal

import numpy as np
import pypulseq as pp

from mrseq.utils import round_to_raster
from mrseq.utils import sys_defaults


def calculate_b_value(
    g_amplitude: float,
    g_total_duration: float,
    g_rise_time: float,
    g_delta_time: float,
) -> float:
    """
    Calculate b-values.

    Calculate the b-value for diffusion-weighted imaging using two
    diffusion gradients (trapezoidal) separated by refocusing pulses.

    A correction for symmetric trapezoidal diffusion gradients is applied according to [MAT1990]_.

    Parameters
    ----------
    g_amplitude
        Gradient amplitude for diffusion weighting in Hz/m.
    g_total_duration
        Total duration of the diffusion gradient in seconds.
    g_rise_time
        Gradient rise time for diffusion weighting in seconds.
    g_delta_time
        Gap between diffusion gradients in seconds.

    Returns
    -------
    b_value
        The calculated b-value in s/mm^2.

    References
    ----------
    .. [MAT1990] Mattiello J, Basser PJ, Lebihan D (1990) Analytical Expressions for the b Matrix in NMR Diffusion
              Imaging and Spectroscopy. JMR 108(2). https://doi.org/10.1006/jmra.1994.1103
    """
    # The correction equation assumes that the duration is from the beginning of the ramp up to the end of the platue.
    g_duration = g_total_duration - g_rise_time
    b_value = (
        (2 * np.pi) ** 2
        * g_amplitude**2
        * (g_duration**2 * (g_delta_time - g_duration / 3) + g_rise_time**3 / 30 - g_duration * g_rise_time**2 / 6)
    )
    b_value = b_value * 1e-6  # Convert from s/m^2 to s/mm^2

    return b_value


def calculate_diffusion_gradient_duration(b_value: float, g_amplitude: float, g_delta_time: float) -> float:
    """
    Calculate diffusion gradient duration.

    Calculate the duration of diffusion gradients based on the b-value,
    gradient amplitude and duration between both gradients.
    The calculation assumes a negligible rise time in comparison to the plateau duration,
    such that the equation for rectangular gradients can be applied.

    b = (2 * np.pi)^2 * g_amplitude^2 * g_duration^2 * (g_delta_time - g_duration/3)

    which cen be reformulated to a cubic equation

    -(2*np.pi)^2 * g_amplitude^2/3 * g_duration^3 + (2 * np.pi)^2 * g_amplitude^2 * g_delta_time * g_duration^2 - b = 0

    The solution for this can be found with np.roots

    Parameters
    ----------
    b_value
        The b-value in s/mm^2.
    amp
        Gradient amplitude for diffusion weighting in Hz/m.
    g_delta_time
        Gap between diffusion gradients in seconds.

    Returns
    -------
    g_duration
        The calculated gradient duration in seconds.
    """
    # Convert b value from s/mm^2 to s/m^2
    coeffs = [
        -((2 * np.pi) ** 2) * g_amplitude**2 / 3,
        (2 * np.pi) ** 2 * g_amplitude**2 * g_delta_time,
        0,
        -b_value * 1e6,
    ]
    roots = np.roots(coeffs)
    # Only keep the real, positive root which is smaller than the g_delta_time
    g_duration = [r.real for r in roots if np.isreal(r) and 0 < r.real < g_delta_time]
    if len(g_duration) == 0:
        raise ValueError('No valid duration found for given parameters')
    return g_duration[0].item()


class DiffusionPrep:
    """
    Multi-echo gradient echo acquisition.

    Attributes
    ----------
    system
        PyPulseq system limits object.
    rf_ref
        180° refocusing pulse
    gz_ref
        Slice encoding gradient for the refocusing pulse
    g_diff
        Diffusion gradient.
    g_delta_time
        Gap between diffusion gradients in seconds.
    max_b_value
        Highest b-value in s/mm^2
    """

    def __init__(
        self,
        system: pp.Opts | None = None,
        fov_z: float = 0.008,
        rf_ref_duration: float = 2e-3,
        rf_ref_bwt: float = 4,
        rf_ref_width_scale_factor: float = 3.5,
        g_amplitude: float | None = None,
        g_slew_rate: float | None = None,
        max_b_value: float = 500,
        g_delta_time: float = 50e-3,
        g_channel: Literal['x', 'y', 'z', 'xy', 'yz', 'xz', 'xyz'] = 'x',
        g_sign: Literal[-1, 1] = 1,
        g_crusher_duration=1.6e-3,
    ):
        """
        Initialize the DiffusionPrep class and compute all required attributes.

        Parameters
        ----------
        system
            PyPulseq system limit object.
        fov_z
            FOV along the slice direction.
        rf_ref_duration
            Duration of the refocusing RF pulse in seconds.
        rf_ref_bwt
            Bandwidth-time product of the refocusing RF pulse.
        rf_ref_width_scale_factor
            Factor to scale the slice thickness of the refocusing pulse.
        g_amplitude
            Amplitude of diffusion gradient. If set to None, maximum amplitude is used.
        g_slew_rate
            Slew rate of diffusion gradient. If set to None, maximum slew rate is used.
        max_b_value
            Highest b-value in s/mm^2
        g_delta_time
            Time between beginning of first and second diffusion gradient. If None, the shortest possible time is used.
        g_channel
            Axis of the diffusion gradients
        g_sign
            Sign of the gradient amplitude, i.e. positive or negative gradient
        g_crusher_duration
            Duration of crusher gradient for b-value = 0
        """
        # set system to default if not provided
        self._system = system if system else sys_defaults

        # Diffusion gradient
        if g_amplitude is None:
            g_amplitude = self._system.max_grad
        if g_slew_rate is None:
            g_slew_rate = self._system.max_slew

        # Calculate duration based on highest b-value
        self._g_duration = round_to_raster(
            calculate_diffusion_gradient_duration(max_b_value, g_amplitude * np.sqrt(len(g_channel)), g_delta_time),
            raster_time=self._system.grad_raster_time,
        )

        self._g_diff = []
        for channel in ['x', 'y', 'z']:
            if channel in g_channel:
                self._g_diff.append(
                    pp.make_trapezoid(
                        channel=channel,
                        system=self._system,
                        amplitude=g_amplitude * g_sign,
                        max_slew=g_slew_rate,
                        duration=self._g_duration,
                    )
                )
                if self._g_diff[-1].flat_time < 0:
                    raise ValueError('Slew rate and duration too small for this b-value.')

        # Refousing pulse
        self._rf_ref, self._gz_ref, _ = pp.make_sinc_pulse(
            flip_angle=np.pi,
            duration=rf_ref_duration,
            slice_thickness=fov_z * rf_ref_width_scale_factor,
            apodization=0.5,
            phase_offset=np.pi / 2,
            time_bw_product=rf_ref_bwt,
            delay=self._system.rf_dead_time,
            system=self._system,
            return_gz=True,
            use='refocusing',
        )

        self._max_b_value = max_b_value

        # Crusher gradient for b_value == 0
        self._g_crush = pp.make_trapezoid(
            channel='z',
            system=self._system,
            amplitude=g_amplitude,
            duration=g_crusher_duration,
            max_slew=g_slew_rate,
        )

        # Calculate timings
        self._g_delta_time = g_delta_time
        min_g_delta_time = pp.calc_duration(*self._g_diff) + pp.calc_duration(self._rf_ref, self._gz_ref)
        if min_g_delta_time > g_delta_time:
            raise ValueError('Time between diffusion gradients is too short to fit in refocusing pulse.')
        self._time_between_diff_and_rf = round_to_raster(
            g_delta_time - min_g_delta_time, raster_time=self._system.block_duration_raster
        )
        self._delay_before_crush = round_to_raster(
            pp.calc_duration(*self._g_diff) - pp.calc_duration(self._g_crush) + self._time_between_diff_and_rf,
            raster_time=self._system.block_duration_raster,
        )
        self._delay_after_crush = round_to_raster(
            pp.calc_duration(*self._g_diff) - pp.calc_duration(self._g_crush),
            raster_time=self._system.block_duration_raster,
        )

        self._time_to_refocusing_pulse = pp.calc_duration(*self._g_diff)
        self._time_to_refocusing_pulse += self._time_between_diff_and_rf
        self._time_to_refocusing_pulse += max(self._rf_ref.delay, self._gz_ref.delay + self._gz_ref.rise_time)
        self._time_to_refocusing_pulse += self._rf_ref.shape_dur / 2

        self._block_duration = (
            2 * pp.calc_duration(*self._g_diff)
            + self._time_between_diff_and_rf
            + pp.calc_duration(self._rf_ref, self._gz_ref)
        )

    def add_diffusion_prep(self, seq: pp.Sequence | None, b_value: float) -> tuple[pp.Sequence, float]:
        """Add a diffusion preparation block to a sequence.

        The diffusion preparation block consists of a diffusion gradient, a 180° refocusing
        pulse and another diffusion gradient.

        Parameters
        ----------
        seq
            PyPulseq Sequence object.
        b_value
            Current b-value in s/mm^2
        """
        if b_value > self._max_b_value:
            raise ValueError(f'Current b-value {b_value} larger than highest value {self._max_b_value}')

        # create new sequence if not provided
        if seq is None:
            seq = pp.Sequence(system=self._system)

        # Add diffusion gradient
        if b_value > 0:
            g_current_diff = [pp.scale_grad(g, np.sqrt(b_value / self._max_b_value)) for g in self._g_diff]
            seq.add_block(*g_current_diff)
            seq.add_block(pp.make_delay(self._time_between_diff_and_rf))

            b_value_calculated = calculate_b_value(
                g_current_diff[0].amplitude * np.sqrt(len(g_current_diff)),
                self._g_duration,
                g_current_diff[0].rise_time,
                self._g_delta_time,
            )
        else:
            seq.add_block(pp.make_delay(self._delay_before_crush))
            seq.add_block(self._g_crush)

            b_value_calculated = calculate_b_value(
                self._g_crush.amplitude,
                pp.calc_duration(self._g_crush),
                self._g_crush.rise_time,
                pp.calc_duration(self._g_crush) + pp.calc_duration(self._rf_ref, self._gz_ref),
            )

        # Add refocusing pulse
        seq.add_block(self._rf_ref, self._gz_ref)

        if b_value > 0:
            seq.add_block(*g_current_diff)
        else:
            seq.add_block(self._g_crush)
            seq.add_block(pp.make_delay(self._delay_after_crush))

        return seq, b_value_calculated
