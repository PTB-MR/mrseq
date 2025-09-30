"""Basic functionality for trajectory calculation."""

from typing import Literal

import numpy as np
import pypulseq as pp

from mrseq.utils import sys_defaults


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
        idx = np.argsort(np.abs(kpe), kind='stable')
        kpe = kpe[idx]
    elif sampling_order == 'high_low':
        idx = np.argsort(-np.abs(kpe), kind='stable')
        kpe = kpe[idx]
    else:
        raise ValueError(f'sampling order {sampling_order} not supported.')

    return kpe, kpe_fully_sampled_center


class MultiGradientEcho:
    """
    Multi-echo gradient echo readout.

    Attributes
    ----------
    system
        PyPulseq system limits object.
    fov
        Field of view in x direction (in meters).
    n_readout
        Number of frequency encoding steps.
    readout_oversampling
        Readout oversampling factor.
    partial_echo_factor
        Partial echo factor, commonly between 0.6 and 1.
    gx_flat_time
        Flat time of the readout gradient.
    gx_pre_duration
        Duration of readout pre-winder gradient (in seconds).
    delta_k
        K-space increment.
    n_readout_post_echo
        Number of readout points after echo.
    n_readout_pre_echo
        Number of readout points before echo.
    n_readout_with_partial_echo
        Total number of readout points with partial echo.
    gx_flat_area
        Flat area of the readout gradient.
    gx_pre_ratio
        Ratio for pre-winder gradient.
    gx_post_ratio
        Ratio for re-winder gradient.
    gx
        Readout gradient object.
    n_readout_with_oversampling
        Number of readout points with oversampling.
    adc
        ADC event object.
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
        fov: float = 0.256,
        n_readout: int = 128,
        readout_oversampling: float = 2.0,
        partial_echo_factor: float = 0.7,
        gx_flat_time: float = 2.0e-3,
        gx_pre_duration: float = 0.8e-3,
    ):
        """
        Initialize the ReadoutGradientADC class and compute all required attributes.

        Parameters
        ----------
        system
            PyPulseq system limits object.
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
        self.system = sys_defaults if system is None else system

        self.fov = fov
        self.n_readout = n_readout
        self.readout_oversampling = readout_oversampling
        self.partial_echo_factor = partial_echo_factor
        self.gx_flat_time = gx_flat_time
        self.gx_pre_duration = gx_pre_duration

        self.delta_k = 1 / self.fov

        self.n_readout_post_echo = int(self.n_readout / 2)
        self.n_readout_post_echo += np.mod(self.n_readout_post_echo + 1, 2)  # make odd
        self.n_readout_pre_echo = int((self.n_readout * self.partial_echo_factor) - self.n_readout_post_echo)
        self.n_readout_pre_echo += np.mod(self.n_readout_pre_echo, 2)  # make even
        self.n_readout_with_partial_echo = self.n_readout_pre_echo + 1 + self.n_readout_post_echo

        self.gx_flat_area = self.n_readout_with_partial_echo * self.delta_k
        self.gx_pre_ratio = (self.n_readout_pre_echo + 1) / self.n_readout_with_partial_echo
        self.gx_post_ratio = self.n_readout_post_echo / self.n_readout_with_partial_echo

        self.gx = pp.make_trapezoid(
            channel='x', flat_area=self.gx_flat_area, flat_time=self.gx_flat_time, system=self.system
        )
        self.n_readout_with_oversampling = int(self.n_readout_with_partial_echo * self.readout_oversampling)
        self.adc = pp.make_adc(
            num_samples=self.n_readout_with_oversampling,
            duration=self.gx.flat_time,
            delay=self.gx.rise_time,
            system=self.system,
        )

        self.gx_pre = pp.make_trapezoid(
            channel='x',
            area=-self.gx.area * self.gx_pre_ratio - self.delta_k / 2,
            duration=self.gx_pre_duration,
            system=self.system,
        )
        self.gx_post = pp.make_trapezoid(
            channel='x',
            area=-self.gx.area * self.gx_post_ratio + self.delta_k / 2,
            duration=self.gx_pre_duration,
            system=self.system,
        )
        self.gx_between = pp.make_trapezoid(
            channel='x', area=self.gx_pre.area - self.gx_post.area, duration=self.gx_pre_duration, system=self.system
        )

    def add_to_seq(self, seq: pp.Sequence, n_echoes: int):
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
        """
        # readout pre-winder
        seq.add_block(self.gx_pre)

        # add readout gradient and ADC
        seq, start_time_of_each_gx = self.add_to_seq_without_pre_post_gradient(seq, n_echoes)

        # readout re-winder
        seq.add_block(self.gx_post)

        return seq, start_time_of_each_gx

    def add_to_seq_without_pre_post_gradient(self, seq: pp.Sequence, n_echoes: int):
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
        """
        # add readout gradient and ADC
        start_time_of_each_gx = []
        for echo_ in range(n_echoes):
            start_time_of_each_gx.append(sum(seq.block_durations.values()))
            gx_sign = (-1) ** echo_
            labels = []
            labels.append(pp.make_label(type='SET', label='REV', value=gx_sign == -1))
            labels.append(pp.make_label(type='SET', label='ECO', value=echo_))
            seq.add_block(pp.scale_grad(self.gx, gx_sign), self.adc, *labels)
            if echo_ < n_echoes - 1:
                seq.add_block(pp.scale_grad(self.gx_between, -gx_sign))

        return seq, start_time_of_each_gx
