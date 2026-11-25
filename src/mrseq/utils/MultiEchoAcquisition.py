"""Basic functionality for trajectory calculation."""

from typing import Literal

import numpy as np
import pypulseq as pp

from mrseq.utils import find_gx_flat_time_on_adc_raster
from mrseq.utils import round_to_raster
from mrseq.utils import sys_defaults


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
            self._delta_te_delay = 0.0
        else:
            self._delta_te_delay = round_to_raster(delta_te - min_delta_te, self._system.block_duration_raster)
            if self._delta_te_delay < 0:
                raise ValueError(
                    f'TE must be larger than {min_delta_te * 1000:.3f} ms. Current value is {delta_te * 1000:.3f} ms.'
                )

    def add_to_seq(self, seq: pp.Sequence, n_echoes: int, polarity: Literal['positive', 'negative'] = 'positive'):
        """Add all gradients and adc to sequence.

        Parameters
        ----------
        seq
            PyPulseq Sequence object.
        n_echoes
            Number of echoes
        polarity
            Polarity of first readout gradient

        Returns
        -------
        seq
            PyPulseq Sequence object.
        time_to_echoes
            Time from beginning of sequence to echoes.
        """
        # readout pre-winder
        seq.add_block(self._gx_pre if polarity == 'positive' else pp.scale_grad(self._gx_pre, -1))

        # add readout gradients and ADCs
        seq, time_to_echoes = self.add_to_seq_without_pre_post_gradient(seq, n_echoes, polarity)

        # readout re-winder
        seq.add_block(self._gx_post if polarity == 'positive' else pp.scale_grad(self._gx_post, -1))

        return seq, time_to_echoes

    def add_to_seq_without_pre_post_gradient(
        self, seq: pp.Sequence, n_echoes: int, polarity: Literal['positive', 'negative'] = 'positive'
    ):
        """Add readout gradients without pre- and re-winder gradients.

        Often the pre- and re-winder gradients are played out at the same time as phase encoding gradients or spoiler
        gradients.

        Parameters
        ----------
        seq
            PyPulseq Sequence object.
        n_echoes
            Number of echoes
        polarity
            Polarity of first readout gradient

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
            gx_sign = (-1) ** echo_ if polarity == 'positive' else (-1) ** (echo_ + 1)
            labels = []
            labels.append(pp.make_label(label='REV', type='SET', value=gx_sign == -1))
            labels.append(pp.make_label(label='ECO', type='SET', value=echo_))
            seq.add_block(pp.scale_grad(self._gx, gx_sign), self._adc, *labels)
            time_to_echoes.append(
                start_of_current_gx + self._adc.delay + self._n_readout_pre_echo * self._adc.dwell + self._adc.dwell / 2
            )
            start_of_current_gx = sum(seq.block_durations.values())
            if echo_ < n_echoes - 1:
                if self._delta_te_delay > 0:
                    seq.add_block(pp.make_delay(self._delta_te_delay))
                seq.add_block(pp.scale_grad(self._gx_between, -gx_sign))

        return seq, time_to_echoes
