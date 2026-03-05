"""Echo Planar Imaging (EPI) readout sequence."""

from typing import Literal

import ismrmrd
import matplotlib.pyplot as plt
import numpy as np
import pypulseq as pp

from mrseq.utils import round_to_raster
from mrseq.utils import sys_defaults


def _trapezoid_area_at_times(
    rise_time: float,
    flat_time: float,
    fall_time: float,
    amplitude: float,
    sample_times: np.ndarray,
) -> np.ndarray:
    """Calculate accumulated gradient area analytically for a trapezoidal gradient.

    Parameters
    ----------
    rise_time
        Gradient rise time (in seconds).
    flat_time
        Gradient flat time (in seconds).
    fall_time
        Gradient fall time (in seconds).
    amplitude
        Gradient amplitude (in Hz/m).
    sample_times
        Array of time points (in seconds) relative to the start of the gradient.

    Returns
    -------
    area
        Accumulated gradient area at each sample time (in 1/m).
    """
    t1 = rise_time
    t2 = t1 + flat_time
    t3 = t2 + fall_time

    area_at_t1 = amplitude * rise_time / 2
    area_at_t2 = area_at_t1 + amplitude * flat_time
    area_at_t3 = area_at_t2 + amplitude * fall_time / 2

    area = np.zeros_like(sample_times)
    for i, t in enumerate(sample_times):
        if t <= 0:
            area[i] = 0.0
        elif t <= t1:
            area[i] = amplitude * t**2 / (2 * rise_time)
        elif t <= t2:
            area[i] = area_at_t1 + amplitude * (t - t1)
        elif t <= t3:
            dt = t - t2
            area[i] = area_at_t2 + amplitude * dt - amplitude * dt**2 / (2 * fall_time)
        else:
            area[i] = area_at_t3

    return area


def _piecewise_linear_area_at_times(
    waveform_times: np.ndarray,
    waveform_values: np.ndarray,
    sample_times: np.ndarray,
) -> np.ndarray:
    """Compute cumulative integral of a piecewise-linear waveform at given sample times.

    Parameters
    ----------
    waveform_times
        Time points of the piecewise-linear waveform (in seconds), sorted ascending.
    waveform_values
        Waveform amplitude values at each time point (in Hz/m).
    sample_times
        Times at which to evaluate the cumulative area (in seconds).

    Returns
    -------
    area
        Cumulative area at each sample time (in 1/m).
    """
    n_knots = len(waveform_times)
    cum_area = np.zeros(n_knots)
    for k in range(1, n_knots):
        dt = waveform_times[k] - waveform_times[k - 1]
        cum_area[k] = cum_area[k - 1] + (waveform_values[k] + waveform_values[k - 1]) / 2 * dt

    area = np.zeros_like(sample_times)
    for i, t in enumerate(sample_times):
        if t <= waveform_times[0]:
            area[i] = 0.0
        elif t >= waveform_times[-1]:
            area[i] = cum_area[-1]
        else:
            k = np.searchsorted(waveform_times, t, side='right')
            base_area = cum_area[k - 1]
            dt_seg = waveform_times[k] - waveform_times[k - 1]
            dt = t - waveform_times[k - 1]
            frac = dt / dt_seg
            wf_at_t = waveform_values[k - 1] + frac * (waveform_values[k] - waveform_values[k - 1])
            area[i] = base_area + (waveform_values[k - 1] + wf_at_t) / 2 * dt

    return area


class EpiReadout:
    """EPI readout module supporting flyback and symmetric readout, ramp sampling, oversampling, and partial Fourier.

    Attributes
    ----------
    system
        System limits.
    ramp_sampling
        If True, ADC is active during gradient ramps (optimized timing).
    readout_type
        Readout type ('flyback' or 'symmetric').
    spoiling_enable
        Enable spoiling gradients (useful for calibration if False).
    adc
        ADC event.
    gx_pre
        Pre-phaser gradient in readout direction.
    gy_pre
        Pre-phaser gradient in phase encoding direction.
    gx
        Readout gradient.
    gy
        Phase encoding gradient.
    gy_blip
        Complete blip gradient used for flyback readout.
    gy_blipup
        Ramp-up part of blip gradient used for symmetric readout.
    gy_blipdown
        Ramp-down part of blip gradient used for symmetric readout.
    gy_blipdownup
        Combined gy_blipdown and gy_blipup gradient used for symmetric readout.
    gz_spoil
        Spoiler gradient in z-direction.
    gx_flyback
        Flyback gradient in x-direction.
    """

    def __init__(
        self,
        system: pp.Opts | None = None,
        fov: float = 128e-3,
        n_readout: int = 64,
        n_phase_encoding: int = 64,
        bandwidth: float = 50e3,
        oversampling: int = 2,
        ramp_sampling: bool = True,
        readout_type: Literal['flyback', 'symmetric'] = 'symmetric',
        partial_fourier_factor: float = 1.0,
        adc_freq_offset: float = 0,
        pe_enable: bool = True,
        spoiling_enable: bool = True,
    ):
        """Initialize EPI Readout.

        Parameters
        ----------
        system
            System limits.
        fov
            Field of view in meters (square).
        n_readout
            Number of readout points.
        n_phase_encoding
            Number of phase encoding points.
        bandwidth
            Total receiver bandwidth in Hz.
        oversampling
            ADC oversampling factor.
        ramp_sampling
            If True, ADC is active during gradient ramps (optimized timing).
        readout_type
            Readout type ('flyback' or 'symmetric').
        partial_fourier_factor
            Partial Fourier factor (0.5 to 1.0).
        adc_freq_offset
            Frequency offset for the ADC.
        pe_enable
            Enable phase encoding (useful for calibration scans if False).
        spoiling_enable
            Enable spoiling gradients.
        """
        if not 0.5 < partial_fourier_factor <= 1.0:
            raise ValueError('Desired partial Fourier factor must be larger than 0.5 and smaller or equal to 1.0.')

        # set system to default if not provided
        if system is None:
            system = sys_defaults

        self.system = system
        self.fov = fov
        self.n_readout = n_readout
        self.n_phase_encoding = n_phase_encoding
        self.bandwidth = bandwidth
        self.oversampling = oversampling
        self.ramp_sampling = ramp_sampling
        self.readout_type = readout_type
        self.partial_fourier_factor = partial_fourier_factor
        self.adc_freq_offset = adc_freq_offset
        self.pe_enable = pe_enable
        self.spoiling_enable = spoiling_enable

        # Derived parameters
        readout_time = n_readout / bandwidth
        delta_kx = 1 / (fov * oversampling)
        delta_ky = 1 / fov

        # Initiate all optional gradients as None
        self.gx_flyback = None
        self.gy_blipdown = None
        self.gy_blipup = None
        self.gy_blipdownup = None
        self.gz_spoil = None

        # Create blip gradient with shortest possible timing
        gy_blip_duration = np.ceil(2 * np.sqrt(delta_ky / system.max_slew) / 10e-6 / 2) * 10e-6 * 2
        gy_blip_half_dur = gy_blip_duration / 2
        self.gy_blip = pp.make_trapezoid(channel='y', system=self.system, area=-delta_ky, duration=gy_blip_duration)

        # Create readout gradient
        gx_encoding_area = n_readout * delta_kx * oversampling
        if self.ramp_sampling:
            # Calculate additional gradient area from gy_blip assuming maximum slew rate
            extra_area = np.power(gy_blip_half_dur, 2) * self.system.max_slew

            # Create gradient with additional area
            gx = pp.make_trapezoid(
                channel='x',
                system=self.system,
                area=gx_encoding_area + extra_area,
                duration=gy_blip_half_dur + readout_time + gy_blip_half_dur,
            )

            if not gx.fall_time == gx.rise_time:
                raise ValueError('Gradient fall time must be equal to rise time for ramp sampling.')

            # Second, correct area taking actual slew rate into account
            gx_slew = gx.amplitude / gx.rise_time
            gx_area_reduced_slew = gx.area - gx_slew * np.power(gy_blip_half_dur, 2)

            gx.amplitude = float(gx.amplitude * gx_encoding_area / gx_area_reduced_slew)
            gx.area = float(gx.amplitude * (gx.rise_time / 2 + gx.flat_time + gx.fall_time / 2))
            gx.flat_area = float(gx.amplitude * gx.flat_time)
            self.gx = gx
        else:
            self.gx = pp.make_trapezoid(
                channel='x',
                system=self.system,
                flat_area=gx_encoding_area,
                flat_time=readout_time,
            )

        # Create ADC event
        adc_dwell = delta_kx / self.gx.amplitude
        adc_dwell = round_to_raster(adc_dwell, self.system.adc_raster_time)

        adc_samples = int(round(readout_time / adc_dwell / 4) * 4)

        adc_time_to_center = adc_dwell * (adc_samples / 2 + 0.5)
        adc_delay = self.gx.rise_time + self.gx.flat_time / 2 - adc_time_to_center + adc_dwell / 2
        # the adc delay has to be rounded to the rf raster time (not the adc raster time)
        adc_delay = round_to_raster(adc_delay, self.system.rf_raster_time)

        self.adc = pp.make_adc(
            num_samples=adc_samples,
            dwell=adc_dwell,
            freq_offset=adc_freq_offset,
            delay=adc_delay,
            system=self.system,
        )

        # Create and align pre-phaser gradients considering partial fourier factor
        # determine the number of "PE" lines after (and including) k-space center (independent of partial fourier)
        self.n_phase_enc_post_center = int(np.ceil(self.n_phase_encoding / 2 + 1))
        # find the closest number of lines to the desired partial fourier factor
        valid_n_phase_total = int(np.ceil(partial_fourier_factor * self.n_phase_encoding))
        # ensure that at least one line before the center is acquired
        self.n_phase_enc_pre_center = max(1, valid_n_phase_total - self.n_phase_enc_post_center)
        # update the total number of "PE" lines
        self.n_phase_enc_total = self.n_phase_enc_pre_center + self.n_phase_enc_post_center
        # recalculate the actual partial fourier factor
        actual_pf_factor = self.n_phase_enc_total / self.n_phase_encoding
        if actual_pf_factor != partial_fourier_factor:
            print(f'Adjusted partial Fourier factor from {partial_fourier_factor} to {actual_pf_factor:.2f}.')
            self.partial_fourier_factor = actual_pf_factor

        # Create pre-phaser gradients
        self.gx_pre = pp.make_trapezoid(
            channel='x',
            system=self.system,
            area=-self.gx.area / 2 - delta_kx / 2,
        )
        self.gy_pre = pp.make_trapezoid(
            channel='y',
            system=self.system,
            area=self.n_phase_enc_pre_center * delta_ky,
        )
        self.gx_pre, self.gy_pre = pp.align(right=[self.gx_pre, self.gy_pre])

        # Create and align "phase encoding" gradients
        if self.readout_type == 'flyback':
            self.gx_flyback = pp.make_trapezoid(channel='x', system=self.system, area=-self.gx.area)
        elif self.readout_type == 'symmetric':
            # Split and align blip gradient in case of symmetric readout
            gy_parts = pp.split_gradient_at(grad=self.gy_blip, time_point=gy_blip_duration / 2, system=self.system)
            self.gy_blipup, self.gy_blipdown, _ = pp.align(right=gy_parts[0], left=[gy_parts[1], self.gx])
            self.gy_blipdownup = pp.add_gradients((self.gy_blipdown, self.gy_blipup), system=self.system)
        else:
            raise NotImplementedError('Currently, only "symmetric" and "flyback" readout types are supported.')

        # Disable phase encoding if self.pe_enable is False
        if not self.pe_enable:
            self.gy_pre.amplitude = 0
            if (
                self.readout_type == 'symmetric'
                and self.gy_blipup is not None
                and self.gy_blipdown is not None
                and self.gy_blipdownup is not None
            ):
                self.gy_blipup.waveform *= 0
                self.gy_blipdown.waveform *= 0
                self.gy_blipdownup.waveform *= 0
            elif self.readout_type == 'flyback':
                self.gy_blip.waveform *= 0

        # Create spoiler gradient if spoiling is enabled
        if self.spoiling_enable:
            self.gz_spoil = pp.make_trapezoid(channel='z', system=self.system, area=4 * delta_ky * n_readout)

    @property
    def time_to_center(self) -> float:
        """Return time from beginning of readout to center of k-space (needed for TE calculations)."""
        # i) add time for pre phaser
        time_to_center = pp.calc_duration(self.gx_pre, self.gy_pre)
        # ii) add time for completed k-space lines before central (ky = 0) line
        if self.readout_type == 'flyback':
            time_to_center += self.n_phase_enc_pre_center * (
                pp.calc_duration(self.gx) + pp.calc_duration(self.gx_flyback, self.gy_blip)
            )
        elif self.readout_type == 'symmetric':
            time_to_center += self.n_phase_enc_pre_center * pp.calc_duration(self.gx)
        # iii) add time before start of ADC of central k-space line
        if self.ramp_sampling:
            time_to_center += self.adc.delay
        else:
            time_to_center += self.gx.rise_time
        # iv) add time from start of ADC (for ky = 0) to timepoint when kx = 0 as well
        time_to_center += self.adc.dwell * (self.adc.num_samples / 2 + 0.5)

        return float(time_to_center)

    @property
    def time_to_center_without_prephaser(self) -> float:
        """Return time from after pre-phasers to center of k-space (needed for TE calculations)."""
        return self.time_to_center - pp.calc_duration(self.gx_pre, self.gy_pre)

    @property
    def total_duration(self) -> float:
        """Return total duration of readout including pre-phaser and optional spoiler."""
        total_duration = pp.calc_duration(self.gx_pre, self.gy_pre)
        if self.readout_type == 'flyback':
            total_duration += self.n_phase_enc_total * (
                pp.calc_duration(self.gx) + pp.calc_duration(self.gx_flyback, self.gy_blip)
            )
            total_duration -= pp.calc_duration(self.gx_flyback, self.gy_blip)
        elif self.readout_type == 'symmetric':
            total_duration += self.n_phase_enc_total * pp.calc_duration(self.gx)
        # add time for spoiler if enabled
        if self.spoiling_enable:
            total_duration += pp.calc_duration(self.gz_spoil)

        return float(total_duration)

    @property
    def total_duration_without_prephaser(self) -> float:
        """Return total duration of readout excluding pre-phasers but including optional spoiler."""
        return self.total_duration - pp.calc_duration(self.gx_pre, self.gy_pre)

    def _readout_block_kxy(
        self,
        pe_idx: int,
        sample_times: np.ndarray,
        ky_base: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute kx/ky at given time points for a single readout block.

        Parameters
        ----------
        pe_idx
            Phase encoding index (0-based).
        sample_times
            Time points relative to the start of the readout block (in seconds).
        ky_base
            Cumulative ky offset at the start of this block (in 1/m).

        Returns
        -------
        kx
            kx positions at each sample time (in 1/m).
        ky
            ky positions at each sample time (in 1/m).
        ky_base_next
            Updated ky_base for the next block (in 1/m).
        """
        n_pe = self.n_phase_enc_total
        kx_offset = self.gx_pre.area

        # kx: accumulated area under the readout gradient
        kx_forward = _trapezoid_area_at_times(
            self.gx.rise_time, self.gx.flat_time, self.gx.fall_time, abs(self.gx.amplitude), sample_times
        )

        if self.readout_type == 'flyback':
            kx = kx_offset + kx_forward
            ky = np.full_like(sample_times, ky_base)
            ky_base_next = ky_base + self.gy_blip.area

        elif self.readout_type == 'symmetric':
            if pe_idx % 2 == 0:
                kx = kx_offset + kx_forward
            else:
                kx = kx_offset + self.gx.area - kx_forward

            # ky changes during readout due to blip overlap
            if pe_idx == 0:
                assert self.gy_blipup is not None
                bu_tt = self.gy_blipup.tt + self.gy_blipup.delay
                ky_blip = _piecewise_linear_area_at_times(bu_tt, self.gy_blipup.waveform, sample_times)
                blip_total = _piecewise_linear_area_at_times(bu_tt, self.gy_blipup.waveform, np.array([bu_tt[-1]]))[0]
            elif pe_idx == n_pe - 1:
                assert self.gy_blipdown is not None
                bd_tt = self.gy_blipdown.tt + self.gy_blipdown.delay
                ky_blip = _piecewise_linear_area_at_times(bd_tt, self.gy_blipdown.waveform, sample_times)
                blip_total = 0.0
            else:
                assert self.gy_blipdownup is not None
                bdu_tt = self.gy_blipdownup.tt + self.gy_blipdownup.delay
                ky_blip = _piecewise_linear_area_at_times(bdu_tt, self.gy_blipdownup.waveform, sample_times)
                blip_total = _piecewise_linear_area_at_times(
                    bdu_tt, self.gy_blipdownup.waveform, np.array([bdu_tt[-1]])
                )[0]

            ky = ky_base + ky_blip
            ky_base_next = ky_base + blip_total
        else:
            raise NotImplementedError(f'Readout type "{self.readout_type}" is not supported.')

        return kx, ky, ky_base_next

    def calculate_trajectory(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute analytical k-space trajectory for the EPI readout.

        Returns the kx and ky positions at each ADC sample for each phase encoding line.
        This replaces the need to call ``seq.calculate_kspace()`` for trajectory information.

        Returns
        -------
        kx
            kx trajectory in 1/m, shape ``(n_phase_enc_total, adc.num_samples)``.
        ky
            ky trajectory in 1/m, shape ``(n_phase_enc_total, adc.num_samples)``.
        """
        n_samples = self.adc.num_samples
        n_pe = self.n_phase_enc_total

        # ADC sample times relative to the start of the gradient block (center of each dwell)
        sample_times = self.adc.delay + (np.arange(n_samples) + 0.5) * self.adc.dwell

        kx = np.zeros((n_pe, n_samples))
        ky = np.zeros((n_pe, n_samples))
        ky_base = self.gy_pre.area

        for pe_idx in range(n_pe):
            kx[pe_idx, :], ky[pe_idx, :], ky_base = self._readout_block_kxy(pe_idx, sample_times, ky_base)

        return kx, ky

    def calculate_full_trajectory(self, n_points_per_segment: int = 200) -> dict:
        """Compute full analytical k-space trajectory including prephaser and transitions.

        This extends ``calculate_trajectory()`` by also computing the k-space path during
        the prephaser gradients and the transitions between readout lines (flyback gradients
        or symmetric ramp transitions). Useful for visualization.

        Parameters
        ----------
        n_points_per_segment
            Number of densely sampled time points per gradient block for smooth trajectory lines.

        Returns
        -------
        result
            Dictionary with keys:

            - ``'kx'``: full kx trajectory as 1-D array (1/m).
            - ``'ky'``: full ky trajectory as 1-D array (1/m).
            - ``'adc_indices'``: indices into kx/ky arrays that correspond to ADC samples.
        """
        n_samples = self.adc.num_samples
        n_pe = self.n_phase_enc_total

        all_kx: list[np.ndarray] = []
        all_ky: list[np.ndarray] = []
        adc_indices: list[int] = []
        current_idx = 0

        # --- Prephaser block ---
        pre_dur = pp.calc_duration(self.gx_pre, self.gy_pre)
        t_pre = np.linspace(0, pre_dur, n_points_per_segment, endpoint=False)
        kx_pre = _trapezoid_area_at_times(
            self.gx_pre.rise_time,
            self.gx_pre.flat_time,
            self.gx_pre.fall_time,
            self.gx_pre.amplitude,
            t_pre - self.gx_pre.delay,
        )
        ky_pre = _trapezoid_area_at_times(
            self.gy_pre.rise_time,
            self.gy_pre.flat_time,
            self.gy_pre.fall_time,
            self.gy_pre.amplitude,
            t_pre - self.gy_pre.delay,
        )
        all_kx.append(kx_pre)
        all_ky.append(ky_pre)
        current_idx += len(t_pre)

        # Current kx/ky offsets at the end of the prephaser
        kx_offset = self.gx_pre.area
        ky_base = self.gy_pre.area

        for pe_idx in range(n_pe):
            # --- Readout block (densely sampled) ---
            block_dur = pp.calc_duration(self.gx)
            t_block = np.linspace(0, block_dur, n_points_per_segment, endpoint=False)

            kx_block, ky_block, ky_base = self._readout_block_kxy(pe_idx, t_block, ky_base)

            all_kx.append(kx_block)
            all_ky.append(ky_block)

            # Record ADC sample indices
            for s in range(n_samples):
                adc_indices.append(current_idx + np.searchsorted(t_block, self.adc.delay + (s + 0.5) * self.adc.dwell))
            current_idx += len(t_block)

            # Flyback + blip transition block (flyback only)
            if self.readout_type == 'flyback' and pe_idx < n_pe - 1:
                assert self.gx_flyback is not None
                fb_dur = pp.calc_duration(self.gx_flyback, self.gy_blip)
                t_fb = np.linspace(0, fb_dur, n_points_per_segment // 2, endpoint=False)
                # kx during flyback
                kx_fb = (
                    kx_offset
                    + self.gx.area
                    + _trapezoid_area_at_times(
                        self.gx_flyback.rise_time,
                        self.gx_flyback.flat_time,
                        self.gx_flyback.fall_time,
                        self.gx_flyback.amplitude,
                        t_fb - self.gx_flyback.delay,
                    )
                )
                # ky during flyback
                ky_pre_blip = ky_base - self.gy_blip.area
                ky_fb = ky_pre_blip + _trapezoid_area_at_times(
                    self.gy_blip.rise_time,
                    self.gy_blip.flat_time,
                    self.gy_blip.fall_time,
                    self.gy_blip.amplitude,
                    t_fb - self.gy_blip.delay,
                )
                all_kx.append(kx_fb)
                all_ky.append(ky_fb)
                current_idx += len(t_fb)

        kx_full = np.concatenate(all_kx)
        ky_full = np.concatenate(all_ky)
        adc_idx = np.array(adc_indices, dtype=int)
        # Clip to valid range
        adc_idx = np.clip(adc_idx, 0, len(kx_full) - 1)

        return {'kx': kx_full, 'ky': ky_full, 'adc_indices': adc_idx}

    def add_to_seq(
        self,
        seq: pp.Sequence,
        add_prephaser: bool = True,
        mrd_dataset: ismrmrd.Dataset | None = None,
    ) -> tuple[pp.Sequence, ismrmrd.Dataset | None]:
        """Add EPI readout blocks to the sequence.

        If `mrd_dataset` is provided, the analytical k-space trajectory is
        computed and written to the MRD file for each readout line.
        """
        # (Re)set phase encoding (LIN) label
        lin_label = pp.make_label(label='LIN', type='SET', value=0)

        # Compute analytical trajectory for MRD writing
        if mrd_dataset is not None:
            kx_traj, ky_traj = self.calculate_trajectory()

        # Add pre-phaser gradients if enabled
        if add_prephaser:
            seq.add_block(self.gx_pre, self.gy_pre)

        for pe_idx in range(self.n_phase_enc_total):
            rev_label = pp.make_label(type='SET', label='REV', value=self.gx.amplitude < 0)
            seg_label = pp.make_label(type='SET', label='SEG', value=self.gx.amplitude < 0)
            if self.readout_type == 'symmetric':
                # Select blip gradient based on phase encoding index
                if pe_idx == 0:
                    gy_blip = self.gy_blipup
                elif pe_idx == self.n_phase_enc_total - 1:
                    gy_blip = self.gy_blipdown
                else:
                    gy_blip = self.gy_blipdownup

                # Add readout block and reverse polarity of readout gradient
                seq.add_block(self.gx, gy_blip, self.adc, lin_label, rev_label, seg_label)
                self.gx.amplitude = -self.gx.amplitude

            elif self.readout_type == 'flyback':
                seq.add_block(self.gx, self.adc, lin_label, rev_label)
                if pe_idx != self.n_phase_enc_total - 1:
                    seq.add_block(self.gx_flyback, self.gy_blip)

            # Write per-readout trajectory to MRD
            if mrd_dataset is not None:
                n_samples = self.adc.num_samples
                traj = np.zeros((n_samples, 2), dtype=np.float32)
                traj[:, 0] = kx_traj[pe_idx] * self.fov * self.oversampling
                traj[:, 1] = ky_traj[pe_idx] * self.fov

                acq = ismrmrd.Acquisition()
                acq.resize(trajectory_dimensions=2, number_of_samples=n_samples)
                acq.traj[:] = traj
                mrd_dataset.append_acquisition(acq)

            lin_label = pp.make_label(label='LIN', type='INC', value=1)

        if self.spoiling_enable:
            seq.add_block(self.gz_spoil)

        return seq, mrd_dataset

    def plot_sequence(self):
        """Plot the sequence."""
        seq = pp.Sequence(self.system)
        seq, _ = self.add_to_seq(seq)
        _, axs1, _, axs2 = seq.plot(grad_disp='mT/m', plot_now=False)
        # add vertical line at time to center to all plots in both figures
        for ax in axs1 + axs2:
            ax.axvline(x=self.time_to_center, color='r', linestyle='--')
        plt.show()

    def plot_trajectory(self, show_adc: bool = True):
        """Plot full k-space trajectory with color-coded direction.

        The trajectory is colored from start (dark) to end (bright) using the
        copper colormap. Small arrows at the midpoint of each readout line
        indicate the traversal direction. ADC sample positions are shown as
        red markers.

        Parameters
        ----------
        show_adc
            If True, ADC sample positions are shown as red markers.

        Returns
        -------
        fig
            Matplotlib figure object.
        """
        from matplotlib.collections import LineCollection

        traj = self.calculate_full_trajectory()
        kx = traj['kx']
        ky = traj['ky']
        adc_idx = traj['adc_indices']

        fig, ax = plt.subplots()

        # Create line segments colored by traversal order
        points = np.column_stack([kx, ky]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        t = np.linspace(0, 1, len(segments))
        lc = LineCollection(segments.tolist(), cmap='copper', linewidth=1.0)
        lc.set_array(t)
        ax.add_collection(lc)

        # ADC samples
        if show_adc:
            ax.plot(kx[adc_idx], ky[adc_idx], 'x', color='red', markersize=3, zorder=3)

        # Direction arrows at the midpoint of each readout line
        n_samples = self.adc.num_samples
        for pe_idx in range(self.n_phase_enc_total):
            start = pe_idx * n_samples
            mid = start + n_samples // 2
            kx_adc = kx[adc_idx]
            ky_adc = ky[adc_idx]
            dx = kx_adc[mid] - kx_adc[mid - 1]
            dy = ky_adc[mid] - ky_adc[mid - 1]
            ax.annotate(
                '',
                xy=(kx_adc[mid] + dx, ky_adc[mid] + dy),
                xytext=(kx_adc[mid], ky_adc[mid]),
                arrowprops={'arrowstyle': '->', 'color': 'blue', 'lw': 1.5},
            )

        ax.set_xlabel('kx (1/m)')
        ax.set_ylabel('ky (1/m)')
        ax.set_aspect('equal', adjustable='datalim')
        ax.autoscale_view()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return fig
