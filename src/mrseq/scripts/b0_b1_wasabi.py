"""Simultaneous B0 and B1 mapping using the WASABI method."""

from pathlib import Path

import numpy as np
import pypulseq as pp

from mrseq.utils import find_gx_flat_time_on_adc_raster
from mrseq.utils import round_to_raster
from mrseq.utils import sys_defaults
from mrseq.utils import write_sequence
from mrseq.utils.constants import GYROMAGNETIC_RATIO_PROTON
from mrseq.utils.trajectory import cartesian_phase_encoding


def wasabi_gre_centric_kernel(
    system: pp.Opts,
    frequency_offsets: np.ndarray,
    norm_offset: float | None,
    t_recovery: float,
    t_recovery_norm: float,
    fov_xy: float,
    n_readout: int,
    readout_oversampling: float,
    n_phase_encoding: int,
    slice_thickness: float,
    rf_prep_duration: float,
    rf_prep_amplitude: float,
    rf_duration: float,
    rf_flip_angle: float,
    rf_bwt: float,
    rf_apodization: float,
    rf_spoiling_inc: float,
    adc_dwell_time: float,
) -> pp.Sequence:
    """Generate a WASABI sequence for simultaneous B0 and B1 mapping using a centric-out cartesian GRE readout.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    frequency_offsets
        Array of frequency offsets (in Hz).
    norm_offset
        Frequency offset of normalization offset (in Hz).
    t_recovery
        Recovery time between frequency offsets (in seconds).
    t_recovery_norm
        Recovery time before normalization offset (in seconds).
    fov_xy
        Field of view in x and y direction (in meters).
    n_readout
        Number of frequency encoding steps.
    readout_oversampling
        Readout oversampling factor, commonly 2. This reduces aliasing artifacts.
    n_phase_encoding
        Number of phase encoding steps.
    slice_thickness
        Slice thickness of the 2D slice (in meters).
    rf_prep_duration
        Duration of WASABI block pulse (in seconds)
    rf_prep_amplitude
        Amplitude of WASABI block pulse (in µT)
    rf_duration
        Duration of the rf excitation pulse (in seconds)
    rf_flip_angle
        Flip angle of rf excitation pulse (in degrees)
    rf_bwt
        Bandwidth-time product of rf excitation pulse (Hz * seconds)
    rf_apodization
        Apodization factor of rf excitation pulse
    rf_spoiling_inc
        Phase increment used for RF spoiling. Set to 0 to disable RF spoiling.
    adc_dwell_time
        Dwell time of ADC.

    Returns
    -------
    seq
        PyPulseq Sequence object.
    """
    # create PyPulseq Sequence object and set system limits
    seq = pp.Sequence(system=system)

    if readout_oversampling < 1:
        raise ValueError('Readout oversampling factor must be >= 1.')

    # add normalization offset to beginning of frequency offsets if specified
    if norm_offset is not None:
        frequency_offsets = np.concatenate(([norm_offset], frequency_offsets))

    # create WASABI block pulse
    rf_prep_flipangle_rad = rf_prep_amplitude * 1e-6 * rf_prep_duration * GYROMAGNETIC_RATIO_PROTON * 2 * np.pi
    rf_prep = pp.make_block_pulse(
        flip_angle=rf_prep_flipangle_rad,
        duration=rf_prep_duration,
        delay=system.rf_dead_time,
        system=system,
        use='preparation',
    )

    # create slice selective excitation pulse and gradients
    rf, gz, gzr = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(rf_flip_angle),
        duration=rf_duration,
        slice_thickness=slice_thickness,
        apodization=rf_apodization,
        time_bw_product=rf_bwt,
        delay=system.rf_dead_time,
        system=system,
        return_gz=True,
        use='excitation',
    )

    # define centric-out phase encoding steps
    kpe, _ = cartesian_phase_encoding(
        n_phase_encoding=n_phase_encoding,
        acceleration=1,
        sampling_order='low_high',
    )

    # create readout gradient and ADC
    delta_k = 1 / (fov_xy * readout_oversampling)
    n_readout_with_oversampling = int(n_readout * readout_oversampling)
    gx = pp.make_trapezoid(
        channel='x',
        flat_area=n_readout_with_oversampling * delta_k,
        flat_time=n_readout_with_oversampling * adc_dwell_time,
        system=system,
    )
    n_readout_with_oversampling = n_readout_with_oversampling + np.mod(n_readout_with_oversampling, 2)  # make even
    adc = pp.make_adc(num_samples=n_readout_with_oversampling, duration=gx.flat_time, delay=gx.rise_time, system=system)

    # create frequency encoding pre- and re-winder gradient
    gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2 - delta_k / 2, system=system)
    gx_post = pp.make_trapezoid(channel='x', area=-gx.area / 2 + delta_k / 2, system=system)
    k0_center_id = np.where((np.arange(n_readout_with_oversampling) - n_readout_with_oversampling / 2) * delta_k == 0)[
        0
    ][0]

    # create readout spoiler gradient
    gz_spoil = pp.make_trapezoid(channel='z', area=4 / slice_thickness, system=system)

    # create post preparation spoiler gradient
    prep_spoil = pp.make_trapezoid(channel='z', area=5 * gz_spoil.area, system=system)

    # calculate minimum echo time
    min_te = (
        rf.shape_dur / 2  # time from center to end of RF pulse
        + max(rf.ringdown_time, gz.fall_time)  # RF ringdown time or gradient fall time
        + pp.calc_duration(gzr, gx_pre)  # slice selection rewinder gradient
        + gx.delay  # potential delay of readout gradient
        + gx.rise_time  # rise time of readout gradient
        + (k0_center_id + 0.5) * adc.dwell  # time from beginning of ADC to time point of k-space center sample
    ).item()

    # calculate minimum repetition time
    min_tr = (
        pp.calc_duration(gz)  # rf pulse
        + pp.calc_duration(gzr, gx_pre)  # slice selection re-phasing gradient and readout pre-winder
        + pp.calc_duration(gx)  # readout gradient
        + pp.calc_duration(gz_spoil, gx_post)  # gradient spoiler or readout-re-winder
    )

    # loop over frequency offsets
    for rep_idx, freq_offset_hz in enumerate(frequency_offsets):
        # set repetition ('REP') label for current frequency offset
        rep_label = pp.make_label(type='SET', label='REP', value=int(rep_idx))

        # add delay for normalization offset
        if rep_idx == 0 and norm_offset is not None:
            seq.add_block(pp.make_delay(t_recovery_norm))
        else:
            seq.add_block(pp.make_delay(t_recovery))

        # update frequency offset of WASABI block pulse and add it to sequence
        rf_prep.freq_offset = freq_offset_hz
        seq.add_block(rf_prep)

        # add post prep spoiler gradient
        seq.add_block(prep_spoil)

        # reset rf spoiling
        rf_phase = 0.0
        rf_inc = 0.0

        # loop over phase encoding steps
        for pe_idx in kpe:
            # set phase encoding ('LIN') label
            pe_label = pp.make_label(type='SET', label='LIN', value=int(pe_idx + np.abs(np.min(kpe))))

            if rf_spoiling_inc > 0:
                rf.phase_offset = np.deg2rad(rf_phase)
                adc.phase_offset = np.deg2rad(rf_phase)

                # update rf pulse properties for the next loop
                rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
                rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

            # calculate phase encoding gradient for current phase encoding step
            gy_pre = pp.make_trapezoid(
                channel='y', area=pe_idx * delta_k, duration=pp.calc_duration(gzr, gx_pre), system=system
            )

            # add slice-selective rf pulse
            seq.add_block(rf, gz)

            # add slice-selection rewinder and readout pre-winder
            seq.add_block(gzr, gx_pre, gy_pre)

            # add readout gradient, ADC and labels
            seq.add_block(gx, adc, pe_label, rep_label)

            # add x and y re-winder and spoiler gradient in z-direction
            gy_post = pp.make_trapezoid(
                channel='y', amplitude=-gy_pre.amplitude, rise_time=gy_pre.rise_time, flat_time=gy_pre.flat_time
            )
            seq.add_block(gx_post, gy_post, gz_spoil)

    # write all required parameters in the seq-file header/definitions
    seq.set_definition('FOV', [fov_xy, fov_xy, slice_thickness])
    seq.set_definition('ReconMatrix', (n_readout, n_phase_encoding, 1))
    seq.set_definition('SliceThickness', slice_thickness)
    seq.set_definition('TE', min_te)
    seq.set_definition('TR', min_tr)
    seq.set_definition('frequency_offsets', frequency_offsets.tolist())
    seq.set_definition('ReadoutOversamplingFactor', readout_oversampling)

    return seq


def main(
    system: pp.Opts | None = None,
    frequency_offsets: np.ndarray | None = None,
    norm_offset: float | None = -35e3,
    t_recovery: float = 3.0,
    t_recovery_norm: float = 12.0,
    fov_xy: float = 200e-3,
    n_readout: int = 128,
    n_phase_encoding: int = 128,
    slice_thickness: float = 8e-3,
    receiver_bandwidth_per_pixel: float = 1000,  # Hz/pixel
    show_plots: bool = True,
    test_report: bool = True,
    timing_check: bool = True,
) -> tuple[pp.Sequence, Path]:
    """Generate a WASABI sequence for simultaneous B0 and B1 mapping.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    frequency_offsets
        Array of frequency offsets (in Hz).
    norm_offset
        Frequency offset of normalization offset (in Hz). Defaults to -35e3 Hz.
    t_recovery
        Recovery time between frequency offsets (in seconds).
    t_recovery_norm
        Recovery time before normalization offset (in seconds).
    fov_xy
        Field of view in x and y direction (in meters).
    n_readout
        Number of frequency encoding steps.
    n_phase_encoding
        Number of phase encoding steps.
    slice_thickness
        Slice thickness of the 2D slice (in meters).
    receiver_bandwidth_per_pixel
        Desired receiver bandwidth per pixel (in Hz/pixel). This is used to calculate the readout duration.
    show_plots
        Toggles sequence plot.
    test_report
        Toggles advanced test report.
    timing_check
        Toggles timing check of the sequence.

    Returns
    -------
    seq
        PyPulseq Sequence object.
    file_path
        Path to the sequence file without suffix (append '.seq' for the actual file).
    """
    if system is None:
        system = sys_defaults

    if frequency_offsets is None:
        frequency_offsets = np.linspace(-240, 240, 31)

    # define settings of rf excitation pulse
    rf_duration = 1.28e-3  # duration of the rf excitation pulse [s]
    rf_flip_angle = 10.0  # flip angle of rf excitation pulse [°]
    rf_bwt = 4  # bandwidth-time product of rf excitation pulse [Hz*s]
    rf_apodization = 0.5  # apodization factor of rf excitation pulse
    readout_oversampling = 2  # readout oversampling factor, commonly 2. This reduces aliasing artifacts.
    rf_spoiling_inc = 117  # RF spoiling phase increment [°]. Set to 0 for no RF spoiling.

    # define ADC and gradient timing
    n_readout_with_oversampling = int(n_readout * readout_oversampling)
    adc_dwell_time = round_to_raster(
        1.0 / (receiver_bandwidth_per_pixel * n_readout_with_oversampling), system.adc_raster_time
    )
    _, adc_dwell_time = find_gx_flat_time_on_adc_raster(
        n_readout_with_oversampling, adc_dwell_time, system.grad_raster_time, system.adc_raster_time
    )

    # WASABI block pulse
    rf_prep_duration: float = 5e-3
    rf_prep_amplitude: float = 3.75

    seq = wasabi_gre_centric_kernel(
        system=system,
        frequency_offsets=frequency_offsets,
        norm_offset=norm_offset,
        t_recovery=t_recovery,
        t_recovery_norm=t_recovery_norm,
        fov_xy=fov_xy,
        n_readout=n_readout,
        readout_oversampling=readout_oversampling,
        n_phase_encoding=n_phase_encoding,
        slice_thickness=slice_thickness,
        rf_prep_duration=rf_prep_duration,
        rf_prep_amplitude=rf_prep_amplitude,
        rf_duration=rf_duration,
        rf_flip_angle=rf_flip_angle,
        rf_bwt=rf_bwt,
        rf_apodization=rf_apodization,
        rf_spoiling_inc=rf_spoiling_inc,
        adc_dwell_time=adc_dwell_time,
    )

    # check timing of the sequence
    if timing_check:
        ok, error_report = seq.check_timing()
        if ok:
            print('\nTiming check passed successfully')
        else:
            print('\nTiming check failed! Error listing follows\n')
            print(error_report)

    # show advanced rest report
    if test_report:
        print('\nCreating advanced test report...')
        print(seq.test_report())

    # define sequence filename
    filename = f'{Path(__file__).stem}_{int(fov_xy * 1000)}fov_{n_readout}nx_{n_phase_encoding}ny'
    filename += f'_{len(seq.definitions["frequency_offsets"])}offsets'

    # save seq-file to disk
    output_path = Path.cwd() / 'output'
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving sequence file '{filename}.seq' into folder '{output_path}'.")
    write_sequence(seq, str(output_path / filename), create_signature=True)

    if show_plots:
        seq.plot()

    return seq, output_path / filename


if __name__ == '__main__':
    main()
