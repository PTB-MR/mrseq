"""Gold standard SE-based sequence for T1rho mapping with one preparation pulse before every readout."""

from pathlib import Path

import numpy as np
import pypulseq as pp

from mrseq.preparations import add_t1rho_prep
from mrseq.utils import round_to_raster
from mrseq.utils import sys_defaults


def t1rho_se_single_line_kernel(
    system: pp.Opts,
    spin_lock_times: np.ndarray,
    te: float | None,
    tr: float,
    fov_xy: float,
    n_readout: int,
    n_phase_encoding: int,
    slice_thickness: float,
    rf_spin_lock_duration: float,
    spin_lock_amplitude: float,
    add_spin_lock_spoiler: bool,
    spin_lock_spoiler_ramp_time: float,
    spin_lock_spoiler_flat_time: float,
    gx_pre_duration: float,
    gx_flat_time: float,
    rf90_duration: float,
    rf90_flip_angle: float,
    rf90_bwt: float,
    rf90_apodization: float,
    rf180_duration: float,
    rf180_flip_angle: float,
    rf180_bwt: float,
    rf180_apodization: float,
) -> tuple[pp.Sequence, float, float]:
    """Generate a SE-based sequence for T1rho mapping with one preparation pulse before every readout.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    spin_lock_times
        Array of spin lock times (in seconds).
    te
        Desired echo time (TE) (in seconds). Minimum echo time is used if set to None.
    tr
        Desired repetition time (TR) (in seconds).
    fov_xy
        Field of view in x and y direction (in meters).
    n_readout
        Number of frequency encoding steps.
    n_phase_encoding
        Number of phase encoding steps.
    slice_thickness
        Slice thickness of the 2D slice (in meters).
    rf_spin_lock_duration
        Duration of 90° tip-down/tip-up pulses for spin lock preparation (in seconds).
    spin_lock_amplitude
        Amplitude of the spin-lock pulse (in T).
    add_spin_lock_spoiler
        Toggles addition of spoiler gradients after the spin lock pulse.
    spin_lock_spoiler_ramp_time
        Duration of gradient spoiler ramps (in seconds).
    spin_lock_spoiler_flat_time
        Duration of gradient spoiler plateau (in seconds).
    gx_pre_duration
        Duration of readout pre-winder gradient (in seconds)
    gx_flat_time
        Flat time of readout gradient (in seconds)
    rf90_duration
        Duration of the rf excitation pulse (in seconds)
    rf90_flip_angle
        Flip angle of rf excitation pulse (in degrees)
    rf90_bwt
        Bandwidth-time product of rf excitation pulse (Hz * seconds)
    rf90_apodization
        Apodization factor of rf excitation pulse
    rf180_duration
        Duration of the rf refocusing pulse (in seconds)
    rf180_flip_angle
        Flip angle of rf refocusing pulse (in degrees)
    rf180_bwt
        Bandwidth-time product of rf refocusing pulse (Hz * seconds)
    rf180_apodization
        Apodization factor of rf refocusing pulse

    Returns
    -------
    seq
        PyPulseq Sequence object
    time_to_first_tr_block
        End point of first TR block.
    min_te
        Shortest possible echo time.
    """
    # create PyPulseq Sequence object and set system limits
    seq = pp.Sequence(system=system)

    # create slice selective 90° RF pulse and gradient
    rf90, gz90, _ = pp.make_sinc_pulse(  # type: ignore
        flip_angle=rf90_flip_angle / 180 * np.pi,
        duration=rf90_duration,
        slice_thickness=slice_thickness,
        apodization=rf90_apodization,
        time_bw_product=rf90_bwt,
        delay=system.rf_dead_time,
        system=system,
        return_gz=True,
        use='excitation',
    )

    # manually create rephasing gradient for 90° pulse with desired duration
    gz90_reph = pp.make_trapezoid(channel='z', system=system, area=-gz90.area / 2, duration=gx_pre_duration)

    # create 180° refocusing pulse and gradient
    rf180, gz180, _ = pp.make_sinc_pulse(  # type: ignore
        flip_angle=rf180_flip_angle / 180 * np.pi,
        duration=rf180_duration,
        slice_thickness=slice_thickness,
        apodization=rf180_apodization,
        time_bw_product=rf180_bwt,
        phase_offset=np.pi / 2,
        delay=system.rf_dead_time,
        system=system,
        return_gz=True,
        use='refocusing',
    )

    # create readout gradient and ADC
    delta_k = 1 / fov_xy
    gx = pp.make_trapezoid(channel='x', flat_area=n_readout * delta_k, flat_time=gx_flat_time, system=system)
    adc = pp.make_adc(num_samples=n_readout, duration=gx.flat_time, delay=gx.rise_time, system=system)

    # create frequency encoding pre- and re-winder gradient
    gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2 - delta_k / 2, duration=gx_pre_duration, system=system)
    gx_post = pp.make_trapezoid(channel='x', area=-gx.area / 2 + delta_k / 2, duration=gx_pre_duration, system=system)

    # calculate gradient areas for (linear) phase encoding direction
    phase_areas = (np.arange(n_phase_encoding) - n_phase_encoding / 2) * delta_k
    k0_center_id = np.where((np.arange(n_readout) - n_readout / 2) * delta_k == 0)[0][0]

    # create spoiler gradients
    gz_spoil = pp.make_trapezoid(channel='z', area=4 / slice_thickness, system=system)

    # calculate minimum echo time
    min_te = (
        rf90.shape_dur / 2  # half of RF pulse duration
        + max(rf90.ringdown_time, gz90.fall_time)  # RF ringdown time or gradient fall time
        + pp.calc_duration(gz90_reph)
        + pp.calc_duration(gz_spoil)
        + pp.calc_duration(rf180, gz180)
        + pp.calc_duration(gz_spoil)
        + pp.calc_duration(gx_pre)
        + gx.delay
        + gx.rise_time
        + (k0_center_id + 0.5) * adc.dwell  # time from beginning of ADC to time point of k-space center sample
    )

    if te is None:
        delay_gz90_reph_and_gz_spoil = delay_gz_spoil_and_gx_pre = 0
    else:
        # delays between excitation and refocusing pulse
        delay_gz90_reph_and_gz_spoil = (
            te / 2
            - rf90.shape_dur / 2  # time from center to end of RF pulse
            - max(rf90.ringdown_time, gz90.fall_time)  # RF ringdown time or gradient fall time
            - pp.calc_duration(gz90_reph)  # slice selection rewinder gradient
            - pp.calc_duration(gz_spoil)  # spoiler gradient before 180° pulse
            - rf180.delay  # potential delay of 180° refocusing pulse
            - rf180.shape_dur / 2  # time from to middle of RF pulse
        )
        delay_gz90_reph_and_gz_spoil = round_to_raster(delay_gz90_reph_and_gz_spoil, system.block_duration_raster)

        delay_gz_spoil_and_gx_pre = (
            te / 2
            - rf180.shape_dur / 2  # time from center to end of refocusing pulse
            - max(rf180.ringdown_time, gz180.fall_time)  # RF ringdown time or gradient fall time
            - pp.calc_duration(gz_spoil)  # spoiler gradient after 180° pulse
            - pp.calc_duration(gx_pre)  # readout pre-winder gradient
            - gx.delay  # potential delay of readout gradient
            - gx.rise_time  # rise time of readout gradient
            - (k0_center_id + 0.5) * adc.dwell  # time from beginning of ADC to time point of k-space center sample
        )

        delay_gz_spoil_and_gx_pre = round_to_raster(delay_gz_spoil_and_gx_pre, system.block_duration_raster)

    print(f'\nMinimum TE: {min_te * 1000:.3f} ms')

    # loop over spin lock times
    for tsl_idx, tsl in enumerate(spin_lock_times):
        # set contrast ('ECO') label for current spin lock time
        contrast_label = pp.make_label(type='SET', label='ECO', value=int(tsl_idx))

        # loop over phase encoding steps
        for pe_idx in np.arange(n_phase_encoding):
            # set phase encoding ('LIN') label
            pe_label = pp.make_label(type='SET', label='LIN', value=int(pe_idx))

            # save start time of current TR block
            _start_time_tr_block = sum(seq.block_durations.values())

            # add T1rho preparation block
            seq, _ = add_t1rho_prep(
                seq=seq,
                system=system,
                duration_90=rf_spin_lock_duration,
                spin_lock_time=tsl,
                spin_lock_amplitude=spin_lock_amplitude,
                add_spoiler=add_spin_lock_spoiler,
                spoiler_ramp_time=spin_lock_spoiler_ramp_time,
                spoiler_flat_time=spin_lock_spoiler_flat_time,
            )

            # add 90° excitation pulse followed by rewinder gradient
            seq.add_block(rf90, gz90)
            seq.add_block(gz90_reph)

            # add gradients and refocusing pulse
            seq.add_block(pp.make_delay(delay_gz90_reph_and_gz_spoil))
            seq.add_block(gz_spoil)
            seq.add_block(rf180, gz180)
            seq.add_block(gz_spoil)
            seq.add_block(pp.make_delay(delay_gz_spoil_and_gx_pre))

            # calculate phase encoding gradient for current phase encoding step
            gy_pre = pp.make_trapezoid(channel='y', area=phase_areas[pe_idx], duration=gx_pre_duration, system=system)

            # add pre-winder gradients and labels
            seq.add_block(gx_pre, gy_pre, pe_label, contrast_label)

            # add readout gradient and ADC
            seq.add_block(gx, adc)

            # add x and y re-winder and spoiler gradient in z-direction
            gy_pre.amplitude = -gy_pre.amplitude
            seq.add_block(gx_post, gy_pre, gz_spoil)

            # calculate TR delay
            duration_tr_block = sum(seq.block_durations.values()) - _start_time_tr_block
            tr_delay = round_to_raster(tr - duration_tr_block, system.block_duration_raster)

            # save time for sequence plot
            if tsl_idx == 0 and pe_idx == 0:
                time_to_first_tr_block = duration_tr_block

            if tr_delay < 0:
                raise ValueError('Desired TR too short for given sequence parameters.')

            seq.add_block(pp.make_delay(tr_delay))

    return seq, time_to_first_tr_block, min_te


def main(
    system: pp.Opts | None = None,
    spin_lock_times: np.ndarray | None = None,
    te: float | None = None,
    tr: float = 8,
    fov_xy: float = 128e-3,
    n_readout: int = 128,
    n_phase_encoding: int = 128,
    slice_thickness: float = 8e-3,
    show_plots: bool = True,
    test_report: bool = True,
    timing_check: bool = True,
) -> pp.Sequence:
    """Generate a SE-based sequence for T1rho mapping with one preparation pulse before every readout.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    spin_lock_times
        Array of inversion times (in seconds).
        Default values [0.025, 0.050, 0.1] s are used if set to None.
    te
        Desired echo time (TE) (in seconds). Minimum echo time is used if set to None.
    tr
        Desired repetition time (TR) (in seconds).
    fov_xy
        Field of view in x and y direction (in meters).
    n_readout
        Number of frequency encoding steps.
    n_phase_encoding
        Number of phase encoding steps.
    slice_thickness
        Slice thickness of the 2D slice (in meters).
    show_plots
        Toggles sequence plot.
    test_report
        Toggles advanced test report.
    timing_check
        Toggles timing check of the sequence.
    """
    if system is None:
        system = sys_defaults

    if spin_lock_times is None:
        spin_lock_times = np.array([0.025, 0.050, 0.1])

    # define settings of spin-lock preparation pulse
    rf_spin_lock_duration = 2e-3
    spin_lock_amplitude = 5.0e-6
    add_spin_lock_spoiler = True
    spin_lock_spoiler_ramp_time = 6e-4
    spin_lock_spoiler_flat_time = 8.4e-3

    # define ADC and gradient timing
    adc_dwell = system.grad_raster_time
    gx_pre_duration = 1.0e-3  # duration of readout pre-winder gradient [s]
    gx_flat_time = n_readout * adc_dwell  # flat time of readout gradient [s]

    # define settings of rf excitation pulse
    rf90_duration = 1.28e-3  # duration of the rf excitation pulse [s]
    rf90_flip_angle = 90  # flip angle of rf excitation pulse [°]
    rf90_bwt = 4  # bandwidth-time product of rf excitation pulse [Hz*s]
    rf90_apodization = 0.5  # apodization factor of rf excitation pulse

    # define settings of rf refocusing pulse
    rf180_duration = 2.56e-3  # duration of the rf refocusing pulse [s]
    rf180_flip_angle = 180  # flip angle of rf refocusing pulse [°]
    rf180_bwt = 4  # bandwidth-time product of rf refocusing pulse [Hz*s]
    rf180_apodization = 0.5  # apodization factor of rf refocusing pulse

    seq, time_to_first_tr_block, min_te = t1rho_se_single_line_kernel(
        system=system,
        spin_lock_times=spin_lock_times,
        te=te,
        tr=tr,
        fov_xy=fov_xy,
        n_readout=n_readout,
        n_phase_encoding=n_phase_encoding,
        slice_thickness=slice_thickness,
        rf_spin_lock_duration=rf_spin_lock_duration,
        spin_lock_amplitude=spin_lock_amplitude,
        add_spin_lock_spoiler=add_spin_lock_spoiler,
        spin_lock_spoiler_ramp_time=spin_lock_spoiler_ramp_time,
        spin_lock_spoiler_flat_time=spin_lock_spoiler_flat_time,
        gx_pre_duration=gx_pre_duration,
        gx_flat_time=gx_flat_time,
        rf90_duration=rf90_duration,
        rf90_flip_angle=rf90_flip_angle,
        rf90_bwt=rf90_bwt,
        rf90_apodization=rf90_apodization,
        rf180_duration=rf180_duration,
        rf180_flip_angle=rf180_flip_angle,
        rf180_bwt=rf180_bwt,
        rf180_apodization=rf180_apodization,
    )

    # check timing of the sequence
    if timing_check and not test_report:
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
    filename = (
        f'{Path(__file__).stem}_{int(fov_xy * 1000)}fov_{n_readout}nx_{n_phase_encoding}ny_{len(spin_lock_times)}TIs'
    )

    # write all required parameters in the seq-file header/definitions
    seq.set_definition('FOV', [fov_xy, fov_xy, slice_thickness])
    seq.set_definition('ReconMatrix', (n_readout, n_phase_encoding, 1))
    seq.set_definition('SliceThickness', slice_thickness)
    seq.set_definition('TE', te or min_te)
    seq.set_definition('TR', tr)
    seq.set_definition('TSL', spin_lock_times)

    # save seq-file to disk
    output_path = Path.cwd() / 'output'
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving sequence file '{filename}.seq' into folder '{output_path}'.")
    seq.write(str(output_path / filename), create_signature=True)

    if show_plots:
        seq.plot(time_range=(0, time_to_first_tr_block))

    return seq


if __name__ == '__main__':
    main()
