"""Gold standard GRE-based inversion recovery sequence with one inversion pulse before every readout."""

from pathlib import Path

import numpy as np
import pypulseq as pp

from mrseq.preparations import add_t1_inv_prep
from mrseq.utils import round_to_raster
from mrseq.utils import sys_defaults
from mrseq.utils import write_sequence


def t1_inv_rec_gre_single_line_kernel(
    system: pp.Opts,
    inversion_times: np.ndarray,
    te: float | None,
    tr: float,
    fov_xy: float,
    n_readout: int,
    n_phase_encoding: int,
    slice_thickness: float,
    rf_inv_duration: float,
    rf_inv_spoil_risetime: float,
    rf_inv_spoil_flattime: float,
    gx_pre_duration: float,
    gx_flat_time: float,
    rf_duration: float,
    rf_flip_angle: float,
    rf_bwt: float,
    rf_apodization: float,
) -> tuple[pp.Sequence, float, float]:
    """Generate a GRE-based inversion recovery sequence with one inversion pulse before every readout.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    inversion_times
        Array of inversion times (in seconds).
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
    rf_inv_duration
        Duration of adiabatic inversion pulse (in seconds)
    rf_inv_spoil_risetime
        Rise time of spoiler after inversion pulse (in seconds)
    rf_inv_spoil_flattime
        Flat time of spoiler after inversion pulse (in seconds)
    gx_pre_duration
        Duration of readout pre-winder gradient (in seconds)
    gx_flat_time
        Flat time of readout gradient (in seconds)
    rf_duration
        Duration of the rf excitation pulse (in seconds)
    rf_flip_angle
        Flip angle of rf excitation pulse (in degrees)
    rf_bwt
        Bandwidth-time product of rf excitation pulse (Hz * seconds)
    rf_apodization
        Apodization factor of rf excitation pulse

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

    # create slice selective excitation pulse and gradients
    rf, gz, gzr = pp.make_sinc_pulse(
        flip_angle=rf_flip_angle / 180 * np.pi,
        duration=rf_duration,
        slice_thickness=slice_thickness,
        apodization=rf_apodization,
        time_bw_product=rf_bwt,
        delay=system.rf_dead_time,
        system=system,
        return_gz=True,
        use='excitation',
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
        rf.shape_dur / 2  # time from center to end of RF pulse
        + max(rf.ringdown_time, gz.fall_time)  # RF ringdown time or gradient fall time
        + pp.calc_duration(gzr)  # slice selection rewinder gradient
        + pp.calc_duration(gx_pre)  # readout pre-winder gradient
        + gx.delay  # potential delay of readout gradient
        + gx.rise_time  # rise time of readout gradient
        + (k0_center_id + 0.5) * adc.dwell  # time from beginning of ADC to time point of k-space center sample
    ).item()

    # calculate delay to achieve desired echo time
    if te is None:
        te_delay = 0.0
    else:
        te_delay = round_to_raster(te - min_te, system.block_duration_raster)
        if te_delay < 0:
            raise ValueError(f'TE must be larger than {min_te * 1000:.3f} ms. Current value is {te * 1000:.3f} ms.')

    print(f'\nMinimum TE: {min_te * 1000:.3f} ms')

    # loop over inversion times
    for ti_idx, ti in enumerate(inversion_times):
        # set contrast ('ECO') label for current inversion time
        contrast_label = pp.make_label(type='SET', label='ECO', value=int(ti_idx))

        # loop over phase encoding steps
        for pe_idx in np.arange(n_phase_encoding):
            # set phase encoding ('LIN') label
            pe_label = pp.make_label(type='SET', label='LIN', value=int(pe_idx))

            # save start time of current TR block
            _start_time_tr_block = sum(seq.block_durations.values())

            # add T1 preparation block
            seq, _, time_since_inversion = add_t1_inv_prep(
                seq=seq,
                system=system,
                rf_duration=rf_inv_duration,
                spoiler_ramp_time=rf_inv_spoil_risetime,
                spoiler_flat_time=rf_inv_spoil_flattime,
            )

            # calculate and add inversion time (TI) delay.
            # TI is defined as time from middle of inversion pulse to middle of excitation pulse.
            ti_delay = ti - time_since_inversion - rf.delay - rf_duration / 2
            ti_delay = round_to_raster(ti_delay, system.block_duration_raster)
            if ti_delay < 0:
                raise ValueError(
                    'Inversion time too short for given RF inversion and post inversion spoiler durations.'
                )
            seq.add_block(pp.make_delay(ti_delay))

            # add rf pulse followed by rewinder gradient
            seq.add_block(rf, gz)
            seq.add_block(gzr)

            # add echo time delay
            seq.add_block(pp.make_delay(te_delay))

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
            if ti_idx == 0 and pe_idx == 0:
                time_to_first_tr_block = duration_tr_block

            if tr_delay < 0:
                raise ValueError('Desired TR too short for given sequence parameters.')

            seq.add_block(pp.make_delay(tr_delay))

    return seq, time_to_first_tr_block, min_te


def main(
    system: pp.Opts | None = None,
    inversion_times: np.ndarray | None = None,
    te: float | None = None,
    tr: float = 8,
    fov_xy: float = 128e-3,
    n_readout: int = 128,
    n_phase_encoding: int = 128,
    slice_thickness: float = 8e-3,
    show_plots: bool = True,
    test_report: bool = True,
    timing_check: bool = True,
    v141_compatibility: bool = True,
) -> tuple[pp.Sequence, Path]:
    """Generate a GRE-based inversion recovery sequence with one inversion pulse before every readout.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    inversion_times
        Array of inversion times (in seconds).
        Default values [0.025, 0.050, 0.3, 0.6, 1.2, 2.4, 4.8] s are used if set to None.
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
    v141_compatibility
        Save the sequence in pulseq v1.4.1 for backwards compatibility.

    Returns
    -------
    seq
        Sequence object of GRE-based T1 inversion recovery sequence.
    file_path
        Path to the sequence file.
    """
    if system is None:
        system = sys_defaults

    if inversion_times is None:
        inversion_times = np.array([0.025, 0.050, 0.3, 0.6, 1.2, 2.4, 4.8])

    # define T1prep settings
    rf_inv_duration = 10.24e-3  # duration of adiabatic inversion pulse [s]
    rf_inv_spoil_risetime = 0.6e-3  # rise time of spoiler after inversion pulse [s]
    rf_inv_spoil_flattime = 8.4e-3  # flat time of spoiler after inversion pulse [s]

    # define ADC and gradient timing
    adc_dwell = system.grad_raster_time
    gx_pre_duration = 1.0e-3  # duration of readout pre-winder gradient [s]
    gx_flat_time = n_readout * adc_dwell  # flat time of readout gradient [s]

    # define settings of rf excitation pulse
    rf_duration = 1.28e-3  # duration of the rf excitation pulse [s]
    rf_flip_angle = 12  # flip angle of rf excitation pulse [°]
    rf_bwt = 4  # bandwidth-time product of rf excitation pulse [Hz*s]
    rf_apodization = 0.5  # apodization factor of rf excitation pulse

    seq, time_to_first_tr_block, min_te = t1_inv_rec_gre_single_line_kernel(
        system=system,
        inversion_times=inversion_times,
        te=te,
        tr=tr,
        fov_xy=fov_xy,
        n_readout=n_readout,
        n_phase_encoding=n_phase_encoding,
        slice_thickness=slice_thickness,
        rf_inv_duration=rf_inv_duration,
        rf_inv_spoil_risetime=rf_inv_spoil_risetime,
        rf_inv_spoil_flattime=rf_inv_spoil_flattime,
        gx_pre_duration=gx_pre_duration,
        gx_flat_time=gx_flat_time,
        rf_duration=rf_duration,
        rf_flip_angle=rf_flip_angle,
        rf_bwt=rf_bwt,
        rf_apodization=rf_apodization,
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
        f'{Path(__file__).stem}_{int(fov_xy * 1000)}fov_{n_readout}nx_{n_phase_encoding}ny_{len(inversion_times)}TIs'
    )

    # write all required parameters in the seq-file header/definitions
    seq.set_definition('FOV', [fov_xy, fov_xy, slice_thickness])
    seq.set_definition('ReconMatrix', (n_readout, n_phase_encoding, 1))
    seq.set_definition('SliceThickness', slice_thickness)
    seq.set_definition('TE', te or min_te)
    seq.set_definition('TR', tr)
    seq.set_definition('TI', inversion_times.tolist())

    # save seq-file to disk
    output_path = Path.cwd() / 'output'
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving sequence file '{filename}.seq' into folder '{output_path}'.")
    write_sequence(seq, str(output_path / filename), create_signature=True, v141_compatibility=v141_compatibility)

    if show_plots:
        seq.plot(time_range=(0, time_to_first_tr_block))

    return seq, output_path / filename


if __name__ == '__main__':
    main()
