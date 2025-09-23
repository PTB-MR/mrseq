"""2D Cartesian FLASH with T2-preparation pulses for T2 mapping."""

from pathlib import Path

import numpy as np
import pypulseq as pp

from mrseq.preparations.t2_prep import add_t2_prep
from mrseq.utils import round_to_raster
from mrseq.utils import sys_defaults
from mrseq.utils.trajectory import cartesian_phase_encoding


def t2_t2prep_flash_kernel(
    system: pp.Opts,
    te: float | None,
    tr: float | None,
    t2_prep_echo_times: np.ndarray,
    n_recovery_cardiac_cycles: int,
    cardiac_trigger_delay: float,
    fov_xy: float,
    n_readout: int,
    readout_oversampling: float,
    acceleration: int,
    n_fully_sampled_center: int,
    slice_thickness: float,
    gx_pre_duration: float,
    gx_flat_time: float,
    rf_duration: float,
    rf_flip_angle: float,
    rf_bwt: float,
    rf_apodization: float,
    rf_spoiling_phase_increment: float,
    gz_spoil_duration: float,
    gz_spoil_area: float,
) -> tuple[pp.Sequence, float, float]:
    """Generate a FLASH sequence with T2-preparation pulses.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    te
        Desired echo time (TE) (in seconds). Minimum echo time is used if set to None.
    tr
        Desired repetition time (TR) (in seconds).
    t2_prep_echo_times
        Echo times of T2-preparation pulse
    n_recovery_cardiac_cycles
        Number of cardiac cycles for magnetization recovery after each T2-pepared acquisition.
    cardiac_trigger_delay
        Delay after cardiac trigger (in seconds).
    fov_xy
        Field of view in x and y direction (in meters).
    n_readout
        Number of frequency encoding steps.
    readout_oversampling
        Readout oversampling factor, commonly 2. This reduces aliasing artifacts.
    acceleration
        Uniform undersampling factor along the phase encoding direction
    n_fully_sampled_center
        Number of phsae encoding points in the fully sampled center. This will reduce the overall undersampling factor.
    slice_thickness
        Slice thickness of the 2D slice (in meters).
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
    rf_spoiling_phase_increment
        RF spoiling phase increment (in degrees). Set to 0 for no RF spoiling.
    gz_spoil_duration
        Duration of spoiler gradient (in seconds)
    gz_spoil_area
        Area of spoiler gradient (in mT/m * s)

    Returns
    -------
    seq
        PyPulseq Sequence object
    min_te
        Shortest possible echo time.
    min_tr
        Shortest possible echo time.

    """
    # create PyPulseq Sequence object and set system limits
    seq = pp.Sequence(system=system)

    # create slice selective excitation pulse and gradients
    rf, gz, gzr = pp.make_sinc_pulse(  # type: ignore
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
    n_readout_with_oversampling = int(n_readout * readout_oversampling)
    n_readout_with_oversampling = n_readout_with_oversampling + np.mod(n_readout_with_oversampling, 2)  # make even
    adc = pp.make_adc(num_samples=n_readout_with_oversampling, duration=gx.flat_time, delay=gx.rise_time, system=system)

    # create frequency encoding pre- and re-winder gradient
    gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2 - delta_k / 2, duration=gx_pre_duration, system=system)
    gx_post = pp.make_trapezoid(channel='x', area=-gx.area / 2 + delta_k / 2, duration=gx_pre_duration, system=system)
    k0_center_id = np.where((np.arange(n_readout) - n_readout / 2) * delta_k == 0)[0][0]

    # create phase encoding steps
    pe_steps, pe_fully_sampled_center = cartesian_phase_encoding(
        n_phase_encoding=n_readout,
        acceleration=acceleration,
        n_fully_sampled_center=n_fully_sampled_center,
        sampling_order='low_high',
    )

    # create spoiler gradients
    gz_spoil = pp.make_trapezoid(channel='z', system=system, area=gz_spoil_area, duration=gz_spoil_duration)

    # calculate minimum echo time
    if te is None:
        gzr_gx_dur = pp.calc_duration(gzr, gx_pre)  # gzr and gx_pre are applied simultaneously
    else:
        gzr_gx_dur = pp.calc_duration(gzr) + pp.calc_duration(gx_pre)  # gzr and gx_pre are applied sequentially

    min_te = (
        rf.shape_dur / 2  # time from center to end of RF pulse
        + max(rf.ringdown_time, gz.fall_time)  # RF ringdown time or gradient fall time
        + gzr_gx_dur  # slice selection re-phasing gradient and readout pre-winder
        + gx.delay  # potential delay of readout gradient
        + gx.rise_time  # rise time of readout gradient
        + (k0_center_id + 0.5) * adc.dwell  # time from beginning of ADC to time point of k-space center sample
    )

    # calculate echo time delay (te_delay)
    te_delay = 0 if te is None else round_to_raster(te - min_te, system.block_duration_raster)
    if not te_delay >= 0:
        raise ValueError(f'TE must be larger than {min_te * 1000:.2f} ms. Current value is {te * 1000:.2f} ms.')

    # calculate minimum repetition time
    min_tr = (
        pp.calc_duration(gz)  # rf pulse
        + gzr_gx_dur  # slice selection re-phasing gradient and readout pre-winder
        + pp.calc_duration(gx)  # readout gradient
        + pp.calc_duration(gz_spoil, gx_post)  # gradient spoiler or readout-re-winder
    )

    # calculate repetition time delay (tr_delay)
    current_min_tr = min_tr + te_delay
    tr_delay = 0 if tr is None else round_to_raster(tr - current_min_tr, system.block_duration_raster)

    if not tr_delay >= 0:
        raise ValueError(f'TR must be larger than {current_min_tr * 1000:.2f} ms. Current value is {tr * 1000:.2f} ms.')

    print(f'\nCurrent echo time = {(min_te + te_delay) * 1000:.2f} ms')
    print(f'Current repetition time = {(current_min_tr + tr_delay) * 1000:.2f} ms')

    # choose initial rf phase offset
    rf_phase = 0
    rf_inc = 0

    # obtain noise samples
    # seq.add_block(pp.make_label(label='LIN', type='SET', value=0), pp.make_label(label='SLC', type='SET', value=0))
    # seq.add_block(adc, pp.make_label(label='NOISE', type='SET', value=True))
    # seq.add_block(pp.make_label(label='NOISE', type='SET', value=False))

    for t2_prep_echo_time in t2_prep_echo_times:
        if t2_prep_echo_time > 0:
            # get prep block duration and calculate corresponding trigger delay
            t2prep_block, prep_dur = add_t2_prep(echo_time=t2_prep_echo_time, system=system)
            current_trig_delay = cardiac_trigger_delay - prep_dur

            # add trigger
            seq.add_block(pp.make_trigger(channel='physio1', duration=current_trig_delay))

            # add all events of T2prep block
            for idx in t2prep_block.block_events:
                seq.add_block(t2prep_block.get_block(idx))

        else:
            seq.add_block(pp.make_trigger(channel='physio1', duration=cardiac_trigger_delay))

        for pe_index_ in pe_steps:
            # calculate current phase_offset if rf_spoiling is activated
            if rf_spoiling_phase_increment > 0:
                rf.phase_offset = rf_phase / 180 * np.pi
                adc.phase_offset = rf_phase / 180 * np.pi

            # add slice selective excitation pulse
            seq.add_block(rf, gz)

            # update rf phase offset for the next excitation pulse
            rf_inc = divmod(rf_inc + rf_spoiling_phase_increment, 360.0)[1]
            rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

            # set labels for the next spoke
            labels = []
            labels.append(pp.make_label(label='LIN', type='SET', value=int(pe_index_ - np.min(pe_steps))))
            labels.append(pp.make_label(label='IMA', type='SET', value=True))
            labels.append(pp.make_label(label='REF', type='SET', value=pe_index_ in pe_fully_sampled_center))
            print(pe_index_ in pe_fully_sampled_center)

            # calculate current phase encoding gradient
            gy_pre = pp.make_trapezoid(channel='y', area=delta_k * pe_index_, duration=gx_pre_duration, system=system)

            if te is not None:
                seq.add_block(gzr)
                seq.add_block(pp.make_delay(te_delay))
                seq.add_block(gx_pre, gy_pre, *labels)
            else:
                seq.add_block(gx_pre, gy_pre, gzr, *labels)

            # add the readout gradient and ADC
            seq.add_block(gx, adc)

            gy_pre.amplitude = -gy_pre.amplitude
            seq.add_block(gx_post, gy_pre, gz_spoil)

            # add delay in case TR > min_TR
            if tr_delay > 0:
                seq.add_block(pp.make_delay(tr_delay))

        for _ in range(n_recovery_cardiac_cycles):
            seq.add_block(pp.make_trigger(channel='physio1', duration=0.2))  # add delay of for magnetization recovery

    return seq, min_te, min_tr


def main(
    system: pp.Opts | None = None,
    te: float | None = None,
    tr: float | None = None,
    t2_prep_echo_times: np.ndarray | None = None,
    cardiac_trigger_delay: float = 0.4,
    fov_xy: float = 128e-3,
    n_readout: int = 128,
    acceleration: int = 2,
    n_fully_sampled_center: int = 12,
    slice_thickness: float = 8e-3,
    show_plots: bool = True,
    test_report: bool = True,
    timing_check: bool = True,
) -> pp.Sequence:
    """Generate a FLASH sequence with T2-preparation pulses.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    te
        Desired echo time (TE) (in seconds). Minimum echo time is used if set to None.
    tr
        Desired repetition time (TR) (in seconds). Minimum repetition time is used if set to None.
    t2_prep_echo_times
        Echo times of T2-preparation pulse. If None, default values of [0, 50, 100]ms are used.
    cardiac_trigger_delay
        Delay after cardiac trigger (in seconds).
    fov_xy
        Field of view in x and y direction (in meters).
    n_readout
        Number of frequency encoding steps.
    acceleration
        Uniform undersampling factor along the phase encoding direction
    n_fully_sampled_center
        Number of phsae encoding points in the fully sampled center. This will reduce the overall undersampling factor.
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

    if t2_prep_echo_times is None:
        t2_prep_echo_times = [0.0, 0.05, 0.1]

    n_recovery_cardiac_cycles = 3

    # define ADC and gradient timing
    adc_dwell = system.grad_raster_time
    gx_pre_duration = 1.0e-3  # duration of readout pre-winder gradient [s]
    gx_flat_time = n_readout * adc_dwell  # flat time of readout gradient [s]

    # define settings of rf excitation pulse
    rf_duration = 1.28e-3  # duration of the rf excitation pulse [s]
    rf_flip_angle = 12  # flip angle of rf excitation pulse [°]
    rf_bwt = 4  # bandwidth-time product of rf excitation pulse [Hz*s]
    rf_apodization = 0.5  # apodization factor of rf excitation pulse
    readout_oversampling = 2  # readout oversampling factor, commonly 2. This reduces aliasing artifacts.

    # define spoiling
    gz_spoil_duration = 0.8e-3  # duration of spoiler gradient [s]
    gz_spoil_area = 4 / slice_thickness  # area / zeroth gradient moment of spoiler gradient
    rf_spoiling_phase_increment = 117  # RF spoiling phase increment [°]. Set to 0 for no RF spoiling.

    # define sequence filename
    filename = f'{Path(__file__).stem}_{int(fov_xy * 1000)}fov_{n_readout}nx_{acceleration}us_'
    filename += f'trig{int(cardiac_trigger_delay * 1000)}ms'

    output_path = Path.cwd() / 'output'
    output_path.mkdir(parents=True, exist_ok=True)

    # delete existing header file
    if (output_path / Path(filename + '.mrd')).exists():
        (output_path / Path(filename + '.mrd')).unlink()

    seq, min_te, min_tr = t2_t2prep_flash_kernel(
        system=system,
        te=te,
        tr=tr,
        t2_prep_echo_times=t2_prep_echo_times,
        n_recovery_cardiac_cycles=n_recovery_cardiac_cycles,
        cardiac_trigger_delay=cardiac_trigger_delay,
        fov_xy=fov_xy,
        n_readout=n_readout,
        readout_oversampling=readout_oversampling,
        acceleration=acceleration,
        n_fully_sampled_center=n_fully_sampled_center,
        slice_thickness=slice_thickness,
        gx_pre_duration=gx_pre_duration,
        gx_flat_time=gx_flat_time,
        rf_duration=rf_duration,
        rf_flip_angle=rf_flip_angle,
        rf_bwt=rf_bwt,
        rf_apodization=rf_apodization,
        rf_spoiling_phase_increment=rf_spoiling_phase_increment,
        gz_spoil_duration=gz_spoil_duration,
        gz_spoil_area=gz_spoil_area,
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

    # write all required parameters in the seq-file header/definitions
    seq.set_definition('FOV', [fov_xy, fov_xy, slice_thickness])
    seq.set_definition('ReconMatrix', (n_readout, n_readout, 1))
    seq.set_definition('SliceThickness', slice_thickness)
    seq.set_definition('TE', te or min_te)
    seq.set_definition('TR', tr or min_tr)

    # save seq-file to disk
    print(f"\nSaving sequence file '{filename}.seq' into folder '{output_path}'.")
    seq.write(str(output_path / filename), create_signature=True)

    if show_plots:
        seq.plot()

    return seq


if __name__ == '__main__':
    main()
