"""2D Cartesian MOLLI with bSSFP mapping for cardiac T1 mapping."""

from pathlib import Path

import numpy as np
import pypulseq as pp

from mrseq.preparations.t1_inv_prep import add_t1_inv_prep
from mrseq.utils import find_gx_flat_time_on_adc_raster
from mrseq.utils import round_to_raster
from mrseq.utils import sys_defaults
from mrseq.utils.trajectory import cartesian_phase_encoding


def t1_molli_bssfp_kernel(
    system: pp.Opts,
    te: float | None,
    tr: float | None,
    inversion_times: np.ndarray,
    cardiac_trigger_delay: float,
    fov_xy: float,
    n_readout: int,
    readout_oversampling: float,
    acceleration: int,
    n_fully_sampled_center: int,
    slice_thickness: float,
    n_bssfp_startup_pulses: int,
    gx_pre_duration: float,
    gx_flat_time: float,
    rf_duration: float,
    rf_flip_angle: float,
    rf_bwt: float,
    rf_apodization: float,
) -> tuple[pp.Sequence, float, float]:
    """Generate a 5(3)3 MOLLI sequence with bSSFP readout for cardiac T1 mapping.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    te
        Desired echo time (TE) (in seconds). Minimum echo time is used if set to None.
    tr
        Desired repetition time (TR) (in seconds). Minimum repetition time is used if set to None.
    inversion_times
        First inversion times for both acquisition blocks (in seconds).
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
    n_bssfp_startup_pulses
        Number of bSSFP startup pulses to reach steady state. A linear flip angle ramp is used during the startup.
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

    print(f'Current receiver bandwidth = {1 / gx.flat_time:.0f} Hz/pixel')

    # create frequency encoding pre- and re-winder gradient
    gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2 - delta_k / 2, duration=gx_pre_duration, system=system)
    gx_post = pp.make_trapezoid(channel='x', area=-gx.area / 2 + delta_k / 2, duration=gx_pre_duration, system=system)
    k0_center_id = np.where((np.arange(n_readout_with_oversampling) - n_readout_with_oversampling / 2) * delta_k == 0)[
        0
    ][0]

    # create phase encoding steps
    pe_steps, pe_fully_sampled_center = cartesian_phase_encoding(
        n_phase_encoding=n_readout,
        acceleration=acceleration,
        n_fully_sampled_center=n_fully_sampled_center,
        sampling_order='low_high',
    )

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
    ).item()

    # calculate echo time delay (te_delay)
    te_delay = 0 if te is None else round_to_raster(te - min_te, system.block_duration_raster)
    if not te_delay >= 0:
        raise ValueError(f'TE must be larger than {min_te * 1000:.3f} ms. Current value is {te * 1000:.3f} ms.')
    current_te = min_te + te_delay

    # calculate minimum repetition time
    min_tr = (
        pp.calc_duration(gz)  # rf pulse
        + gzr_gx_dur  # slice selection re-phasing gradient and readout pre-winder
        + pp.calc_duration(gx)  # readout gradient
        + pp.calc_duration(gzr, gx_post)  # readout or slice rewinder
    )

    # calculate repetition time delay (tr_delay)
    current_min_tr = min_tr + te_delay
    tr_delay = 0 if tr is None else round_to_raster(tr - current_min_tr, system.block_duration_raster)
    if not tr_delay >= 0:
        raise ValueError(f'TR must be larger than {current_min_tr * 1000:.3f} ms. Current value is {tr * 1000:.3f} ms.')
    current_tr = current_min_tr + tr_delay

    print(f'\nCurrent echo time = {current_te * 1000:.3f} ms')
    print(f'Current repetition time = {current_tr * 1000:.3f} ms')
    print(f'Acquisition window per cardiac cycle = {current_tr * len(pe_steps) * 1000:.3f} ms')

    # obtain noise samples
    seq.add_block(pp.make_label(label='LIN', type='SET', value=0), pp.make_label(label='SLC', type='SET', value=0))
    seq.add_block(adc, pp.make_label(label='NOISE', type='SET', value=True))
    seq.add_block(pp.make_label(label='NOISE', type='SET', value=False))
    seq.add_block(pp.make_delay(system.rf_dead_time))

    # Create inversion pulse
    t1_inv_prep, block_duration, time_since_inversion = add_t1_inv_prep(system=system)

    # In the first part 5 images are acquired in 5 cardiac cycles, followed by 3 cardiac cycles without data
    # acquisition for signal recovery. Then 3 images are acquired in 3 cardiac cycles in the second part.
    contrast_index = 0
    for part_idx, n_cycles in enumerate((5, 3)):
        for cardiac_index in range(n_cycles):
            if cardiac_index == 0:
                # waiting time to achieve inversion time
                delay_after_inversion_pulse = (
                    inversion_times[part_idx]
                    - time_since_inversion
                    - n_bssfp_startup_pulses * current_tr
                    - current_te
                    - rf.shape_dur / 2
                    - max(rf.delay, gz.rise_time)
                )

                # add trigger
                current_trig_delay = round_to_raster(
                    cardiac_trigger_delay - delay_after_inversion_pulse - block_duration,
                    raster_time=system.block_duration_raster,
                )
                seq.add_block(pp.make_trigger(channel='physio1', duration=current_trig_delay))

                # add inversion pulse
                for idx in t1_inv_prep.block_events:
                    seq.add_block(t1_inv_prep.get_block(idx))

                # wait until inversion time is reached
                seq.add_block(
                    pp.make_delay(
                        round_to_raster(delay_after_inversion_pulse, raster_time=system.block_duration_raster)
                    )
                )
            else:
                current_trig_delay = round_to_raster(
                    cardiac_trigger_delay
                    - n_bssfp_startup_pulses * current_tr
                    - current_te
                    - rf.shape_dur / 2
                    - max(rf.delay, gz.rise_time),
                    raster_time=system.block_duration_raster,
                )
                seq.add_block(pp.make_trigger(channel='physio1', duration=current_trig_delay))

            rf_signal = rf.signal.copy()
            for pe_index in range(-n_bssfp_startup_pulses, len(pe_steps)):
                # add slice selective excitation pulse
                if pe_index < 0:
                    # use linear flip angle ramp for bSSFP startup pulses
                    rf.signal = rf_signal * 1 / n_bssfp_startup_pulses * (n_bssfp_startup_pulses + pe_index + 1)
                else:
                    rf.signal = rf_signal
                if np.mod(pe_index, 2) == 0:
                    rf.phase_offset = -np.pi
                    adc.phase_offset = -np.pi
                else:
                    rf.phase_offset = 0.0
                    adc.phase_offset = 0.0
                seq.add_block(rf, gz)

                # set labels for the next spoke
                labels = []
                labels.append(pp.make_label(label='LIN', type='SET', value=int(pe_steps[pe_index] - np.min(pe_steps))))
                labels.append(
                    pp.make_label(label='IMA', type='SET', value=pe_steps[pe_index] in pe_fully_sampled_center)
                )
                labels.append(pp.make_label(type='SET', label='ECO', value=int(contrast_index)))

                # calculate current phase encoding gradient
                gy_pre = pp.make_trapezoid(
                    channel='y',
                    area=delta_k * pe_steps[pe_index if pe_index >= 0 else 0],
                    duration=gx_pre_duration,
                    system=system,
                )

                if te is not None:
                    seq.add_block(gzr)
                    seq.add_block(pp.make_delay(te_delay))
                    seq.add_block(gx_pre, gy_pre, *labels)
                else:
                    seq.add_block(gx_pre, gy_pre, gzr, *labels)

                # add the readout gradient and ADC
                if pe_index >= 0:
                    seq.add_block(gx, adc)
                else:
                    seq.add_block(gx)

                gy_pre.amplitude = -gy_pre.amplitude
                seq.add_block(gx_post, gy_pre, gzr)

                # add delay in case TR > min_TR
                if tr_delay > 0:
                    seq.add_block(pp.make_delay(tr_delay))

            contrast_index += 1

        # add three cardiac cycles for signal recovery
        if part_idx == 0:
            for _cardiac_index in range(3):
                seq.add_block(
                    pp.make_trigger(
                        channel='physio1',
                        duration=round_to_raster(cardiac_trigger_delay, raster_time=system.block_duration_raster),
                    )
                )

    return seq, min_te, min_tr


def main(
    system: pp.Opts | None = None,
    te: float | None = None,
    tr: float | None = None,
    inversion_times: np.ndarray | None = None,
    cardiac_trigger_delay: float = 0.4,
    fov_xy: float = 128e-3,
    n_readout: int = 128,
    acceleration: int = 2,
    n_fully_sampled_center: int = 12,
    slice_thickness: float = 8e-3,
    receiver_bandwidth_per_pixel: float = 1000,  # Hz/pixel
    show_plots: bool = True,
    test_report: bool = True,
    timing_check: bool = True,
) -> pp.Sequence:
    """Generate a 5(3)3 MOLLI sequence with bSSFP readout for cardiac T1 mapping.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    te
        Desired echo time (TE) (in seconds). Minimum echo time is used if set to None.
    tr
        Desired repetition time (TR) (in seconds). Minimum repetition time is used if set to None.
    inversion_times
        First inversion times for both acquisition blocks (in seconds).
        If None, default values of [100, 180]ms are used.
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
    receiver_bandwidth_per_pixel
        Desired receiver bandwidth per pixel (in Hz/pixel). This is used to calculate the readout duration.
    show_plots
        Toggles sequence plot.
    test_report
        Toggles advanced test report.
    timing_check
        Toggles timing check of the sequence.
    """
    if system is None:
        system = sys_defaults

    if inversion_times is None:
        inversion_times = [0.1, 0.18]

    # define settings of rf excitation pulse
    rf_duration = 0.5e-3  # duration of the rf excitation pulse [s]
    rf_flip_angle = 35  # flip angle of rf excitation pulse [°]
    rf_bwt = 1.5  # bandwidth-time product of rf excitation pulse [Hz*s]
    rf_apodization = 0.5  # apodization factor of rf excitation pulse
    readout_oversampling = 2  # readout oversampling factor, commonly 2. This reduces aliasing artifacts.

    # define ADC and gradient timing
    n_readout_with_oversampling = int(n_readout * readout_oversampling)
    adc_dwell_time = round_to_raster(
        1.0 / (receiver_bandwidth_per_pixel * n_readout_with_oversampling), system.adc_raster_time
    )
    gx_pre_duration = 0.72e-3  # duration of readout pre-winder gradient [s]
    gx_flat_time, adc_dwell_time = find_gx_flat_time_on_adc_raster(
        n_readout_with_oversampling, adc_dwell_time, system.grad_raster_time, system.adc_raster_time
    )

    n_bssfp_startup_pulses = 11  # number of bSSFP startup pulses to reach steady state.

    # define sequence filename
    filename = f'{Path(__file__).stem}_{int(fov_xy * 1000)}fov_{n_readout}nx_{acceleration}us_'
    filename += f'trig{int(cardiac_trigger_delay * 1000)}ms'

    output_path = Path.cwd() / 'output'
    output_path.mkdir(parents=True, exist_ok=True)

    seq, min_te, min_tr = t1_molli_bssfp_kernel(
        system=system,
        te=te,
        tr=tr,
        inversion_times=inversion_times,
        cardiac_trigger_delay=cardiac_trigger_delay,
        fov_xy=fov_xy,
        n_readout=n_readout,
        readout_oversampling=readout_oversampling,
        acceleration=acceleration,
        n_fully_sampled_center=n_fully_sampled_center,
        slice_thickness=slice_thickness,
        n_bssfp_startup_pulses=n_bssfp_startup_pulses,
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

    # write all required parameters in the seq-file header/definitions
    seq.set_definition('FOV', [fov_xy, fov_xy, slice_thickness])
    seq.set_definition('ReconMatrix', (n_readout, n_readout, 1))
    seq.set_definition('SliceThickness', slice_thickness)
    seq.set_definition('TE', te or min_te)
    seq.set_definition('TR', tr or min_tr)
    seq.set_definition('TI', inversion_times)
    seq.set_definition('ReadoutOversamplingFactor', readout_oversampling)

    # save seq-file to disk
    print(f"\nSaving sequence file '{filename}.seq' into folder '{output_path}'.")
    seq.write(str(output_path / filename), create_signature=True)

    if show_plots:
        seq.plot()

    return seq, output_path / filename


if __name__ == '__main__':
    main()
