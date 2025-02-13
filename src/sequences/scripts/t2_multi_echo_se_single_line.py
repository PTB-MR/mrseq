"""Gold standard SE-based multi-echo sequence."""

from pathlib import Path

import numpy as np
import pypulseq as pp

from sequences.utils import round_to_raster
from sequences.utils import sys_defaults


def main(
    system: pp.Opts | None = None,
    echo_times: np.ndarray | None = None,
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
    """Generate a SE-based multi-echo sequence for T2 mapping.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    echo_times
        Array of echo times (in seconds).
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

    if echo_times is None:
        echo_times = np.array([0.024, 0.05, 0.1, 0.2, 0.4])

    # create PyPulseq Sequence object and set system limits
    seq = pp.Sequence(system=system)

    # define ADC and gradient timing
    adc_dwell = 20e-6  # 10 µs results in 100 kHz bandwidth
    gx_pre_duration = 1.0e-3  # duration of readout pre-winder gradient [s]
    gx_flat_time = n_readout * adc_dwell  # flat time of readout gradient [s]

    # define spoiler gradient settings
    gz_spoil_duration = 3.2e-3  # duration of spoiler gradient [s]
    gz_spoil_area = 4 / slice_thickness  # area / zeroth gradient moment of spoiler gradient

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

    # create slice selection 90° and 180° pulse and gradient
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

    # spoiler along slice direction before and after 180°-SE-refocusing pulse
    gz_spoil = pp.make_trapezoid(channel='z', system=system, area=gz_spoil_area, duration=gz_spoil_duration)

    # loop over all echo times
    for te_idx, te in enumerate(echo_times):
        contrast_label = pp.make_label(type='SET', label='ECO', value=int(te_idx))

        delay_gz90_reph_and_gz_spoil = (
            te / 2
            - rf90.shape_dur / 2  # time from center to end of rf pulse
            - max(rf90.ringdown_time, gz90.fall_time)  # RF ringdown time or gradient fall time
            - pp.calc_duration(gz90_reph)  # rephasing gradient duration
            - pp.calc_duration(gz_spoil)  # spoiler gradient duration
            - rf180.delay  # delay of 180° refocusing pulse
            - rf180.shape_dur / 2  # time from center to end of rf pulse
        )
        delay_gz90_reph_and_gz_spoil = round_to_raster(delay_gz90_reph_and_gz_spoil, system.block_duration_raster)
        if delay_gz90_reph_and_gz_spoil < 0:
            raise ValueError('Echo time too short for given sequence parameters.')

        delay_gz_spoil_and_gx_pre = (
            te / 2
            - rf180.shape_dur / 2  # time from center to end of refocusing pulse
            - max(rf180.ringdown_time, gz180.fall_time)  # RF ringdown time or gradient fall time
            - pp.calc_duration(gz_spoil)  # spoiler gradient after 180° pulse
            - pp.calc_duration(gx_pre)  # readout pre-winder gradient
            - gx.delay  # potential delay of readout gradient
            - gx.rise_time  # rise time of readout gradient
            - (k0_center_id + 0.5) * adc.dwell  # time from begin of ADC to time point of k-space center sample
        )
        delay_gz_spoil_and_gx_pre = round_to_raster(delay_gz_spoil_and_gx_pre, system.block_duration_raster)
        if delay_gz_spoil_and_gx_pre < 0:
            raise ValueError('Echo time too short for given sequence parameters.')

        # loop over phase encoding steps
        for pe_idx in np.arange(n_phase_encoding):
            # set phase encoding ('LIN') label
            pe_label = pp.make_label(type='SET', label='LIN', value=int(pe_idx))

            # save start time of current TR block
            _start_time_tr_block = sum(seq.block_durations.values())

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
            if show_plots and te_idx == 0 and pe_idx == 0:
                upper_time_limit = duration_tr_block

            if tr_delay < 0:
                raise ValueError('Desired TR too short for given sequence parameters.')

            seq.add_block(pp.make_delay(tr_delay))

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
    filename = f'{Path(__file__).stem}_{int(fov_xy * 1000)}fov_{n_readout}nx_{n_phase_encoding}ny_{len(echo_times)}TEs'

    # write all required parameters in the seq-file header/definitions
    seq.set_definition('FOV', [fov_xy, fov_xy, slice_thickness])
    seq.set_definition('ReconMatrix', (n_readout, n_phase_encoding, 1))
    seq.set_definition('SliceThickness', slice_thickness)
    seq.set_definition('TE', echo_times)
    seq.set_definition('TR', tr)

    # save seq-file to disk
    output_path = Path.cwd() / 'output'
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving sequence file '{filename}.seq' into folder '{output_path}'.")
    seq.write(str(output_path / filename), create_signature=True)

    # plot first TR block
    if show_plots:
        seq.plot(time_range=(0, upper_time_limit))

    return seq


if __name__ == '__main__':
    main()
