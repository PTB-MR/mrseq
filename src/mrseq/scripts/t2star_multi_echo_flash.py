"""2D Cartesian FLASH with multi-echo readout for T2* mapping."""

from pathlib import Path

import ismrmrd
import numpy as np
import pypulseq as pp

from mrseq.utils import find_gx_flat_time_on_adc_raster
from mrseq.utils import round_to_raster
from mrseq.utils import sys_defaults
from mrseq.utils.create_ismrmrd_header import create_header
from mrseq.utils.trajectory import MultiEchoAcquisition
from mrseq.utils.trajectory import cartesian_phase_encoding


def t2star_multi_echo_flash_kernel(
    system: pp.Opts,
    te: float | None,
    delta_te: float | None,
    tr: float | None,
    n_echoes: int,
    n_recovery_cardiac_cycles: int,
    cardiac_trigger_delay: float,
    fov_xy: float,
    n_readout: int,
    readout_oversampling: float,
    partial_echo_factor: float,
    acceleration: int,
    n_fully_sampled_center: int,
    n_pe_points_per_cardiac_cycle: int,
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
    mrd_header_file: str | None,
) -> tuple[pp.Sequence, float, float]:
    """Generate a FLASH sequence with multiple echoes.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    te
        Desired echo time (TE) (in seconds). Minimum echo time is used if set to None.
    delta_te
            Desired echo spacing (in seconds). Minimum echo spacing is used if set to None.
    tr
        Desired repetition time (TR) (in seconds).
    n_echoes
        Number of echoes.
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
    partial_echo_factor
        Partial echo factor, commonly between 0.7 and 1. This reduces the echo time.
    acceleration
        Uniform undersampling factor along the phase encoding direction
    n_fully_sampled_center
        Number of phsae encoding points in the fully sampled center. This will reduce the overall undersampling factor.
    n_pe_points_per_cardiac_cycle
        Number of phase encoding points per cardiac cycle.
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
    mrd_header_file
        Filename of the ISMRMRD header file to be created. If None, no header file is created.

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

    multi_echo_gradient = MultiEchoAcquisition(
        system=system,
        delta_te=delta_te,
        fov=fov_xy,
        n_readout=n_readout,
        readout_oversampling=readout_oversampling,
        partial_echo_factor=partial_echo_factor,
        gx_flat_time=gx_flat_time,
        gx_pre_duration=gx_pre_duration,
    )

    # create phase encoding steps
    pe_steps, pe_fully_sampled_center = cartesian_phase_encoding(
        n_phase_encoding=n_readout,
        acceleration=acceleration,
        n_fully_sampled_center=n_fully_sampled_center,
        sampling_order='low_high',
        n_phase_encoding_per_shot=n_pe_points_per_cardiac_cycle,
    )

    # create spoiler gradients
    gz_spoil = pp.make_trapezoid(channel='z', system=system, area=gz_spoil_area, duration=gz_spoil_duration)

    # calculate minimum echo time
    if te is None:
        gzr_gx_dur = pp.calc_duration(gzr, multi_echo_gradient._gx_pre)  # gzr and gx_pre are applied simultaneously
    else:
        gzr_gx_dur = pp.calc_duration(gzr) + pp.calc_duration(
            multi_echo_gradient._gx_pre
        )  # gzr and gx_pre are applied sequentially

    min_te = (
        rf.shape_dur / 2  # time from center to end of RF pulse
        + max(rf.ringdown_time, gz.fall_time)  # RF ringdown time or gradient fall time
        + gzr_gx_dur  # slice selection re-phasing gradient and readout pre-winder
        + multi_echo_gradient._gx.delay  # potential delay of readout gradient
        + multi_echo_gradient._gx.rise_time  # rise time of readout gradient
        + (multi_echo_gradient._n_readout_pre_echo + 0.5) * multi_echo_gradient._adc.dwell
    )

    # calculate echo time delay (te_delay)
    te_delay = 0 if te is None else round_to_raster(te - min_te, system.block_duration_raster)
    if not te_delay >= 0:
        raise ValueError(f'TE must be larger than {min_te * 1000:.2f} ms. Current value is {te * 1000:.2f} ms.')
    current_te = min_te + te_delay

    # calculate minimum repetition time
    min_tr = (
        pp.calc_duration(gz)  # rf pulse
        + gzr_gx_dur  # slice selection re-phasing gradient and readout pre-winder
        + pp.calc_duration(multi_echo_gradient._gx) * n_echoes  # readout gradient
        + pp.calc_duration(multi_echo_gradient._gx_between) * (n_echoes - 1)  # readout gradient
        + pp.calc_duration(gz_spoil, multi_echo_gradient._gx_post)  # gradient spoiler or readout-re-winder
    )

    # calculate repetition time delay (tr_delay)
    current_min_tr = min_tr + te_delay
    tr_delay = 0 if tr is None else round_to_raster(tr - current_min_tr, system.block_duration_raster)
    if not tr_delay >= 0:
        raise ValueError(f'TR must be larger than {current_min_tr * 1000:.2f} ms. Current value is {tr * 1000:.2f} ms.')
    current_tr = current_min_tr + tr_delay

    print(f'\nCurrent echo time = {current_te * 1000:.2f} ms')
    print(f'Current repetition time = {current_tr * 1000:.2f} ms')
    print(f'Acquisition window per cardiac cycle = {current_tr * n_pe_points_per_cardiac_cycle * 1000:.2f} ms')

    # create header
    if mrd_header_file:
        hdr = create_header(
            traj_type='cartesian',
            fov=fov_xy,
            res=fov_xy / n_readout,
            slice_thickness=slice_thickness,
            dt=multi_echo_gradient._adc.dwell,
            n_k1=len(pe_steps),
        )

        # write header to file
        prot = ismrmrd.Dataset(mrd_header_file, 'w')
        prot.write_xml_header(hdr.toXML('utf-8'))

    # obtain noise samples
    seq.add_block(pp.make_label(label='LIN', type='SET', value=0), pp.make_label(label='SLC', type='SET', value=0))
    seq.add_block(multi_echo_gradient._adc, pp.make_label(label='NOISE', type='SET', value=True))
    seq.add_block(pp.make_label(label='NOISE', type='SET', value=False))
    seq.add_block(pp.make_delay(system.rf_dead_time))

    if mrd_header_file:
        acq = ismrmrd.Acquisition()
        acq.resize(trajectory_dimensions=2, number_of_samples=multi_echo_gradient._adc.num_samples)
        # prot.append_acquisition(acq)

    # choose initial rf phase offset
    rf_phase = 0
    rf_inc = 0

    for cardiac_cycle_idx in range(len(pe_steps) // n_pe_points_per_cardiac_cycle):
        seq.add_block(
            pp.make_trigger(
                channel='physio1',
                duration=round_to_raster(
                    cardiac_trigger_delay - current_te / 2, raster_time=system.block_duration_raster
                ),
            )
        )

        for shot_idx in range(n_pe_points_per_cardiac_cycle):
            pe_index = pe_steps[shot_idx + n_pe_points_per_cardiac_cycle * cardiac_cycle_idx]
            # calculate current phase_offset if rf_spoiling is activated
            if rf_spoiling_phase_increment > 0:
                rf.phase_offset = rf_phase / 180 * np.pi
                multi_echo_gradient._adc.phase_offset = rf_phase / 180 * np.pi

            # add slice selective excitation pulse
            seq.add_block(rf, gz)

            # update rf phase offset for the next excitation pulse
            rf_inc = divmod(rf_inc + rf_spoiling_phase_increment, 360.0)[1]
            rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

            # set labels for the next spoke
            labels = []
            labels.append(pp.make_label(label='LIN', type='SET', value=int(pe_index - np.min(pe_steps))))
            labels.append(pp.make_label(label='IMA', type='SET', value=pe_index in pe_fully_sampled_center))

            # calculate current phase encoding gradient
            gy_pre = pp.make_trapezoid(channel='y', area=1 / fov_xy * pe_index, duration=gx_pre_duration, system=system)

            if te is not None:
                seq.add_block(gzr)
                seq.add_block(pp.make_delay(te_delay))
                seq.add_block(multi_echo_gradient._gx_pre, gy_pre, *labels)
            else:
                seq.add_block(multi_echo_gradient._gx_pre, gy_pre, gzr, *labels)

            # add readout gradients and ADCs
            seq, _ = multi_echo_gradient.add_to_seq_without_pre_post_gradient(seq, n_echoes)

            gy_pre.amplitude = -gy_pre.amplitude
            seq.add_block(multi_echo_gradient._gx_post, gy_pre, gz_spoil)

            # add delay in case TR > min_TR
            if tr_delay > 0:
                seq.add_block(pp.make_delay(tr_delay))

            if mrd_header_file:
                # add acquisitions to metadata
                k0_trajectory = np.linspace(
                    -multi_echo_gradient._n_readout_pre_echo,
                    multi_echo_gradient._n_readout_post_echo,
                    multi_echo_gradient._n_readout_with_partial_echo,
                )
                cart_trajectory = np.zeros((multi_echo_gradient._n_readout_with_partial_echo, 2), dtype=np.float32)

                for echo_ in range(n_echoes):
                    gx_sign = (-1) ** echo_
                    cart_trajectory[:, 0] = k0_trajectory * gx_sign
                    cart_trajectory[:, 1] = pe_index

                    acq = ismrmrd.Acquisition()
                    acq.resize(trajectory_dimensions=2, number_of_samples=multi_echo_gradient._adc.num_samples)
                    acq.traj[:] = cart_trajectory
                    prot.append_acquisition(acq)

    # close ISMRMRD file
    if mrd_header_file:
        prot.close()

    return seq, min_te, min_tr


def main(
    system: pp.Opts | None = None,
    te: float | None = None,
    delta_te: float | None = None,
    tr: float | None = None,
    n_echoes: int = 3,
    cardiac_trigger_delay: float = 0.4,
    fov_xy: float = 128e-3,
    n_readout: int = 128,
    partial_echo_factor: float = 0.7,
    acceleration: int = 2,
    n_fully_sampled_center: int = 12,
    n_pe_points_per_cardiac_cycle: int = 16,
    slice_thickness: float = 8e-3,
    receiver_bandwidth_per_pixel: float = 800,  # Hz/pixel
    show_plots: bool = True,
    test_report: bool = True,
    timing_check: bool = True,
) -> pp.Sequence:
    """Generate a FLASH sequence with multiple echoes.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    te
        Desired echo time (TE) (in seconds). Minimum echo time is used if set to None.
    delta_te
            Desired echo spacing (in seconds). Minimum echo spacing is used if set to None.
    tr
        Desired repetition time (TR) (in seconds). Minimum repetition time is used if set to None.
    n_echoes
        Number of echoes.
    cardiac_trigger_delay
        Delay after cardiac trigger (in seconds).
    fov_xy
        Field of view in x and y direction (in meters).
    n_readout
        Number of frequency encoding steps.
    partial_echo_factor
        Partial echo factor, commonly between 0.75 and 1. This reduces the echo
    acceleration
        Uniform undersampling factor along the phase encoding direction
    n_fully_sampled_center
        Number of phsae encoding points in the fully sampled center. This will reduce the overall undersampling factor.
    n_pe_points_per_cardiac_cycle
        Number of phase encoding points per cardiac cycle.
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

    if partial_echo_factor > 1 or partial_echo_factor < 0.5:
        raise ValueError('Partial echo factor has to be within 0.5 and 1')

    n_recovery_cardiac_cycles = 3

    # define settings of rf excitation pulse
    rf_duration = 1.28e-3  # duration of the rf excitation pulse [s]
    rf_flip_angle = 12  # flip angle of rf excitation pulse [°]
    rf_bwt = 4  # bandwidth-time product of rf excitation pulse [Hz*s]
    rf_apodization = 0.5  # apodization factor of rf excitation pulse
    readout_oversampling = 1  # readout oversampling factor, commonly 2. This reduces aliasing artifacts.

    # this is just approximately, the final calculation is done in the kernel
    n_readout_with_oversampling = int(n_readout * readout_oversampling * partial_echo_factor)
    # define ADC and gradient timing
    adc_dwell_time = 1.0 / (receiver_bandwidth_per_pixel * n_readout_with_oversampling)
    gx_pre_duration = 0.8e-3  # duration of readout pre-winder gradient [s]
    gx_flat_time, adc_dwell_time = find_gx_flat_time_on_adc_raster(
        n_readout_with_oversampling, adc_dwell_time, system.grad_raster_time, system.adc_raster_time
    )

    # define spoiling
    gz_spoil_duration = 0.8e-3  # duration of spoiler gradient [s]
    gz_spoil_area = 4 / slice_thickness  # area / zeroth gradient moment of spoiler gradient
    rf_spoiling_phase_increment = 117  # RF spoiling phase increment [°]. Set to 0 for no RF spoiling.

    # define sequence filename
    filename = f'{Path(__file__).stem}_{int(fov_xy * 1000)}fov_{n_readout}nx_{acceleration}us_'
    filename += f'trig{int(cardiac_trigger_delay * 1000)}ms'

    output_path = Path.cwd() / 'output'
    output_path.mkdir(parents=True, exist_ok=True)

    seq, min_te, min_tr = t2star_multi_echo_flash_kernel(
        system=system,
        te=te,
        delta_te=delta_te,
        tr=tr,
        n_echoes=n_echoes,
        n_recovery_cardiac_cycles=n_recovery_cardiac_cycles,
        cardiac_trigger_delay=cardiac_trigger_delay,
        fov_xy=fov_xy,
        n_readout=n_readout,
        partial_echo_factor=partial_echo_factor,
        readout_oversampling=readout_oversampling,
        acceleration=acceleration,
        n_fully_sampled_center=n_fully_sampled_center,
        n_pe_points_per_cardiac_cycle=n_pe_points_per_cardiac_cycle,
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
        mrd_header_file=output_path / Path(filename + '_header.h5'),
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
    seq.set_definition('ReadoutOversamplingFactor', readout_oversampling)

    # save seq-file to disk
    print(f"\nSaving sequence file '{filename}.seq' into folder '{output_path}'.")
    seq.write(str(output_path / filename), create_signature=True)

    if show_plots:
        seq.plot()

    return seq


if __name__ == '__main__':
    main()
