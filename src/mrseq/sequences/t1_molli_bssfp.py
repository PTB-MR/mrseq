"""2D Cartesian MOLLI with bSSFP mapping for cardiac T1 mapping."""

from pathlib import Path

import ismrmrd
import numpy as np
import pypulseq as pp
from raw2ismrmrd.utils import Fov
from raw2ismrmrd.utils import Limits
from raw2ismrmrd.utils import MatrixSize
from raw2ismrmrd.utils import create_header

from mrseq.preparations import add_t1_inv_prep
from mrseq.utils import MultiEchoAcquisition
from mrseq.utils import cartesian_phase_encoding
from mrseq.utils import find_gx_flat_time_on_adc_raster
from mrseq.utils import round_to_raster
from mrseq.utils import sys_defaults
from mrseq.utils import write_sequence


def t1_molli_bssfp_kernel(
    system: pp.Opts,
    te: float | None,
    tr: float | None,
    inversion_times: np.ndarray,
    min_cardiac_trigger_delay: float,
    use_soft_delay: bool,
    fov_xy: float,
    n_readout: int,
    partial_echo_factor: float,
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
    rf_inv_duration: float,
    rf_inv_spoil_risetime: float,
    rf_inv_spoil_flattime: float,
    rf_inv_mu: float,
    mrd_header_file: str | Path | None,
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
    min_cardiac_trigger_delay
        Minimum delay after cardiac trigger (in seconds).
        The total trigger delay is implemented as a soft delay and can be chosen by the user in the UI.
    use_soft_delay
        Use soft-delay functionality available from pulseq version 1.5.0 to allow UI adaption of trigger delay.
        If set to false only the min_cardiac_trigger_delay is used.
    fov_xy
        Field of view in x and y direction (in meters).
    n_readout
        Number of frequency encoding steps.
    readout_oversampling
        Readout oversampling factor, commonly 2. This reduces aliasing artifacts.
    partial_echo_factor
        Partial echo factor between 0 and 1, which reduces the echo time by acquiring only a part of the readout.
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
    rf_inv_duration
        Duration of adiabatic inversion pulse (in seconds)
    rf_inv_spoil_risetime
        Rise time of spoiler after inversion pulse (in seconds)
    rf_inv_spoil_flattime
        Flat time of spoiler after inversion pulse (in seconds)
    rf_inv_mu
        Constant determining amplitude of frequency sweep of adiabatic inversion pulse
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
    if partial_echo_factor > 1 or partial_echo_factor < 0.5:
        raise ValueError('Partial echo factor has to be within 0.5 and 1')

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
    multi_echo_gradient = MultiEchoAcquisition(
        system=system,
        delta_te=None,
        fov=fov_xy,
        n_readout=n_readout,
        readout_oversampling=readout_oversampling,
        partial_echo_factor=partial_echo_factor,
        gx_flat_time=gx_flat_time,
        gx_pre_duration=gx_pre_duration,
    )

    print(f'Current receiver bandwidth = {1 / multi_echo_gradient._gx.flat_time:.0f} Hz/pixel')

    # create phase encoding steps
    pe_steps, pe_fully_sampled_center = cartesian_phase_encoding(
        n_phase_encoding=n_readout,
        acceleration=acceleration,
        n_fully_sampled_center=n_fully_sampled_center,
        sampling_order='low_high',
    )

    # phase encoding gradient for max ky position
    gy_pre_max = pp.make_trapezoid(
        channel='y',
        area=1 / fov_xy * n_readout / 2,
        duration=gx_pre_duration,
        system=system,
    )

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
    ).item()

    # calculate echo time delay (te_delay)
    if te is None:
        te_delay = 0.0
    else:
        te_delay = round_to_raster(te - min_te, system.block_duration_raster)
        if not te_delay >= 0:
            raise ValueError(f'TE must be larger than {min_te * 1000:.3f} ms. Current value is {te * 1000:.3f} ms.')
    current_te = min_te + te_delay

    # calculate minimum repetition time
    min_tr = (
        pp.calc_duration(gz)  # rf pulse
        + gzr_gx_dur  # slice selection re-phasing gradient and readout pre-winder
        + pp.calc_duration(multi_echo_gradient._gx)  # readout gradient
        + pp.calc_duration(gzr, multi_echo_gradient._gx_post)  # gradient spoiler or readout-re-winder
    ).item()

    # calculate repetition time delay (tr_delay)
    current_min_tr = min_tr + te_delay
    if tr is None:
        tr_delay = 0.0
    else:
        tr_delay = round_to_raster(tr - current_min_tr, system.block_duration_raster)
        if not tr_delay >= 0:
            raise ValueError(
                f'TR must be larger than {current_min_tr * 1000:.3f} ms. Current value is {tr * 1000:.3f} ms.'
            )
    current_tr = current_min_tr + tr_delay

    print(f'\nCurrent echo time = {current_te * 1000:.3f} ms')
    print(f'Current repetition time = {current_tr * 1000:.3f} ms')
    print(f'Acquisition window per cardiac cycle = {current_tr * len(pe_steps) * 1000:.3f} ms')

    # create header
    if mrd_header_file:
        hdr = create_header(
            traj_type='other',
            encoding_fov=Fov(x=fov_xy, y=fov_xy, z=slice_thickness),
            recon_fov=Fov(x=fov_xy, y=fov_xy, z=slice_thickness),
            encoding_matrix=MatrixSize(n_x=int(n_readout * readout_oversampling), n_y=n_readout, n_z=1),
            recon_matrix=MatrixSize(n_x=n_readout, n_y=n_readout, n_z=1),
            dwell_time=multi_echo_gradient._adc.dwell,
            k1_limits=Limits(min=0, max=len(pe_steps), center=0),
            h1_resonance_freq=system.gamma * system.B0,
        )

        # write header to file
        prot = ismrmrd.Dataset(mrd_header_file, 'w')
        prot.write_xml_header(hdr.toXML('utf-8'))

    # create trigger soft delay (total duration: user_input/1.0 - min_cardiac_trigger_delay)
    if use_soft_delay:
        trig_soft_delay = pp.make_soft_delay(
            hint='trig_delay',
            offset=-min_cardiac_trigger_delay,
            factor=1.0,
            default_duration=0.8 - min_cardiac_trigger_delay,
        )

    # obtain noise samples
    seq.add_block(pp.make_label(label='LIN', type='SET', value=0), pp.make_label(label='SLC', type='SET', value=0))
    seq.add_block(multi_echo_gradient._adc, pp.make_label(label='NOISE', type='SET', value=True))
    seq.add_block(pp.make_label(label='NOISE', type='SET', value=False))
    seq.add_block(pp.make_delay(system.rf_dead_time))

    if mrd_header_file:
        acq = ismrmrd.Acquisition()
        acq.resize(trajectory_dimensions=2, number_of_samples=multi_echo_gradient._adc.num_samples)
        prot.append_acquisition(acq)

    # Create inversion pulse
    t1_inv_prep, block_duration, time_since_inversion = add_t1_inv_prep(
        system=system,
        rf_duration=rf_inv_duration,
        spoiler_flat_time=rf_inv_spoil_flattime,
        spoiler_ramp_time=rf_inv_spoil_risetime,
        rf_mu=rf_inv_mu,
    )

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
                constant_trig_delay = round_to_raster(
                    min_cardiac_trigger_delay - delay_after_inversion_pulse - block_duration,
                    raster_time=system.block_duration_raster,
                )

                if constant_trig_delay < 0:
                    raise ValueError('Minimum trigger delay is to small for selected inversion times.')

                # add trigger and constant part of trigger delay
                seq.add_block(pp.make_trigger(channel='physio1', duration=constant_trig_delay))

                # add variable part of trigger delay (soft delay)
                if use_soft_delay:
                    seq.add_block(trig_soft_delay)

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
                constant_trig_delay = round_to_raster(
                    min_cardiac_trigger_delay
                    - n_bssfp_startup_pulses * current_tr
                    - current_te
                    - rf.shape_dur / 2
                    - max(rf.delay, gz.rise_time),
                    raster_time=system.block_duration_raster,
                )
                # add trigger and constant part of trigger delay
                seq.add_block(pp.make_trigger(channel='physio1', duration=constant_trig_delay))

                # add variable part of trigger delay (soft delay)
                if use_soft_delay:
                    seq.add_block(trig_soft_delay)

            rf_signal = rf.signal.copy()
            for idx in range(-n_bssfp_startup_pulses, len(pe_steps)):
                pe_index_ = pe_steps[idx if idx >= 0 else 0]

                # add slice selective excitation pulse
                if idx < 0:
                    # use linear flip angle ramp for bSSFP startup pulses
                    rf.signal = rf_signal * 1 / n_bssfp_startup_pulses * (n_bssfp_startup_pulses + idx + 1)
                else:
                    rf.signal = rf_signal

                rf.phase_offset = np.mod(rf.phase_offset + np.pi, 2 * np.pi)
                multi_echo_gradient._adc.phase_offset = np.mod(multi_echo_gradient._adc.phase_offset + np.pi, 2 * np.pi)

                seq.add_block(rf, gz)

                # set labels for the next spoke
                labels = []
                labels.append(pp.make_label(label='LIN', type='SET', value=int(pe_index_ - np.min(pe_steps))))
                labels.append(pp.make_label(label='IMA', type='SET', value=pe_index_ in pe_fully_sampled_center))
                labels.append(pp.make_label(type='SET', label='ECO', value=int(contrast_index)))

                if te is not None:
                    seq.add_block(gzr)
                    seq.add_block(pp.make_delay(te_delay))
                    seq.add_block(multi_echo_gradient._gx_pre, pp.scale_grad(gy_pre_max, pe_index_ / (n_readout / 2)))
                else:
                    seq.add_block(
                        multi_echo_gradient._gx_pre, pp.scale_grad(gy_pre_max, pe_index_ / (n_readout / 2)), gzr
                    )

                # add the readout gradient and ADC
                if idx >= 0:
                    seq.add_block(multi_echo_gradient._gx, multi_echo_gradient._adc, *labels)
                else:
                    seq.add_block(multi_echo_gradient._gx)

                seq.add_block(
                    multi_echo_gradient._gx_post, pp.scale_grad(gy_pre_max, -pe_index_ / (n_readout / 2)), gzr
                )

                # add delay in case TR > min_TR
                if tr_delay > 0:
                    seq.add_block(pp.make_delay(tr_delay))

                if mrd_header_file and idx >= 0:
                    # add acquisitions to metadata
                    k0_trajectory = np.linspace(
                        -multi_echo_gradient._n_readout_pre_echo,
                        multi_echo_gradient._n_readout_post_echo,
                        multi_echo_gradient._n_readout_with_partial_echo,
                    )
                    cart_trajectory = np.zeros((multi_echo_gradient._n_readout_with_partial_echo, 2), dtype=np.float32)
                    cart_trajectory[:, 0] = k0_trajectory
                    cart_trajectory[:, 1] = pe_index_

                    acq = ismrmrd.Acquisition()
                    acq.resize(trajectory_dimensions=2, number_of_samples=multi_echo_gradient._adc.num_samples)
                    acq.traj[:] = cart_trajectory
                    prot.append_acquisition(acq)

            contrast_index += 1

        # add three cardiac cycles for signal recovery
        if part_idx == 0:
            for _cardiac_index in range(3):
                # add trigger and constant part of trigger delay
                seq.add_block(pp.make_trigger(channel='physio1', duration=min_cardiac_trigger_delay))

                # add variable part of trigger delay (soft delay)
                if use_soft_delay:
                    seq.add_block(trig_soft_delay)

    # write all required parameters in the seq-file header/definitions
    seq.set_definition('FOV', [fov_xy, fov_xy, slice_thickness])
    seq.set_definition('ReconMatrix', (n_readout, n_readout, 1))
    seq.set_definition('SliceThickness', slice_thickness)
    seq.set_definition('TE', te or min_te)
    seq.set_definition('TR', tr or min_tr)
    seq.set_definition('TI', inversion_times.tolist())
    seq.set_definition('ReadoutOversamplingFactor', readout_oversampling)

    return seq, min_te, min_tr


def main(
    system: pp.Opts | None = None,
    te: float | None = None,
    tr: float | None = None,
    inversion_times: np.ndarray | None = None,
    fov_xy: float = 128e-3,
    n_readout: int = 128,
    acceleration: int = 2,
    n_fully_sampled_center: int = 12,
    partial_echo_factor: float = 1.0,
    slice_thickness: float = 8e-3,
    receiver_bandwidth_per_pixel: float = 1000,  # Hz/pixel
    show_plots: bool = True,
    test_report: bool = True,
    timing_check: bool = True,
    v141_compatibility: bool = True,
) -> tuple[pp.Sequence, Path]:
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
        If None, default values of [100, 180] ms are used.
    fov_xy
        Field of view in x and y direction (in meters).
    n_readout
        Number of frequency encoding steps.
    acceleration
        Uniform undersampling factor along the phase encoding direction
    n_fully_sampled_center
        Number of phsae encoding points in the fully sampled center. This will reduce the overall undersampling factor.
    partial_echo_factor
        Partial echo factor between 0 and 1, which reduces the echo time by acquiring only a part of the readout.
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
    v141_compatibility
        Save the sequence in pulseq v1.4.1 for backwards compatibility.


    Returns
    -------
    seq
        Sequence object of cardiac MOLLI T1 mapping sequence.
    file_path
        Path to the sequence file.
    """
    if system is None:
        system = sys_defaults

    if inversion_times is None:
        inversion_times = np.asarray([0.1, 0.18])

    # define T1prep settings
    rf_inv_duration = 12e-3  # duration of adiabatic inversion pulse [s]
    rf_inv_spoil_risetime = 0.6e-3  # rise time of spoiler after inversion pulse [s]
    rf_inv_spoil_flattime = 8.4e-3  # flat time of spoiler after inversion pulse [s]
    rf_inv_mu = 4.9  # constant determining amplitude of frequency sweep of adiabatic inversion pulse

    # define settings of rf excitation pulse
    rf_duration = 0.5e-3  # duration of the rf excitation pulse [s]
    rf_flip_angle = 35  # flip angle of rf excitation pulse [°]
    rf_bwt = 1.5  # bandwidth-time product of rf excitation pulse [Hz*s]
    rf_apodization = 0.5  # apodization factor of rf excitation pulse
    readout_oversampling = 2  # readout oversampling factor, commonly 2. This reduces aliasing artifacts.

    # this is just approximately, the final calculation is done in the kernel
    n_readout_with_oversampling = int(n_readout * readout_oversampling * partial_echo_factor)
    # define ADC and gradient timing
    adc_dwell_time = 1.0 / (receiver_bandwidth_per_pixel * n_readout_with_oversampling)
    gx_pre_duration = 0.72e-3  # duration of readout pre-winder gradient [s]
    gx_flat_time, adc_dwell_time = find_gx_flat_time_on_adc_raster(
        n_readout_with_oversampling, adc_dwell_time, system.grad_raster_time, system.adc_raster_time
    )

    n_bssfp_startup_pulses = 11  # number of bSSFP startup pulses to reach steady state.

    # define sequence filename
    filename = f'{Path(__file__).stem}_{int(fov_xy * 1000)}fov_{n_readout}nx_{acceleration}us'

    output_path = Path.cwd() / 'output'
    output_path.mkdir(parents=True, exist_ok=True)

    # delete existing header file
    if (output_path / Path(filename + '_header.h5')).exists():
        (output_path / Path(filename + '_header.h5')).unlink()

    seq, _min_te, _min_tr = t1_molli_bssfp_kernel(
        system=system,
        te=te,
        tr=tr,
        inversion_times=inversion_times,
        min_cardiac_trigger_delay=np.max(inversion_times)
        + 0.02,  # max inversion time + approx inversion pulse duration
        use_soft_delay=True,
        fov_xy=fov_xy,
        n_readout=n_readout,
        readout_oversampling=readout_oversampling,
        partial_echo_factor=partial_echo_factor,
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
        rf_inv_duration=rf_inv_duration,
        rf_inv_spoil_risetime=rf_inv_spoil_risetime,
        rf_inv_spoil_flattime=rf_inv_spoil_flattime,
        rf_inv_mu=rf_inv_mu,
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

    # save seq-file to disk
    print(f"\nSaving sequence file '{filename}.seq' into folder '{output_path}'.")
    write_sequence(seq, str(output_path / filename), create_signature=True, v141_compatibility=v141_compatibility)

    if show_plots:
        seq.plot()

    return seq, output_path / filename


if __name__ == '__main__':
    main()
