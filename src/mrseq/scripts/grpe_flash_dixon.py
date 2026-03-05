"""3D FLASH sequence with radial phase encoding."""

from pathlib import Path

import ismrmrd
import numpy as np
import pypulseq as pp
from pypulseq.rotate import rotate

from mrseq.utils import MultiEchoAcquisition
from mrseq.utils import find_gx_flat_time_on_adc_raster
from mrseq.utils import round_to_raster
from mrseq.utils import sys_defaults
from mrseq.utils import write_sequence
from mrseq.utils.constants import GOLDEN_ANGLE_HALF_CIRCLE
from mrseq.utils.ismrmrd import Fov
from mrseq.utils.ismrmrd import Limits
from mrseq.utils.ismrmrd import MatrixSize
from mrseq.utils.ismrmrd import create_header


def grpe_flash_dixon_kernel(
    system: pp.Opts,
    te: float | None,
    delta_te: float | None,
    n_echoes: int,
    tr: float | None,
    fov_x: float,
    fov_y: float,
    fov_z: float,
    n_readout: int,
    n_rpe_points: int,
    n_rpe_points_per_shot: int,
    n_rpe_spokes: int,
    readout_oversampling: float,
    partial_echo_factor: float,
    partial_fourier_factor: float,
    n_dummy_spokes: int,
    gx_pre_duration: float,
    gx_flat_time: float,
    rf_duration: float,
    rf_flip_angle: float,
    rf_bwt: float,
    rf_apodization: float,
    rf_spoiling_phase_increment: float,
    gx_spoil_duration: float,
    gx_spoil_area: float,
    mrd_header_file: str | Path | None,
) -> tuple[pp.Sequence, float, float, float]:
    """Generate a 3D FLASH sequence with golden radial phase encoding.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    te
        Desired echo time (TE) (in seconds). Minimum echo time is used if set to None.
    delta_te
        Desired echo spacing (in seconds). Minimum echo spacing is used if set to None.
    n_echoes
        Number of echoes.
    tr
        Desired repetition time (TR) (in seconds).
    fov_x
        Field of view in x direction (in meters).
    fov_y
        Field of view in y direction (in meters).
    fov_z
        Field of view in z direction (in meters).
    n_readout
        Number of frequency encoding steps.
    n_rpe_points
        Number of radial phase encoding points (points along one RPE line).
    n_rpe_points_per_shot
        Shots are interleaved groups of points along RPE lines. Each shot obtains the k-space center at the cost of a
        point of the highest k-space frequency. Fat-saturation pulses are applied prior to each shot.
    n_rpe_spokes
        Number of radial phase encoding spokes (number of RPE lines).
    readout_oversampling
        Readout oversampling factor, commonly 2. This reduces aliasing artifacts.
    partial_echo_factor
        Partial echo factor along readout (between 0.5 and 1).
    partial_fourier_factor
        Partial Fourier factor along RPE lines (between 0.5 and 1).
    n_dummy_spokes
        Number of dummy RPE spokes before data acquisition to ensure steady state.
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
    gx_spoil_duration
        Duration of spoiler gradient (in seconds)
    gx_spoil_area
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
        Shortest possible repetition time.
    delta_te
        Time between echoes.

    """
    if readout_oversampling < 1:
        raise ValueError('Readout oversampling factor must be >= 1.')

    if n_dummy_spokes < 0:
        raise ValueError('Number of dummy spokes must be >= 0.')

    spoke_angle = GOLDEN_ANGLE_HALF_CIRCLE
    rpe_radial_shift = [0, 0.5, 0.25, 0.75]

    if partial_echo_factor > 1 or partial_echo_factor < 0.5:
        raise ValueError('Partial echo factor has to be within 0.5 and 1')
    if partial_fourier_factor > 1 or partial_fourier_factor < 0.5:
        raise ValueError('Partial Fourier factor has to be within 0.5 and 1')

    # create PyPulseq Sequence object and set system limits
    seq = pp.Sequence(system=system)

    # create slab selective excitation pulse and gradients
    rf, gz, gzr = pp.make_sinc_pulse(
        flip_angle=rf_flip_angle / 180 * np.pi,
        duration=rf_duration,
        slice_thickness=fov_z,
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
        fov=fov_x,
        n_readout=n_readout,
        readout_oversampling=readout_oversampling,
        partial_echo_factor=partial_echo_factor,
        gx_flat_time=gx_flat_time,
        gx_pre_duration=gx_pre_duration,
    )

    # calculate gradient areas for phase encoding along each RPE line in a low-high order
    delta_ky = 1 / fov_y
    n_rpe_points_with_partial_fourier = int(
        ((n_rpe_points * partial_fourier_factor) // n_rpe_points_per_shot) * n_rpe_points_per_shot
    )
    n_shots_per_rpe_spoke = n_rpe_points_with_partial_fourier // n_rpe_points_per_shot
    print(
        f'Number of phase encoding points {n_rpe_points_with_partial_fourier}',
        f'with partial Fourier factor {partial_fourier_factor}',
    )
    enc_steps_pe = np.arange(0, n_rpe_points_with_partial_fourier)
    phase_areas = (enc_steps_pe - n_rpe_points / 2) * delta_ky
    centric_idx = np.argsort(np.abs(phase_areas), kind='stable')
    enc_steps_pe = enc_steps_pe[centric_idx]

    # Interleave shots
    enc_steps_pe_interleaved = []
    for step in range(n_shots_per_rpe_spoke):
        enc_steps_pe_interleaved.append(enc_steps_pe[step::n_shots_per_rpe_spoke])
    enc_steps_pe = np.concatenate(enc_steps_pe_interleaved)

    # create spoiler gradients
    gx_spoil = pp.make_trapezoid(channel='x', system=system, area=gx_spoil_area, duration=gx_spoil_duration)

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
        if te_delay < 0:
            raise ValueError(f'TE must be larger than {min_te * 1000:.3f} ms. Current value is {te * 1000:.3f} ms.')

    # calculate minimum repetition time
    min_tr = (
        pp.calc_duration(gz)  # rf pulse
        + gzr_gx_dur  # slice selection re-phasing gradient and readout pre-winder
        + pp.calc_duration(multi_echo_gradient._gx) * n_echoes  # readout gradient
        + pp.calc_duration(multi_echo_gradient._gx_between) * (n_echoes - 1)  # readout gradient
        + pp.calc_duration(gx_spoil, multi_echo_gradient._gx_post)  # gradient spoiler or readout-re-winder
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

    print(f'\nCurrent echo time = {(min_te + te_delay) * 1000:.3f} ms')
    print(f'Current repetition time = {(current_min_tr + tr_delay) * 1000:.3f} ms')

    # choose initial rf phase offset
    rf_phase = 0.0
    rf_inc = 0.0

    # create header
    if mrd_header_file:
        hdr = create_header(
            traj_type='other',
            encoding_fov=Fov(x=fov_x * readout_oversampling, y=fov_y, z=fov_y),
            recon_fov=Fov(x=fov_x, y=fov_y, z=fov_y),
            encoding_matrix=MatrixSize(n_x=int(n_readout * readout_oversampling), n_y=n_rpe_points, n_z=n_rpe_points),
            recon_matrix=MatrixSize(n_x=n_readout, n_y=n_rpe_points, n_z=n_rpe_points),
            dwell_time=multi_echo_gradient._adc.dwell,
            slice_limits=Limits(min=0, max=0, center=0),
            k1_limits=Limits(min=0, max=n_rpe_points, center=0),
            k2_limits=Limits(min=0, max=n_rpe_spokes, center=0),
        )

        # write header to file
        prot = ismrmrd.Dataset(mrd_header_file, 'w')
        prot.write_xml_header(hdr.toXML('utf-8'))

    # obtain noise samples
    seq.add_block(pp.make_label(label='LIN', type='SET', value=0), pp.make_label(label='PAR', type='SET', value=0))
    seq.add_block(multi_echo_gradient._adc, pp.make_label(label='NOISE', type='SET', value=True))
    seq.add_block(pp.make_label(label='NOISE', type='SET', value=False))
    seq.add_block(pp.make_delay(system.rf_dead_time))

    if mrd_header_file:
        acq = ismrmrd.Acquisition()
        acq.resize(trajectory_dimensions=3, number_of_samples=multi_echo_gradient._adc.num_samples)
        prot.append_acquisition(acq)

    # Define sequence blocks
    for se_index in np.arange(-n_dummy_spokes, n_rpe_spokes):
        se_label = pp.make_label(type='SET', label='PAR', value=int(se_index))
        for shot_index in range(n_shots_per_rpe_spoke):
            for pe_index in range(n_rpe_points_per_shot):
                pe = int(enc_steps_pe[shot_index * n_rpe_points_per_shot + pe_index])
                pe_label = pp.make_label(type='SET', label='LIN', value=pe)

                # set rf pulse properties and add rf pulse block event
                if rf_spoiling_phase_increment > 0:
                    rf.phase_offset = rf_phase / 180 * np.pi
                    multi_echo_gradient._adc.phase_offset = rf_phase / 180 * np.pi

                # add rf pulse
                seq.add_block(rf, gz)

                # update rf pulse properties for the next loop
                rf_inc = divmod(rf_inc + rf_spoiling_phase_increment, 360.0)[1]
                rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

                # area of phase encoding gradient
                current_phase_area = phase_areas[pe]
                # do not shift k-space center for respiratory navigator calculation
                if np.abs(current_phase_area) > 1e-5:
                    pe_shift = rpe_radial_shift[int(np.mod(se_index, len(rpe_radial_shift)))]
                else:
                    pe_shift = 0.0

                # phase encoding along pe
                gy_pre = pp.make_trapezoid(
                    channel='y',
                    area=current_phase_area + pe_shift * delta_ky,
                    duration=pp.calc_duration(multi_echo_gradient._gx_pre),
                    system=system,
                )

                # calculate rotated phase encoding gradient
                rotation_angle_rad = spoke_angle * se_index
                gy_pre_rotated = rotate(gy_pre, angle=rotation_angle_rad, axis='x')
                gy_pre = gy_pre_rotated[0]
                if len(gy_pre_rotated) == 2:
                    gz_pre = gy_pre_rotated[1]
                else:
                    gz_pre = pp.make_trapezoid(
                        channel='z',
                        area=0,
                        duration=pp.calc_duration(multi_echo_gradient._gx_pre),
                        system=system,
                    )
                assert gy_pre.channel == 'y' and gz_pre.channel == 'z'

                # combine slice rephaser and slice encoding gradient
                gz_r_pre = pp.make_trapezoid(
                    channel='z',
                    area=gz_pre.area + gzr.area,
                    duration=pp.calc_duration(multi_echo_gradient._gx_pre),
                    system=system,
                )

                label_contents = [pe_label, se_label]
                seq.add_block(multi_echo_gradient._gx_pre, gy_pre, gz_r_pre, *label_contents)

                # add delay due to TE
                if te_delay > 0:
                    seq.add_block(pp.make_delay(te_delay))

                # add readout gradients and ADCs
                if se_index >= 0:
                    seq, _ = multi_echo_gradient.add_to_seq_without_pre_post_gradient(seq, n_echoes)
                else:
                    # the most accurate way to get the duration of the readout block is to add it to a dummy sequence
                    seq_dummy = pp.Sequence(system=system)
                    seq_dummy, _ = multi_echo_gradient.add_to_seq_without_pre_post_gradient(seq_dummy, n_echoes)
                    readout_duration = sum(seq_dummy.block_durations.values())
                    seq.add_block(pp.make_delay(readout_duration))

                # add re-winder and spoiler gradients
                gy_pre.amplitude = -gy_pre.amplitude
                gz_pre.amplitude = -gz_pre.amplitude
                seq.add_block(gx_spoil, gy_pre, gz_pre)

                # add delay in case TR > min_TR
                if tr_delay > 0:
                    seq.add_block(pp.make_delay(tr_delay))

                if mrd_header_file and se_index >= 0:
                    # add acquisitions to metadata
                    k0_trajectory = np.linspace(
                        -multi_echo_gradient._n_readout_pre_echo,
                        multi_echo_gradient._n_readout_post_echo,
                        multi_echo_gradient._n_readout_with_partial_echo,
                    )
                    grpe_trajectory = np.zeros((multi_echo_gradient._n_readout_with_partial_echo, 3), dtype=np.float32)

                    for echo_ in range(n_echoes):
                        gx_sign = (-1) ** echo_
                        grpe_trajectory[:, 0] = k0_trajectory * gx_sign
                        grpe_trajectory[:, 1] = (pe - n_rpe_points / 2 + pe_shift) * np.cos(rotation_angle_rad)
                        grpe_trajectory[:, 2] = (pe - n_rpe_points / 2 + pe_shift) * np.sin(rotation_angle_rad)

                        acq = ismrmrd.Acquisition()
                        acq.resize(trajectory_dimensions=3, number_of_samples=multi_echo_gradient._adc.num_samples)
                        acq.traj[:] = grpe_trajectory
                        prot.append_acquisition(acq)

    # obtain echoes with positive and negative gradient polarity to be able to correct for any differences between
    # positive and negative gradient waveforms in the bi-polar readout
    seq.add_block(pp.make_label(type='SET', label='NAV', value=True))
    for average_idx in range(4):
        average_label = pp.make_label(type='SET', label='AVG', value=average_idx)
        for polarity_idx, polarity in enumerate(['positive', 'negative']):
            polarity_label = pp.make_label(type='SET', label='REP', value=polarity_idx)
            # set rf pulse properties and add rf pulse block event
            if rf_spoiling_phase_increment > 0:
                rf.phase_offset = rf_phase / 180 * np.pi
                multi_echo_gradient._adc.phase_offset = rf_phase / 180 * np.pi

            # add rf pulse
            seq.add_block(rf, gz)

            # update rf pulse properties for the next loop
            rf_inc = divmod(rf_inc + rf_spoiling_phase_increment, 360.0)[1]
            rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

            seq.add_block(
                multi_echo_gradient._gx_pre
                if polarity == 'positive'
                else pp.scale_grad(multi_echo_gradient._gx_pre, -1),
                average_label,
                polarity_label,
            )

            # add delay due to TE
            if te_delay > 0:
                seq.add_block(pp.make_delay(te_delay))

            # add readout gradients and ADCs
            seq, time_to_echoes = multi_echo_gradient.add_to_seq_without_pre_post_gradient(seq, n_echoes, polarity)  # type: ignore[arg-type]

            # add spoiler gradients
            seq.add_block(gx_spoil)

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
                grpe_trajectory = np.zeros((multi_echo_gradient._n_readout_with_partial_echo, 3), dtype=np.float32)

                for echo_ in range(n_echoes):
                    gx_sign = (-1) ** echo_ if polarity == 'positive' else (-1) ** (echo_ + 1)
                    grpe_trajectory[:, 0] = k0_trajectory * gx_sign

                    acq = ismrmrd.Acquisition()
                    acq.resize(trajectory_dimensions=3, number_of_samples=multi_echo_gradient._adc.num_samples)
                    acq.traj[:] = grpe_trajectory
                    prot.append_acquisition(acq)

    seq.add_block(pp.make_label(label='NAV', type='SET', value=False))

    # close ISMRMRD file
    if mrd_header_file:
        prot.close()

    delta_te_array = np.diff(time_to_echoes)
    return seq, float(min_te), float(min_tr), float(delta_te_array[0])


def main(
    system: pp.Opts | None = None,
    tr: float | None = None,
    rf_flip_angle: float = 12,
    fov_x: float = 128e-3,
    fov_y: float = 128e-3,
    fov_z: float = 128e-3,
    n_readout: int = 128,
    n_rpe_points: int = 128,
    n_rpe_points_per_shot: int = 8,
    n_rpe_spokes: int = 16,
    partial_echo_factor: float = 0.7,
    partial_fourier_factor: float = 0.7,
    receiver_bandwidth_per_pixel: float = 1200,  # Hz/pixel
    n_dummy_spokes: int = 2,
    show_plots: bool = True,
    test_report: bool = True,
    timing_check: bool = True,
    v141_compatibility: bool = True,
) -> pp.Sequence:
    """Generate a 3D FLASH sequence with radial phase encoding.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    tr
        Desired repetition time (TR) (in seconds). Minimum repetition time is used if set to None.
    rf_flip_angle
        Flip angle of rf excitation pulse (in degrees)
    fov_x
        Field of view in x direction (in meters).
    fov_y
        Field of view in y direction (in meters).
    fov_z
        Field of view in z direction (in meters).
    n_readout
        Number of frequency encoding steps.
    n_rpe_points
        Number of radial phase encoding points (points along one RPE line).
    n_rpe_points_per_shot
        Shots are interleaved groups of points along RPE lines. Each shot obtains the k-space center at the cost of a
        point of the highest k-space frequency. Fat-saturation pulses are applied prior to each shot.
    n_rpe_spokes
        Number of radial phase encoding spokes (number of RPE lines).
    partial_echo_factor
        Partial echo factor along the readout (between 0.5 and 1).
    partial_fourier_factor
        Partial Fourier factor along RPE lines (between 0.5 and 1).
    receiver_bandwidth_per_pixel
        Desired receiver bandwidth per pixel (in Hz/pixel). This is used to calculate the readout duration.
    n_dummy_spokes
        Number of dummy RPE spokes before data acquisition to ensure steady state
    show_plots
        Toggles sequence plot.
    test_report
        Toggles advanced test report.
    timing_check
        Toggles timing check of the sequence.
    v141_compatibility
        Save the sequence in pulseq v1.4.1 for backwards compatibility.
    """
    if system is None:
        system = sys_defaults

    # define settings of rf excitation pulse
    rf_duration = 0.6e-3  # duration of the rf excitation pulse [s]
    rf_bwt = 2  # bandwidth-time product of rf excitation pulse [Hz*s]
    rf_apodization = 0.5  # apodization factor of rf excitation pulse
    readout_oversampling = 2  # readout oversampling factor, commonly 2. This reduces aliasing artifacts.

    # this is just approximately, the final calculation is done in the kernel
    n_readout_with_oversampling = int(n_readout * readout_oversampling * partial_echo_factor)
    # define ADC and gradient timing
    adc_dwell_time = 1.0 / (receiver_bandwidth_per_pixel * n_readout_with_oversampling)
    gx_pre_duration = 1.0e-3  # duration of readout pre-winder gradient [s]
    gx_flat_time, adc_dwell_time = find_gx_flat_time_on_adc_raster(
        n_readout_with_oversampling, adc_dwell_time, system.grad_raster_time, system.adc_raster_time
    )

    te = None  # shortest possible echo time
    delta_te = None  # shortest possible delta echo time
    n_echoes = 3

    # define spoiling
    gx_spoil_duration = 1.9e-3  # duration of spoiler gradient [s]
    gx_spoil_area = readout_oversampling * n_readout * 1 / fov_x  # area / zeroth gradient moment of spoiler gradient
    rf_spoiling_phase_increment = 117  # RF spoiling phase increment [°]. Set to 0 for no RF spoiling.

    # define sequence filename
    filename = f'{Path(__file__).stem}_fov{int(fov_x * 1000)}_{int(fov_y * 1000)}_{int(fov_z * 1000)}mm_'
    filename += f'{n_readout}_{n_rpe_points}_{n_rpe_spokes}_3ne'

    output_path = Path.cwd() / 'output'
    output_path.mkdir(parents=True, exist_ok=True)

    # delete existing header file
    if (output_path / Path(filename + '_header.h5')).exists():
        (output_path / Path(filename + '_header.h5')).unlink()

    seq, min_te, min_tr, delta_te = grpe_flash_dixon_kernel(
        system=system,
        te=te,
        delta_te=delta_te,
        n_echoes=n_echoes,
        tr=tr,
        fov_x=fov_x,
        fov_y=fov_y,
        fov_z=fov_z,
        n_readout=n_readout,
        n_rpe_points=n_rpe_points,
        n_rpe_points_per_shot=n_rpe_points_per_shot,
        n_rpe_spokes=n_rpe_spokes,
        readout_oversampling=readout_oversampling,
        partial_echo_factor=partial_echo_factor,
        partial_fourier_factor=partial_fourier_factor,
        n_dummy_spokes=n_dummy_spokes,
        gx_pre_duration=gx_pre_duration,
        gx_flat_time=gx_flat_time,
        rf_duration=rf_duration,
        rf_flip_angle=rf_flip_angle,
        rf_bwt=rf_bwt,
        rf_apodization=rf_apodization,
        rf_spoiling_phase_increment=rf_spoiling_phase_increment,
        gx_spoil_duration=gx_spoil_duration,
        gx_spoil_area=gx_spoil_area,
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
    seq.set_definition('FOV', [fov_x, fov_y, fov_z])
    seq.set_definition('ReconMatrix', (n_readout, n_rpe_points, n_rpe_points))
    seq.set_definition('SliceThickness', fov_z)
    seq.set_definition('TE', [(te or min_te) + idx * delta_te for idx in range(n_echoes)])
    seq.set_definition('TR', tr or min_tr)
    seq.set_definition('ReadoutOversamplingFactor', readout_oversampling)

    # save seq-file to disk
    print(f"\nSaving sequence file '{filename}.seq' into folder '{output_path}'.")
    write_sequence(seq, str(output_path / filename), create_signature=True, v141_compatibility=v141_compatibility)

    if show_plots:
        seq.plot(time_range=(0, (n_dummy_spokes * n_rpe_points + 20) * (tr or min_tr)))

    return seq, output_path / filename


if __name__ == '__main__':
    main()
