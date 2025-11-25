"""M2D radial FLASH sequence."""

from pathlib import Path

import ismrmrd
import numpy as np
import pypulseq as pp

from mrseq.utils import find_gx_flat_time_on_adc_raster
from mrseq.utils import round_to_raster
from mrseq.utils import sys_defaults
from mrseq.utils.constants import GOLDEN_ANGLE_HALF_CIRCLE
from mrseq.utils.ismrmrd import Fov
from mrseq.utils.ismrmrd import Limits
from mrseq.utils.ismrmrd import MatrixSize
from mrseq.utils.ismrmrd import create_header


def radial_flash_kernel(
    system: pp.Opts,
    te: float | None,
    tr: float | None,
    fov_xy: float,
    n_readout: int,
    n_spokes: int,
    spoke_angle: float,
    readout_oversampling: float,
    slice_thickness: float,
    n_slices: int,
    n_dummy_excitations: int,
    gx_pre_duration: float,
    gx_flat_time: float,
    rf_duration: float,
    rf_flip_angle: float,
    rf_bwt: float,
    rf_apodization: float,
    rf_spoiling_phase_increment: float,
    gz_spoil_duration: float,
    gz_spoil_area: float,
    mrd_header_file: str | Path | None,
) -> tuple[pp.Sequence, float, float]:
    """Generate a radial FLASH sequence.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    te
        Desired echo time (TE) (in seconds). Minimum echo time is used if set to None.
    tr
        Desired repetition time (TR) (in seconds).
    fov_xy
        Field of view in x and y direction (in meters).
    n_readout
        Number of frequency encoding steps.
    n_spokes
        Number of radial spokes.
    spoke_angle
        Angle between successive radial spokes (in radian).
    readout_oversampling
        Readout oversampling factor, commonly 2. This reduces aliasing artifacts.
    slice_thickness
        Slice thickness of the 2D slice (in meters).
    n_slices
        Number of slices.
    n_dummy_excitations
        Number of dummy excitations before data acquisition to ensure steady state.
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
        Shortest possible repetition time.

    """
    if readout_oversampling < 1:
        raise ValueError('Readout oversampling factor must be >= 1.')

    if n_dummy_excitations < 0:
        raise ValueError('Number of dummy excitations must be >= 0.')

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
    delta_k = 1 / (fov_xy * readout_oversampling)
    n_readout_with_oversampling = int(n_readout * readout_oversampling)
    gx = pp.make_trapezoid(
        channel='x', flat_area=n_readout_with_oversampling * delta_k, flat_time=gx_flat_time, system=system
    )
    n_readout_with_oversampling = n_readout_with_oversampling + np.mod(n_readout_with_oversampling, 2)  # make even
    adc = pp.make_adc(num_samples=n_readout_with_oversampling, duration=gx.flat_time, delay=gx.rise_time, system=system)

    # create frequency encoding pre- and re-winder gradient
    gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2 - delta_k / 2, duration=gx_pre_duration, system=system)
    gx_post = pp.make_trapezoid(channel='x', area=-gx.area / 2 + delta_k / 2, duration=gx_pre_duration, system=system)
    k0_center_id = np.where((np.arange(n_readout_with_oversampling) - n_readout_with_oversampling / 2) * delta_k == 0)[
        0
    ][0]

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
        + pp.calc_duration(gx)  # readout gradient
        + pp.calc_duration(gz_spoil, gx_post)  # gradient spoiler or readout-re-winder
    )

    # calculate repetition time delay (tr_delay)
    current_min_tr = min_tr + te_delay
    if tr is None:
        tr_delay = 0.0
    else:
        tr_delay = round_to_raster(tr - current_min_tr, system.block_duration_raster)
        if not tr_delay >= 0:
            raise ValueError(
                f'TR must be larger than {current_min_tr * 1000:.2f} ms. Current value is {tr * 1000:.3f} ms.'
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
            encoding_fov=Fov(x=fov_xy * readout_oversampling, y=fov_xy, z=slice_thickness),
            recon_fov=Fov(x=fov_xy, y=fov_xy, z=slice_thickness),
            encoding_matrix=MatrixSize(n_x=n_readout_with_oversampling, n_y=n_readout_with_oversampling, n_z=1),
            recon_matrix=MatrixSize(n_x=n_readout, n_y=n_readout, n_z=1),
            dwell_time=adc.dwell,
            slice_limits=Limits(min=0, max=n_slices, center=0),
            k1_limits=Limits(min=0, max=n_spokes, center=0),
            k2_limits=Limits(min=0, max=1, center=0),
        )

        # write header to file
        prot = ismrmrd.Dataset(mrd_header_file, 'w')
        prot.write_xml_header(hdr.toXML('utf-8'))

    # obtain noise samples
    seq.add_block(pp.make_label(label='LIN', type='SET', value=0), pp.make_label(label='SLC', type='SET', value=0))
    seq.add_block(adc, pp.make_label(label='NOISE', type='SET', value=True))
    seq.add_block(pp.make_label(label='NOISE', type='SET', value=False))
    seq.add_block(pp.make_delay(system.rf_dead_time))

    if mrd_header_file:
        acq = ismrmrd.Acquisition()
        acq.resize(trajectory_dimensions=2, number_of_samples=adc.num_samples)
        prot.append_acquisition(acq)

    for slice_ in range(n_slices):
        for spoke_ in range(-n_dummy_excitations, n_spokes):
            # calculate current phase_offset if rf_spoiling is activated
            if rf_spoiling_phase_increment > 0:
                rf.phase_offset = rf_phase / 180 * np.pi
                adc.phase_offset = rf_phase / 180 * np.pi

            # set frequency offset for current slice
            rf.freq_offset = gz.amplitude * slice_thickness * (slice_ - (n_slices - 1) / 2)

            # add slice selective excitation pulse
            seq.add_block(rf, gz)

            # update rf phase offset for the next excitation pulse
            rf_inc = divmod(rf_inc + rf_spoiling_phase_increment, 360.0)[1]
            rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

            # calculate rotation angle for the current spoke
            rotation_angle_rad = spoke_angle * spoke_

            if te_delay > 0:
                seq.add_block(gzr)
                seq.add_block(pp.make_delay(te_delay))
                seq.add_block(*pp.rotate(gx_pre, angle=rotation_angle_rad, axis='z'))
            else:
                seq.add_block(*pp.rotate(gx_pre, gzr, angle=rotation_angle_rad, axis='z'))

            # rotate and add the readout gradient and ADC
            if spoke_ >= 0:
                labels = []
                labels.append(pp.make_label(label='LIN', type='SET', value=spoke_))
                labels.append(pp.make_label(label='SLC', type='SET', value=slice_))
                seq.add_block(*pp.rotate(gx, adc, angle=rotation_angle_rad, axis='z'), *labels)
            else:
                seq.add_block(pp.make_delay(pp.calc_duration(gx, adc)))

            seq.add_block(*pp.rotate(gx_post, gz_spoil, angle=rotation_angle_rad, axis='z'))

            # add delay in case TR > min_TR
            if tr_delay > 0:
                seq.add_block(pp.make_delay(tr_delay))

            if mrd_header_file and spoke_ >= 0:
                # add acquisitions to metadata
                k_radial_line = np.linspace(
                    -n_readout_with_oversampling // 2,
                    (n_readout_with_oversampling // 2) - 1,
                    n_readout_with_oversampling,
                )
                radial_trajectory = np.zeros((n_readout_with_oversampling, 2), dtype=np.float32)

                radial_trajectory[:, 0] = k_radial_line * np.cos(rotation_angle_rad)
                radial_trajectory[:, 1] = k_radial_line * np.sin(rotation_angle_rad)

                acq = ismrmrd.Acquisition()
                acq.resize(trajectory_dimensions=2, number_of_samples=adc.num_samples)
                acq.traj[:] = radial_trajectory
                prot.append_acquisition(acq)

    # close ISMRMRD file
    if mrd_header_file:
        prot.close()

    return seq, min_te, min_tr


def main(
    system: pp.Opts | None = None,
    te: float | None = None,
    tr: float | None = None,
    rf_flip_angle: float = 12,
    fov_xy: float = 128e-3,
    n_readout: int = 128,
    n_spokes: int = 128,
    slice_thickness: float = 8e-3,
    n_slices: int = 1,
    receiver_bandwidth_per_pixel: float = 800,  # Hz/pixel
    n_dummy_excitations: int = 20,
    show_plots: bool = True,
    test_report: bool = True,
    timing_check: bool = True,
) -> tuple[pp.Sequence, Path]:
    """Generate a radial FLASH sequence.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    te
        Desired echo time (TE) (in seconds). Minimum echo time is used if set to None.
    tr
        Desired repetition time (TR) (in seconds). Minimum repetition time is used if set to None.
    rf_flip_angle
        Flip angle of rf excitation pulse (in degrees)
    fov_xy
        Field of view in x and y direction (in meters).
    n_readout
        Number of frequency encoding steps.
    n_spokes
        Number of radial lines.
    slice_thickness
        Slice thickness of the 2D slice (in meters).
    n_slices
        Number of slices.
    receiver_bandwidth_per_pixel
        Desired receiver bandwidth per pixel (in Hz/pixel). This is used to calculate the readout duration.
    n_dummy_excitations
        Number of dummy excitations before data acquisition to ensure steady state.
    show_plots
        Toggles sequence plot.
    test_report
        Toggles advanced test report.
    timing_check
        Toggles timing check of the sequence.

    Returns
    -------
    seq
        Sequence object of radial FLASH sequence.
    file_path
        Path to the sequence file.
    """
    if system is None:
        system = sys_defaults

    # define settings of rf excitation pulse
    rf_duration = 1.28e-3  # duration of the rf excitation pulse [s]
    rf_bwt = 4  # bandwidth-time product of rf excitation pulse [Hz*s]
    rf_apodization = 0.5  # apodization factor of rf excitation pulse
    readout_oversampling = 2  # readout oversampling factor, commonly 2. This reduces aliasing artifacts.
    spoke_angle = GOLDEN_ANGLE_HALF_CIRCLE

    # define ADC and gradient timing
    n_readout_with_oversampling = int(n_readout * readout_oversampling)
    adc_dwell_time = 1.0 / (receiver_bandwidth_per_pixel * n_readout_with_oversampling)
    gx_pre_duration = 1.0e-3  # duration of readout pre-winder gradient [s]
    gx_flat_time, adc_dwell_time = find_gx_flat_time_on_adc_raster(
        n_readout_with_oversampling, adc_dwell_time, system.grad_raster_time, system.adc_raster_time
    )

    # define spoiling
    gz_spoil_duration = 0.8e-3  # duration of spoiler gradient [s]
    gz_spoil_area = 4 / slice_thickness  # area / zeroth gradient moment of spoiler gradient
    rf_spoiling_phase_increment = 117  # RF spoiling phase increment [°]. Set to 0 for no RF spoiling.

    # define sequence filename
    filename = f'{Path(__file__).stem}_{int(fov_xy * 1000)}fov_{n_readout}nx_{n_spokes}na_{n_slices}ns'

    output_path = Path.cwd() / 'output'
    output_path.mkdir(parents=True, exist_ok=True)

    # delete existing header file
    if (output_path / Path(filename + '_header.h5')).exists():
        (output_path / Path(filename + '_header.h5')).unlink()

    seq, min_te, min_tr = radial_flash_kernel(
        system=system,
        te=te,
        tr=tr,
        fov_xy=fov_xy,
        n_readout=n_readout,
        n_spokes=n_spokes,
        spoke_angle=spoke_angle,
        readout_oversampling=readout_oversampling,
        slice_thickness=slice_thickness,
        n_slices=n_slices,
        n_dummy_excitations=n_dummy_excitations,
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
    seq.set_definition('FOV', [fov_xy, fov_xy, slice_thickness * n_slices])
    seq.set_definition('ReconMatrix', (n_readout, n_readout, 1))
    seq.set_definition('SliceThickness', slice_thickness)
    seq.set_definition('TE', te or min_te)
    seq.set_definition('TR', tr or min_tr)
    seq.set_definition('ReadoutOversamplingFactor', readout_oversampling)

    # save seq-file to disk
    print(f"\nSaving sequence file '{filename}.seq' into folder '{output_path}'.")
    seq.write(str(output_path / filename), create_signature=True)

    if show_plots:
        seq.plot(time_range=(0, 10 * (tr or min_tr)))

    return seq, output_path / filename


if __name__ == '__main__':
    main()
