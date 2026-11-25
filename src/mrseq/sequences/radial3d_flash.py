"""3D radial (kooshball) FLASH sequence."""

from pathlib import Path
from typing import Literal

import ismrmrd
import numpy as np
import pypulseq as pp
from raw2ismrmrd.utils import Fov
from raw2ismrmrd.utils import Limits
from raw2ismrmrd.utils import MatrixSize
from raw2ismrmrd.utils import create_header

from mrseq.utils import find_gx_flat_time_on_adc_raster
from mrseq.utils import radial_3d_spoke_directions
from mrseq.utils import radial_3d_trajectory
from mrseq.utils import rotation_matrix_to_direction
from mrseq.utils import round_to_raster
from mrseq.utils import sys_defaults
from mrseq.utils import write_sequence


def radial3d_flash_kernel(
    system: pp.Opts,
    te: float | None,
    tr: float | None,
    fov: float,
    n_readout: int,
    n_spokes: int,
    spoke_type: Literal['full', 'half'],
    distribution: Literal['golden', 'spiral', 'phyllotaxis'],
    n_interleaves: int | None,
    excitation: Literal['block', 'slab'],
    readout_oversampling: float,
    n_dummy_excitations: int,
    gx_pre_duration: float,
    gx_flat_time: float,
    rf_duration: float,
    rf_flip_angle: float,
    rf_bwt: float,
    rf_apodization: float,
    rf_spoiling_phase_increment: float,
    gz_spoil_duration: float | None,
    gz_spoil_area: float,
    mrd_header_file: str | Path | None,
) -> tuple[pp.Sequence, float, float]:
    """Generate a 3D radial (kooshball) FLASH sequence.

    For half spokes, the k-space center is acquired at the first sample of the flat top (no ramp sampling).

    Parameters
    ----------
    system
        PyPulseq system limits object.
    te
        Desired echo time (TE) (in seconds). Minimum echo time is used if set to None.
    tr
        Desired repetition time (TR) (in seconds).
    fov
        Isotropic field of view (in meters).
    n_readout
        Number of frequency encoding steps.
    n_spokes
        Number of radial spokes.
    spoke_type
        'full' spokes pass through the k-space center, 'half' spokes start at the k-space center.
    distribution
        Spoke distribution: 'golden', 'spiral' or 'phyllotaxis'.
    n_interleaves
        Number of interleaves the spokes are split into.
    excitation
        Excitation type: 'block' (non-selective block pulse) or 'slab' (slice selective sinc pulse covering the whole
        FOV in z direction).
    readout_oversampling
        Readout oversampling factor, commonly 2. This reduces aliasing artifacts.
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
        Bandwidth-time product of rf excitation pulse (Hz * seconds). Only used for excitation = 'slab'.
    rf_apodization
        Apodization factor of rf excitation pulse. Only used for excitation = 'slab'.
    rf_spoiling_phase_increment
        RF spoiling phase increment (in degrees). Set to 0 for no RF spoiling.
    gz_spoil_duration
        Duration of spoiler gradient (in seconds). Set to None for minimum duration.
    gz_spoil_area
        Area of spoiler gradient (in 1/m). Set to 0 for no spoiler gradient.
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

    # create excitation pulse and gradients
    gz = gzr = None
    if excitation == 'slab':
        rf, gz, gzr = pp.make_sinc_pulse(
            flip_angle=rf_flip_angle / 180 * np.pi,
            duration=rf_duration,
            slice_thickness=fov,
            apodization=rf_apodization,
            time_bw_product=rf_bwt,
            delay=system.rf_dead_time,
            system=system,
            return_gz=True,
            use='excitation',
        )
    elif excitation == 'block':
        rf = pp.make_block_pulse(
            flip_angle=rf_flip_angle / 180 * np.pi,
            duration=rf_duration,
            delay=system.rf_dead_time,
            system=system,
            use='excitation',
        )
    else:
        raise ValueError(f'Unknown excitation type {excitation}.')

    # create readout gradient and ADC
    delta_k = 1 / (fov * readout_oversampling)
    if spoke_type == 'full':
        n_readout_with_oversampling = int(n_readout * readout_oversampling)
        n_readout_with_oversampling = n_readout_with_oversampling + np.mod(n_readout_with_oversampling, 2)  # make even
        gx = pp.make_trapezoid(
            channel='x', flat_area=n_readout_with_oversampling * delta_k, flat_time=gx_flat_time, system=system
        )
        adc = pp.make_adc(
            num_samples=n_readout_with_oversampling, duration=gx.flat_time, delay=gx.rise_time, system=system
        )
        k0_center_id = np.where(
            (np.arange(n_readout_with_oversampling) - n_readout_with_oversampling / 2) * delta_k == 0
        )[0][0]
    elif spoke_type == 'half':
        n_readout_with_oversampling = int(np.ceil(n_readout * readout_oversampling / 2))
        gx = pp.make_trapezoid(
            channel='x', flat_area=n_readout_with_oversampling * delta_k, flat_time=gx_flat_time, system=system
        )
        adc = pp.make_adc(
            num_samples=n_readout_with_oversampling, duration=gx.flat_time, delay=gx.rise_time, system=system
        )
        k0_center_id = 0
    else:
        raise ValueError(f'Unknown spoke type {spoke_type}.')

    # create frequency encoding pre- and re-winder gradient
    gx_pre = None
    if spoke_type == 'full':
        gx_pre = pp.make_trapezoid(
            channel='x', area=-gx.area / 2 - delta_k / 2, duration=gx_pre_duration, system=system
        )
        gx_post = pp.make_trapezoid(
            channel='x', area=-gx.area / 2 + delta_k / 2, duration=gx_pre_duration, system=system
        )
    else:
        # full rewind of the readout gradient, so the encoding train is balanced
        gx_post = pp.make_trapezoid(channel='x', area=-gx.area, duration=gx_pre_duration, system=system)

    # create spoiler gradient
    gz_spoil = None
    if gz_spoil_area != 0:
        gz_spoil = pp.make_trapezoid(channel='z', system=system, area=gz_spoil_area, duration=gz_spoil_duration)

    # calculate duration of events between excitation and readout
    pre_readout_events = [event for event in (gzr, gx_pre) if event is not None]
    if te is None:
        gzr_gx_dur = pp.calc_duration(*pre_readout_events) if pre_readout_events else 0.0
    else:
        gzr_gx_dur = sum(pp.calc_duration(event) for event in pre_readout_events)

    # calculate minimum echo time
    min_te = float(
        rf.shape_dur / 2  # time from center to end of RF pulse
        + max(rf.ringdown_time, gz.fall_time if gz is not None else 0)  # RF ringdown time or gradient fall time
        + gzr_gx_dur  # slab selection re-phasing gradient and readout pre-winder
        + gx.delay  # potential delay of readout gradient
        + gx.rise_time  # rise time of readout gradient
        + (k0_center_id + 0.5) * adc.dwell  # time from beginning of ADC to time point of k-space center sample
    )

    # calculate echo time delay (te_delay)
    if te is None:
        te_delay = 0.0
    else:
        te_delay = round_to_raster(te - min_te, system.block_duration_raster)
        if te_delay < 0:
            raise ValueError(f'TE must be larger than {min_te * 1000:.3f} ms. Current value is {te * 1000:.3f} ms.')

    # calculate minimum repetition time
    min_tr = (
        pp.calc_duration(*[event for event in (rf, gz) if event is not None])  # rf pulse
        + gzr_gx_dur  # slab selection re-phasing gradient and readout pre-winder
        + pp.calc_duration(gx)  # readout gradient
        + pp.calc_duration(gx_post)  # readout re-winder
        + (pp.calc_duration(gz_spoil) if gz_spoil is not None else 0.0)  # gradient spoiler
    )

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

    # calculate spoke directions
    directions = radial_3d_spoke_directions(
        n_spokes, spoke_type=spoke_type, distribution=distribution, n_interleaves=n_interleaves
    )

    # create header
    if mrd_header_file:
        if spoke_type == 'full':
            n_encoding = n_readout_with_oversampling
        else:
            n_encoding = int(n_readout * readout_oversampling)
            n_encoding = n_encoding + np.mod(n_encoding, 2)  # make even
        hdr = create_header(
            traj_type='other',
            encoding_fov=Fov(x=fov * readout_oversampling, y=fov, z=fov),
            recon_fov=Fov(x=fov, y=fov, z=fov),
            encoding_matrix=MatrixSize(n_x=n_encoding, n_y=n_encoding, n_z=n_encoding),
            recon_matrix=MatrixSize(n_x=n_readout, n_y=n_readout, n_z=n_readout),
            dwell_time=adc.dwell,
            slice_limits=Limits(min=0, max=0, center=0),
            k1_limits=Limits(min=0, max=n_spokes, center=0),
        )

        # write header to file
        prot = ismrmrd.Dataset(mrd_header_file, 'w')
        prot.write_xml_header(hdr.toXML('utf-8'))

    # obtain noise samples
    seq.add_block(pp.make_label(label='LIN', type='SET', value=0), pp.make_label(label='SET', type='SET', value=0))
    seq.add_block(adc, pp.make_label(label='NOISE', type='SET', value=True))
    seq.add_block(pp.make_label(label='NOISE', type='SET', value=False))
    seq.add_block(pp.make_delay(system.rf_dead_time))

    if mrd_header_file:
        acq = ismrmrd.Acquisition()
        acq.resize(trajectory_dimensions=3, number_of_samples=adc.num_samples)
        prot.append_acquisition(acq)

    for spoke_ in range(-n_dummy_excitations, n_spokes):
        # calculate current phase_offset if rf_spoiling is activated
        if rf_spoiling_phase_increment > 0:
            rf.phase_offset = rf_phase / 180 * np.pi
            adc.phase_offset = rf_phase / 180 * np.pi

        # add excitation pulse
        if gz is not None:
            seq.add_block(rf, gz)
        else:
            seq.add_block(rf)

        # update rf phase offset for the next excitation pulse
        rf_inc = divmod(rf_inc + rf_spoiling_phase_increment, 360.0)[1]
        rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

        # calculate rotation matrix for the current spoke
        rot = rotation_matrix_to_direction(directions[max(spoke_, 0)])

        if te_delay > 0:
            if gzr is not None:
                seq.add_block(gzr)
            seq.add_block(pp.make_delay(te_delay))
            if gx_pre is not None:
                seq.add_block(*pp.rotate_3d(gx_pre, rotation=rot, system=system))
        elif pre_readout_events:
            seq.add_block(*pp.rotate_3d(*pre_readout_events, rotation=rot, system=system))

        # rotate and add the readout gradient and ADC
        if spoke_ >= 0:
            labels = []
            labels.append(pp.make_label(label='LIN', type='SET', value=spoke_))
            labels.append(pp.make_label(label='SET', type='SET', value=spoke_ % n_interleaves if n_interleaves else 0))
            seq.add_block(*pp.rotate_3d(gx, rotation=rot, system=system), adc, *labels)
        else:
            seq.add_block(*pp.rotate_3d(gx, rotation=rot, system=system))

        seq.add_block(*pp.rotate_3d(gx_post, rotation=rot, system=system))

        # the spoiler stays on the z-axis and is not rotated
        if gz_spoil is not None:
            seq.add_block(gz_spoil)

        # add delay in case TR > min_TR
        if tr_delay > 0:
            seq.add_block(pp.make_delay(tr_delay))

        if mrd_header_file and spoke_ >= 0:
            # add acquisitions to metadata
            radial_trajectory = radial_3d_trajectory(directions[spoke_][None], adc.num_samples, spoke_type=spoke_type)[
                0
            ].astype(np.float32)

            acq = ismrmrd.Acquisition()
            acq.resize(trajectory_dimensions=3, number_of_samples=adc.num_samples)
            acq.traj[:] = radial_trajectory
            prot.append_acquisition(acq)

    # close ISMRMRD file
    if mrd_header_file:
        prot.close()

    # write all required parameters in the seq-file header/definitions
    seq.set_definition('FOV', [fov, fov, fov])
    seq.set_definition('ReconMatrix', (n_readout, n_readout, n_readout))
    seq.set_definition('TE', te or min_te)
    seq.set_definition('TR', tr or min_tr)
    seq.set_definition('ReadoutOversamplingFactor', readout_oversampling)
    seq.set_definition('SpokeType', spoke_type)
    seq.set_definition('SpokeDistribution', distribution)

    return seq, min_te, min_tr


def main(
    system: pp.Opts | None = None,
    te: float | None = None,
    tr: float | None = None,
    rf_flip_angle: float = 12,
    fov: float = 128e-3,
    n_readout: int = 64,
    n_spokes: int | None = None,
    spoke_type: Literal['full', 'half'] = 'full',
    distribution: Literal['golden', 'spiral', 'phyllotaxis'] = 'golden',
    n_interleaves: int | None = None,
    excitation: Literal['block', 'slab'] = 'block',
    receiver_bandwidth_per_pixel: float = 800,  # Hz/pixel
    n_dummy_excitations: int = 20,
    show_plots: bool = True,
    test_report: bool = True,
    timing_check: bool = True,
    v141_compatibility: bool = True,
) -> tuple[pp.Sequence, Path]:
    """Generate a 3D radial (kooshball) FLASH sequence.

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
    fov
        Isotropic field of view (in meters).
    n_readout
        Number of frequency encoding steps.
    n_spokes
        Number of radial spokes. If None, the spokes are calculated to fulfill the Nyquist criterion for a sphere.
    spoke_type
        'full' spokes pass through the k-space center, 'half' spokes start at the k-space center.
    distribution
        Spoke distribution: 'golden', 'spiral' or 'phyllotaxis'.
    n_interleaves
        Number of interleaves the spokes are split into.
    excitation
        Excitation type: 'block' (non-selective) or 'slab' (slab selective covering the whole FOV in z direction).
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
    v141_compatibility
        Save the sequence in pulseq v1.4.1 for backwards compatibility.

    Returns
    -------
    seq
        Sequence object of 3D radial FLASH sequence.
    file_path
        Path to the sequence file.
    """
    if system is None:
        system = sys_defaults

    if n_spokes is None:
        # spokes to fulfill the Nyquist criterion for a spherical k-space
        if spoke_type == 'full':
            n_spokes = int(np.pi * n_readout**2 / 2)
        else:
            n_spokes = int(np.pi * n_readout**2)

    # define settings of rf excitation pulse
    rf_duration = 0.2e-3 if excitation == 'block' else 1.28e-3  # duration of the rf excitation pulse [s]
    rf_bwt = 4  # bandwidth-time product of rf excitation pulse [Hz*s]
    rf_apodization = 0.5  # apodization factor of rf excitation pulse
    readout_oversampling = 2  # readout oversampling factor, commonly 2. This reduces aliasing artifacts.

    # define ADC and gradient timing
    n_readout_with_oversampling = int(n_readout * readout_oversampling)
    adc_dwell_time = 1.0 / (receiver_bandwidth_per_pixel * n_readout_with_oversampling)
    gx_pre_duration = 1.0e-3  # duration of readout pre-winder gradient [s]
    gx_flat_time, adc_dwell_time = find_gx_flat_time_on_adc_raster(
        n_readout_with_oversampling, adc_dwell_time, system.grad_raster_time, system.adc_raster_time
    )

    # define spoiling
    gz_spoil_duration = None  # duration of spoiler gradient [s]
    gz_spoil_area = 4 * n_readout / fov  # area of spoiler gradient, 4 cycles per voxel
    rf_spoiling_phase_increment = 84  # RF spoiling phase increment [°]. Set to 0 for no RF spoiling.

    # define sequence filename
    filename = f'{Path(__file__).stem}_{int(fov * 1000)}fov_{n_readout}nx_{n_spokes}na_{spoke_type}_{distribution}'

    output_path = Path.cwd() / 'output'
    output_path.mkdir(parents=True, exist_ok=True)

    # delete existing header file
    if (output_path / Path(filename + '_header.h5')).exists():
        (output_path / Path(filename + '_header.h5')).unlink()

    seq, _min_te, min_tr = radial3d_flash_kernel(
        system=system,
        te=te,
        tr=tr,
        fov=fov,
        n_readout=n_readout,
        n_spokes=n_spokes,
        spoke_type=spoke_type,
        distribution=distribution,
        n_interleaves=n_interleaves,
        excitation=excitation,
        readout_oversampling=readout_oversampling,
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

    # save seq-file to disk
    print(f"\nSaving sequence file '{filename}.seq' into folder '{output_path}'.")
    write_sequence(seq, str(output_path / filename), create_signature=True, v141_compatibility=v141_compatibility)

    if show_plots:
        seq.plot(time_range=(0, 10 * (tr or min_tr)))

    return seq, output_path / filename


if __name__ == '__main__':
    main()
