"""3D Echo Planar Imaging (EPI) FID sequence."""

from pathlib import Path
from typing import Literal

import ismrmrd
import numpy as np
import pypulseq as pp
from raw2ismrmrd.utils import Fov
from raw2ismrmrd.utils import Limits
from raw2ismrmrd.utils import MatrixSize
from raw2ismrmrd.utils import create_header

from mrseq.utils import round_to_raster
from mrseq.utils import sys_defaults
from mrseq.utils import write_sequence
from mrseq.utils.EpiReadout import EpiReadout


def epi3d_fid_kernel(
    system: pp.Opts,
    te: float | None,
    tr: float | None,
    fov_xy: float,
    fov_z: float,
    n_readout: int,
    n_phase_encoding: int,
    n_partitions: int,
    bandwidth: float,
    rf_duration: float,
    rf_flip_angle: float,
    rf_bwt: float,
    rf_apodization: float,
    readout_type: Literal['symmetric', 'flyback'],
    readout_oversampling: Literal[1, 2, 4],
    ramp_sampling: bool,
    partial_fourier_factor: float,
    partial_fourier_factor_z: float,
    add_spoiler: bool,
    add_noise_acq: bool,
    add_navigator_acq: bool,
    mrd_header_file: str | Path | None,
) -> tuple[pp.Sequence, float, float]:
    """Generate a 3D Echo Planar Imaging (EPI) FID sequence.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    te
        Desired echo time (TE) (in seconds). Minimum echo time is used if set to None.
    tr
        Desired repetition time (TR) (in seconds). Minimum repetition time is used if set to None.
    fov_xy
        Field of view in x and y direction (in meters).
    fov_z
        Field of view in z (partition) direction (in meters). Defines the slab thickness.
    n_readout
        Number of frequency encoding steps.
    n_phase_encoding
        Number of phase encoding steps.
    n_partitions
        Number of partitions (kz encoding steps).
    bandwidth
        Total receiver bandwidth (in Hz).
    rf_duration
        Duration of the rf excitation pulse (in seconds).
    rf_flip_angle
        Flip angle of rf excitation pulse (in degrees).
    rf_bwt
        Bandwidth-time product of rf excitation pulse (in Hz * seconds).
    rf_apodization
        Apodization factor of rf excitation pulse.
    readout_type
        Readout type ('symmetric' or 'flyback').
    readout_oversampling
        Readout oversampling factor. Can be 1 (no oversampling), 2, or 4.
    ramp_sampling
        If True, ADC is active during gradient ramps for optimized timing.
    partial_fourier_factor
        Desired partial Fourier factor in "phase encoding" direction. Must be larger than 0.5 and smaller or equal to 1.
        The actual partial Fourier factor might slightly deviate from the desired value.
    partial_fourier_factor_z
        Desired partial Fourier factor in partition direction. Must be larger than 0.5 and smaller or equal to 1.
        The actual partial Fourier factor might slightly deviate from the desired value.
    add_spoiler
        If True, a spoiler gradients will be added to the sequence after the EPI readout.
    add_noise_acq
        If True, noise acquisitions will be added at the beginning of the sequence.
    add_navigator_acq
        If True, 3 navigator acquisitions will be added to the sequence to allow for ghost corrections.
        The navigator acquisitions are added between the rf excitation pulse and the EPI readout.
        Be aware that navigator acquisitions will increase the minimum echo and repetition times.
    mrd_header_file
        Filename of the ISMRMRD header file to be created. If None, no header file is created.

    Returns
    -------
    seq
        PyPulseq Sequence object.
    min_te
        Shortest possible echo time.
    min_tr
        Shortest possible repetition time.

    """
    # create PyPulseq Sequence object and set system limits
    seq = pp.Sequence(system=system)

    # define number of navigator acquisitions
    n_navigator_acq = 3 if add_navigator_acq else 0

    # create EpiReadout object
    epi3d = EpiReadout(
        system=system,
        fov=fov_xy,
        n_readout=n_readout,
        n_phase_encoding=n_phase_encoding,
        bandwidth=bandwidth,
        oversampling=readout_oversampling,
        readout_type=readout_type,
        ramp_sampling=ramp_sampling,
        partial_fourier_factor=partial_fourier_factor,
        partial_fourier_factor_z=partial_fourier_factor_z,
        pe_enable=True,
        spoiling_enable=add_spoiler,
        n_partitions=n_partitions,
        fov_z=fov_z,
    )

    # create slab-selective excitation pulse and gradients
    rf, gz, gzr = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(rf_flip_angle),
        duration=rf_duration,
        slice_thickness=fov_z,
        apodization=rf_apodization,
        time_bw_product=rf_bwt,
        delay=system.rf_dead_time,
        system=system,
        return_gz=True,
        use='excitation',
    )

    # precompute duration of the merged slab rewinder + kz prephaser (same for all partitions)
    gz_pre_dur = pp.calc_duration(
        pp.make_trapezoid(
            channel='z',
            system=system,
            area=abs(gzr.area) + float(np.max(np.abs(epi3d.kz_areas))),
        )
    )
    gz_reph_pre_template = pp.make_trapezoid(
        channel='z',
        system=system,
        area=gzr.area + epi3d.kz_areas[0],
        duration=gz_pre_dur,
    )

    # duration of the pre-readout gradient block (between excitation and EPI readout)
    if add_navigator_acq:
        t_pre = pp.calc_duration(gzr, epi3d.gx_pre)
        t_pre += n_navigator_acq * pp.calc_duration(epi3d.gx)
        t_pre += pp.calc_duration(epi3d.gy_pre, epi3d.gz_pre)
    else:
        t_pre = pp.calc_duration(gz_reph_pre_template, epi3d.gx_pre, epi3d.gy_pre)

    # calculate echo time delay
    min_te = rf.shape_dur / 2  # time from center to end of RF pulse
    min_te += max(rf.ringdown_time, gz.fall_time)  # RF ringdown time or gradient fall time
    min_te += t_pre
    min_te += epi3d.time_to_center_without_prephaser

    if te is None:
        te_delay = 0.0
    else:
        te_delay = round_to_raster(te - min_te, system.block_duration_raster)
        if te_delay < 0:
            raise ValueError(f'TE must be larger than {min_te * 1000:.3f} ms. Current value is {te * 1000:.3f} ms.')

    # calculate repetition time delay (tr_delay) for current TE settings
    min_tr = pp.calc_duration(rf, gz)
    min_tr += t_pre
    min_tr += epi3d.total_duration_without_prephaser
    min_tr += te_delay

    if tr is None:
        tr_delay = 0.0
    else:
        tr_delay = round_to_raster(tr - min_tr, system.block_duration_raster)
        if tr_delay < 0:
            raise ValueError(f'TR must be larger than {min_tr * 1000:.2f} ms. Current value is {tr * 1000:.3f} ms.')

    # create header
    prot = None
    if mrd_header_file:
        hdr = create_header(
            traj_type='other',
            encoding_fov=Fov(x=fov_xy * readout_oversampling, y=fov_xy, z=fov_z),
            recon_fov=Fov(x=fov_xy, y=fov_xy, z=fov_z),
            encoding_matrix=MatrixSize(
                n_x=n_readout * readout_oversampling, n_y=epi3d.n_phase_encoding, n_z=epi3d.n_partitions_total
            ),
            recon_matrix=MatrixSize(n_x=n_readout, n_y=n_phase_encoding, n_z=n_partitions),
            dwell_time=epi3d.adc.dwell,
            slice_limits=Limits(min=0, max=1, center=0),
            k1_limits=Limits(min=0, max=epi3d.n_phase_enc_total, center=epi3d.n_phase_enc_pre_center + 1),
            k2_limits=Limits(min=0, max=epi3d.n_partitions_total, center=epi3d.n_partitions_pre_center + 1),
        )

        # write header to file
        prot = ismrmrd.Dataset(mrd_header_file, 'w')
        prot.write_xml_header(hdr.toXML('utf-8'))

    # obtain noise samples if selected
    if add_noise_acq:
        seq.add_block(
            pp.make_label(label='LIN', type='SET', value=0),
            pp.make_label(label='SLC', type='SET', value=0),
            pp.make_label(label='NOISE', type='SET', value=True),
        )
        seq.add_block(
            epi3d.adc, pp.make_delay(round_to_raster(pp.calc_duration(epi3d.adc), system.block_duration_raster, 'ceil'))
        )
        seq.add_block(pp.make_label(label='NOISE', type='SET', value=False))
        seq.add_block(pp.make_delay(system.rf_dead_time))

        # Write noise trajectory to MRD (zero trajectory — no gradients active)
        if mrd_header_file:
            assert prot is not None
            n_samples = epi3d.adc.num_samples
            acq = ismrmrd.Acquisition()
            acq.resize(trajectory_dimensions=epi3d.trajectory_dimensions, number_of_samples=n_samples)
            acq.traj[:] = np.zeros((n_samples, epi3d.trajectory_dimensions), dtype=np.float32)
            prot.append_acquisition(acq)

    # single slab (slice index 0)
    slice_label = pp.make_label(label='SLC', type='SET', value=0)

    for par in range(epi3d.n_partitions_total):
        par_label = pp.make_label(label='PAR', type='SET', value=par)

        # add slab-selective excitation pulse and set slice/partition labels
        seq.add_block(rf, gz, slice_label, par_label)

        # add navigator scans for ghost correction
        if add_navigator_acq:
            # invert pre-winder for odd number of navigators
            gx_pre_factor = -1 if n_navigator_acq % 2 == 1 else 1
            seq, prot = epi3d.add_navigator_to_seq(
                seq,
                gzr,
                n_navigator_acq,
                n_phase_encoding,
                gx_pre_factor,
                mrd_dataset=prot,
            )

            # add echo time delay
            seq.add_block(pp.make_delay(te_delay))

            # add remaining pre-phasers and reset labels
            # (kx pre-winder is part of the navigators, navigators stay at kz = 0)
            seq.add_block(
                epi3d.gy_pre,
                epi3d.gz_pre_for_partition(par),
                pp.make_label(label='NAV', type='SET', value=0),
            )
        else:
            # add echo time delay
            seq.add_block(pp.make_delay(te_delay))

            # merge slab rewinder and kz pre-phaser into one trapezoid per partition
            # and align it with the readout pre-winder gradients
            gz_reph_pre = pp.make_trapezoid(
                channel='z',
                system=system,
                area=gzr.area + epi3d.kz_areas[par],
                duration=gz_pre_dur,
            )
            gz_reph_pre, gx_pre, gy_pre = pp.align(left=[gz_reph_pre], right=[epi3d.gx_pre, epi3d.gy_pre])
            seq.add_block(gz_reph_pre, gx_pre, gy_pre)

        # add EPI readout block without pre-phaser gradients
        # (trajectory is written per-readout inside add_to_seq when mrd_dataset is provided)
        seq, prot = epi3d.add_to_seq(seq, add_prephaser=False, mrd_dataset=prot, partition_idx=par)

        # add repetition time delay
        if tr_delay > 0:
            seq.add_block(pp.make_delay(tr_delay))

    # close ISMRMRD file
    if mrd_header_file:
        assert prot is not None
        prot.close()

    # set gridding definitions extracted from EpiReadout
    if ramp_sampling:
        gridding_params = [
            epi3d.gx.rise_time,
            epi3d.gx.flat_time,
            epi3d.gx.fall_time,
            epi3d.adc.delay - epi3d.gx.delay,
            epi3d.adc.num_samples * epi3d.adc.dwell,
        ]

        seq.set_definition(key='TargetGriddedSamples', value=n_readout * readout_oversampling)
        seq.set_definition(key='TrapezoidGriddingParameters', value=gridding_params)

    # write all required parameters in the seq-file header/definitions
    seq.set_definition('FOV', [fov_xy, fov_xy, fov_z])
    seq.set_definition('ReconMatrix', (n_readout, n_phase_encoding, n_partitions))
    seq.set_definition('SliceThickness', fov_z / n_partitions)
    seq.set_definition('TE', te or float(min_te))
    seq.set_definition('TR', tr or float(min_tr))
    seq.set_definition('ReadoutOversamplingFactor', readout_oversampling)

    return seq, float(min_te), float(min_tr)


def main(
    system: pp.Opts | None = None,
    te: float | None = None,
    tr: float | None = None,
    fov_xy: float = 200e-3,
    fov_z: float = 64e-3,
    n_readout: int = 64,
    n_phase_encoding: int = 64,
    n_partitions: int = 16,
    bandwidth: float = 100e3,
    readout_type: Literal['symmetric', 'flyback'] = 'symmetric',
    readout_oversampling: Literal[1, 2, 4] = 2,
    ramp_sampling: bool = True,
    partial_fourier_factor: float = 0.7,
    partial_fourier_factor_z: float = 1.0,
    add_navigator_acq: bool = True,
    add_noise_acq: bool = True,
    show_plots: bool = True,
    test_report: bool = True,
    timing_check: bool = True,
    v141_compatibility: bool = True,
) -> tuple[pp.Sequence, Path]:
    """Generate a 3D Echo Planar Imaging (EPI) FID sequence.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    te
        Desired echo time (TE) (in seconds). Minimum echo time is used if set to None.
    tr
        Desired repetition time (TR) (in seconds). Minimum repetition time is used if set to None.
    fov_xy
        Field of view in x and y direction (in meters).
    fov_z
        Field of view in z (partition) direction (in meters).
    n_readout
        Number of frequency encoding steps.
    n_phase_encoding
        Number of phase encoding steps.
    n_partitions
        Number of partitions (kz encoding steps).
    bandwidth
        Total receiver bandwidth (in Hz).
    readout_type
        Readout type ('symmetric' or 'flyback').
    readout_oversampling
        Readout oversampling factor. Can be 1 (no oversampling), 2, or 4.
    ramp_sampling
        If True, ADC is active during gradient ramps for optimized timing.
    partial_fourier_factor
        Desired partial Fourier factor in "phase encoding" direction.
    partial_fourier_factor_z
        Desired partial Fourier factor in partition direction.
    add_navigator_acq
        If True, navigator acquisitions will be added for ghost corrections.
    add_noise_acq
        If True, noise acquisitions will be added at the beginning of the sequence.
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
        Sequence object of 3D EPI FID sequence.
    file_path
        Path to the sequence file.
    """
    if system is None:
        system = sys_defaults

    # define settings of rf excitation pulse
    rf_flip_angle = 30.0  # flip angle of the rf excitation pulse [°]
    rf_duration = 1.28e-3  # duration of the rf excitation pulse [s]
    rf_bwt = 4.0  # bandwidth-time product of rf excitation pulse [Hz*s]
    rf_apodization = 0.5  # apodization factor of rf excitation pulse

    # define spoiling settings
    enable_gradient_spoiling = True

    # define sequence filename
    rs_string = 'rs' if ramp_sampling else 'nors'  # ramp sampling
    pf_string = f'{partial_fourier_factor}pf'.replace('.', 'p')  # partial fourier factor
    pfz_string = f'{partial_fourier_factor_z}pfz'.replace('.', 'p')  # partial fourier factor along kz
    readout_string = 'sym' if readout_type == 'symmetric' else 'flyb'  # readout type
    noise_string = 'withnoise' if add_noise_acq else 'nonoise'  # noise acquisition
    nav_string = 'withnav' if add_navigator_acq else 'nonav'  # navigator acquisition

    filename = f'{Path(__file__).stem}_{int(fov_xy * 1000)}fov_{n_readout}px_{n_partitions}par'
    filename += f'_{readout_string}_fid_{readout_oversampling}ro_{rs_string}_{pf_string}_{pfz_string}'
    filename += f'_{noise_string}_{nav_string}'

    output_path = Path.cwd() / 'output'
    output_path.mkdir(parents=True, exist_ok=True)

    # delete existing header file
    if (output_path / Path(filename + '_header.h5')).exists():
        (output_path / Path(filename + '_header.h5')).unlink()

    mrd_file = output_path / Path(filename + '_header.h5')

    seq, _min_te, _min_tr = epi3d_fid_kernel(
        system=system,
        te=te,
        tr=tr,
        fov_xy=fov_xy,
        fov_z=fov_z,
        n_readout=n_readout,
        n_phase_encoding=n_phase_encoding,
        n_partitions=n_partitions,
        bandwidth=bandwidth,
        readout_oversampling=readout_oversampling,
        rf_duration=rf_duration,
        rf_flip_angle=rf_flip_angle,
        rf_bwt=rf_bwt,
        rf_apodization=rf_apodization,
        readout_type=readout_type,
        ramp_sampling=ramp_sampling,
        partial_fourier_factor=partial_fourier_factor,
        partial_fourier_factor_z=partial_fourier_factor_z,
        add_spoiler=enable_gradient_spoiling,
        add_noise_acq=add_noise_acq,
        add_navigator_acq=add_navigator_acq,
        mrd_header_file=mrd_file,
    )

    # check timing of the sequence
    if timing_check and not test_report:
        ok, error_report = seq.check_timing()
        if ok:
            print('\nTiming check passed successfully')
        else:
            print('\nTiming check failed! Error listing follows\n')
            print(error_report)

    # show advanced test report
    if test_report:
        print('\nCreating advanced test report...')
        print(seq.test_report())

    # save seq-file to disk
    print(f"\nSaving sequence file '{filename}.seq' into folder '{output_path}'.")
    write_sequence(seq, str(output_path / filename), create_signature=True, v141_compatibility=v141_compatibility)

    if show_plots:
        seq.plot(time_range=(0, tr or _min_tr), plot_now=True)

    return seq, output_path / filename


if __name__ == '__main__':
    main()
