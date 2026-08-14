"""ADC mapping using a 2D Echo Planar Imaging (EPI) spin echo (SE) sequence."""

from collections.abc import Sequence
from math import floor
from pathlib import Path
from typing import Literal

import ismrmrd
import numpy as np
import pypulseq as pp
from raw2ismrmrd.utils import Fov
from raw2ismrmrd.utils import Limits
from raw2ismrmrd.utils import MatrixSize
from raw2ismrmrd.utils import create_header

from mrseq.preparations.diffusion_prep import DiffusionPrep
from mrseq.utils import round_to_raster
from mrseq.utils import sys_defaults
from mrseq.utils import write_sequence
from mrseq.utils.EpiReadout import EpiReadout


def adc_epi2d_se_kernel(
    system: pp.Opts,
    te: float | None,
    tr: float | None,
    fov_xy: float,
    n_readout: int,
    n_phase_encoding: int,
    bandwidth: float,
    slice_thickness: float,
    n_slices: int,
    rf_ex_duration: float,
    rf_ex_bwt: float,
    rf_ex_apodization: float,
    readout_type: Literal['symmetric', 'flyback'],
    readout_oversampling: Literal[1, 2, 4],
    n_repetitions: int,
    ramp_sampling: bool,
    partial_fourier_factor: float,
    gz_crusher_duration: float,
    b_values: Sequence[float],
    g_diff_max_amplitude: float,
    g_diff_max_slew_rate: float,
    g_diff_delta_time: float,
    ge_segment_delay: float,
    mrd_header_file: str | Path | None,
) -> tuple[pp.Sequence, float, float]:
    """Generate a 2D Echo Planar Imaging (EPI) sequence for ADC mapping..

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
    n_phase_encoding
        Number of phase encoding steps.
    bandwidth
        Total receiver bandwidth in Hz.
    slice_thickness
        Slice thickness of the 2D slice (in meters).
    n_slices
        Number of slices.
    rf_ex_duration
        Duration of the rf excitation pulse (in seconds)
    rf_ex_bwt
        Bandwidth-time product of rf excitation pulse (Hz * seconds)
    rf_ex_apodization
        Apodization factor of rf excitation pulse
    rf_ref_width_scale_factor
        Factor to scale the slice thickness of the refocusing pulse.
    readout_type
        Readout type ('symmetric' or 'flyback').
    readout_oversampling
        Readout oversampling factor. Can be 1 (no oversampling), 2, or 4.
    ramp_sampling
        If True, ADC is active during gradient ramps for optimized timing.
    partial_fourier_factor
        Desired partial Fourier factor in "phase encoding" direction. Must be larger than 0.5 and smaller or equal to 1.
        The actual partial Fourier factor might slightly deviate from the desired value.
    gz_crusher_duration
        Duration of the crusher gradients applied around the 180° pulse.
    b_values
        b-values for diffusion weighting gradients in s/mm^2
    g_diff_max_amplitude
        Max amplitude of diffusion gradient.
    g_diff_max_slew_rate
        Max slew rate of diffusion gradient.
    g_diff_delta_time
        Time between beginning of first and second diffusion gradient.
    ge_segment_delay
        Delay time at the end of each segment for GE scanners.
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
    # Direction and sign of diffusion gradients
    diff_directions: list[Literal['x', 'y', 'z', 'xy', 'yz', 'xz', 'xyz']] = ['xy', 'yz', 'xz', 'xy', 'yz', 'xz']
    diff_sign: list[Literal[-1, 1]] = [1, 1, 1, -1, -1, -1]

    # Time delay between second diffusion gradient and adc to minimize impact of eddy currents
    t_diff_gradient_adc = 1e-3

    # create PyPulseq Sequence object and set system limits
    seq = pp.Sequence(system=system)

    # define number of navigator acquisitions
    n_navigator_acq = 3

    # create diffusion gradient object for timing calculations
    diff_prep = DiffusionPrep(
        system,
        fov_z=slice_thickness,
        rf_ref_duration=rf_ex_duration * 2,
        rf_ref_bwt=rf_ex_bwt,
        rf_ref_width_scale_factor=2,
        g_crusher_duration=gz_crusher_duration,
        g_amplitude=g_diff_max_amplitude,
        g_slew_rate=g_diff_max_slew_rate,
        g_delta_time=g_diff_delta_time,
        max_b_value=max(b_values),
        g_channel='xy',
        g_sign=1,
    )

    # create EpiReadout object
    epi2d = EpiReadout(
        system=system,
        fov=fov_xy,
        n_readout=n_readout,
        n_phase_encoding=n_phase_encoding,
        bandwidth=bandwidth,
        oversampling=readout_oversampling,
        readout_type=readout_type,
        ramp_sampling=ramp_sampling,
        partial_fourier_factor=partial_fourier_factor,
        pe_enable=True,
        spoiling_enable=True,
    )

    # create slice selective excitation pulse and gradients
    rf, gz, gzr = pp.make_sinc_pulse(
        flip_angle=np.pi / 2,
        duration=rf_ex_duration,
        slice_thickness=slice_thickness,
        apodization=rf_ex_apodization,
        time_bw_product=rf_ex_bwt,
        delay=system.rf_dead_time,
        system=system,
        return_gz=True,
        use='excitation',
    )

    # calculate echo time delay(s)
    t_exc_to_ref = rf.shape_dur / 2
    t_exc_to_ref += max(rf.ringdown_time, gz.fall_time)
    if n_navigator_acq > 0:
        t_exc_to_ref += pp.calc_duration(gzr, epi2d.gx_pre)
        t_exc_to_ref += n_navigator_acq * pp.calc_duration(epi2d.gx)
        t_exc_to_ref += pp.calc_duration(epi2d.gy_pre)
    else:
        t_exc_to_ref += pp.calc_duration(gzr, epi2d.gx_pre, epi2d.gy_pre)
    t_exc_to_ref += diff_prep._time_to_refocusing_pulse

    t_ref_to_kcenter = diff_prep._block_duration - diff_prep._time_to_refocusing_pulse
    t_ref_to_kcenter += t_diff_gradient_adc
    t_ref_to_kcenter += ge_segment_delay
    t_ref_to_kcenter += epi2d.time_to_center_without_prephaser

    # calculate minimum echo time
    min_te = 2 * round_to_raster(max(t_exc_to_ref, t_ref_to_kcenter), system.block_duration_raster)

    # calculate echo time delays for minimum echo time
    te_delay1 = round_to_raster(min_te / 2 - t_exc_to_ref, system.block_duration_raster)
    te_delay2 = round_to_raster(min_te / 2 - t_ref_to_kcenter + t_diff_gradient_adc, system.block_duration_raster)

    if te is not None:
        if te > min_te:
            # calculate echo time delays for desired echo time
            additional_te_delay_half = round_to_raster((te - min_te) / 2, system.block_duration_raster)
            te_delay1 += additional_te_delay_half
            te_delay2 += additional_te_delay_half
        else:
            raise ValueError(f'Desired TE ({te * 1000:.3f} ms) is smaller than minimum TE ({min_te * 1000:.3f} ms).')

    t_exc_to_ref = t_exc_to_ref + te_delay1  # might NOT be on block_raster
    t_ref_to_kcenter = t_ref_to_kcenter + te_delay2  # might NOT be on block_raster

    # calculate repetition time delay (tr_delay) for current TE settings
    min_tr = pp.calc_duration(rf, gz)
    if n_navigator_acq > 0:
        min_tr += pp.calc_duration(gzr, epi2d.gx_pre)
        min_tr += n_navigator_acq * pp.calc_duration(epi2d.gx)
        min_tr += pp.calc_duration(epi2d.gy_pre)
    else:
        min_tr += pp.calc_duration(gzr, epi2d.gx_pre, epi2d.gy_pre)
    min_tr += te_delay1
    min_tr += 2 * diff_prep._time_to_refocusing_pulse
    min_tr += te_delay2
    min_tr += epi2d.total_duration_without_prephaser
    min_tr += ge_segment_delay

    if tr is None:
        tr_delay = 0.0
    else:
        tr_delay = round_to_raster(tr - min_tr - ge_segment_delay, system.block_duration_raster)
        if tr_delay < 0:
            raise ValueError(f'TR must be larger than {min_tr * 1000:.2f} ms. Current value is {tr * 1000:.3f} ms.')

    print(f'\nCurrent echo time = {(t_exc_to_ref + t_ref_to_kcenter) * 1000:.3f} ms')
    print(f'Current repetition time = {(min_tr + tr_delay + ge_segment_delay) * 1000:.3f} ms')

    # create header
    if mrd_header_file:
        hdr = create_header(
            traj_type='other',
            encoding_fov=Fov(x=fov_xy * readout_oversampling, y=fov_xy, z=slice_thickness),
            recon_fov=Fov(x=fov_xy, y=fov_xy, z=slice_thickness),
            encoding_matrix=MatrixSize(n_x=n_readout * readout_oversampling, n_y=epi2d.n_phase_encoding, n_z=1),
            recon_matrix=MatrixSize(n_x=n_readout, n_y=n_phase_encoding, n_z=1),
            dwell_time=epi2d.adc.dwell,
            slice_limits=Limits(min=0, max=n_slices, center=n_slices // 2),
            k1_limits=Limits(min=0, max=epi2d.n_phase_enc_total, center=epi2d.n_phase_enc_pre_center + 1),
            k2_limits=Limits(min=0, max=1, center=0),
        )

        # write header to file
        prot = ismrmrd.Dataset(mrd_header_file, 'w')
        prot.write_xml_header(hdr.toXML('utf-8'))

    # Precompute analytical navigator trajectory (single kx line, no ky blip)
    if n_navigator_acq > 0:
        from mrseq.utils.EpiReadout import _trapezoid_area_at_times

        nav_sample_times = epi2d.adc.delay + (np.arange(epi2d.adc.num_samples) + 0.5) * epi2d.adc.dwell
        nav_kx_forward = _trapezoid_area_at_times(
            epi2d.gx.rise_time, epi2d.gx.flat_time, epi2d.gx.fall_time, abs(epi2d.gx.amplitude), nav_sample_times
        )

    b_values_calculated = []
    for rep in range(n_repetitions):
        rep_label = pp.make_label(type='SET', label='REP', value=int(rep))

        diff_prep = DiffusionPrep(
            system,
            rf_ref_duration=rf_ex_duration * 2,
            rf_ref_bwt=rf_ex_bwt,
            rf_ref_width_scale_factor=2,
            g_crusher_duration=gz_crusher_duration,
            g_amplitude=g_diff_max_amplitude,
            g_slew_rate=g_diff_max_slew_rate,
            g_delta_time=g_diff_delta_time,
            max_b_value=max(b_values),
            g_channel=diff_directions[np.mod(rep, len(diff_directions))],
            g_sign=diff_sign[np.mod(rep, len(diff_directions))],
        )

        for b_idx, b_value in enumerate(b_values):
            dw_label = pp.make_label(type='SET', label='ECO', value=int(b_idx))

            for slice_ in range(n_slices):
                # define slice label
                slice_label = pp.make_label(label='SLC', type='SET', value=slice_)

                # set frequency offset for current slice
                rf.freq_offset = gz.amplitude * slice_thickness * (slice_ - (n_slices - 1) / 2)

                # add slice selective excitation pulse and set slice label
                seq.add_block(
                    rf,
                    gz,
                    slice_label,
                    dw_label,
                    rep_label,
                    pp.make_label(type='SET', label='TRID', value=100 + rep * len(b_values) + b_idx),
                )

                # add navigator scans for ghost correction
                if n_navigator_acq > 0:
                    # add slice selection rewinder and readout pre-winder in x direction
                    # (gy_pre will be added after navigators)
                    gzr, gx_pre = pp.align(left=[gzr], right=[epi2d.gx_pre])
                    seq.add_block(
                        gzr,
                        gx_pre,
                        pp.make_label(label='NAV', type='SET', value=1),
                        pp.make_label(label='LIN', type='SET', value=floor(n_phase_encoding / 2)),
                    )

                    # Navigator kx offset: starts from -gx_pre.area (reversed pre-winder)
                    nav_kx_offset = epi2d.gx_pre.area

                    # add navigator acquisitions
                    for n in range(n_navigator_acq):
                        gx_sign = (-1) ** n
                        seq.add_block(
                            pp.scale_grad(epi2d.gx, gx_sign),
                            epi2d.adc,
                            pp.make_label(label='REV', type='SET', value=gx_sign < 0),
                            pp.make_label(label='SEG', type='SET', value=gx_sign < 0),
                            # pp.make_label(label='AVG', type='SET', value=(n + 1) == 3),
                        )

                        # Write navigator trajectory to MRD
                        if mrd_header_file:
                            assert prot is not None
                            n_samples = epi2d.adc.num_samples
                            traj = np.zeros((n_samples, 2), dtype=np.float32)
                            nav_kx = nav_kx_offset + gx_sign * nav_kx_forward
                            traj[:, 0] = nav_kx * fov_xy * readout_oversampling
                            acq = ismrmrd.Acquisition()
                            acq.resize(trajectory_dimensions=2, number_of_samples=n_samples)
                            acq.traj[:] = traj
                            prot.append_acquisition(acq)

                        # Update kx offset for next navigator (accumulate the signed gx area)
                        nav_kx_offset += gx_sign * epi2d.gx.area

                    # add gy_pre and reset labels
                    seq.add_block(
                        pp.scale_grad(epi2d.gy_pre, -1),
                        pp.make_label(label='NAV', type='SET', value=0),
                        pp.make_label(label='AVG', type='SET', value=0),
                    )
                else:
                    seq.add_block(gzr, pp.scale_grad(epi2d.gx_pre, -1), pp.scale_grad(epi2d.gy_pre, -1))

                if te_delay1 > 0:
                    seq.add_block(pp.make_delay(te_delay1))

                # add diffusion gradient and refocusing pulse
                seq, b_value_calc = diff_prep.add_diffusion_prep(seq, b_value=b_value)

                seq.add_block(pp.make_delay(te_delay2), pp.make_label(type='SET', label='TRID', value=1))

                # add EPI readout block without pre-phaser gradients
                # (trajectory is written per-readout inside add_to_seq when mrd_dataset is provided)
                seq, prot = epi2d.add_to_seq(seq, add_prephaser=False, mrd_dataset=prot)

                # add repetition time delay
                if tr_delay > 0:
                    seq.add_block(pp.make_delay(tr_delay))

            if rep == 0:
                b_values_calculated.append(b_value_calc)

    # obtain noise samples if selected
    seq.add_block(
        pp.make_delay(0.1),
        pp.make_label(label='LIN', type='SET', value=0),
        pp.make_label(label='SLC', type='SET', value=0),
        pp.make_label(type='SET', label='TRID', value=9999),
        pp.make_label(label='NOISE', type='SET', value=True),
    )
    seq.add_block(
        epi2d.adc, pp.make_delay(round_to_raster(pp.calc_duration(epi2d.adc), system.block_duration_raster, 'ceil'))
    )
    seq.add_block(pp.make_label(label='NOISE', type='SET', value=False))
    seq.add_block(pp.make_delay(system.rf_dead_time))

    # Write noise trajectory to MRD (zero trajectory — no gradients active)
    if mrd_header_file:
        assert prot is not None
        n_samples = epi2d.adc.num_samples
        acq = ismrmrd.Acquisition()
        acq.resize(trajectory_dimensions=2, number_of_samples=n_samples)
        acq.traj[:] = np.zeros((n_samples, 2), dtype=np.float32)
        prot.append_acquisition(acq)

    # close ISMRMRD file
    if mrd_header_file:
        assert prot is not None
        prot.close()

    # set gridding definitions extracted from EpiReadout
    if ramp_sampling:
        gridding_params = [
            epi2d.gx.rise_time,
            epi2d.gx.flat_time,
            epi2d.gx.fall_time,
            epi2d.adc.delay - epi2d.gx.delay,
            epi2d.adc.num_samples * epi2d.adc.dwell,
        ]

        seq.set_definition(key='TargetGriddedSamples', value=n_readout * readout_oversampling)
        seq.set_definition(key='TrapezoidGriddingParameters', value=gridding_params)

    # write all required parameters in the seq-file header/definitions
    seq.set_definition('FOV', [fov_xy, fov_xy, slice_thickness * n_slices])
    seq.set_definition('ReconMatrix', (n_readout, n_phase_encoding, n_slices))
    seq.set_definition('SliceThickness', slice_thickness)
    seq.set_definition('TE', float(t_exc_to_ref + t_ref_to_kcenter))
    seq.set_definition('TR', tr or float(min_tr))
    seq.set_definition('ReadoutOversamplingFactor', readout_oversampling)
    seq.set_definition('b-values', [int(b) for b in b_values_calculated])
    print(b_values_calculated)
    return seq, float(min_te), float(min_tr)


def main(
    system: pp.Opts | None = None,
    te: float | None = None,
    tr: float | None = None,
    fov_xy: float = 200e-3,
    n_readout: int = 64,
    n_phase_encoding: int = 64,
    n_slices: int = 3,
    slice_thickness: float = 8e-3,
    bandwidth: float = 100e3,
    readout_type: Literal['symmetric', 'flyback'] = 'symmetric',
    readout_oversampling: Literal[1, 2, 4] = 2,
    ramp_sampling: bool = True,
    partial_fourier_factor: float = 1.0,
    b_values: Sequence[float] = [0, 200, 400, 600],
    show_plots: bool = True,
    test_report: bool = True,
    timing_check: bool = True,
    v141_compatibility: bool = True,
) -> tuple[pp.Sequence, Path]:
    """Generate a 2D Echo Planar Imaging (EPI) spin echo (SE) sequence for ADC mapping.

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
    n_readout
        Number of frequency encoding steps.
    n_phase_encoding
        Number of phase encoding steps.
    n_slices
        Number of slices.
    slice_thickness
        Slice thickness of the 2D slice (in meters).
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
    b_values
        b-values for diffusion weighting gradients in s/mm^2
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
        Sequence object of 2D EPI SE sequence.
    file_path
        Path to the sequence file.
    """
    if system is None:
        system = sys_defaults

    if isinstance(b_values, float):
        b_values = (b_values,)

    # define settings of rf excitation pulse
    rf_ex_duration = 1.28e-3  # duration of the rf excitation pulse [s]
    rf_ex_bwt = 4.0  # bandwidth-time product of rf excitation pulse [Hz*s]
    rf_ex_apodization = 0.5  # apodization factor of rf excitation pulse

    gz_crusher_duration = 1.6e-3  # duration of crusher gradients [s]

    # diffusion gradients
    g_diff_delta_time = 34e-3

    # define sequence filename
    rs_string = 'rs' if ramp_sampling else 'nors'  # ramp sampling
    pf_string = f'{partial_fourier_factor}pf'.replace('.', 'p')  # partial fourier factor
    readout_string = 'sym' if readout_type == 'symmetric' else 'flyb'  # readout type

    filename = f'{Path(__file__).stem}_{int(fov_xy * 1000)}fov_{n_readout}px'
    filename += f'_{readout_string}_se_{readout_oversampling}ro_{rs_string}_{pf_string}'

    output_path = Path.cwd() / 'output'
    output_path.mkdir(parents=True, exist_ok=True)

    # delete existing header file
    if (output_path / Path(filename + '_header.h5')).exists():
        (output_path / Path(filename + '_header.h5')).unlink()

    mrd_file = output_path / Path(filename + '_header.h5')

    seq, _min_te, min_tr = adc_epi2d_se_kernel(
        system=system,
        te=te,
        tr=tr,
        fov_xy=fov_xy,
        n_readout=n_readout,
        n_phase_encoding=n_phase_encoding,
        bandwidth=bandwidth,
        readout_oversampling=readout_oversampling,
        n_repetitions=1,
        slice_thickness=slice_thickness,
        n_slices=n_slices,
        rf_ex_duration=rf_ex_duration,
        rf_ex_bwt=rf_ex_bwt,
        rf_ex_apodization=rf_ex_apodization,
        readout_type=readout_type,
        ramp_sampling=ramp_sampling,
        partial_fourier_factor=partial_fourier_factor,
        gz_crusher_duration=gz_crusher_duration,
        b_values=b_values,
        g_diff_max_amplitude=system.max_grad * 0.8,
        g_diff_max_slew_rate=system.max_slew * 0.8,
        g_diff_delta_time=g_diff_delta_time,
        ge_segment_delay=0.0,
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
        seq.plot(time_range=(0, 10 * (tr or min_tr)), plot_now=True)

    return seq, output_path / filename


if __name__ == '__main__':
    main()
