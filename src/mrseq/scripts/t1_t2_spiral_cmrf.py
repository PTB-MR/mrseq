"""Cardiac MR Fingerprinting sequence with spiral readout."""

from pathlib import Path
from typing import Literal

import ismrmrd
import numpy as np
import pypulseq as pp

from mrseq.preparations import add_t1_inv_prep
from mrseq.preparations import add_t2_prep
from mrseq.utils import round_to_raster
from mrseq.utils import spiral_acquisition
from mrseq.utils import sys_defaults
from mrseq.utils.ismrmrd import Fov
from mrseq.utils.ismrmrd import Limits
from mrseq.utils.ismrmrd import MatrixSize
from mrseq.utils.ismrmrd import create_header


def t1_t2_spiral_cmrf_kernel(
    system: pp.Opts,
    t2_prep_echo_times: np.ndarray,
    tr: float | None,
    min_cardiac_trigger_delay: float,
    fov_xy: float,
    n_readout: int,
    readout_oversampling: Literal[1, 2, 4],
    spiral_undersampling: int,
    slice_thickness: float,
    rf_inv_duration: float,
    rf_inv_spoil_risetime: float,
    rf_inv_spoil_flattime: float,
    rf_duration: float,
    rf_bwt: float,
    rf_apodization: float,
    mrd_header_file: str | None,
) -> tuple[pp.Sequence, float, float]:
    """Generate a cardiac MR Fingerprinting sequence with spiral readout.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    t2_prep_echo_times
        Array of three T2prep echo times (in seconds).
    tr
        Desired repetition time (TR) (in seconds).
    min_cardiac_trigger_delay
        Minimum delay after cardiac trigger (in seconds).
        The total trigger delay is implemented as a soft delay and can be chosen by the user in the UI.
    fov_xy
        Field of view in x and y direction (in meters).
    n_readout
        Number of readout points.
    readout_oversampling
        Readout oversampling. Determines the number of ADC samples along a spiral and the bandwidth.
    spiral_undersampling
        Undersampling in the periphery of the variable density spiral.
    slice_thickness
        Slice thickness of the 2D slice (in meters).
    rf_inv_duration
        Duration of adiabatic inversion pulse (in seconds)
    rf_inv_spoil_risetime
        Rise time of spoiler after inversion pulse (in seconds)
    rf_inv_spoil_flattime
        Flat time of spoiler after inversion pulse (in seconds)
    rf_duration
        Duration of the rf excitation pulse (in seconds)
    rf_bwt
        Bandwidth-time product of rf excitation pulse (Hz * seconds)
    rf_apodization
        Apodization factor of rf excitation pulse
    mrd_header_file
        Filename of the ISMRMRD header file. If None, no header file is created.

    Returns
    -------
    seq
        PyPulseq Sequence object
    time_to_first_tr_block
        End point of first TR block.
    min_te
        Shortest possible echo time.

    """
    if readout_oversampling < 1:
        raise ValueError('Readout oversampling factor must be >= 1.')

    # create PyPulseq Sequence object and set system limits
    seq = pp.Sequence(system=system)

    if len(t2_prep_echo_times) != 3:
        raise ValueError('t2_prep_echo_times must be an array of three echo times.')

    # cMRF specific settings
    n_blocks = 15  # number of heartbeat blocks
    minimum_time_to_set_label = 1e-5  # minimum time to set a label (in seconds)

    # create flip angle pattern
    max_flip_angles_deg = [12.5, 18.75, 25, 25, 25, 12.5, 18.75, 25, 25.0, 25, 12.5, 18.75, 25, 25, 25]
    flip_angles = np.deg2rad(
        np.concatenate(
            [
                np.concatenate((np.linspace(4, max_angle, 16), np.full((31,), max_angle)))
                for max_angle in max_flip_angles_deg
            ]
        )
    )

    # make sure the number of blocks fits the total number of flip angles / repetitions
    if not flip_angles.size % n_blocks == 0:
        raise ValueError('Number of repetitions must be a multiple of the number of blocks.')

    # calculate number of shots / repetitions per block
    n_shots_per_block = flip_angles.size // n_blocks

    # create rf dummy pulse (required for some timing calculations)
    rf_dummy, gz_dummy, gzr_dummy = pp.make_sinc_pulse(
        flip_angle=np.pi,
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
    gx, gy, adc, trajectory, time_to_echo = spiral_acquisition(
        system,
        n_readout,
        fov_xy,
        spiral_undersampling,
        readout_oversampling,
        n_spirals=None,
        max_pre_duration=0.0,
        spiral_type='out',
    )
    delta_array = 2 * np.pi / len(gx) * np.arange(len(gx))  # angle difference between subsequent spirals

    # create gradient spoiler
    gz_spoil_area = 4 / slice_thickness - gz_dummy.area / 2
    gz_spoil = pp.make_trapezoid(channel='z', area=gz_spoil_area, system=system)

    # calculate minimum echo time (TE) for sequence header
    min_te = pp.calc_duration(gz_dummy) / 2 + pp.calc_duration(gzr_dummy) + time_to_echo
    min_te = round_to_raster(min_te, system.grad_raster_time)

    # calculate minimum repetition time (TR)
    min_tr = (
        pp.calc_duration(rf_dummy, gz_dummy)  # rf pulse
        + pp.calc_duration(gzr_dummy)  # slice selection re-phasing gradient
        + pp.calc_duration(gx[0])  # readout
        + pp.calc_duration(gz_spoil)  # gz_spoil durations
        + minimum_time_to_set_label  # min time to set labels
    )

    # ensure minimum TR is on gradient raster
    min_tr = round_to_raster(min_tr, system.grad_raster_time)

    # calculate TR delay
    if tr is None:
        tr_delay = minimum_time_to_set_label
    else:
        tr_delay = round_to_raster((tr - min_tr + minimum_time_to_set_label), system.grad_raster_time)
        if not tr_delay >= 0:
            raise ValueError(f'TR must be larger than {min_tr * 1000:.3f} ms. Current value is {tr * 1000:.3f} ms.')

    # print TE / TR values
    final_tr = min_tr if tr is None else (min_tr - minimum_time_to_set_label) + tr_delay
    print('\n Manual timing calculations:')
    print(f'\n shortest possible TR = {min_tr * 1000:.3f} ms')
    print(f'\n final TR = {final_tr * 1000:.3f} ms')

    # create header
    if mrd_header_file:
        hdr = create_header(
            traj_type='other',
            encoding_fov=Fov(x=fov_xy, y=fov_xy, z=slice_thickness),
            recon_fov=Fov(x=fov_xy, y=fov_xy, z=slice_thickness),
            encoding_matrix=MatrixSize(n_x=int(n_readout), n_y=int(n_readout), n_z=1),
            recon_matrix=MatrixSize(n_x=n_readout, n_y=n_readout, n_z=1),
            dwell_time=adc.dwell,
            slice_limits=Limits(min=0, max=1, center=0),
            k1_limits=Limits(min=0, max=len(gx), center=0),
            k2_limits=Limits(min=0, max=1, center=0),
        )

        # write header to file
        prot = ismrmrd.Dataset(mrd_header_file, 'w')
        prot.write_xml_header(hdr.toXML('utf-8'))

    # create trigger soft delay (total duration: user_input/1.0 - min_cardiac_trigger_delay)
    trig_soft_delay = pp.make_soft_delay(
        hint='trig_delay',
        offset=-min_cardiac_trigger_delay,
        factor=1.0,
        default_duration=0.5 - min_cardiac_trigger_delay,
    )

    # obtain noise samples
    seq.add_block(pp.make_label(label='LIN', type='SET', value=0), pp.make_label(label='SLC', type='SET', value=0))
    seq.add_block(adc, pp.make_label(label='NOISE', type='SET', value=True))
    seq.add_block(pp.make_label(label='NOISE', type='SET', value=False))
    seq.add_block(pp.make_delay(system.rf_dead_time))

    # add noise acquisition to ISMRMRD file
    if mrd_header_file:
        acq = ismrmrd.Acquisition()
        acq.resize(trajectory_dimensions=2, number_of_samples=adc.num_samples)
        prot.append_acquisition(acq)

    # initialize LIN label
    seq.add_block(pp.make_delay(minimum_time_to_set_label), pp.make_label(label='LIN', type='SET', value=0))

    # initialize repetition counter
    rep_counter = 0

    # loop over all blocks
    for block in range(n_blocks):
        # add inversion pulse for every fifth block
        if block % 5 == 0:
            # get prep block duration and calculate corresponding trigger delay
            t1prep_block, prep_dur, time_since_inversion = add_t1_inv_prep(
                rf_duration=rf_inv_duration,
                spoiler_ramp_time=rf_inv_spoil_risetime,
                spoiler_flat_time=rf_inv_spoil_flattime,
                system=system,
            )
            constant_trig_delay = min_cardiac_trigger_delay - prep_dur

            # add trigger and constant part of trigger delay
            seq.add_block(pp.make_trigger(channel='physio1', duration=constant_trig_delay))

            # add variable part of trigger delay (soft delay)
            seq.add_block(trig_soft_delay)

            # add all events of T1prep block
            for idx in t1prep_block.block_events:
                seq.add_block(t1prep_block.get_block(idx))

        # add no preparation for every block following an inversion block
        elif block % 5 == 1:
            # add trigger and trigger delay(s)
            seq.add_block(pp.make_trigger(channel='physio1', duration=min_cardiac_trigger_delay))
            seq.add_block(trig_soft_delay)

        # add T2prep for every other block
        else:
            # get echo time for current block
            echo_time = t2_prep_echo_times[block % 5 - 2]

            # get prep block duration and calculate corresponding trigger delay
            t2prep_block, prep_dur = add_t2_prep(echo_time=echo_time, system=system)
            constant_trig_delay = min_cardiac_trigger_delay - prep_dur

            # add trigger and constant part of trigger delay
            seq.add_block(pp.make_trigger(channel='physio1', duration=constant_trig_delay))

            # add variable part of trigger delay (soft delay)
            seq.add_block(trig_soft_delay)

            # add all events of T2prep block
            for idx in t2prep_block.block_events:
                seq.add_block(t2prep_block.get_block(idx))

        # loop over shots / repetitions per block
        for _ in range(n_shots_per_block):
            # get current flip angle
            fa = flip_angles[rep_counter]

            # calculate theoretical golden angle rotation for current shot
            golden_angle = (rep_counter * 2 * np.pi * (1 - 2 / (1 + np.sqrt(5)))) % (2 * np.pi)

            # find closest unique spiral to current golden angle rotation
            diff = np.abs(delta_array - golden_angle)
            spiral_idx = np.argmin(diff)

            # create slice selective rf pulse for current shot
            rf_n, gz_n, gzr_n = pp.make_sinc_pulse(
                flip_angle=fa,
                duration=rf_duration,
                slice_thickness=slice_thickness,
                apodization=rf_apodization,
                time_bw_product=rf_bwt,
                delay=system.rf_dead_time,
                system=system,
                return_gz=True,
                use='excitation',
            )

            # add slice selective excitation pulse
            seq.add_block(rf_n, gz_n)

            # add slice selection re-phasing gradient
            seq.add_block(gzr_n)

            # add readout gradients and ADC
            seq.add_block(gx[spiral_idx], gy[spiral_idx], adc)

            # add spoiler
            seq.add_block(gz_spoil)

            # add TR delay and LIN label
            seq.add_block(pp.make_delay(tr_delay), pp.make_label(label='LIN', type='INC', value=1))

            if mrd_header_file:
                # add acquisitions to metadata
                spiral_trajectory = np.zeros((trajectory.shape[1], 2), dtype=np.float32)

                # the spiral trajectory is calculated in units of delta_k. for image reconstruction we use delta_k = 1
                spiral_trajectory[:, 0] = trajectory[spiral_idx, :, 0] * fov_xy
                spiral_trajectory[:, 1] = trajectory[spiral_idx, :, 1] * fov_xy

                acq = ismrmrd.Acquisition()
                acq.resize(trajectory_dimensions=2, number_of_samples=adc.num_samples)
                acq.traj[:] = spiral_trajectory
                prot.append_acquisition(acq)

            # increment repetition counter
            rep_counter += 1

    # close ISMRMRD header file
    if mrd_header_file:
        prot.close()

    return seq, time_since_inversion, min_te


def main(
    system: pp.Opts | None = None,
    t2_prep_echo_times: np.ndarray | None = None,
    tr: float = 10e-3,
    fov_xy: float = 128e-3,
    spiral_undersampling: int = 4,
    n_readout: int = 128,
    slice_thickness: float = 8e-3,
    show_plots: bool = True,
    test_report: bool = True,
    timing_check: bool = True,
) -> pp.Sequence:
    """Generate a cardiac MR Fingerprinting sequence with spiral readout.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    t2_prep_echo_times
        Array of three T2prep echo times (in seconds). Default: [0.03, 0.05, 0.1] s if set to None.
    tr
        Desired repetition time (TR) (in seconds).
    min_cardiac_trigger_delay
        Minimum delay after cardiac trigger (in seconds).
    fov_xy
        Field of view in x and y direction (in meters).
    spiral_undersampling
        Undersampling factor in the periphery of the variable density spiral.
    n_readout
        Number of readout points.
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
        t2_prep_echo_times = np.array([0.03, 0.05, 0.1])  # [s]

    # define T1prep settings
    rf_inv_duration = 10.24e-3  # duration of adiabatic inversion pulse [s]
    rf_inv_spoil_risetime = 0.6e-3  # rise time of spoiler after inversion pulse [s]
    rf_inv_spoil_flattime = 8.4e-3  # flat time of spoiler after inversion pulse [s]

    # define settings of rf excitation pulse
    rf_duration = 0.8e-3  # duration of the rf excitation pulse [s]
    rf_bwt = 8  # bandwidth-time product of rf excitation pulse [Hz*s]
    rf_apodization = 0.5  # apodization factor of rf excitation pulse
    readout_oversampling: Literal[1, 2, 4] = 2

    # define sequence filename
    filename = f'{Path(__file__).stem}_{fov_xy * 1000:.0f}fov_{n_readout}px_variable_trig_delay'

    output_path = Path.cwd() / 'output'
    output_path.mkdir(parents=True, exist_ok=True)

    # delete existing header file
    if (output_path / Path(filename + '_header.h5')).exists():
        (output_path / Path(filename + '_header.h5')).unlink()

    seq, inversion_time, te = t1_t2_spiral_cmrf_kernel(
        system=system,
        t2_prep_echo_times=t2_prep_echo_times,
        tr=tr,
        min_cardiac_trigger_delay=np.max(t2_prep_echo_times) + 0.05,  # max T2prep echo time and buffer for spoiler
        fov_xy=fov_xy,
        n_readout=n_readout,
        readout_oversampling=readout_oversampling,
        spiral_undersampling=spiral_undersampling,
        slice_thickness=slice_thickness,
        rf_inv_duration=rf_inv_duration,
        rf_inv_spoil_risetime=rf_inv_spoil_risetime,
        rf_inv_spoil_flattime=rf_inv_spoil_flattime,
        rf_duration=rf_duration,
        rf_bwt=rf_bwt,
        rf_apodization=rf_apodization,
        mrd_header_file=str(output_path / Path(filename + '_header.h5')),
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

    # write all important parameters into the seq-file definitions
    seq.set_definition('Name', 'cMRF_spiral')
    seq.set_definition('FOV', [fov_xy, fov_xy, slice_thickness])
    seq.set_definition('TE', te)
    seq.set_definition('TI', inversion_time)
    seq.set_definition('TR', tr)
    seq.set_definition('t2prep_te', [0, 0, t2_prep_echo_times[0], t2_prep_echo_times[1], t2_prep_echo_times[2]])
    seq.set_definition('t1prep_ti', [inversion_time, 0, 0, 0, 0])
    seq.set_definition('slice_thickness', slice_thickness)
    seq.set_definition('sampling_scheme', 'spiral')
    seq.set_definition('number_of_readouts', int(n_readout))

    # save seq-file to disk
    output_path = Path.cwd() / 'output'
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving sequence file '{filename}.seq' into folder '{output_path}'.")
    seq.write(str(output_path / filename), create_signature=True)

    if show_plots:
        seq.plot()

    return seq, output_path / filename


if __name__ == '__main__':
    main()
