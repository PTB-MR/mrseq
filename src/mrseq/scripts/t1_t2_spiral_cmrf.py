"""Cardiac MR Fingerprinting sequence with spiral readout."""

from pathlib import Path

import ismrmrd
import numpy as np
import pypulseq as pp

from mrseq.preparations import add_t1_inv_prep
from mrseq.preparations import add_t2_prep
from mrseq.utils import round_to_raster
from mrseq.utils import sys_defaults
from mrseq.utils.create_ismrmrd_header import create_header
from mrseq.utils.vds import variable_density_spiral_trajectory


def t1_t2_spiral_cmrf_kernel(
    system: pp.Opts,
    t2_prep_echo_times: np.ndarray,
    tr: float,
    cardiac_trigger_delay: float,
    fov_xy: float,
    variable_fov_coefficient: float,
    n_xy: int,
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
    cardiac_trigger_delay
        Delay after cardiac trigger (in seconds).
    fov_xy
        Field of view in x and y direction (in meters).
    variable_fov_coefficient
        Coefficient for variable density spiral trajectory.
    n_xy
        Number of voxel along x and y direction.
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
    # create PyPulseq Sequence object and set system limits
    seq = pp.Sequence(system=system)

    if len(t2_prep_echo_times) != 3:
        raise ValueError('t2_prep_echo_times must be an array of three echo times.')

    # cMRF specific settings
    n_blocks = 15  # number of heartbeat blocks
    n_unique_spirals = 48  # number of unique spiral interleaves
    minimum_time_to_set_label = 1e-5  # minimum time to set a label (in seconds)

    # define VDS readout parameters
    if fov_xy == 128e-3 and n_xy == 128:
        n_spirals_for_traj_calc = 16
    elif fov_xy == 300e-3 and n_xy == 192:
        n_spirals_for_traj_calc = 24
    else:
        print('Please double check the number of spirals for trajectory calculation.')
    # FOV decreases linearly from fov_coefficients[0] to fov_coefficients[0]-fov_coefficients[1].
    fov_coefficients = [fov_xy, variable_fov_coefficient * fov_xy]

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
    rf_dummy, gz_dummy, gzr_dummy = pp.make_sinc_pulse(  # type: ignore
        flip_angle=90 / 180 * np.pi,
        duration=rf_duration,
        slice_thickness=slice_thickness,
        apodization=rf_apodization,
        time_bw_product=rf_bwt,
        delay=system.rf_dead_time,
        system=system,
        return_gz=True,
        use='excitation',
    )

    # calculate variable density spiral (VDS) trajectory
    max_kspace_radius = 0.5 / fov_xy * n_xy
    k, g, s, timing, r, theta = variable_density_spiral_trajectory(
        system=system,
        sampling_period=system.grad_raster_time,
        n_interleaves=n_spirals_for_traj_calc,
        fov_coefficients=fov_coefficients,
        max_kspace_radius=max_kspace_radius,
    )

    # calculate angular increment
    delta_unique_spirals = 2 * np.pi / n_unique_spirals
    delta_array = np.arange(0, 2 * np.pi, delta_unique_spirals)

    # calculate ADC
    adc_dwell = system.grad_raster_time
    adc_total_samples = np.shape(g)[0] - 1
    assert adc_total_samples <= 8192, 'ADC samples exceed maximum value of 8192.'
    adc = pp.make_adc(num_samples=adc_total_samples, dwell=adc_dwell, delay=system.adc_dead_time, system=system)

    # Pre-calculate the n_unique_spirals gradient waveforms, k-space trajectories, and rewinders
    n_points_g = np.shape(g)[0]
    n_points_k = np.shape(k)[0]

    spiral_readout_grad = np.zeros((n_unique_spirals, 2, n_points_g))
    spiral_trajectory = np.zeros((n_unique_spirals, 2, n_points_k))
    gx_readout_list = []
    gy_readout_list = []
    gx_rewinder_list = []
    gy_rewinder_list = []
    max_rewinder_duration = 0

    # iterate over all unique spirals
    for n, delta in enumerate(delta_array):
        exp_delta = np.exp(1j * delta)
        exp_delta_pi = np.exp(1j * (delta + np.pi))

        spiral_readout_grad[n, 0, :] = np.real(g * exp_delta)
        spiral_readout_grad[n, 1, :] = np.imag(g * exp_delta)
        spiral_trajectory[n, 0, :] = np.real(k * exp_delta_pi)
        spiral_trajectory[n, 1, :] = np.imag(k * exp_delta_pi)

        gx_readout = pp.make_arbitrary_grad(
            channel='x',
            waveform=spiral_readout_grad[n, 0],
            first=0,
            delay=adc.delay,
            system=system,
        )

        gy_readout = pp.make_arbitrary_grad(
            channel='y',
            waveform=spiral_readout_grad[n, 1],
            first=0,
            delay=adc.delay,
            system=system,
        )

        gx_rewinder, _, _ = pp.make_extended_trapezoid_area(
            area=-gx_readout.area,
            channel='x',
            grad_start=gx_readout.last,
            grad_end=0,
            system=system,
        )

        gy_rewinder, _, _ = pp.make_extended_trapezoid_area(
            area=-gy_readout.area,
            channel='y',
            grad_start=gy_readout.last,
            grad_end=0,
            system=system,
        )

        gx_readout_list.append(gx_readout)
        gy_readout_list.append(gy_readout)
        gx_rewinder_list.append(gx_rewinder)
        gy_rewinder_list.append(gy_rewinder)

        # update maximum rewinder duration
        max_rewinder_duration = max(max_rewinder_duration, pp.calc_duration(gx_rewinder, gy_rewinder))

    # gradient spoiling
    gz_spoil_area = 4 / slice_thickness - gz_dummy.area / 2
    gz_spoil = pp.make_trapezoid(channel='z', area=gz_spoil_area, system=system)

    # update maximum rewinder duration including spoiling gradient
    max_rewinder_duration = max(max_rewinder_duration, pp.calc_duration(gz_spoil))

    # calculate minimum echo time (TE) for sequence header
    min_te = pp.calc_duration(gz_dummy) / 2 + pp.calc_duration(gzr_dummy) + adc.delay
    min_te = round_to_raster(min_te, system.grad_raster_time)

    # calculate minimum repetition time (TR)
    min_tr = (
        pp.calc_duration(rf_dummy, gz_dummy)  # rf pulse
        + pp.calc_duration(gzr_dummy)  # slice selection re-phasing gradient
        + pp.calc_duration(gx_readout_list[0])  # readout
        + max_rewinder_duration  # max of rewinder gradients / gz_spoil durations
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
        raise ValueError(f'TR must be larger than {min_tr * 1000:.2f} ms. Current value is {tr * 1000:.2f} ms.')

    # print TE / TR values
    final_tr = min_tr if tr is None else (min_tr - minimum_time_to_set_label) + tr_delay
    print('\n Manual timing calculations:')
    print(f'\n shortest possible TR = {min_tr * 1000:.2f} ms')
    print(f'\n final TR = {final_tr * 1000:.2f} ms')

    # create header
    if mrd_header_file:
        hdr = create_header(
            traj_type='radial',
            fov=fov_xy,
            res=fov_xy / n_xy,
            slice_thickness=slice_thickness,
            dt=adc.dwell,
            n_k1=len(flip_angles),
        )

        # write header to file
        prot = ismrmrd.Dataset(mrd_header_file, 'w')
        prot.write_xml_header(hdr.toXML('utf-8'))

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
            current_trig_delay = cardiac_trigger_delay - prep_dur

            # add trigger
            seq.add_block(pp.make_trigger(channel='physio1', duration=current_trig_delay))

            # add all events of T1prep block
            for idx in t1prep_block.block_events:
                seq.add_block(t1prep_block.get_block(idx))

        # add no preparation for every block following an inversion block
        elif block % 5 == 1:
            # add trigger with chosen trigger delay
            seq.add_block(pp.make_trigger(channel='physio1', duration=cardiac_trigger_delay))

        # add T2prep for every other block
        else:
            # get echo time for current block
            echo_time = t2_prep_echo_times[block % 5 - 2]

            # get prep block duration and calculate corresponding trigger delay
            t2prep_block, prep_dur = add_t2_prep(echo_time=echo_time, system=system)
            current_trig_delay = cardiac_trigger_delay - prep_dur

            # add trigger
            seq.add_block(pp.make_trigger(channel='physio1', duration=current_trig_delay))

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
            idx = np.argmin(diff)

            # create slice selective rf pulse for current shot
            rf_n, gz_n, gzr_n = pp.make_sinc_pulse(  # type: ignore
                flip_angle=fa / 180 * np.pi,
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
            seq.add_block(gx_readout_list[idx], gy_readout_list[idx], adc)

            # add rewinder gradients and spoiler
            gx_rewinder = gx_rewinder_list[idx]
            gy_rewinder = gy_rewinder_list[idx]
            seq.add_block(gx_rewinder, gy_rewinder, gz_spoil)

            # calculate rewinder delay for current shot
            current_rewinder_duration = max(pp.calc_duration(gx_rewinder), pp.calc_duration(gy_rewinder))
            rewinder_delay = max_rewinder_duration - current_rewinder_duration

            # add TR delay and LIN label
            seq.add_block(pp.make_delay(rewinder_delay + tr_delay), pp.make_label(label='LIN', type='INC', value=1))

            if mrd_header_file:
                # add trajectory to ISMRMRD header
                acq = ismrmrd.Acquisition()
                acq.resize(trajectory_dimensions=2, number_of_samples=adc.num_samples)
                traj_ismrmrd = np.stack(
                    [spiral_trajectory[idx, 0, 0:-1] * fov_xy, spiral_trajectory[idx, 1, 0:-1] * fov_xy]
                ).T
                acq.traj[:] = traj_ismrmrd
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
    cardiac_trigger_delay: float = 0.4,
    fov_xy: float = 128e-3,
    variable_fov_coefficient: float = -0.75,
    n_xy: int = 128,
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
    cardiac_trigger_delay
        Delay after cardiac trigger (in seconds).
    fov_xy
        Field of view in x and y direction (in meters).
    variable_fov_coefficient
        Coefficient for variable density spiral trajectory.
    n_xy
        Number of voxel along x and y direction.
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

    # define sequence filename
    filename = f'{Path(__file__).stem}_{fov_xy * 1000:.0f}fov_{n_xy}px_'
    filename += f'trig{int(cardiac_trigger_delay * 1000)}ms'

    output_path = Path.cwd() / 'output'
    output_path.mkdir(parents=True, exist_ok=True)

    # delete existing header file
    if (output_path / Path(filename + '.mrd')).exists():
        (output_path / Path(filename + '.mrd')).unlink()

    seq, inversion_time, te = t1_t2_spiral_cmrf_kernel(
        system=system,
        t2_prep_echo_times=t2_prep_echo_times,
        tr=tr,
        cardiac_trigger_delay=cardiac_trigger_delay,
        fov_xy=fov_xy,
        variable_fov_coefficient=variable_fov_coefficient,
        n_xy=n_xy,
        slice_thickness=slice_thickness,
        rf_inv_duration=rf_inv_duration,
        rf_inv_spoil_risetime=rf_inv_spoil_risetime,
        rf_inv_spoil_flattime=rf_inv_spoil_flattime,
        rf_duration=rf_duration,
        rf_bwt=rf_bwt,
        rf_apodization=rf_apodization,
        mrd_header_file=str(output_path / Path(filename + '.mrd')),
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
    seq.set_definition('number_of_readouts', int(n_xy))

    # save seq-file to disk
    output_path = Path.cwd() / 'output'
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving sequence file '{filename}.seq' into folder '{output_path}'.")
    seq.write(str(output_path / filename), create_signature=True)

    if show_plots:
        seq.plot()

    return seq


if __name__ == '__main__':
    main()
