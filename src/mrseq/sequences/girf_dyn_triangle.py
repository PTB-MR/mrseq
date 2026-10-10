"""GIRF sequence creator with triangular gradients."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pypulseq as pp

from mrseq.utils import round_to_raster
from mrseq.utils import sys_defaults
from mrseq.utils import write_sequence


def girf_triangle_kernel(
    system: pp.Opts,
    rf_flip_angle: float,
    rf_duration: float,
    rf_bwt: float,
    apodization: float,
    n_avg: int,
    adc_duration: float,
    dwell_time: float,
    slice_thickness: float,
    slice_pos: Sequence[float],
    g_amplitude_coeff: Sequence[float],
    tr: float,
    eddy_current_delay: float,
    rise_times: Sequence[float],
    gradient_pre_emphasis_delay: float,
) -> pp.Sequence:
    """Generate a GIRF sequence with triangular gradients.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    rf_flip_angle
        Flip angle of RF excitation pulse (in degrees).
    rf_duration
        Duration of RF excitation pulse (in seconds).
    rf_bwt
        Bandwidth-time product of RF excitation pulse (Hz * seconds).
    apodization
        Apodization factor of RF excitation pulse.
    n_avg
        Number of averages.
    adc_duration
        Duration of ADC acquisition (in seconds).
    dwell_time
        ADC dwell time (in seconds).
    slice_thickness
        Slice thickness (in meters).
    slice_pos
        List of slice positions (in meters).
    g_amplitude_coeff
        List of amplitude coefficients for gradient encoding.
    tr
        Repetition time (in seconds).
    eddy_current_delay
        Delay between slice refocusing gradient and ADC to minimize eddy current effects (in seconds).
    rise_times
        List of gradient rise times (in seconds).
    gradient_pre_emphasis_delay
        The gradient is delayed by this value relative to the ADC to ensure the full gradient signal is caught, even
        if the gradient starts earlier than the nominal gradient due to the pre-emphasis of the gradient signal.

    Returns
    -------
    seq
        PyPulseq Sequence object.
    """
    if n_avg < 0:
        raise ValueError('Number of averages must be >= 0.')

    if not rise_times:
        raise ValueError('Rise times list cannot be empty.')

    # Create PyPulseq Sequence object
    seq = pp.Sequence(system=system)

    # Create and set sinc pulse parameters
    rf, gsl, gsl_r = pp.make_sinc_pulse(
        flip_angle=rf_flip_angle / 180 * np.pi,
        delay=system.rf_dead_time,
        duration=rf_duration,
        slice_thickness=slice_thickness,
        apodization=apodization,
        time_bw_product=rf_bwt,
        system=system,
        return_gz=True,
    )

    # Verify triangular gradients are always shorter than readout. We use 2xgradient_pre_emphasis_delay to also capture
    # delayed effects at the end of the gradient due to the pre-emphasis.
    if adc_duration < (max(rise_times) * 2 + system.adc_dead_time + 2 * gradient_pre_emphasis_delay):
        raise ValueError('ADC too short for current rise times and system settings.')

    # Calculate ADC parameters
    n_readout = int(np.round(adc_duration / dwell_time))
    adc = pp.make_adc(
        num_samples=n_readout,
        duration=adc_duration,
        system=system,
        delay=system.adc_dead_time,
    )

    # Build sequence with nested loops
    for avg in range(n_avg):
        avg_label = pp.make_label(type='SET', label='AVG', value=avg)

        for idx_rise_time, rise_time_val in enumerate(rise_times):
            seg_label = pp.make_label(type='SET', label='SEG', value=idx_rise_time)
            rise_time_val = np.round(rise_time_val, decimals=6)

            for idx_grad, grad_channel in enumerate(['x', 'y', 'z']):
                grad_label = pp.make_label(type='SET', label='PHS', value=idx_grad)

                for idx_slice_pos, slice_pos_val in enumerate(slice_pos):
                    slice_label = pp.make_label(type='SET', label='SLC', value=idx_slice_pos)

                    for idx_amp_fac, amp_fac in enumerate(g_amplitude_coeff):
                        amp_fac_label = pp.make_label(type='SET', label='REP', value=idx_amp_fac)

                        # Calculate amplitude from coefficient, slew rate, and rise time
                        amp = amp_fac * system.max_slew * rise_time_val

                        # Set RF frequency offset for current slice
                        rf.freq_offset = gsl.amplitude * slice_pos_val
                        gsl.channel = grad_channel
                        gsl_r.channel = grad_channel

                        # Add RF pulse with slice selection
                        seq.add_block(rf, gsl, avg_label, seg_label, grad_label, slice_label, amp_fac_label)
                        seq.add_block(gsl_r)

                        # Add eddy current compensation delay
                        seq.add_block(pp.make_delay(eddy_current_delay))

                        # Create triangle gradient (trapezoid with no flat time)
                        g_triangle = pp.make_trapezoid(
                            channel=grad_channel,
                            system=system,
                            rise_time=rise_time_val,
                            flat_time=0,
                            amplitude=amp,
                            delay=round_to_raster(gradient_pre_emphasis_delay + adc.delay, system.grad_raster_time),
                        )
                        seq.add_block(adc, g_triangle)

                        # Add TR delay
                        seq.add_block(pp.make_delay(tr))

        seq.add_block(avg_label)

    # Set sequence definitions
    seq.set_definition('FOV', [0.5, 0.5, slice_thickness])
    seq.set_definition('ReconMatrix', (n_readout, 1, 1))
    seq.set_definition('SliceThickness', slice_thickness)
    seq.set_definition('SlicePos', slice_pos)
    seq.set_definition('TR', tr)
    seq.set_definition('RiseTimes', rise_times)
    seq.set_definition('AdcDuration', adc_duration)
    seq.set_definition('DwellTime', dwell_time)
    seq.set_definition('SlewRate', system.max_slew)
    seq.set_definition('GradientPreEmphasisDelay', gradient_pre_emphasis_delay)
    seq.set_definition('GradAmplitudeCoeff', g_amplitude_coeff)

    return seq


def main(
    system: pp.Opts | None = None,
    n_avg: int = 9,
    tr: float = 2.0,
    slice_thickness: float = 1.5e-3,
    slice_pos: list[float] | None = None,
    show_plots: bool = True,
    test_report: bool = True,
    timing_check: bool = True,
    v141_compatibility: bool = True,
) -> tuple[pp.Sequence, Path]:
    """Generate a sequence with triangular gradients to estimate the GIRF.

    This sequence allows for the calculation of a gradient impulse response function (GIRF) using the Dyn
    method [DUY1998]_ .


    .. [DUY1998] Dyn J, Yang Y, Frank JA, and van der Veen JW (1998), Simple Correction Method fork-Space Trajectory
       Deviations in MRI. JMR 132, 150-153. https://doi.org/10.1006/jmre.1998.1396

    Parameters
    ----------
    system
        PyPulseq system limits object. Uses default system if None.
    n_avg
        Number of averages.
    tr
        Repetition time (in seconds).
    slice_thickness
        Slice thickness (in meters).
    slice_pos
        List of slice positions (in meters).
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
        Sequence object of GIRF triangle sequence.
    file_path
        Path to the sequence file.

    """
    if system is None:
        system = sys_defaults

    if slice_pos is None:
        slice_pos = [0.04]

    # Define RF excitation pulse parameters
    rf_flip_angle = 90  # flip angle [degrees]
    rf_duration = 8.4e-3  # duration of the RF excitation pulse [s]
    rf_bwt = 4  # bandwidth-time product of RF excitation pulse [Hz*s]
    apodization = 0.5  # apodization factor of RF excitation pulse

    # Define ADC and gradient timing
    adc_duration = 0.06  # ADC acquisition duration [s]
    dwell_time = 10e-6  # ADC dwell time [s]

    # Define sequence parameters
    g_amplitude_coeff = [-1.0, 0.0]  # amplitude coefficients for gradient encoding
    eddy_current_delay = 10e-6  # eddy current compensation delay [s]
    rise_times = [5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 1e-4, 1.1e-4, 1.2e-4, 1.3e-4, 1.4e-4, 1.5e-4, 1.6e-4]  # rise times [s]
    gradient_pre_emphasis_delay = 50e-6

    # Define sequence filename
    filename = f'{Path(__file__).stem}'

    output_path = Path.cwd() / 'output'
    output_path.mkdir(parents=True, exist_ok=True)

    seq = girf_triangle_kernel(
        system=system,
        rf_flip_angle=rf_flip_angle,
        rf_duration=rf_duration,
        rf_bwt=rf_bwt,
        apodization=apodization,
        n_avg=n_avg,
        adc_duration=adc_duration,
        dwell_time=dwell_time,
        slice_thickness=slice_thickness,
        slice_pos=slice_pos,
        g_amplitude_coeff=g_amplitude_coeff,
        tr=tr,
        eddy_current_delay=eddy_current_delay,
        rise_times=rise_times,
        gradient_pre_emphasis_delay=gradient_pre_emphasis_delay,
    )

    # Check sequence timing
    if timing_check and not test_report:
        ok, error_report = seq.check_timing()
        if ok:
            print('\nTiming check passed successfully')
        else:
            print('\nTiming check failed! Error listing follows\n')
            print(error_report)

    # Show advanced test report
    if test_report:
        print('\nCreating advanced test report...')
        print(seq.test_report())

    # Save sequence file to disk
    print(f"\nSaving sequence file '{filename}.seq' into folder '{output_path}'.")
    write_sequence(seq, str(output_path / filename), create_signature=True, v141_compatibility=v141_compatibility)

    if show_plots:
        seq.plot(grad_disp='mT/m')

    return seq, output_path / filename


if __name__ == '__main__':
    main()
