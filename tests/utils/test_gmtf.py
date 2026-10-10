"""Tests for GMTF estimation and correction."""

import numpy as np
import pypulseq as pp
import pytest
from mrseq.utils.Gmtf import Gmtf
from mrseq.utils.Gmtf import build_input_triangles
from mrseq.utils.Gmtf import calc_kspace_from_grad_waveforms
from mrseq.utils.Gmtf import convert_waveforms_to_ppoly


def create_test_sequence() -> tuple[pp.Sequence, np.ndarray, float]:
    """
    Build a simple test sequence: RF excitation, a negative pre-winder followed by a readout gradient.

    Returns
    -------
    seq
        The assembled Pulseq sequence.
    trajectory
        Ground-truth k-space trajectory (1/m).
    dwell_time
        ADC dwell time
    """
    system = pp.Opts()
    seq = pp.Sequence(system=system)
    rf = pp.make_block_pulse(flip_angle=np.pi / 2, duration=1e-3, system=system)
    gx_pre = pp.make_trapezoid(channel='x', area=-300, system=system)
    gx = pp.make_trapezoid(channel='x', area=-2 * gx_pre.area, system=system)
    n_samples = 100
    adc = pp.make_adc(num_samples=n_samples, duration=gx.flat_time, delay=gx.rise_time, system=system)

    seq.add_block(rf)
    seq.add_block(pp.make_trapezoid(channel='x', area=0, duration=0.01, system=system))
    seq.add_block(gx_pre)
    seq.add_block(gx, adc)
    k_traj_adc = seq.calculate_kspace()[0]
    return seq, k_traj_adc, adc.dwell


def test_build_input_triangles_peak_amplitude():
    """Peak of each triangle should equal enumerate_coeff[0] * slew_rate * rise_time."""
    rise_time = 0.002
    slew_rate = 150.0
    coeff = 0.8
    triangles = build_input_triangles(
        [rise_time], slew_rate=slew_rate, g_delay=0.0, enumerate_coeff=[coeff], dwell_time=1e-5, n_samples=2000
    )
    expected_amplitude = coeff * slew_rate * rise_time
    assert np.isclose(triangles.max(), expected_amplitude, rtol=1e-6)


def test_recovers_original_values_at_breakpoints():
    """Evaluating the PPoly at the original time points should reproduce the original amplitudes."""
    time = np.array([0.0, 1.0, 2.0, 3.0])
    amplitude = np.array([0.0, 2.0, -1.0, 0.0])
    gw_data = [np.stack((time, amplitude))]
    result = convert_waveforms_to_ppoly(gw_data)
    assert result[0] is not None
    np.testing.assert_allclose(result[0](time), amplitude, atol=1e-10)


def test_linear_interpolation_between_breakpoints():
    """Midpoint between two breakpoints should be the linear average of their amplitudes."""
    time = np.array([0.0, 2.0])
    amplitude = np.array([0.0, 4.0])
    gw_data = [np.stack((time, amplitude))]
    result = convert_waveforms_to_ppoly(gw_data)
    assert result[0] is not None
    assert np.isclose(result[0](1.0), 2.0, atol=1e-10)


def test_error_message_identifies_correct_channel_for_nan():
    """Test correct channel for invalid input."""
    gw_data = [
        np.array([[0.0, 1.0], [0.0, 1.0]]),
        np.array([[0.0, 1.0], [0.0, np.nan]]),
    ]
    with pytest.raises(ValueError, match='channel 1'):
        convert_waveforms_to_ppoly(gw_data)


def test_error_message_identifies_correct_channel_for_inf():
    """Test correct channel for invalid input."""
    gw_data = [
        np.array([[0.0, 1.0], [0.0, 1.0]]),
        np.array([[0.0, 1.0], [0.0, 1.0]]),
        np.array([[0.0, 1.0], [0.0, np.inf]]),
    ]
    with pytest.raises(ValueError, match='channel 2'):
        convert_waveforms_to_ppoly(gw_data)


def test_convert_waveforms_to_ppoly():
    """Test correct transformation to ppoly."""
    seq, _, _ = create_test_sequence()
    ppoly_gradients_pypulseq = seq.get_gradients()
    waveform_gradients = seq.waveforms()
    ppoly_gradients = convert_waveforms_to_ppoly(waveform_gradients)

    eval_points = np.linspace(0, 1, 100)
    assert ppoly_gradients[1] == ppoly_gradients_pypulseq[1]  # None
    assert ppoly_gradients[2] == ppoly_gradients_pypulseq[2]  # None
    assert ppoly_gradients[0] is not None
    np.testing.assert_allclose(ppoly_gradients[0](eval_points), ppoly_gradients_pypulseq[0](eval_points))


def test_calc_kspace_from_grad_waveforms():
    """Correct k-space trajectory from gradient."""
    seq, _, _ = create_test_sequence()
    ppoly_gradients_pypulseq = seq.get_gradients()
    k_traj_adc_pp, k_traj_pp, _, _, _ = seq.calculate_kspace()
    k_traj_adc, k_traj, _, _, _ = calc_kspace_from_grad_waveforms(ppoly_gradients_pypulseq, seq)
    np.testing.assert_allclose(k_traj_adc, k_traj_adc_pp)
    np.testing.assert_allclose(k_traj, k_traj_pp)


def test_gmtf_construction():
    """Test GMTF constructor."""
    n = 50
    gmtf_x = np.random.rand(n) + 1j * np.random.rand(n)
    gmtf_y = np.random.rand(n) + 1j * np.random.rand(n)
    gmtf_z = np.random.rand(n) + 1j * np.random.rand(n)
    frequency = np.linspace(-1000, 1000, n)
    gmtf = Gmtf(gmtf_x=gmtf_x, gmtf_y=gmtf_y, gmtf_z=gmtf_z, frequency=frequency)
    assert gmtf.gmtf.shape == (3, n)
    np.testing.assert_array_equal(gmtf.gmtf[0], gmtf_x)
    np.testing.assert_array_equal(gmtf.gmtf[1], gmtf_y)
    np.testing.assert_array_equal(gmtf.gmtf[2], gmtf_z)
    assert gmtf.grad_output is None
    assert gmtf.grad_input is None


@pytest.mark.parametrize('scaling', (1, 0.8, 0.5))
@pytest.mark.parametrize('shift', (0, -2, 3))
def test_gmtf_trajectory_scaling(scaling, shift):
    """GMTF with different scaling and linear shifts."""
    n = 500
    seq, ktraj_orig, dwell_time = create_test_sequence()
    frequency = np.linspace(-1e5, 1e5, n)
    linear_phase = np.exp(-2j * np.pi * frequency * -shift * dwell_time)
    gmtf = Gmtf(gmtf_x=np.ones(n) * scaling * linear_phase, gmtf_y=np.ones(n), gmtf_z=np.ones(n), frequency=frequency)
    ktraj_gmtf = gmtf.correct_gradients(seq)
    delta_k = np.diff(ktraj_orig)[0, 0]  # shift is in units of delta k
    np.testing.assert_allclose(ktraj_gmtf[0], (ktraj_orig[0] + shift * delta_k) * scaling, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(ktraj_gmtf[1], ktraj_orig[1], rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(ktraj_gmtf[2], ktraj_orig[2], rtol=1e-3, atol=1e-2)
