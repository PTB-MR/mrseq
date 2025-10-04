"""Tests for sequence helper functions."""

import numpy as np
import pypulseq as pp
import pytest
from mrseq.utils.trajectory import MultiEchoAcquisition
from mrseq.utils.trajectory import cartesian_phase_encoding


@pytest.mark.parametrize('n_phase_encoding', [50, 51, 100])
@pytest.mark.parametrize('acceleration', [1, 2, 3, 4, 6])
@pytest.mark.parametrize('n_fully_sampled_center', [0, 8, 9])
def test_cartesian_phase_encoding_identical_points(
    n_phase_encoding: int, acceleration: int, n_fully_sampled_center: int
):
    """Test that linear, low-high and high-low cover same phase encoding points."""
    pe_linear, pe_center_linear = cartesian_phase_encoding(
        n_phase_encoding, acceleration, n_fully_sampled_center, sampling_order='linear'
    )
    pe_low_high, pe_center_low_high = cartesian_phase_encoding(
        n_phase_encoding, acceleration, n_fully_sampled_center, sampling_order='low_high'
    )
    pe_high_low, pe_center_high_low = cartesian_phase_encoding(
        n_phase_encoding, acceleration, n_fully_sampled_center, sampling_order='high_low'
    )

    np.testing.assert_allclose(pe_linear, np.sort(pe_low_high))
    np.testing.assert_allclose(pe_linear, np.sort(pe_high_low))
    np.testing.assert_allclose(pe_center_linear, np.sort(pe_center_low_high))
    np.testing.assert_allclose(pe_center_linear, np.sort(pe_center_high_low))


@pytest.mark.parametrize('pattern', ['linear', 'low_high', 'high_low', 'random'])
def test_cartesian_phase_encoding_acceleration(pattern: str):
    """Test correct undersampling factor."""
    n_pe_full = 100
    acceleration = 4

    pe, _ = cartesian_phase_encoding(n_phase_encoding=n_pe_full, acceleration=acceleration, sampling_order=pattern)
    assert len(pe) == n_pe_full // acceleration


@pytest.mark.parametrize('pattern', ['linear', 'low_high', 'high_low', 'random'])
@pytest.mark.parametrize('n_phase_encoding_per_shot', [3, 8, 11, 13])
def test_cartesian_phase_encoding_integer_shots(pattern: str, n_phase_encoding_per_shot: int):
    """Test that the total number of phase encoding points lead to an integer number."""
    n_pe_full = 100
    acceleration = 4

    pe, _ = cartesian_phase_encoding(
        n_phase_encoding=n_pe_full,
        acceleration=acceleration,
        sampling_order=pattern,
        n_phase_encoding_per_shot=n_phase_encoding_per_shot,
    )
    assert np.mod(len(pe), n_phase_encoding_per_shot) == 0


def test_multi_gradient_echo(system_defaults):
    """Test multi-echo gradient echo readout as part of a simple sequence."""
    seq = pp.Sequence(system=system_defaults)
    rf = pp.make_block_pulse(
        flip_angle=np.pi,
        delay=system_defaults.rf_dead_time,
        duration=2e-3,
        phase_offset=0.0,
        system=system_defaults,
    )
    seq.add_block(rf)
    mecho = MultiEchoAcquisition(system=seq.system)
    seq, _ = mecho.add_to_seq(seq, n_echoes=3)
    ok, error_report = seq.check_timing()
    if not ok:
        print('\nTiming check failed! Error listing follows\n')
        print(error_report)
    assert ok


@pytest.mark.parametrize('delta_te', [3e-3, 5.34e-3])
def test_multi_gradient_echo_set_delta_te(delta_te, system_defaults):
    """Test pre-defined delta te."""
    seq = pp.Sequence(system=system_defaults)
    mecho = MultiEchoAcquisition(system=seq.system, delta_te=delta_te)
    seq, time_to_echoes = mecho.add_to_seq(seq, 6)
    np.testing.assert_allclose(np.diff(time_to_echoes), delta_te)


def test_multi_gradient_echo_error_on_short_delta_te(system_defaults):
    """Test if error is raised on too short delta echo time."""
    with pytest.raises(ValueError):
        MultiEchoAcquisition(system=system_defaults, delta_te=1e-6)


@pytest.mark.parametrize('n_echoes', [1, 3, 8])
@pytest.mark.parametrize('readout_oversampling', [1, 1.5, 2])
@pytest.mark.parametrize('n_readout', [64, 128, 200])
@pytest.mark.parametrize('partial_echo_factor', [1.0, 0.8, 0.7])
def test_multi_gradient_echo_timing(n_echoes, readout_oversampling, n_readout, partial_echo_factor, system_defaults):
    """Test that zero crossing of gradient moment coincides with echo time and correct adc sample."""
    seq = pp.Sequence(system=system_defaults)
    mecho = MultiEchoAcquisition(
        system=seq.system,
        n_readout=n_readout,
        readout_oversampling=readout_oversampling,
        partial_echo_factor=partial_echo_factor,
    )
    seq, time_to_echoes = mecho.add_to_seq(seq, n_echoes)

    from scipy.signal import argrelextrema

    # Get full waveform for readout gradient
    w = seq.waveforms_and_times()
    gx_waveform = w[0][0][1]
    gx_waveform_time = w[0][0][0]

    # Find k0-crossings
    max_grad = np.max(np.abs(gx_waveform))
    dt = np.arange(gx_waveform_time[0], gx_waveform_time[-1], step=mecho._adc.dwell / 10)
    gx_waveform_intp = np.interp(dt, gx_waveform_time, gx_waveform / max_grad)
    m0_intp = np.cumsum(gx_waveform_intp) / len(gx_waveform_intp)
    k0_idx = argrelextrema(np.abs(m0_intp), np.less, order=100)[0]

    # Remove k0-crossings at the beginning and end of the block
    k0_idx = [ki for ki in k0_idx if (ki > 100 and ki < len(dt) - 100)]

    # Zero-crossing of the readout gradient should be within +/- .5 adc dwell time of the k-space center sample which is
    # the (_n_readout_pre_echo + 1)th sample. This should also be the same as the time_to_echo - time
    assert n_echoes == len(k0_idx)
    current_time = pp.calc_duration(mecho._gx_pre)
    for echo in range(n_echoes):
        time_of_k0_adc_sample = (
            current_time + mecho._adc.delay + mecho._n_readout_pre_echo * mecho._adc.dwell + mecho._adc.dwell / 2
        )
        print(dt[k0_idx[echo]], time_of_k0_adc_sample, mecho._adc.dwell)
        assert np.isclose(dt[k0_idx[echo]], time_of_k0_adc_sample, atol=mecho._adc.dwell / 2)
        assert np.isclose(dt[k0_idx[echo]], time_to_echoes[echo], atol=mecho._adc.dwell / 2)

        current_time += pp.calc_duration(mecho._gx) + pp.calc_duration(mecho._gx_between)

    assert seq is not None
