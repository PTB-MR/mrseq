"""Tests for sequence helper functions."""

import numpy as np
import pypulseq as pp
import pytest
from mrseq.utils import spiral_acquisition
from mrseq.utils.trajectory import cartesian_phase_encoding
from mrseq.utils.trajectory import undersampled_variable_density_spiral


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


def get_interp_waveform_for_gx_gy(seq: pp.Sequence, dt: np.ndarray | None = None, scale: float = 1.0):
    """Interpolate gradient waveforms for the x and y axes.

    Parameters
    ----------
    seq
        The PyPulseq sequence object containing gradient waveforms.
    dt
        Desired time points for interpolation. If None, a default time array is generated.
    scale
        Scaling factor for the gradient waveforms. Default is 1.

    Returns
    -------
    gx_waveform_intp
        Interpolated gradient waveform for the x-axis.
    gy_waveform_intp
        Interpolated gradient waveform for the y-axis.
    dt
        Time points corresponding to the interpolated waveforms.
    """
    w = seq.waveforms_and_times()
    gx_waveform = w[0][0][1] * scale
    gx_waveform_time = w[0][0][0]

    gy_waveform = w[0][1][1] * scale
    gy_waveform_time = w[0][1][0]

    if dt is None:
        dt = np.arange(
            min(gx_waveform_time[0], gy_waveform_time[0]), max(gx_waveform_time[-1], gy_waveform_time[-1]), step=1e-7
        )
    gx_waveform_intp = np.interp(dt, gx_waveform_time, gx_waveform)
    gy_waveform_intp = np.interp(dt, gy_waveform_time, gy_waveform)

    return gx_waveform_intp, gy_waveform_intp, dt


@pytest.mark.parametrize('n_readout', (128, 256))
@pytest.mark.parametrize('fov', (128e-3, 320e-3))
@pytest.mark.parametrize('undersampling_factor', (1, 2, 3, 4))
def test_undersampled_variable_density_spiral(
    system_defaults: pp.Opts, n_readout: int, fov: float, undersampling_factor: float
):
    """Test spiral for different undersampling factors."""
    traj, _grad, _s, _timing, _r, _theta, n_spirals, _fov_scaling_center, _fov_scaling_edge = (
        undersampled_variable_density_spiral(system_defaults, n_readout, fov, undersampling_factor)
    )
    total_number_of_points = len(traj) * n_spirals
    assert np.round(n_readout**2 / total_number_of_points) == undersampling_factor


@pytest.mark.parametrize('n_readout', (64, 256))
@pytest.mark.parametrize('fov', (128e-3, 320e-3))
@pytest.mark.parametrize('n_spirals', (14, None))
@pytest.mark.parametrize('undersampling_factor', (1, 3))
@pytest.mark.parametrize('readout_oversampling', (1, 2, 4))
@pytest.mark.parametrize('spiral_type', ('out', 'in-out'))
def test_spiral_acquisition(
    system_defaults: pp.Opts,
    n_readout: int,
    fov: float,
    undersampling_factor: float,
    n_spirals: int,
    readout_oversampling: int,
    spiral_type: str,
):
    """Test spiral trajectories for different parameter combinations."""
    g_pre_duration = 2e-3  # make this duration long to work for all combinations

    gx, gy, adc, trajectory, time_to_echo = spiral_acquisition(
        system_defaults,
        n_readout,
        fov,
        undersampling_factor,
        readout_oversampling,
        n_spirals,
        g_pre_duration,
        spiral_type=spiral_type,
    )

    # Verify timing for each spiral arm
    for spiral_idx in range(len(gx)):
        seq = pp.Sequence(system=system_defaults)
        seq.add_block(gx[spiral_idx], gy[spiral_idx], adc)

        # Get full waveform for readout gradient
        gx_waveform_intp, gy_waveform_intp, dt = get_interp_waveform_for_gx_gy(seq)
        max_grad = np.max(np.abs(gx_waveform_intp))
        gx_waveform_intp /= max_grad
        gy_waveform_intp /= max_grad

        m0_intp = (np.abs(np.cumsum(gx_waveform_intp)) + np.abs(np.cumsum(gy_waveform_intp))) / (
            2 * len(gx_waveform_intp)
        )

        if spiral_type == 'out':
            k0_idx = np.argmin(m0_intp[:-100])
        else:
            k0_idx = np.argmin(m0_intp[100:-100]) + 100
            assert m0_intp[0] < 1e-3
        assert m0_intp[-1] < 1e-3
        assert np.isclose(dt[k0_idx], time_to_echo, atol=system_defaults.grad_raster_time)

        k_traj_adc = seq.calculate_kspace()[0]
        # Ignore first and last elements because they are extrapolated for readout_oversampling > 1
        k_traj_spiral = trajectory[spiral_idx, :, :]
        if readout_oversampling > 1:
            k_traj_adc = k_traj_adc[:, readout_oversampling // 2 : -readout_oversampling // 2]
            k_traj_spiral = k_traj_spiral[readout_oversampling // 2 : -readout_oversampling // 2, :]
        k_traj_adc /= np.max(np.abs(k_traj_adc))
        k_traj_spiral /= np.max(np.abs(k_traj_spiral))
        np.testing.assert_allclose(k_traj_adc[0, :], k_traj_spiral[:, 0], atol=2.5e-3)
        np.testing.assert_allclose(k_traj_adc[1, :], k_traj_spiral[:, 1], atol=2.5e-3)

    # Verify entire trajectory
    seq = pp.Sequence(system=system_defaults)
    for idx in range(len(gx)):
        seq.add_block(gx[idx], gy[idx], adc)
        seq.add_block(pp.make_delay(1e-3))

    ok, error_report = seq.check_timing()
    if not ok:
        print('\nTiming check failed! Error listing follows\n')
        print(error_report)
    assert ok
