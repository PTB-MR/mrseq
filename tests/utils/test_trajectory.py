"""Tests for sequence helper functions."""

import numpy as np
import pytest
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
