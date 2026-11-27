"""Tests for the diffusion preparation block."""

import pytest
from mrseq.preparations.diffusion_prep import DiffusionPrep
from mrseq.preparations.diffusion_prep import calculate_b_value
from mrseq.preparations.diffusion_prep import calculate_diffusion_gradient_duration


def test_add_t1rho_prep_system_defaults_if_none(system_defaults):
    """Test if system defaults are used if no system limits are provided."""
    diff_prep1 = DiffusionPrep(system=system_defaults)
    diff_prep2 = DiffusionPrep(system=None)

    assert diff_prep1._time_to_refocusing_pulse == diff_prep2._time_to_refocusing_pulse


@pytest.mark.parametrize('g_amplitude', (4e5, 8e5, 1e6, 3e6))
@pytest.mark.parametrize('g_duration', (5e-3, 10e-3, 20e-3, 40e-3))
@pytest.mark.parametrize('g_delta_time', (20e-3, 40e-3, 80e-3, 100e-3))
@pytest.mark.parametrize('g_rise_time', (0.1e-4, 0.2e-4, 0.5e-4, 1e-4))
def test_duration_calculation(g_amplitude, g_duration, g_rise_time, g_delta_time):
    """Test the gradient duration calculation."""
    if g_duration < g_delta_time:
        b_value = calculate_b_value(g_amplitude, g_duration, g_rise_time, g_delta_time)
        g_duration_calc = calculate_diffusion_gradient_duration(b_value, g_amplitude, g_delta_time)
        assert g_duration == pytest.approx(g_duration_calc, rel=3e-2)  # difference due to finite rise time


@pytest.mark.parametrize('g_amplitude', (8e5, 1e6))
@pytest.mark.parametrize('g_delta_time', (50e-3, 80e-3))
def test_block_duration(system_defaults, g_amplitude, g_delta_time):
    """Test the block duration calculation."""
    diff_prep = DiffusionPrep(system_defaults, g_amplitude=g_amplitude, g_delta_time=g_delta_time)
    seq, _ = diff_prep.add_diffusion_prep(seq=None, b_value=200)
    seq_b0, _ = diff_prep.add_diffusion_prep(seq=None, b_value=0)
    assert sum(seq.block_durations.values()) == pytest.approx(sum(seq_b0.block_durations.values()), 1e-3)
    assert sum(seq.block_durations.values()) == pytest.approx(diff_prep._block_duration, 1e-3)


def test_too_short_delta_time(system_defaults):
    """Test if error is raised on short delta time."""
    with pytest.raises(ValueError):
        DiffusionPrep(system_defaults, g_amplitude=4e5, g_delta_time=50e-3)
