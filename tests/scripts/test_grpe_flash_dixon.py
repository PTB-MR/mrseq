"""Tests for 3D FLASH sequence with golden radial phase encoding."""

import pytest
from mrseq.scripts.grpe_flash_dixon import main as create_seq

EXPECTED_DUR = 17.84831  # defined 2025-11-09


def test_default_seq_duration(system_defaults):
    """Test if default values result in expected sequence duration."""
    seq, _ = create_seq(system=system_defaults, show_plots=False)
    duration = seq.duration()[0]
    assert duration == pytest.approx(EXPECTED_DUR)


def test_seq_creation_error_on_short_tr(system_defaults):
    """Test if error is raised on too short repetition time."""
    with pytest.raises(ValueError):
        create_seq(system=system_defaults, tr=5e-3, show_plots=False)


def test_seq_duration_vary_params_without_effect(system_defaults):
    """Test if sequence duration is as expected."""
    seq, _ = create_seq(
        system=system_defaults,
        n_rpe_points_per_shot=4,  # default 8
        show_plots=False,
        test_report=False,
        timing_check=False,
    )
    duration = seq.duration()[0]
    assert duration == pytest.approx(EXPECTED_DUR)


def test_seq_creation_error_on_wrong_partial_echo_factor(system_defaults):
    """Test if error is raised on wrong partial echo factor."""
    with pytest.raises(ValueError):
        create_seq(system=system_defaults, partial_echo_factor=1.1, show_plots=False)
    with pytest.raises(ValueError):
        create_seq(system=system_defaults, partial_echo_factor=0.4, show_plots=False)


def test_seq_creation_error_on_wrong_partial_fourier_factor(system_defaults):
    """Test if error is raised on wrong partial Fourier factor."""
    with pytest.raises(ValueError):
        create_seq(system=system_defaults, partial_fourier_factor=1.1, show_plots=False)
    with pytest.raises(ValueError):
        create_seq(system=system_defaults, partial_fourier_factor=0.4, show_plots=False)
