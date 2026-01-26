"""Tests for 2D Cartesian FLASH with T2-preparation pulses for T2 mapping."""

import numpy as np
import pytest
from mrseq.scripts.t2_t2_prep_flash import main as create_seq

EXPECTED_DUR = 8.17667  # defined 2026-01-26


def test_default_seq_duration(system_defaults):
    """Test if default values result in expected sequence duration."""
    seq, _ = create_seq(system=system_defaults, show_plots=False)
    duration = seq.duration()[0]
    assert duration == pytest.approx(EXPECTED_DUR)


def test_seq_creation_error_on_short_te(system_defaults):
    """Test if error is raised on too short echo time."""
    with pytest.raises(ValueError):
        create_seq(system=system_defaults, te=1e-3, show_plots=False)


def test_seq_creation_error_on_short_tr(system_defaults):
    """Test if error is raised on too short repetition time."""
    with pytest.raises(ValueError):
        create_seq(system=system_defaults, tr=2e-3, show_plots=False)


def test_seq_duration_vary_params_without_effect(system_defaults):
    """Test if sequence duration is as expected."""
    seq, _ = create_seq(
        system=system_defaults,
        t2_prep_echo_times=np.array([0.05, 0.1, 0.2]),
        show_plots=False,
        test_report=False,
        timing_check=False,
    )
    duration = seq.duration()[0]
    assert duration == pytest.approx(EXPECTED_DUR)
