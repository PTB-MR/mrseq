"""Tests for radial FLASH sequence."""

import numpy as np
import pytest
from mrseq.scripts.radial_flash import main as create_seq

EXPECTED_DUR = 0.75043  # defined 2025-10-17


def test_default_seq_duration(system_defaults):
    """Test if default values result in expected sequence duration."""
    seq = create_seq(system=system_defaults, show_plots=False)
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


def test_seq_predefined_echo_time(system_defaults):
    """Test sequence with predefined echo time."""
    seq = create_seq(
        system=system_defaults,
        te=3e-3,
        show_plots=False,
        test_report=False,
        timing_check=False,
    )
    assert seq


def test_seq_m2d(system_defaults):
    """Test if sequence with predefined echo time."""
    n_slices = 10
    seq = create_seq(
        system=system_defaults,
        n_slices=n_slices,
        show_plots=False,
        test_report=False,
        timing_check=False,
    )
    duration = seq.duration()[0]
    # we need a larger tolerance because noise samples are only acquired once and not for each slice
    np.testing.assert_allclose(duration / n_slices, EXPECTED_DUR, rtol=0.01)
