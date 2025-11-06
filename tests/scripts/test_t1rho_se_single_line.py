"""Tests for Gold standard SE-based T1rho mapping sequence."""

import pytest
from mrseq.scripts.t1rho_se_single_line import main as create_seq

EXPECTED_DUR = 3072.000000  # defined 2025-02-20


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
        create_seq(system=system_defaults, tr=5e-3, show_plots=False)


def test_seq_duration_vary_params_without_effect(system_defaults):
    """Test if sequence duration is as expected."""
    seq, _ = create_seq(
        system=system_defaults,
        te=10e-3,  # default None
        fov_xy=192e-3,  # default 128e-3
        n_readout=192,  # default 128
        slice_thickness=6e-3,  # default 8e-3
        show_plots=False,
        test_report=False,
        timing_check=False,
    )
    duration = seq.duration()[0]
    assert duration == pytest.approx(EXPECTED_DUR)
