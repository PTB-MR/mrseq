"""Tests for cardiac MR Fingerprinting sequence with spiral readout."""

import pytest
from mrseq.scripts.t1_t2_spiral_cmrf import main as create_seq

EXPECTED_DUR = 14.53304  # defined 2025-12-12


def test_default_seq_duration(system_defaults):
    """Test if default values result in expected sequence duration."""
    seq, _ = create_seq(system=system_defaults, show_plots=False)
    duration = seq.duration()[0]
    assert duration == pytest.approx(EXPECTED_DUR)


def test_seq_creation_error_on_short_tr(system_defaults):
    """Test if error is raised on too short repetition time."""
    with pytest.raises(ValueError):
        create_seq(system=system_defaults, tr=3e-3, show_plots=False)
