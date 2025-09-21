"""Tests for cardiac MR Fingerprinting sequence with spiral readout."""

import pytest
from mrseq.scripts.t1_t2_spiral_cmrf import main as create_seq

EXPECTED_DUR = 7056.000000  # defined 2025-02-21


def test_default_seq_duration(system_defaults):
    """Test if default values result in expected sequence duration."""
    seq = create_seq(system=system_defaults, show_plots=False)
    duration = seq.duration()[0]
    assert duration == pytest.approx(EXPECTED_DUR)


def test_seq_creation_error_on_short_tr(system_defaults):
    """Test if error is raised on too short repetition time."""
    with pytest.raises(ValueError):
        create_seq(system=system_defaults, tr=5e-3, show_plots=False)
