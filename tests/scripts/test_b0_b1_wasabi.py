"""Tests for WASABI sequence."""

import pytest
from mrseq.scripts.b0_b1_wasabi import main as create_seq

EXPECTED_DUR = 122.1104  # defined 2026-02-05


def test_default_seq_duration(system_defaults):
    """Test if default values result in expected sequence duration."""
    seq, _ = create_seq(system=system_defaults, show_plots=False)
    duration = seq.duration()[0]
    assert duration == pytest.approx(EXPECTED_DUR)
