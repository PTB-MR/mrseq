"""Tests for GIRF Dyn triangle sequence."""

import pytest
from mrseq.scripts.girf_dyn_triangle import main as create_seq

EXPECTED_DUR = 447.42456  # defined 2026-03-19


def test_default_seq_duration(system_defaults):
    """Test if default values result in expected sequence duration."""
    seq, _ = create_seq(system=system_defaults, show_plots=False)
    duration = seq.duration()[0]
    assert duration == pytest.approx(EXPECTED_DUR)
