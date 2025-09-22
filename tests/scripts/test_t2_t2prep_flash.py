"""Tests for 2D Cartesian FLASH with T2-preparation pulses for T2 mapping."""

import pytest
from mrseq.scripts.t2_t2prep_flash import main as create_seq

EXPECTED_DUR = 5120.000970  # defined 2025-02-06


def test_default_seq_duration(system_defaults):
    """Test if default values result in expected sequence duration."""
    seq = create_seq(system=system_defaults, show_plots=False)
    duration = seq.duration()[0]
    assert duration == pytest.approx(EXPECTED_DUR)
