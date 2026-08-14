"""Tests for 2D EPI SE for ADC mapping."""

import pytest
from mrseq.sequences.adc_epi2d_se import main as create_seq

EXPECTED_DUR = 1.38768  # defined 2026-03-18


def test_default_seq_duration(system_defaults):
    """Test if default values result in expected sequence duration."""
    seq, _ = create_seq(system=system_defaults, show_plots=False)
    duration = seq.duration()[0]
    assert duration == pytest.approx(EXPECTED_DUR)
