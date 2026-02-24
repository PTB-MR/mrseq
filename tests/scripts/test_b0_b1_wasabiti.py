"""Tests for WASABI sequence."""

import numpy as np
import pytest
from mrseq.scripts.b0_b1_wasabiti import main as create_seq

EXPECTED_DUR = 123.2429  # defined 2026-02-24


def test_default_seq_duration(system_defaults):
    """Test if default values result in expected sequence duration."""
    seq, _ = create_seq(system=system_defaults, show_plots=False)
    duration = seq.duration()[0]
    assert duration == pytest.approx(EXPECTED_DUR)


def test_seq_error_mismatch_t_offset(system_defaults):
    """Test if error is raised on mismatch between recovery times and offsets."""
    with pytest.raises(ValueError):
        create_seq(system=system_defaults, frequency_offsets=np.ones(3), t_recovery=np.ones(2), show_plots=False)
