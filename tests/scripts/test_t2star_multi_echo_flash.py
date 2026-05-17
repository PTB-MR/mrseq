"""Tests for 2D Cartesian FLASH with T2-preparation pulses for T2 mapping."""

import numpy as np
import pytest
from mrseq.scripts.t2star_multi_echo_flash import main as create_seq
from mrseq.utils.system_defaults import sys_a
from mrseq.utils.system_defaults import sys_b

EXPECTED_DUR = 2.83989  # defined 2025-11-22


def test_default_seq_duration(system_defaults):
    """Test if default values result in expected sequence duration."""
    seq, _ = create_seq(system=system_defaults, show_plots=False)
    duration = seq.duration()[0]
    assert duration == pytest.approx(EXPECTED_DUR)


@pytest.mark.parametrize('system', [sys_a, sys_b])
def test_seq_duration(system):
    """Test system dependance of sequence."""
    seq, _ = create_seq(system=system, show_plots=False)
    duration = seq.duration()[0]
    assert np.abs(duration - EXPECTED_DUR) / EXPECTED_DUR < 0.1


def test_seq_creation_error_on_short_te(system_defaults):
    """Test if error is raised on too short echo time."""
    with pytest.raises(ValueError):
        create_seq(system=system_defaults, te=1e-3, show_plots=False)


def test_seq_creation_error_on_short_tr(system_defaults):
    """Test if error is raised on too short repetition time."""
    with pytest.raises(ValueError):
        create_seq(system=system_defaults, tr=2e-3, show_plots=False)


def test_seq_creation_error_on_wrong_partial_echo_factor(system_defaults):
    """Test if error is raised on wrong partial echo factor."""
    with pytest.raises(ValueError):
        create_seq(system=system_defaults, partial_echo_factor=1.1, show_plots=False)
    with pytest.raises(ValueError):
        create_seq(system=system_defaults, partial_echo_factor=0.4, show_plots=False)
