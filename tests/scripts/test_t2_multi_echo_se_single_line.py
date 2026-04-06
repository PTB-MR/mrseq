"""Tests for Gold standard multi-echo SE sequence."""

import numpy as np
import pytest
from mrseq.scripts.t2_multi_echo_se_single_line import main as create_seq
from mrseq.utils.system_defaults import sys_a
from mrseq.utils.system_defaults import sys_b

EXPECTED_DUR = 5120.000970  # defined 2025-02-06


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
    assert np.abs(duration - EXPECTED_DUR) / EXPECTED_DUR < 0.05


def test_seq_creation_error_on_short_te(system_defaults):
    """Test if error is raised on too short echo time."""
    with pytest.raises(ValueError):
        create_seq(system=system_defaults, echo_times=np.array([1e-3, 2e-3]), show_plots=False)


def test_seq_creation_error_on_short_tr(system_defaults):
    """Test if error is raised on too short repetition time."""
    with pytest.raises(ValueError):
        create_seq(system=system_defaults, tr=5e-3, show_plots=False)


def test_seq_duration_vary_params_without_changing_duration(system_defaults):
    """Test if sequence duration is as expected."""
    seq, _ = create_seq(
        system=system_defaults,
        fov_xy=192e-3,  # default 128e-3
        n_readout=192,  # default 128
        slice_thickness=6e-3,  # default 8e-3
        show_plots=False,
        test_report=False,
        timing_check=False,
    )
    duration = seq.duration()[0]
    assert duration == pytest.approx(EXPECTED_DUR)
