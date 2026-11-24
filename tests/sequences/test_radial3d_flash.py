"""Tests for 3D radial (kooshball) FLASH sequence."""

from typing import Any

import numpy as np
import pytest
from mrseq.sequences.radial3d_flash import main as create_seq

DEFAULT_KWARGS: dict[str, Any] = {
    'n_readout': 32,
    'n_spokes': 200,
    'n_dummy_excitations': 5,
    'show_plots': False,
    'test_report': False,
    'timing_check': False,
}

EXPECTED_DUR = 0.98338  # defined 2026-11-24


def test_default_seq_duration(system_defaults):
    """Test if default values result in expected sequence duration."""
    seq, _ = create_seq(system=system_defaults, **DEFAULT_KWARGS)
    duration = seq.duration()[0]
    assert duration == pytest.approx(EXPECTED_DUR)


def test_seq_creation_error_on_short_te(system_defaults):
    """Test if error is raised on too short echo time."""
    with pytest.raises(ValueError):
        create_seq(system=system_defaults, te=1e-4, **DEFAULT_KWARGS)


def test_seq_creation_error_on_short_tr(system_defaults):
    """Test if error is raised on too short repetition time."""
    with pytest.raises(ValueError):
        create_seq(system=system_defaults, tr=1e-3, **DEFAULT_KWARGS)


def test_seq_predefined_echo_time(system_defaults):
    """Test sequence with predefined echo time."""
    seq, _ = create_seq(system=system_defaults, te=3e-3, **DEFAULT_KWARGS)
    assert seq


@pytest.mark.parametrize('excitation', ['block', 'slab'])
@pytest.mark.parametrize('spoke_type', ['full', 'half'])
def test_seq_variants(system_defaults, excitation, spoke_type):
    """Test that all excitation and spoke type combinations build and pass timing check."""
    seq, _ = create_seq(
        system=system_defaults,
        excitation=excitation,
        spoke_type=spoke_type,
        **DEFAULT_KWARGS,
    )
    assert seq.check_timing()[0]

    n_adc_blocks = sum(1 for i in range(1, len(seq.block_durations) + 1) if seq.get_block(i).adc is not None)
    assert n_adc_blocks == DEFAULT_KWARGS['n_spokes'] + 1  # spokes + noise scan


def test_seq_half_spoke_adc_samples(system_defaults):
    """Test that half spokes acquire half of the full spoke samples."""
    seq, _ = create_seq(system=system_defaults, spoke_type='half', **DEFAULT_KWARGS)
    expected_samples = int(np.ceil(DEFAULT_KWARGS['n_readout'] * 2 / 2))
    assert seq.get_block(2).adc.num_samples == expected_samples


def test_seq_encoding_train_balanced(system_defaults):
    """Test that the k-space center sample of each full spoke is at k = 0."""
    n_readout = DEFAULT_KWARGS['n_readout']
    readout_oversampling = 2
    fov = 128e-3
    seq, _ = create_seq(system=system_defaults, spoke_type='full', fov=fov, **DEFAULT_KWARGS)

    k_adc = seq.calculate_kspace()[0]  # shape (3, n_adc_samples)
    n_samples = n_readout * readout_oversampling
    k0_index = n_samples // 2
    delta_k = 1 / (fov * readout_oversampling)

    n_spokes = DEFAULT_KWARGS['n_spokes']
    k_spokes = k_adc[:, -n_spokes * n_samples :].reshape(3, n_spokes, n_samples).transpose(1, 2, 0)
    np.testing.assert_array_less(np.abs(k_spokes[:, k0_index, :]), 0.5 * delta_k)
    assert np.abs(np.abs(k_spokes).max() - n_samples / 2 * delta_k) < delta_k
