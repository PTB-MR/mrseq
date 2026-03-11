"""Tests for EPI readout."""

from typing import Literal

import numpy as np
import pypulseq as pp
import pytest
from mrseq.utils.EpiReadout import EpiReadout


@pytest.mark.parametrize('fov', (64e-3, 128e-3))
@pytest.mark.parametrize('n_readout', (32, 64))
@pytest.mark.parametrize('n_phase_encoding', (32, 64))
@pytest.mark.parametrize('bandwidth', (50e3, 40e3))
@pytest.mark.parametrize('oversampling', (1, 2))
@pytest.mark.parametrize('ramp_sampling', (True, False))
@pytest.mark.parametrize('readout_type', ('flyback', 'symmetric'))
@pytest.mark.parametrize('partial_fourier_factor', (0.75, 1.0))
@pytest.mark.parametrize('spoiling_enable', (True, False))
def test_epi_time_to_center(
    system_defaults: pp.Opts | None,
    fov: float,
    n_readout,
    n_phase_encoding,
    bandwidth: float,
    oversampling: int,
    ramp_sampling: bool,
    readout_type: Literal['flyback', 'symmetric'],
    partial_fourier_factor: float,
    spoiling_enable: bool,
):
    """Test epi readout for different parameter combinations."""

    def get_time_to_k0(seq):
        """Find time when gradients are 0 (k-space center)."""
        k_traj_adc, _, _, _, t_adc = seq.calculate_kspace()
        m0 = np.sqrt(k_traj_adc[0] ** 2 + k_traj_adc[1] ** 2)
        k0_idx = np.argmin(m0[100:-100]) + 100
        return t_adc[k0_idx], np.median(np.diff(t_adc))

    # create EpiReadout object
    epi2d = EpiReadout(
        system=system_defaults,
        fov=fov,
        n_readout=n_readout,
        n_phase_encoding=n_phase_encoding,
        bandwidth=bandwidth,
        oversampling=oversampling,
        readout_type=readout_type,
        ramp_sampling=ramp_sampling,
        partial_fourier_factor=partial_fourier_factor,
        adc_freq_offset=0.0,
        pe_enable=True,
        spoiling_enable=spoiling_enable,
    )

    # Complete EPI readout
    seq = pp.Sequence(system=system_defaults)
    seq, _ = epi2d.add_to_seq(seq)
    assert sum(seq.block_durations.values()) == pytest.approx(epi2d.total_duration, abs=1e-6)
    time_to_k0, dwell_time = get_time_to_k0(seq)
    assert np.isclose(time_to_k0, epi2d.time_to_center, atol=dwell_time)

    # EPI readout without prephaseser gradients
    seq = pp.Sequence(system=system_defaults)
    seq, _ = epi2d.add_to_seq(seq, add_prephaser=False)
    assert sum(seq.block_durations.values()) == pytest.approx(epi2d.total_duration_without_prephaser, abs=1e-6)


@pytest.mark.parametrize('fov', (64e-3, 128e-3))
@pytest.mark.parametrize('n_readout', (32, 64))
@pytest.mark.parametrize('n_phase_encoding', (32, 64))
@pytest.mark.parametrize('bandwidth', (50e3, 40e3))
@pytest.mark.parametrize('oversampling', (1, 2))
@pytest.mark.parametrize('ramp_sampling', (True, False))
@pytest.mark.parametrize('readout_type', ('flyback', 'symmetric'))
@pytest.mark.parametrize('partial_fourier_factor', (0.75, 1.0))
def test_epi_analytical_trajectory(
    system_defaults: pp.Opts | None,
    fov: float,
    n_readout: int,
    n_phase_encoding: int,
    bandwidth: float,
    oversampling: int,
    ramp_sampling: bool,
    readout_type: Literal['flyback', 'symmetric'],
    partial_fourier_factor: float,
):
    """Test that the analytical trajectory matches calculate_kspace()."""
    epi2d = EpiReadout(
        system=system_defaults,
        fov=fov,
        n_readout=n_readout,
        n_phase_encoding=n_phase_encoding,
        bandwidth=bandwidth,
        oversampling=oversampling,
        readout_type=readout_type,
        ramp_sampling=ramp_sampling,
        partial_fourier_factor=partial_fourier_factor,
        adc_freq_offset=0.0,
        pe_enable=True,
        spoiling_enable=True,
    )

    # Build sequence and get PyPulseq trajectory
    seq = pp.Sequence(system=system_defaults)
    seq, _ = epi2d.add_to_seq(seq)
    k_traj_adc, _, _, _, _ = seq.calculate_kspace()

    # Get analytical trajectory
    kx_analytical, ky_analytical = epi2d.calculate_trajectory()

    n_samples = epi2d.adc.num_samples
    n_pe = epi2d.n_phase_enc_total

    # The prephaser block has no ADC, so calculate_kspace ADC samples start at the first readout
    for pe_idx in range(n_pe):
        start = pe_idx * n_samples
        end = start + n_samples
        kx_pulseq = k_traj_adc[0, start:end]
        ky_pulseq = k_traj_adc[1, start:end]

        np.testing.assert_allclose(kx_analytical[pe_idx], kx_pulseq, atol=1e-9)
        np.testing.assert_allclose(ky_analytical[pe_idx], ky_pulseq, atol=1e-9)
