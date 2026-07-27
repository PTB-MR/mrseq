"""Tests for EPI SE sequence."""

from typing import Literal

import ismrmrd
import numpy as np
import pytest
from mrseq.scripts.epi2d_se import epi2d_se_kernel
from mrseq.scripts.epi2d_se import main as create_seq
from mrseq.utils.EpiReadout import EpiReadout

EXPECTED_DUR = 0.04846  # defined 2026-05-19


def test_default_seq_duration(system_defaults):
    """Test if default values result in expected sequence duration."""
    seq, _ = create_seq(system=system_defaults, show_plots=False)
    duration = seq.duration()[0]
    assert duration == pytest.approx(EXPECTED_DUR)


def test_seq_creation_error_on_short_te(system_defaults):
    """Test if error is raised on too short echo time."""
    with pytest.raises(ValueError):
        create_seq(system=system_defaults, te=1e-3, show_plots=False)


def test_seq_creation_error_on_short_tr(system_defaults):
    """Test if error is raised on too short repetition time."""
    with pytest.raises(ValueError):
        create_seq(system=system_defaults, tr=2e-3, show_plots=False)


@pytest.mark.parametrize('add_noise_acq', [True, False])
@pytest.mark.parametrize('readout_type', ('flyback', 'symmetric'))
@pytest.mark.parametrize('add_navigator_acq', [True, False])
@pytest.mark.parametrize('partial_fourier_factor', [0.75, 1.0])
@pytest.mark.parametrize('ramp_sampling', [True, False])
@pytest.mark.parametrize('oversampling', [1, 2])
@pytest.mark.parametrize('n_slices', [1, 4])
def test_mrd_trajectory(
    system_defaults,
    readout_type: Literal['flyback', 'symmetric'],
    add_navigator_acq: bool,
    add_noise_acq: bool,
    partial_fourier_factor: float,
    ramp_sampling: bool,
    oversampling: Literal[1, 2],
    n_slices: int,
    tmp_path,
):
    """Test that the MRD trajectory matches the analytical and PyPulseq trajectories."""
    fov = 200e-3
    n_readout = 32
    n_phase_encoding = 32

    mrd_file = tmp_path / 'test_epi2d_se_traj.h5'

    seq, _, _ = epi2d_se_kernel(
        system=system_defaults,
        te=None,
        tr=None,
        fov_xy=fov,
        n_readout=n_readout,
        n_phase_encoding=n_phase_encoding,
        bandwidth=100e3,
        slice_thickness=5e-3,
        n_slices=n_slices,
        rf_duration=2e-3,
        rf_flip_angle=90,
        rf_bwt=4,
        rf_apodization=0.5,
        readout_type=readout_type,
        readout_oversampling=oversampling,
        ramp_sampling=ramp_sampling,
        partial_fourier_factor=partial_fourier_factor,
        add_spoiler=True,
        add_noise_acq=add_noise_acq,
        add_navigator_acq=add_navigator_acq,
        mrd_header_file=mrd_file,
    )

    # Read MRD trajectories
    ds = ismrmrd.Dataset(mrd_file, 'w', False)
    n_acq = ds.number_of_acquisitions()
    traj_mrd = [ds.read_acquisition(i).traj.copy() for i in range(n_acq)]
    ds.close()

    # Recreate EpiReadout for reference
    epi2d = EpiReadout(
        system=system_defaults,
        fov=fov,
        n_readout=n_readout,
        n_phase_encoding=n_phase_encoding,
        bandwidth=100e3,
        oversampling=oversampling,
        ramp_sampling=ramp_sampling,
        readout_type=readout_type,
        partial_fourier_factor=partial_fourier_factor,
        pe_enable=True,
        spoiling_enable=True,
    )
    n_samples = epi2d.adc.num_samples
    n_nav = 3 if add_navigator_acq else 0
    n_noise = 1 if add_noise_acq else 0
    n_pre_per_slice = n_nav
    n_epi = epi2d.n_phase_enc_total
    assert n_acq == n_noise + (n_pre_per_slice + n_epi) * n_slices

    # Compare Pulseq trajectories against MRD trajectories including noise and navigator acquisitions
    k_traj_adc, _, _, _, _ = seq.calculate_kspace()
    for i in range(n_acq):
        start = i * n_samples
        end = start + n_samples
        kx_calc = (k_traj_adc[0, start:end] * fov * oversampling).astype(np.float32)
        ky_calc = (k_traj_adc[1, start:end] * fov).astype(np.float32)
        np.testing.assert_array_almost_equal(traj_mrd[i][:, 0], kx_calc, decimal=10)
        np.testing.assert_array_almost_equal(traj_mrd[i][:, 1], ky_calc, decimal=10)

    # Compare analytical trajectories against MRD trajectories without noise and navigator acquisitions
    kx_analytical, ky_analytical = epi2d.calculate_trajectory()
    for s in range(n_slices):
        for pe_idx in range(n_epi):
            mrd_idx = n_noise + s * (n_nav + n_epi) + n_nav + pe_idx
            kx_expected = (kx_analytical[pe_idx] * fov * oversampling).astype(np.float32)
            ky_expected = (ky_analytical[pe_idx] * fov).astype(np.float32)
            np.testing.assert_array_equal(traj_mrd[mrd_idx][:, 0], kx_expected)
            np.testing.assert_array_equal(traj_mrd[mrd_idx][:, 1], ky_expected)
