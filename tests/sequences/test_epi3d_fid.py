"""Tests for 3D EPI FID sequence."""

from typing import Literal

import ismrmrd
import numpy as np
import pytest
from mrseq.sequences.epi3d_fid import epi3d_fid_kernel
from mrseq.sequences.epi3d_fid import main as create_seq
from mrseq.utils.EpiReadout import EpiReadout

EXPECTED_DUR = 0.60944  # defined 2026-11-20


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


def test_kernel_errors(system_defaults):
    """Test if error is raised for invalid 3D parameters."""
    kwargs = {
        'system': system_defaults,
        'te': None,
        'tr': None,
        'fov_xy': 200e-3,
        'fov_z': 64e-3,
        'n_readout': 32,
        'n_phase_encoding': 32,
        'n_partitions': 4,
        'bandwidth': 100e3,
        'rf_duration': 2e-3,
        'rf_flip_angle': 90,
        'rf_bwt': 4,
        'rf_apodization': 0.5,
        'readout_type': 'symmetric',
        'readout_oversampling': 2,
        'ramp_sampling': True,
        'partial_fourier_factor': 1.0,
        'partial_fourier_factor_z': 1.0,
        'add_spoiler': True,
        'add_noise_acq': False,
        'add_navigator_acq': False,
        'mrd_header_file': None,
    }
    # n_partitions < 1 raises ValueError (propagates from EpiReadout)
    with pytest.raises(ValueError):
        epi3d_fid_kernel(**(kwargs | {'n_partitions': 0}))


@pytest.mark.parametrize('readout_type', ('flyback', 'symmetric'))
@pytest.mark.parametrize('add_navigator_acq', [True, False])
@pytest.mark.parametrize('partial_fourier_factor', [0.75, 1.0])
@pytest.mark.parametrize('partial_fourier_factor_z', [0.75, 1.0])
@pytest.mark.parametrize('n_partitions', [4, 5])
def test_mrd_trajectory(
    system_defaults,
    readout_type: Literal['flyback', 'symmetric'],
    add_navigator_acq: bool,
    partial_fourier_factor: float,
    partial_fourier_factor_z: float,
    n_partitions: int,
    tmp_path,
):
    """Test that the 3D MRD trajectory and labels match the PyPulseq and analytical trajectories."""
    fov = 200e-3
    fov_z = 64e-3
    n_readout = 32
    n_phase_encoding = 32
    oversampling: Literal[1, 2] = 2
    ramp_sampling = True

    mrd_file = tmp_path / 'test_epi3d_fid_traj.h5'

    seq, _, _ = epi3d_fid_kernel(
        system=system_defaults,
        te=None,
        tr=None,
        fov_xy=fov,
        fov_z=fov_z,
        n_readout=n_readout,
        n_phase_encoding=n_phase_encoding,
        n_partitions=n_partitions,
        bandwidth=100e3,
        rf_duration=2e-3,
        rf_flip_angle=90,
        rf_bwt=4,
        rf_apodization=0.5,
        readout_type=readout_type,
        readout_oversampling=oversampling,
        ramp_sampling=ramp_sampling,
        partial_fourier_factor=partial_fourier_factor,
        partial_fourier_factor_z=partial_fourier_factor_z,
        add_spoiler=True,
        add_noise_acq=True,
        add_navigator_acq=add_navigator_acq,
        mrd_header_file=mrd_file,
    )

    # Read MRD trajectories
    ds = ismrmrd.Dataset(mrd_file, 'w', False)
    n_acq = ds.number_of_acquisitions()
    traj_mrd = [ds.read_acquisition(i).traj.copy() for i in range(n_acq)]
    ds.close()

    # Recreate EpiReadout for reference
    epi = EpiReadout(
        system=system_defaults,
        fov=fov,
        n_readout=n_readout,
        n_phase_encoding=n_phase_encoding,
        bandwidth=100e3,
        oversampling=oversampling,
        ramp_sampling=ramp_sampling,
        readout_type=readout_type,
        partial_fourier_factor=partial_fourier_factor,
        partial_fourier_factor_z=partial_fourier_factor_z,
        pe_enable=True,
        spoiling_enable=True,
        n_partitions=n_partitions,
        fov_z=fov_z,
    )
    n_samples = epi.adc.num_samples
    n_nav = 3 if add_navigator_acq else 0
    n_noise = 1
    n_epi = epi.n_phase_enc_total
    assert n_acq == n_noise + (n_nav + n_epi) * epi.n_partitions_total

    # Expected partition index for each acquisition
    par_expected = np.concatenate([[0]] + [np.full(n_nav + n_epi, par) for par in range(epi.n_partitions_total)])
    labels = seq.evaluate_labels(evolution='adc')
    np.testing.assert_array_equal(labels['PAR'], par_expected)

    # Expected kz for each acquisition (noise and navigator acquisitions are at kz = 0)
    kz_expected = np.concatenate(
        [[0.0]]
        + [
            np.concatenate([np.zeros(n_nav), np.full(n_epi, epi.kz_areas[par])])
            for par in range(epi.n_partitions_total)
        ]
    )

    # Check MRD header encoding limits and matrix size
    dset = ismrmrd.Dataset(mrd_file, 'w', False)
    hdr = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
    assert hdr.encoding[0].encodingLimits.kspace_encoding_step_2.maximum == epi.n_partitions_total
    assert hdr.encoding[0].encodingLimits.kspace_encoding_step_2.center == epi.n_partitions_pre_center + 1
    assert hdr.encoding[0].encodedSpace.matrixSize.z == n_partitions
    dset.close()

    # Compare Pulseq trajectories against MRD trajectories including noise and navigator acquisitions
    k_traj_adc, _, _, _, _ = seq.calculate_kspace()
    for i in range(n_acq):
        start = i * n_samples
        end = start + n_samples
        kx_calc = (k_traj_adc[0, start:end] * fov * oversampling).astype(np.float32)
        ky_calc = (k_traj_adc[1, start:end] * fov).astype(np.float32)
        kz_calc = (k_traj_adc[2, start:end] * fov_z).astype(np.float32)
        assert traj_mrd[i].shape[1] == 3
        np.testing.assert_array_almost_equal(traj_mrd[i][:, 0], kx_calc, decimal=10)
        np.testing.assert_array_almost_equal(traj_mrd[i][:, 1], ky_calc, decimal=10)
        # kz of MRD trajectory matches Pulseq (navigators are at kz = 0)
        np.testing.assert_allclose(traj_mrd[i][:, 2], kz_calc, atol=0.01)
        # kz of MRD trajectory matches the expected partition encoding
        np.testing.assert_allclose(traj_mrd[i][:, 2], kz_expected[i] * fov_z, atol=0.01)

    # Compare analytical trajectories against MRD trajectories without noise and navigator acquisitions
    kx_analytical, ky_analytical = epi.calculate_trajectory()
    for par in range(epi.n_partitions_total):
        for pe_idx in range(n_epi):
            mrd_idx = n_noise + par * (n_nav + n_epi) + n_nav + pe_idx
            kx_expected = (kx_analytical[pe_idx] * fov * oversampling).astype(np.float32)
            ky_expected = (ky_analytical[pe_idx] * fov).astype(np.float32)
            np.testing.assert_array_equal(traj_mrd[mrd_idx][:, 0], kx_expected)
            np.testing.assert_array_equal(traj_mrd[mrd_idx][:, 1], ky_expected)
            np.testing.assert_array_equal(traj_mrd[mrd_idx][:, 2], np.float32(epi.kz_areas[par] * fov_z))
