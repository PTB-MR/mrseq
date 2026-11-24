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


@pytest.mark.parametrize('readout_type', ('flyback', 'symmetric'))
@pytest.mark.parametrize('ramp_sampling', (True, False))
@pytest.mark.parametrize('spoiling_enable', (True, False))
@pytest.mark.parametrize('n_partitions', (6, 7))
@pytest.mark.parametrize('partial_fourier_factor_z', (0.75, 1.0))
def test_epi_3d_partition_encoding(
    system_defaults: pp.Opts | None,
    readout_type: Literal['flyback', 'symmetric'],
    ramp_sampling: bool,
    spoiling_enable: bool,
    n_partitions: int,
    partial_fourier_factor_z: float,
):
    """Test partition (kz) encoding of the 3D EPI readout for each partition."""
    fov = 128e-3
    fov_z = 64e-3
    delta_kz = 1 / fov_z

    epi = EpiReadout(
        system=system_defaults,
        fov=fov,
        n_readout=32,
        n_phase_encoding=32,
        bandwidth=50e3,
        oversampling=2,
        ramp_sampling=ramp_sampling,
        readout_type=readout_type,
        partial_fourier_factor=1.0,
        partial_fourier_factor_z=partial_fourier_factor_z,
        spoiling_enable=spoiling_enable,
        n_partitions=n_partitions,
        fov_z=fov_z,
    )

    assert epi.trajectory_dimensions == 3
    assert epi.n_partitions_total == epi.n_partitions_pre_center + epi.n_partitions_post_center
    assert epi.n_partitions_total <= n_partitions
    if partial_fourier_factor_z == 1.0:
        assert epi.n_partitions_total == n_partitions
    else:
        assert epi.n_partitions_total < n_partitions
    assert epi.kz_areas[0] == pytest.approx(-epi.n_partitions_pre_center * delta_kz)
    assert epi.kz_areas[-1] == pytest.approx((epi.n_partitions_post_center - 1) * delta_kz)

    for p in range(epi.n_partitions_total):
        seq = pp.Sequence(system=system_defaults)
        seq, _ = epi.add_to_seq(seq, partition_idx=p)
        assert sum(seq.block_durations.values()) == pytest.approx(epi.total_duration, abs=1e-6)

        k_traj_adc, k_traj, _, _, t_adc = seq.calculate_kspace()
        # all ADC samples are at the encoded kz position
        np.testing.assert_allclose(k_traj_adc[2], epi.kz_areas[p], atol=1e-3 * delta_kz)
        # kz is rewound to the spoiler area at the end
        np.testing.assert_allclose(k_traj[2, -1], epi.spoil_area, atol=1e-3 * delta_kz)

        if p == epi.n_partitions_pre_center:
            # center partition: time to k-space center using 3D norm
            m0 = np.sqrt(k_traj_adc[0] ** 2 + k_traj_adc[1] ** 2 + k_traj_adc[2] ** 2)
            k0_idx = np.argmin(m0[100:-100]) + 100
            time_to_k0 = t_adc[k0_idx]
            dwell_time = np.median(np.diff(t_adc))
            assert np.isclose(time_to_k0, epi.time_to_center, atol=dwell_time)


def test_epi_3d_mrd_trajectory(system_defaults: pp.Opts | None, tmp_path):
    """Test that the 3D trajectory written to the MRD dataset is correct."""
    import ismrmrd
    from raw2ismrmrd.utils import Fov
    from raw2ismrmrd.utils import MatrixSize
    from raw2ismrmrd.utils import create_header

    fov = 128e-3
    fov_z = 64e-3
    epi = EpiReadout(
        system=system_defaults,
        fov=fov,
        n_readout=32,
        n_phase_encoding=32,
        bandwidth=50e3,
        oversampling=2,
        n_partitions=4,
        fov_z=fov_z,
    )

    mrd_file = tmp_path / 'traj.h5'
    dset = ismrmrd.Dataset(str(mrd_file), 'w')
    hdr = create_header(
        traj_type='other',
        encoding_fov=Fov(x=fov * epi.oversampling, y=fov, z=fov_z),
        recon_fov=Fov(x=fov, y=fov, z=fov_z),
        encoding_matrix=MatrixSize(n_x=32 * epi.oversampling, n_y=32, n_z=4),
        recon_matrix=MatrixSize(n_x=32, n_y=32, n_z=4),
        dwell_time=epi.adc.dwell,
    )
    dset.write_xml_header(hdr.toXML('utf-8'))
    for p in (0, 3):
        _, dset = epi.add_to_seq(pp.Sequence(system=system_defaults), mrd_dataset=dset, partition_idx=p)
    dset.close()

    kx_traj, ky_traj = epi.calculate_trajectory()
    n_pe = epi.n_phase_enc_total
    dset = ismrmrd.Dataset(str(mrd_file), 'w', False)
    for i, p in enumerate((0, 3)):
        for pe_idx in range(n_pe):
            acq = dset.read_acquisition(i * n_pe + pe_idx)
            traj = acq.traj
            assert traj.shape == (epi.adc.num_samples, 3)
            np.testing.assert_allclose(traj[:, 0], kx_traj[pe_idx] * fov * epi.oversampling, rtol=1e-6)
            np.testing.assert_allclose(traj[:, 1], ky_traj[pe_idx] * fov, rtol=1e-6)
            np.testing.assert_allclose(traj[:, 2], np.float32(epi.kz_areas[p] * fov_z), rtol=1e-6)
    dset.close()


def test_epi_3d_errors(system_defaults: pp.Opts | None):
    """Test error handling of the 3D partition encoding parameters."""
    with pytest.raises(ValueError, match='fov_z'):
        EpiReadout(system=system_defaults, n_partitions=4)
    with pytest.raises(ValueError, match='partitions'):
        EpiReadout(system=system_defaults, n_partitions=0)
    with pytest.raises(ValueError, match='n_partitions > 1'):
        EpiReadout(system=system_defaults, n_partitions=1, partial_fourier_factor_z=0.75)
    with pytest.raises(ValueError, match='partial Fourier'):
        EpiReadout(system=system_defaults, n_partitions=4, fov_z=64e-3, partial_fourier_factor_z=0.4)

    epi = EpiReadout(system=system_defaults, n_partitions=4, fov_z=64e-3)
    for bad_idx in (-1, epi.n_partitions_total, epi.n_partitions_total + 5):
        with pytest.raises(ValueError, match='partition_idx'):
            epi.add_to_seq(pp.Sequence(system=system_defaults), partition_idx=bad_idx)
