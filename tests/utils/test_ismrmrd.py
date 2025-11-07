"""Tests for ISMRMrd header creation and combination functions."""

import ismrmrd
import pytest
from mrseq.utils.ismrmrd import Fov
from mrseq.utils.ismrmrd import Limits
from mrseq.utils.ismrmrd import MatrixSize
from mrseq.utils.ismrmrd import combine_ismrmrd_files
from mrseq.utils.ismrmrd import create_header

from tests.utils._IsmrmrdTestData import IsmrmrdTestData


@pytest.fixture
def mock_fov():
    """FOV fixture."""
    return Fov(x=0.2, y=0.2, z=0.2)


@pytest.fixture
def mock_matrix_size():
    """Matrix size fixture."""
    return MatrixSize(n_x=128, n_y=128, n_z=1)


@pytest.fixture
def mock_limits():
    """Limits fixture."""
    return Limits(min=0, max=127, center=64)


@pytest.fixture(scope='session')
def ismrmrd_test_data_cartesian(tmp_path_factory):
    """ISMRMRD test data with cartesian trajectory."""
    ismrmrd_filename = tmp_path_factory.mktemp('mrseq') / 'ismrmrd_cart.mrd'
    ismrmrd_data = IsmrmrdTestData(filename=ismrmrd_filename, trajectory_type='cartesian', matrix_size=128)
    return ismrmrd_data


@pytest.fixture(scope='session')
def ismrmrd_test_data_radial(tmp_path_factory):
    """ISMRMRD test data with radial trajectory."""
    ismrmrd_filename = tmp_path_factory.mktemp('mrseq') / 'ismrmrd_radial.mrd'
    ismrmrd_data = IsmrmrdTestData(filename=ismrmrd_filename, trajectory_type='radial', matrix_size=128)
    return ismrmrd_data


def test_create_header(mock_fov, mock_matrix_size, mock_limits):
    """Test the create_header function."""
    header = create_header(
        traj_type='cartesian',
        encoding_fov=mock_fov,
        recon_fov=mock_fov,
        encoding_matrix=mock_matrix_size,
        recon_matrix=mock_matrix_size,
        dwell_time=0.000002,
        k1_limits=mock_limits,
        k2_limits=mock_limits,
        slice_limits=mock_limits,
    )
    assert isinstance(header, ismrmrd.xsd.ismrmrdHeader)
    assert header.experimentalConditions.H1resonanceFrequency_Hz == 127729200
    assert header.encoding[0].trajectory == ismrmrd.xsd.trajectoryType('cartesian')
    assert header.encoding[0].encodedSpace.fieldOfView_mm.x == pytest.approx(mock_fov.x * 1e3)
    assert header.encoding[0].encodedSpace.matrixSize.x == mock_matrix_size.n_x
    assert header.encoding[0].encodingLimits.kspace_encoding_step_1.minimum == mock_limits.min


def test_combine_ismrmrd_files(ismrmrd_test_data_cartesian, ismrmrd_test_data_radial):
    """Test the combine_ismrmrd_files function."""
    _ = combine_ismrmrd_files(ismrmrd_test_data_cartesian.filename, ismrmrd_test_data_radial.filename)
    with ismrmrd.File(str(ismrmrd_test_data_cartesian.filename).replace('.mrd', '_with_traj.mrd'), 'r') as file:
        dataset = file[list(file.keys())[-1]]
        ismrmrd_header = dataset.header
    assert ismrmrd_header.encoding[0].trajectory == ismrmrd.xsd.trajectoryType(ismrmrd_test_data_radial.trajectory_type)
