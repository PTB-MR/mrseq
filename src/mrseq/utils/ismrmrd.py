"""Utilities to deal with creating ISMRMRD files."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import ismrmrd

T_traj = Literal['cartesian', 'epi', 'radial', 'spiral', 'other']


@dataclass(slots=True)
class Limits:
    """Limits dataclass with min, max, and center attributes."""

    min: int = 0
    max: int = 0
    center: int = 0


@dataclass(slots=True)
class Fov:
    """Fov (x, y, z)."""

    x: float
    y: float
    z: float


@dataclass(slots=True)
class MatrixSize:
    """Matrix size (x, y, z)."""

    n_x: int
    n_y: int
    n_z: int


def m_to_mm(value: float) -> float:
    """Convert meters to millimeters."""
    return value * 1e3


def create_header(
    traj_type: T_traj,
    encoding_fov: Fov,
    recon_fov: Fov,
    encoding_matrix: MatrixSize,
    recon_matrix: MatrixSize,
    dwell_time: float,
    k1_limits: Limits,
    k2_limits: Limits,
    slice_limits: Limits,
    h1_resonance_freq: float = 127729200,  # 3T
) -> ismrmrd.xsd.ismrmrdHeader:
    """
    Create an ISMRMRD header based on the given parameters.

    This code is adjusted from https://github.com/mrphysics-bonn/spiral-pypulseq-example

    Parameters
    ----------
    traj_type
        Trajectory type.
    encoding_fov
        Field of view encoded by the gradients in meters.
    recon_fov
        Field of view for reconstruction (e.g. without readout oversampling) in meters.
    encoding_matrix
        Matrix size of encoded k-space.
    recon_matrix
        Matrix size for reconstruction.
    dwell_time
        Dwell time in seconds.
    k1_limits
        Min, max, and center limits for k1.
    k2_limits
        Min, max, and center limits for k2.
    h1_resonance_freq
        Resonance frequency of water nuclei.

    Returns
    -------
        created ISMRMRD header.
    """
    hdr = ismrmrd.xsd.ismrmrdHeader()

    # experimental conditions
    exp = ismrmrd.xsd.experimentalConditionsType()
    exp.H1resonanceFrequency_Hz = h1_resonance_freq
    hdr.experimentalConditions = exp

    # user parameters
    dtime = ismrmrd.xsd.userParameterDoubleType()
    dtime.name = 'dwellTime_us'
    dtime.value_ = dwell_time * 1e6
    hdr.userParameters = ismrmrd.xsd.userParametersType()
    hdr.userParameters.userParameterDouble.append(dtime)

    # encoding
    encoding = ismrmrd.xsd.encodingType()
    encoding.trajectory = ismrmrd.xsd.trajectoryType(traj_type)

    # set fov and matrix size
    efov = ismrmrd.xsd.fieldOfViewMm(m_to_mm(encoding_fov.x), m_to_mm(encoding_fov.y), m_to_mm(encoding_fov.z))
    rfov = ismrmrd.xsd.fieldOfViewMm(m_to_mm(recon_fov.x), m_to_mm(recon_fov.y), m_to_mm(recon_fov.z))

    ematrix = ismrmrd.xsd.matrixSizeType(int(encoding_matrix.n_x), int(encoding_matrix.n_y), int(encoding_matrix.n_z))
    rmatrix = ismrmrd.xsd.matrixSizeType(int(recon_matrix.n_x), int(recon_matrix.n_y), int(recon_matrix.n_z))

    # set encoded and recon spaces
    escape = ismrmrd.xsd.encodingSpaceType()
    escape.matrixSize = ematrix
    escape.fieldOfView_mm = efov
    rspace = ismrmrd.xsd.encodingSpaceType()
    rspace.matrixSize = rmatrix
    rspace.fieldOfView_mm = rfov
    encoding.encodedSpace = escape
    encoding.reconSpace = rspace

    # encoding limits
    limits = ismrmrd.xsd.encodingLimitsType()
    limits.slice = ismrmrd.xsd.limitType(slice_limits.min, slice_limits.max, slice_limits.center)
    limits.kspace_encoding_step_1 = ismrmrd.xsd.limitType(k1_limits.min, k1_limits.max, k1_limits.center)
    limits.kspace_encoding_step_2 = ismrmrd.xsd.limitType(k2_limits.min, k2_limits.max, k2_limits.center)
    encoding.encodingLimits = limits

    # append encoding
    hdr.encoding.append(encoding)

    return hdr


def read_ismrmrd_dataset(fname: Path) -> tuple:
    """Read ismrmrd dataset from file.

    Parameters
    ----------
    fname
        file path to the ismrmrd dataset.

    Returns
    -------
        ismrmrd header and list of acquisitions.
    """
    with ismrmrd.File(str(fname), 'r') as file:
        ds = file[list(file.keys())[-1]]
        header = ds.header
        acqs = ds.acquisitions[:]

    return header, acqs


def insert_traj_from_meta(
    data_acqs: list[ismrmrd.acquisition.Acquisition],
    meta_acqs: list[ismrmrd.acquisition.Acquisition],
) -> list[ismrmrd.acquisition.Acquisition]:
    """
    Insert trajectory information from the meta file into the data file.

    Parameters
    ----------
    data_acqs : list
        list of acquisitions from the data file.
    meta_acqs : list
        list of acquisitions from the meta file.

    Returns
    -------
    list of acquisitions with the trajectory information from the meta file.
    """
    if not len(data_acqs) == len(meta_acqs):
        raise ValueError('Number of acquisitions in data and meta file do not match.')

    for i, (acq_d, acq_m) in enumerate(zip(data_acqs, meta_acqs, strict=False)):
        if not acq_d.number_of_samples == acq_m.number_of_samples:
            raise ValueError(f'Number of samples in acquisition {i} do not match.')

        # insert trajectory information from meta file
        acq_d.resize(
            number_of_samples=acq_d.number_of_samples,
            active_channels=acq_d.active_channels,
            trajectory_dimensions=acq_m.trajectory_dimensions,
        )
        acq_d.traj[:] = acq_m.traj[:]
        data_acqs[i] = acq_d

    return data_acqs


def update_header_from_meta(
    data_header: ismrmrd.xsd.ismrmrdHeader,
    meta_header: ismrmrd.xsd.ismrmrdHeader,
    enc_idx: int = 0,
) -> ismrmrd.xsd.ismrmrdHeader:
    """Update the header of the data file with the information from the meta file.

    Parameters
    ----------
    data_header : ismrmrd.xsd.ismrmrdHeader
        Header of the ISMRMRD data file.
    meta_header : ismrmrd.xsd.ismrmrdHeader
        Header of the ISMRMRD meta file created with the seq-file.
    enc_idx : int, optional
        Encoding index, by default 0

    Returns
    -------
    ismrmrd.xsd.ismrmrdHeader
        Updated header.
    """

    # Helper function to copy attributes if they are not None
    def copy_attributes(source, target, attributes):
        for attr in attributes:
            value = getattr(source, attr, None)
            if value is not None:
                setattr(target, attr, value)

    # Define the attributes to update for encodedSpace and reconSpace
    attributes_to_update = ['matrixSize', 'fieldOfView_mm']

    # Update encodedSpace
    copy_attributes(
        meta_header.encoding[enc_idx].encodedSpace,
        data_header.encoding[enc_idx].encodedSpace,
        attributes_to_update,
    )

    # Update reconSpace
    copy_attributes(
        meta_header.encoding[enc_idx].reconSpace,
        data_header.encoding[enc_idx].reconSpace,
        attributes_to_update,
    )

    # Update trajectory type
    if meta_header.encoding[enc_idx].trajectory is not None:
        data_header.encoding[enc_idx].trajectory = meta_header.encoding[enc_idx].trajectory

    return data_header


def combine_ismrmrd_files(data_file: Path, meta_file: Path, filename_ext: str = '_with_traj.mrd') -> ismrmrd.Dataset:
    """Combine ismrmrd data file and meta file.

    Parameters
    ----------
    data_file
        path to the ismrmrd data file
    meta_file
        path to the ismrmrd meta file
    filename_ext, optional
        filename extension of the output file, by default '_with_traj.mrd'

    Returns
    -------
        combined ismrmrd file from data and meta file.
    """
    filename_out = data_file.parent / (data_file.stem + filename_ext)

    data_header, data_acqs = read_ismrmrd_dataset(data_file)
    meta_header, meta_acqs = read_ismrmrd_dataset(meta_file)

    new_acqs = insert_traj_from_meta(data_acqs, meta_acqs)
    new_header = update_header_from_meta(data_header, meta_header)

    # Create new file
    ds = ismrmrd.Dataset(filename_out)
    ds.write_xml_header(new_header.toXML())

    # add acquisitions with trajectory information
    for acq in new_acqs:
        ds.append_acquisition(acq)

    ds.close()

    return ds
