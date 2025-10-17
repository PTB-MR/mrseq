"""
Create an ISMRMRD header based on the given parameters.

This code is adjusted from https://github.com/mrphysics-bonn/spiral-pypulseq-example
"""

from dataclasses import dataclass
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
    efov = ismrmrd.xsd.fieldOfViewMm()
    efov.x = m_to_mm(encoding_fov.x)
    efov.y = m_to_mm(encoding_fov.y)
    efov.z = m_to_mm(encoding_fov.z)
    rfov = ismrmrd.xsd.fieldOfViewMm()
    rfov.x = m_to_mm(recon_fov.x)
    rfov.y = m_to_mm(recon_fov.y)
    rfov.z = m_to_mm(recon_fov.z)

    ematrix = ismrmrd.xsd.matrixSizeType()
    ematrix.x = encoding_matrix.n_x
    ematrix.y = encoding_matrix.n_y
    ematrix.z = encoding_matrix.n_z
    rmatrix = ismrmrd.xsd.matrixSizeType()
    rmatrix.x = recon_matrix.n_x
    rmatrix.y = recon_matrix.n_y
    rmatrix.z = recon_matrix.n_z

    # set encoded and recon spaces
    escape = ismrmrd.xsd.encodingSpaceType()
    escape.matrixSize = ematrix
    escape.fieldOfView_mm = efov
    rspace = ismrmrd.xsd.encodingSpaceType()
    rspace.matrixSize = rmatrix
    rspace.fieldOfView_mm = rfov
    # encoding.encodedSpace = escape
    # encoding.reconSpace = rspace

    # encoding limits
    limits = ismrmrd.xsd.encodingLimitsType()
    limits.slice = ismrmrd.xsd.limitType()
    limits.slice.minimum = slice_limits.min
    limits.slice.maximum = slice_limits.max
    limits.slice.center = slice_limits.center

    limits.kspace_encoding_step_1 = ismrmrd.xsd.limitType()
    limits.kspace_encoding_step_1.minimum = k1_limits.min
    limits.kspace_encoding_step_1.maximum = k1_limits.max
    limits.kspace_encoding_step_1.center = k1_limits.center

    limits.kspace_encoding_step_2 = ismrmrd.xsd.limitType()
    limits.kspace_encoding_step_2.minimum = k2_limits.min
    limits.kspace_encoding_step_2.maximum = k2_limits.max
    limits.kspace_encoding_step_2.center = k2_limits.center
    # encoding.encodingLimits = limits

    # append encoding
    hdr.encoding.append(encoding)

    return hdr
