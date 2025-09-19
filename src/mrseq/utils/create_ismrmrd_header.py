"""
Create an ISMRMRD header based on the given parameters.

This code is adjusted from https://github.com/mrphysics-bonn/spiral-pypulseq-example
"""

from typing import Literal

import ismrmrd

T_traj = Literal['cartesian', 'epi', 'radial', 'spiral', 'other']


def create_header(
    traj_type: T_traj,
    fov: float,
    res: float,
    dt: float,
    slice_thickness: float,
    n_k1: int,
) -> ismrmrd.xsd.ismrmrdHeader:
    """
    Create an ISMRMRD header based on the given parameters.

    Parameters
    ----------
    traj_type : str
        Trajectory type.
    fov : float
        Field of view in meters.
    res : float
        Resolution in meters.
    dt : float
        Dwell time in seconds.
    slice_thickness : float
        Slice thickness in meters.
    n_k1 : int
        Number of k1 encodes. (spokes for radial, interleaves for spiral, etc.)

    Returns
    -------
        created ISMRMRD header.
    """
    hdr = ismrmrd.xsd.ismrmrdHeader()

    # experimental conditions
    exp = ismrmrd.xsd.experimentalConditionsType()
    exp.H1resonanceFrequency_Hz = 127729200  # 3T
    hdr.experimentalConditions = exp

    # user parameters
    dtime = ismrmrd.xsd.userParameterDoubleType()
    dtime.name = 'dwellTime_us'
    dtime.value_ = dt * 1e6
    hdr.userParameters = ismrmrd.xsd.userParametersType()
    hdr.userParameters.userParameterDouble.append(dtime)

    # encoding
    encoding = ismrmrd.xsd.encodingType()
    encoding.trajectory = ismrmrd.xsd.trajectoryType(traj_type)

    # set fov and matrix size
    efov = ismrmrd.xsd.fieldOfViewMm()
    efov.x = fov * 1e3  # convert to mm
    efov.y = fov * 1e3  # convert to mm
    efov.z = slice_thickness * 1e3  # convert to mm
    rfov = ismrmrd.xsd.fieldOfViewMm()
    rfov.x = fov * 1e3  # convert to mm
    rfov.y = fov * 1e3  # convert to mm
    rfov.z = slice_thickness * 1e3  # convert to mm
    ematrix = ismrmrd.xsd.matrixSizeType()
    ematrix.x = int(fov / res + 0.5)  # both in m
    ematrix.y = int(fov / res + 0.5)  # both in m
    ematrix.z = 1  # 2D
    rmatrix = ismrmrd.xsd.matrixSizeType()
    rmatrix.x = int(fov / res + 0.5)  # both in m
    rmatrix.y = int(fov / res + 0.5)  # both in m
    rmatrix.z = 1

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
    limits.slice = ismrmrd.xsd.limitType()
    limits.slice.minimum = 0
    limits.slice.maximum = 0
    limits.slice.center = 0

    limits.kspace_encoding_step_1 = ismrmrd.xsd.limitType()
    limits.kspace_encoding_step_1.minimum = 0
    limits.kspace_encoding_step_1.maximum = n_k1 - 1
    limits.kspace_encoding_step_1.center = int(n_k1 / 2)
    encoding.encodingLimits = limits

    # append encoding
    hdr.encoding.append(encoding)

    return hdr
