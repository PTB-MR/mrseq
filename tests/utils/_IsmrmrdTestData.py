"""Create ismrmrd datasets."""

from pathlib import Path
from typing import Literal

import ismrmrd
import ismrmrd.xsd


class IsmrmrdTestData:
    """Test data in ISMRMRD format for testing.

    This is based on
    https://github.com/ismrmrd/ismrmrd-python-tools/blob/master/generate_cartesian_shepp_logan_dataset.py
    """

    def __init__(
        self,
        filename: str | Path,
        matrix_size: int = 64,
        n_coils: int = 4,
        trajectory_type: Literal['cartesian', 'radial'] = 'cartesian',
    ):
        """Initialize IsmrmrdRawTestData.

        Parameters
        ----------
        filename
            full path and filename
        matrix_size
            size of image matrix
        n_coils
            number of coils
        trajectory_type
            cartesian
        """
        self.filename = filename
        self.matrix_size = matrix_size
        self.n_coils = n_coils
        self.trajectory_type = trajectory_type

        # The number of points in image space (x,y) and kspace (fe,pe)
        n_x = self.matrix_size
        n_y = self.matrix_size
        n_freq_encoding = self.matrix_size

        # Open the dataset
        dataset = ismrmrd.Dataset(self.filename, 'dataset', create_if_needed=True)

        # Create the XML header and write it to the file
        header = ismrmrd.xsd.ismrmrdHeader()

        # Experimental Conditions
        exp = ismrmrd.xsd.experimentalConditionsType()
        exp.H1resonanceFrequency_Hz = 128000000
        header.experimentalConditions = exp

        # Acquisition System Information
        sys = ismrmrd.xsd.acquisitionSystemInformationType()
        sys.receiverChannels = self.n_coils
        header.acquisitionSystemInformation = sys

        # Sequence Information
        seq = ismrmrd.xsd.sequenceParametersType()
        seq.TR = [89.6]
        seq.TE = [2.3]
        seq.TI = [0.0]
        seq.flipAngle_deg = 12.0
        seq.echo_spacing = 5.6
        header.sequenceParameters = seq

        # Encoding
        encoding = ismrmrd.xsd.encodingType()
        encoding.trajectory = ismrmrd.xsd.trajectoryType(self.trajectory_type)

        # Encoded and recon spaces
        encoding_fov = ismrmrd.xsd.fieldOfViewMm()
        encoding_matrix = ismrmrd.xsd.matrixSizeType()
        if self.trajectory_type == 'radial':
            encoding_fov.y = matrix_size
            encoding_fov.x = matrix_size
            encoding_fov.z = 5
            encoding_matrix.x = matrix_size
            encoding_matrix.y = matrix_size
            encoding_matrix.z = 1
        else:
            encoding_fov.x = matrix_size
            encoding_fov.y = matrix_size
            encoding_fov.z = 5
            encoding_matrix.x = matrix_size
            encoding_matrix.y = matrix_size
            encoding_matrix.z = 1

        encoding_space = ismrmrd.xsd.encodingSpaceType()
        encoding_space.matrixSize = encoding_matrix
        encoding_space.fieldOfView_mm = encoding_fov
        encoding.encodedSpace = encoding_space

        recon_fov = ismrmrd.xsd.fieldOfViewMm()
        recon_fov.x = matrix_size
        recon_fov.y = matrix_size
        recon_fov.z = 5

        recon_matrix = ismrmrd.xsd.matrixSizeType()
        recon_matrix.x = n_x
        recon_matrix.y = n_y
        recon_matrix.z = 1

        recon_space = ismrmrd.xsd.encodingSpaceType()
        recon_space.matrixSize = recon_matrix
        recon_space.fieldOfView_mm = recon_fov
        encoding.reconSpace = recon_space

        # Encoding limits
        limits = ismrmrd.xsd.encodingLimitsType()

        limits1 = ismrmrd.xsd.limitType()
        limits1.minimum = 0
        limits1.center = n_y // 2
        limits1.maximum = n_y - 1
        limits.kspace_encoding_step_1 = limits1

        encoding.encodingLimits = limits
        header.encoding.append(encoding)

        dataset.write_xml_header(header.toXML('utf-8'))

        # Create an acquisition and reuse it
        acq = ismrmrd.Acquisition()
        acq.resize(n_freq_encoding, self.n_coils, trajectory_dimensions=2)
        acq.version = 1
        acq.available_channels = self.n_coils
        acq.center_sample = round(n_freq_encoding / 2)
        acq.read_dir = (-0.33, 0.38, -0.86)
        acq.phase_dir = (0.75, 0.66, 0.0)
        acq.slice_dir = (-0.57, 0.65, 0.5)

        time_stamp = 10000

        # Write out a few noise scans
        for scan_counter in range(5):
            acq.scan_counter = scan_counter
            acq.acquisition_time_stamp = time_stamp
            acq.clearAllFlags()
            acq.setFlag(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)
            dataset.append_acquisition(acq)
            time_stamp += 2

        # Clean up
        dataset.close()
