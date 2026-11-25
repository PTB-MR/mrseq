"""3D radial (kooshball) trajectory calculation."""

from typing import Literal

import numpy as np

from mrseq.utils.constants import GOLDEN_ANGLE_FULL_CIRCLE
from mrseq.utils.constants import GOLDEN_MEANS_3D


def radial_3d_spoke_directions(
    n_spokes: int,
    spoke_type: Literal['full', 'half'] = 'full',
    distribution: Literal['golden', 'spiral', 'phyllotaxis'] = 'golden',
    n_interleaves: int | None = None,
) -> np.ndarray:
    """Calculate unit vectors of the readout direction of each spoke of a 3D radial trajectory.

    Parameters
    ----------
    n_spokes
        number of radial spokes
    spoke_type
        'full' spokes pass through the k-space center (directions on the hemisphere kz >= 0), 'half' spokes start at
        the k-space center (directions on the full sphere)
    distribution
        spoke distribution: 'golden' (3D golden means, Chan et al., MRM 2009), 'spiral' (Archimedean spiral on the
        sphere, Wong & Roos, MRM 1994) or 'phyllotaxis' (spiral phyllotaxis, Piccini et al., MRM 2011)
    n_interleaves
        number of interleaves the spokes are split into. Ignored for 'golden', where any contiguous subset is already
        quasi-uniform

    Returns
    -------
    directions
        unit vectors of shape (n_spokes, 3) in acquisition order
    """
    if n_spokes < 1:
        raise ValueError('Number of spokes must be >= 1.')
    if spoke_type not in ('full', 'half'):
        raise ValueError(f'Unknown spoke type {spoke_type}.')
    if distribution not in ('golden', 'spiral', 'phyllotaxis'):
        raise ValueError(f'Unknown spoke distribution {distribution}.')
    if n_interleaves is not None and distribution != 'golden' and n_spokes % n_interleaves != 0:
        raise ValueError('Number of spokes must be a multiple of the number of interleaves.')

    n = np.arange(n_spokes)

    if distribution == 'golden':
        phi1, phi2 = GOLDEN_MEANS_3D
        z = np.mod(n * phi1, 1)
        azimuth = 2 * np.pi * np.mod(n * phi2, 1)
        if spoke_type == 'half':
            z = 2 * z - 1
    else:
        # uniform z sampling: hemisphere for 'full' spokes, full sphere for 'half' spokes
        if spoke_type == 'half':
            z = -1 + (2 * n + 1) / n_spokes
        else:
            z = (2 * n + 1) / (2 * n_spokes)

        if distribution == 'spiral':
            # discrete form of the Wong-Roos spiral on the sphere
            c = 4 * np.pi if spoke_type == 'half' else 2 * np.pi
            azimuth_step = np.sqrt(c / n_spokes) / np.sqrt(np.maximum(1 - z**2, 1e-12))
            azimuth = np.concatenate(([0.0], np.cumsum(azimuth_step[:-1])))
        else:  # phyllotaxis
            azimuth = n * GOLDEN_ANGLE_FULL_CIRCLE

    r = np.sqrt(np.maximum(1 - z**2, 0))
    directions = np.stack((r * np.cos(azimuth), r * np.sin(azimuth), z), axis=-1)

    if n_interleaves is not None and distribution != 'golden' and n_interleaves > 1:
        order = np.arange(n_spokes).reshape(-1, n_interleaves).T.ravel()
        directions = directions[order]

    return directions / np.linalg.norm(directions, axis=-1, keepdims=True)


def rotation_matrix_to_direction(direction: np.ndarray) -> np.ndarray:
    """Calculate rotation matrix that maps the x-axis onto the given direction.

    Parameters
    ----------
    direction
        target direction of shape (3,), does not need to be normalized

    Returns
    -------
    rotation_matrix
        rotation matrix of shape (3, 3) with determinant +1, for use with pypulseq.rotate_3d
    """
    direction = np.asarray(direction, dtype=np.float64)
    if direction.shape != (3,):
        raise ValueError('Direction must have shape (3,).')
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        raise ValueError('Direction must have a non-zero norm.')
    d = direction / norm

    if abs(d[2]) < 0.9:
        e2 = np.cross([0.0, 0.0, 1.0], d)
    else:
        e2 = np.cross([0.0, 1.0, 0.0], d)
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(d, e2)

    return np.stack((d, e2, e3), axis=1)


def radial_3d_trajectory(
    directions: np.ndarray,
    n_samples: int,
    spoke_type: Literal['full', 'half'] = 'full',
) -> np.ndarray:
    """Calculate k-space trajectory in units of delta_k.

    Parameters
    ----------
    directions
        unit vectors of the spoke directions, shape (n_spokes, 3)
    n_samples
        number of samples per spoke
    spoke_type
        'full' spokes range from -n_samples // 2 to n_samples // 2 - 1, 'half' spokes range from 0 to n_samples - 1

    Returns
    -------
    trajectory
        k-space trajectory of shape (n_spokes, n_samples, 3) in units of delta_k
    """
    if spoke_type == 'full':
        k_radial = np.arange(n_samples) - n_samples // 2
    elif spoke_type == 'half':
        k_radial = np.arange(n_samples)
    else:
        raise ValueError(f'Unknown spoke type {spoke_type}.')

    return k_radial[None, :, None] * directions[:, None, :]
