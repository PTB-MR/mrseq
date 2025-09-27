"""Basic functionality for trajectory calculation."""

from typing import Literal

import numpy as np


def cartesian_phase_encoding(
    n_phase_encoding: int,
    acceleration: int = 1,
    n_fully_sampled_center: int = 0,
    sampling_order: Literal['linear', 'low_high', 'high_low', 'random'] = 'linear',
    n_phase_encoding_per_shot: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate Cartesian sampling trajectory.

    Parameters
    ----------
    n_phase_encoding
        number of phase encoding points before undersampling
    acceleration
        undersampling factor
    n_fully_sampled_center
        number of phsae encoding points in the fully sampled center. This will reduce the overall undersampling factor.
    sampling_order
        order how phase encoding points are sampled
    n_phase_encoding_per_shot
        used to ensure that all phase encoding points can be acquired in an integer number of shots. If None, this
        parameter is ignored, i.e. equal to n_phase_encoding_per_shot = 1
    """
    if sampling_order == 'random':
        # Linear order of a fully sampled kpe dimension. Undersampling is done later.
        kpe = np.arange(0, n_phase_encoding)
    else:
        # Always include k-space center and more points on the negative side of k-space
        kpe_pos = np.arange(0, n_phase_encoding // 2, acceleration)
        kpe_neg = -np.arange(acceleration, n_phase_encoding // 2 + 1, acceleration)
        kpe = np.concatenate((kpe_neg, kpe_pos), axis=0)

    # Ensure fully sampled center
    kpe_fully_sampled_center = np.arange(
        -n_fully_sampled_center // 2, -n_fully_sampled_center // 2 + n_fully_sampled_center
    )
    kpe = np.unique(np.concatenate((kpe, kpe_fully_sampled_center)))

    # Always acquire more to ensure desired resolution
    if n_phase_encoding_per_shot and sampling_order != 'random':
        kpe_extended = np.arange(-n_phase_encoding, n_phase_encoding)
        kpe_extended = kpe_extended[np.argsort(np.abs(kpe_extended), kind='stable')]
        idx = 0
        while np.mod(len(kpe), n_phase_encoding_per_shot) > 0:
            kpe = np.unique(np.concatenate((kpe, (kpe_extended[idx],))))
            idx += 1

    # Different temporal orders of phase encoding points
    if sampling_order == 'random':
        perm = np.random.permutation(kpe)
        npe = len(perm) // acceleration
        if n_phase_encoding_per_shot:
            npe += n_phase_encoding_per_shot - np.mod(npe, n_phase_encoding_per_shot)
        kpe = kpe[perm[:npe]]
    elif sampling_order == 'linear':
        kpe = np.sort(kpe)
    elif sampling_order == 'low_high':
        idx = np.argsort(np.abs(kpe), kind='stable')
        kpe = kpe[idx]
    elif sampling_order == 'high_low':
        idx = np.argsort(-np.abs(kpe), kind='stable')
        kpe = kpe[idx]
    else:
        raise ValueError(f'sampling order {sampling_order} not supported.')

    return kpe, kpe_fully_sampled_center
