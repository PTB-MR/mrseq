"""Tests for 3D radial sampling."""

import numpy as np
import pytest
from mrseq.utils.radial_sampling import radial_3d_spoke_directions
from mrseq.utils.radial_sampling import radial_3d_trajectory
from mrseq.utils.radial_sampling import rotation_matrix_to_direction


def nearest_neighbor_distance(directions: np.ndarray, include_antipodes: bool) -> np.ndarray:
    """Great-circle distance of each direction to its nearest neighbor."""
    points = np.concatenate((directions, -directions)) if include_antipodes else directions
    # great-circle distance on the unit sphere: arccos of the clipped dot product
    dot = np.clip(directions @ points.T, -1.0, 1.0)
    dist = np.arccos(dot)
    dist[dist < 1e-12] = np.inf  # exclude the direction itself
    return dist.min(axis=1)


@pytest.mark.parametrize('distribution', ['golden', 'spiral', 'phyllotaxis'])
@pytest.mark.parametrize('spoke_type', ['full', 'half'])
def test_radial_3d_spoke_directions_unit_norm(distribution, spoke_type):
    """Test that spoke directions are unit vectors."""
    n_spokes = 100
    directions = radial_3d_spoke_directions(n_spokes, spoke_type=spoke_type, distribution=distribution)
    assert directions.shape == (n_spokes, 3)
    assert directions.dtype == np.float64
    np.testing.assert_allclose(np.linalg.norm(directions, axis=-1), 1.0, atol=1e-12)


@pytest.mark.parametrize('distribution', ['golden', 'spiral', 'phyllotaxis'])
def test_radial_3d_spoke_directions_full_on_hemisphere(distribution):
    """Test that full spokes lie on the kz >= 0 hemisphere."""
    directions = radial_3d_spoke_directions(200, spoke_type='full', distribution=distribution)
    assert np.all(directions[:, 2] >= 0)


@pytest.mark.parametrize('distribution', ['golden', 'spiral', 'phyllotaxis'])
def test_radial_3d_spoke_directions_half_on_sphere(distribution):
    """Test that half spokes cover the full sphere."""
    directions = radial_3d_spoke_directions(200, spoke_type='half', distribution=distribution)
    assert np.any(directions[:, 2] < 0)
    assert np.any(directions[:, 2] > 0)


@pytest.mark.parametrize(
    ('distribution', 'cv_limit'),
    [('golden', 0.5), ('spiral', 0.35), ('phyllotaxis', 0.35)],
)
@pytest.mark.parametrize('spoke_type', ['full', 'half'])
def test_radial_3d_spoke_directions_uniformity(distribution, cv_limit, spoke_type):
    """Test quasi-uniformity via nearest-neighbor distances."""
    n_spokes = 2000
    directions = radial_3d_spoke_directions(n_spokes, spoke_type=spoke_type, distribution=distribution)
    nn = nearest_neighbor_distance(directions, include_antipodes=spoke_type == 'full')
    assert nn.std() / nn.mean() < cv_limit
    n_eff = 2 * n_spokes if spoke_type == 'full' else n_spokes
    expected_spacing = np.sqrt(4 * np.pi / n_eff)
    assert 0.4 * expected_spacing < nn.mean() < 1.2 * expected_spacing


@pytest.mark.parametrize('spoke_type', ['full', 'half'])
def test_radial_3d_spoke_directions_golden_subset_uniform(spoke_type):
    """Test that the first quarter of golden spokes is already quasi-uniform."""
    directions = radial_3d_spoke_directions(2000, spoke_type=spoke_type, distribution='golden')
    subset = directions[: len(directions) // 4]
    nn = nearest_neighbor_distance(subset, include_antipodes=spoke_type == 'full')
    # larger limit than for the full set: golden subsets can contain a near-duplicate spoke
    assert nn.std() / nn.mean() < 0.7


@pytest.mark.parametrize('distribution', ['spiral', 'phyllotaxis'])
def test_radial_3d_spoke_directions_interleaves(distribution):
    """Test that interleaves are a permutation of the un-interleaved directions."""
    n_spokes = 800
    n_interleaves = 8
    single = radial_3d_spoke_directions(n_spokes, spoke_type='full', distribution=distribution)
    interleaved = radial_3d_spoke_directions(
        n_spokes, spoke_type='full', distribution=distribution, n_interleaves=n_interleaves
    )
    np.testing.assert_allclose(np.sort(interleaved, axis=0), np.sort(single, axis=0))
    spokes_per_interleave = n_spokes // n_interleaves
    for k in range(n_interleaves):
        np.testing.assert_allclose(
            interleaved[k * spokes_per_interleave : (k + 1) * spokes_per_interleave],
            single[k::n_interleaves],
        )


@pytest.mark.parametrize('distribution', ['spiral', 'phyllotaxis'])
def test_radial_3d_spoke_directions_interleaves_error(distribution):
    """Test error if number of spokes is not a multiple of interleaves."""
    with pytest.raises(ValueError):
        radial_3d_spoke_directions(803, spoke_type='full', distribution=distribution, n_interleaves=8)


def test_radial_3d_spoke_directions_errors():
    """Test errors for invalid input."""
    with pytest.raises(ValueError):
        radial_3d_spoke_directions(0)
    with pytest.raises(ValueError):
        radial_3d_spoke_directions(10, spoke_type='quarter')  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        radial_3d_spoke_directions(10, distribution='random')  # type: ignore[arg-type]


def test_rotation_matrix_to_direction():
    """Test that the rotation matrix maps the x-axis onto the target direction."""
    rng = np.random.default_rng(42)
    directions = rng.normal(size=(200, 3))
    directions = np.concatenate((directions, [[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]))
    for direction in directions:
        d = direction / np.linalg.norm(direction)
        rot = rotation_matrix_to_direction(direction)
        np.testing.assert_allclose(rot @ [1, 0, 0], d, atol=1e-12)
        np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(rot), 1.0, atol=1e-12)


def test_rotation_matrix_to_direction_errors():
    """Test errors for invalid directions."""
    with pytest.raises(ValueError):
        rotation_matrix_to_direction(np.zeros(3))
    with pytest.raises(ValueError):
        rotation_matrix_to_direction(np.ones(2))


def test_radial_3d_trajectory():
    """Test the k-space trajectory in units of delta_k."""
    directions = radial_3d_spoke_directions(10, spoke_type='full')
    n_samples = 64

    traj = radial_3d_trajectory(directions, n_samples, spoke_type='full')
    assert traj.shape == (10, n_samples, 3)
    k_along = traj @ directions[0]
    assert k_along[0].min() == -n_samples // 2
    assert k_along[0].max() == n_samples // 2 - 1
    assert np.any(np.linalg.norm(traj, axis=-1) == 0.0)

    traj = radial_3d_trajectory(directions, n_samples, spoke_type='half')
    k_along = traj @ directions[0]
    assert k_along[0].min() == 0
    assert k_along[0].max() == n_samples - 1

    with pytest.raises(ValueError):
        radial_3d_trajectory(directions, n_samples, spoke_type='quarter')  # type: ignore[arg-type]
