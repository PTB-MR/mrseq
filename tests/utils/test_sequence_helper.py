"""Tests for sequence helper functions."""

import pytest
from mrseq.utils import find_gx_flat_time_on_adc_raster
from mrseq.utils import round_to_raster


@pytest.mark.parametrize(
    'value, raster_time, expected',
    [
        (+1.0e-6, 1e-6, 1e-6),
        (+1.1e-6, 1e-6, 2e-6),
        (-1.1e-6, 1e-6, -1e-6),
        (+1.0e-9, 1e-6, 1e-6),
        (0.0, 1e-6, 0.0),  # Zero case
        (2.0e-6, 1e-6, 2.0e-6),  # Already a multiple
        (-2.0e-6, 1e-6, -2.0e-6),  # Negative multiple
        (-1.9e-6, 1e-6, -1e-6),  # Negative rounding up
        (1e6, 1e-6, 1e6),  # Large value
        (1e-12, 1e-6, 1e-6),  # Very small value
        (2.5e-6, 1e-6, 3e-6),  # Non-integer multiple
        (1.0000000001e-6, 1e-6, 2e-6),  # Floating-point precision up
        (0.9999999999e-6, 1e-6, 1e-6),  # Floating-point precision down
    ],
)
def test_round_to_raster_ceil(value: float, raster_time: float, expected: float):
    """Test the rounding to raster with ceil mode."""
    assert round_to_raster(value, raster_time, 'ceil') == expected


@pytest.mark.parametrize(
    'value, raster_time, expected',
    [
        (+1.0e-6, 1e-6, 1e-6),
        (+1.1e-6, 1e-6, 1e-6),
        (-1.1e-6, 1e-6, -2e-6),
        (+1.0e-9, 1e-6, 0.0),
        (0.0, 1e-6, 0.0),  # Zero case
        (2.0e-6, 1e-6, 2.0e-6),  # Already a multiple
        (-2.0e-6, 1e-6, -2.0e-6),  # Negative multiple
        (-1.9e-6, 1e-6, -2e-6),  # Negative rounding down
        (1e6, 1e-6, 1e6),  # Large value
        (1e-12, 1e-6, 0.0),  # Very small value
        (2.5e-6, 1e-6, 2e-6),  # Non-integer multiple
        (1.0000000001e-6, 1e-6, 1e-6),  # Floating-point precision down
        (0.9999999999e-6, 1e-6, 0.0),  # Floating-point precision down
    ],
)
def test_round_to_raster_floor(value: float, raster_time: float, expected: float):
    """Test rounding to raster with floor mode."""
    assert round_to_raster(value, raster_time, 'floor') == expected


@pytest.mark.parametrize(
    'value, raster_time, expected',
    [
        (+1.0e-6, 1e-6, 1e-6),
        (+1.1e-6, 1e-6, 1e-6),
        (-1.1e-6, 1e-6, -1e-6),
        (+1.0e-9, 1e-6, 0.0),
        (0.0, 1e-6, 0.0),  # Zero case
        (2.0e-6, 1e-6, 2.0e-6),  # Already a multiple
        (-2.0e-6, 1e-6, -2.0e-6),  # Negative multiple
        (-1.9e-6, 1e-6, -2e-6),  # Negative rounding
        (1e6, 1e-6, 1e6),  # Large value
        (1e-12, 1e-6, 0.0),  # Very small value
        (1.5000000001e-6, 1e-6, 2e-6),  # Floating-point precision up
        (1.4999999999e-6, 1e-6, 1e-6),  # Floating-point precision down
    ],
)
def test_round_to_raster_round(value: float, raster_time: float, expected: float):
    """Test rounding to raster with default 'round' mode."""
    assert round_to_raster(value, raster_time) == expected


def test_round_to_raster_invalid_method():
    """Test that an invalid method raises a ValueError."""
    with pytest.raises(ValueError, match='Unknown rounding method'):
        round_to_raster(1.0, 1.0, 'invalid')


@pytest.mark.parametrize(
    'n_readout, adc_dwell_time, grad_raster_time, adc_raster_time, expected_gx_flat_time, expected_adc_dwell_time',
    [
        # Basic case where values align perfectly
        (128, 1e-6, 1e-6, 1e-6, 128e-6, 1e-6),
        # Case where adc_dwell_time needs to be adjusted down
        (128, 1.1e-6, 1e-6, 1e-6, 128e-6, 1e-6),
        # Case where adc_dwell_time needs to be adjusted up
        (128, 0.9e-6, 1e-6, 1e-6, 128e-6, 1e-6),
        # Case with larger raster times
        (256, 2e-6, 2e-6, 1e-6, 512e-6, 2e-6),
        # Case with smaller raster times
        (64, 0.5e-6, 1e-6, 0.5e-6, 32e-6, 0.5e-6),
        # Case where adc_dwell_time is already on raster
        (100, 2e-6, 1e-6, 1e-6, 200e-6, 2e-6),
        # Case with very small tolerance
        (128, 1.0000001e-6, 1e-6, 1e-6, 128e-6, 1e-6),
        # Case with very large values
        (1000, 10e-6, 5e-6, 1e-6, 10000e-6, 10e-6),
    ],
)
def test_find_gx_flat_time_on_adc_raster(
    n_readout, adc_dwell_time, grad_raster_time, adc_raster_time, expected_gx_flat_time, expected_adc_dwell_time
):
    """Test find_gx_flat_time_on_adc_raster with various inputs."""
    gx_flat_time, adjusted_adc_dwell_time = find_gx_flat_time_on_adc_raster(
        n_readout, adc_dwell_time, grad_raster_time, adc_raster_time
    )
    assert gx_flat_time == pytest.approx(expected_gx_flat_time, rel=1e-9)
    assert adjusted_adc_dwell_time == pytest.approx(expected_adc_dwell_time, rel=1e-9)


def test_find_gx_flat_time_on_adc_raster_no_solution():
    """Test that find_gx_flat_time_on_adc_raster raises ValueError when no solution is found."""
    with pytest.raises(ValueError, match=r'No adc_dwell_time found within search range.'):
        find_gx_flat_time_on_adc_raster(
            n_readout=128,
            adc_dwell_time=1e-6,
            grad_raster_time=1e-6,
            adc_raster_time=1e-6,
            max_m=1,  # Restrict search range to make solution impossible
        )
