"""Define default system limitations."""

from pypulseq.opts import Opts

sys_defaults = Opts(
    max_grad=30,
    grad_unit='mT/m',
    max_slew=120,
    slew_unit='T/m/s',
    rf_ringdown_time=30e-6,
    rf_dead_time=100e-6,
    adc_dead_time=10e-6,
)

sys_a = Opts(
    max_grad=30,
    grad_unit='mT/m',
    max_slew=120,
    slew_unit='T/m/s',
    grad_raster_time=8e-6,
    rf_ringdown_time=60e-6,
    rf_raster_time=2e-6,
    rf_dead_time=100e-6,
    adc_dead_time=40e-6,
    adc_raster_time=2e-6,
    block_duration_raster=4e-6,
)

sys_b = Opts(
    max_grad=30,
    grad_unit='mT/m',
    max_slew=120,
    slew_unit='T/m/s',
    grad_raster_time=10e-6,
    rf_ringdown_time=30e-6,
    rf_raster_time=1e-6,
    rf_dead_time=100e-6,
    adc_dead_time=10e-6,
    adc_raster_time=1e-7,
    block_duration_raster=10e-6,
)
