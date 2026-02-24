"""Adiabatic hyperbolic secant full passage pulse."""

from types import SimpleNamespace

import numpy as np
from pypulseq import Opts


def calculate_phase(
    frequency: np.ndarray, duration: float, samples: int, shift_idx: int = -1, pos_offsets: bool = False
) -> np.ndarray:
    """Calculate phase modulation for a given frequency modulation."""
    phase = np.zeros_like(frequency)
    for i in range(1, samples):
        phase[i] = phase[i - 1] + (frequency[i] * duration / samples)
    phase_shift = phase[shift_idx]
    for i in range(samples):
        phase[i] = np.fmod(phase[i] + 1e-12 - phase_shift, 2 * np.pi)
    if not pos_offsets:
        phase += 2 * np.pi
    return phase


def create_arbitrary_pulse_with_phase(
    signal: np.ndarray,
    flip_angle: float,
    freq_offset: float = 0,
    phase_offset: float = 0,
    system: Opts | None = None,
) -> SimpleNamespace:
    """Create a RF pulse with arbitrary shape and phase."""
    if not system:
        system = Opts()

    signal *= flip_angle / (2 * np.pi)
    t = np.arange(1, len(signal) + 1) * system.rf_raster_time

    rf = SimpleNamespace()
    rf.type = 'rf'
    rf.signal = signal
    rf.t = t
    rf.shape_dur = t[-1]
    rf.freq_offset = freq_offset
    rf.phase_offset = phase_offset
    rf.dead_time = system.rf_dead_time
    rf.ringdown_time = system.rf_ringdown_time
    rf.delay = system.rf_dead_time
    """If rf.ringdown_time > 0:

    t_fill = np.arange(1, round(rf.ringdown_time /
    system.rf_raster_time) + 1) * system.rf_raster_time rf.t =
    np.concatenate((rf.t, rf.t[-1] + t_fill)) rf.signal =
    np.concatenate((rf.signal, np.zeros(len(t_fill))))
    """

    return rf


def calculate_amplitude(t: np.ndarray, amp: float, mu: float, bandwidth: float) -> np.ndarray:
    """Calculate amplitude modulation for a hypsec inversion pulse."""
    return np.divide(amp, np.cosh((bandwidth * np.pi / mu) * t))


def calculate_frequency(t: np.ndarray, mu: float, bandwidth: float) -> np.ndarray:
    """Calculate frequency modulation for a hypsec inversion pulse."""
    beta = bandwidth * np.pi / mu
    return bandwidth * np.pi * np.tanh(beta * t)


def make_hypsec_180(
    amp: float,
    pulse_duration: float = 8e-3,
    mu: float = 6,
    bandwidth: float = 1200,
    system: Opts | None = None,
) -> SimpleNamespace:
    """Create a hypsec full passage pulse."""
    if not system:
        system = Opts()

    samples = int(pulse_duration / system.rf_raster_time)
    t_pulse = np.arange(-samples // 2, samples // 2) / samples * pulse_duration
    w1 = calculate_amplitude(t=t_pulse, amp=1, mu=mu, bandwidth=bandwidth)
    freq = calculate_frequency(t=t_pulse, mu=mu, bandwidth=bandwidth)
    phase = calculate_phase(frequency=freq, duration=pulse_duration, samples=samples)
    signal = np.multiply(w1, np.exp(1j * phase))
    flip_angle = amp * 1e-6 * system.gamma * 2 * np.pi  # factor 1e-6 converts from µT to T
    hypsec_180 = create_arbitrary_pulse_with_phase(signal=signal, flip_angle=flip_angle, system=system)
    return hypsec_180


def make_hypsec_90(
    amp: float,
    pulse_duration: float = 8e-3,
    mu: float = 6,
    bandwidth: float = 1200,
    system: Opts | None = None,
) -> SimpleNamespace:
    """Create a hypsec half passage pulse."""
    if not system:
        system = Opts()

    samples = int(pulse_duration / system.rf_raster_time)
    t_pulse = np.divide(np.arange(1, samples + 1), samples) * pulse_duration
    t_pulse -= t_pulse[-1]
    w1 = calculate_amplitude(t=t_pulse, amp=1, mu=mu, bandwidth=bandwidth)
    freq = calculate_frequency(t=t_pulse, mu=mu, bandwidth=bandwidth)
    freq = freq - freq[-1]  # ensure phase ends with 0 for tip-down pulse
    phase = calculate_phase(frequency=freq, duration=pulse_duration, samples=samples)
    signal = np.multiply(w1, np.exp(1j * phase))
    flip_angle = amp * 1e-6 * system.gamma * 2 * np.pi  # factor 1e-6 converts from µT to T
    hypsec_90 = create_arbitrary_pulse_with_phase(signal=signal, flip_angle=flip_angle, system=system)
    # hypsec_90 = make_arbitrary_rf(signal=signal, flip_angle=flip_angle, system=system)
    return hypsec_90
