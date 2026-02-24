"""Cmposite or adiabatic saturation pulse trains."""

import numpy as np
import pypulseq as pp

from mrseq.utils import sys_defaults
from mrseq.utils.hyperbolic_secant_pulse import make_hypsec_90


def add_adia_sat_block(
    seq: pp.Sequence | None = None,
    system: pp.Opts | None = None,
    n_pulses: int = 3,
    max_b1: float = 20,
) -> tuple[pp.Sequence, float]:
    """Add adiabatic saturation pulse train to a PyPulseq Sequence object.

    Parameters
    ----------
    seq
        PyPulseq Sequence object
    system
        system limits
    n_pulses
        number of rf pulses in pulse train
    max_b1
        maximum B1 field amplitude in µT


    Returns
    -------
    seq
        PyPulseq Sequence object
    last_spoil_dur
        duration of the last spoiler gradient in s
    """
    # set system to default if not provided
    if system is None:
        system = sys_defaults

    # create new sequence if not provided
    if seq is None:
        seq = pp.Sequence(system=system)

    # spoilers
    spoil_amp0 = 0.8 * system.max_grad  # Hz/m
    spoil_amp1 = -0.7 * system.max_grad  # Hz/m
    spoil_amp2 = 0.6 * system.max_grad  # Hz/m

    rise_time = 1.0e-3  # spoiler rise time in seconds
    spoil_dur = 5.5e-3  # complete spoiler duration in seconds

    gx_spoil0, gy_spoil0, gz_spoil0 = (
        pp.make_trapezoid(channel=c, system=system, amplitude=spoil_amp0, duration=spoil_dur, rise_time=rise_time)
        for c in ['x', 'y', 'z']
    )
    gx_spoil1, gy_spoil1, gz_spoil1 = (
        pp.make_trapezoid(channel=c, system=system, amplitude=spoil_amp1, duration=spoil_dur, rise_time=rise_time)
        for c in ['x', 'y', 'z']
    )
    gx_spoil2, gy_spoil2, gz_spoil2 = (
        pp.make_trapezoid(channel=c, system=system, amplitude=spoil_amp2, duration=spoil_dur, rise_time=rise_time)
        for c in ['x', 'y', 'z']
    )

    hypsec_90 = make_hypsec_90(amp=max_b1, system=system)

    for i in range(n_pulses):
        seq.add_block(hypsec_90)
        if i % 3 == 0:
            seq.add_block(gx_spoil0, gy_spoil1, gz_spoil2)
        elif i % 2 == 0:
            seq.add_block(gx_spoil2, gy_spoil0, gz_spoil1)
        else:
            seq.add_block(gx_spoil1, gy_spoil2, gz_spoil0)

    return seq, spoil_dur


def add_sat_pulse_train(
    seq: pp.Sequence | None = None,
    system: pp.Opts | None = None,
    n_pulses: int = 6,
    max_b1: float = 14,
    grad_strength_factor: float = 1.0,
) -> tuple[pp.Sequence, float]:
    """Create a saturation pulse train according to Chow, K. et al.

    For more information see:
    Saturation pulse design for quantitative myocardial T1 mapping. J. Cardiovasc. Magn. Reason. 17, 1-15 (2015).

    Parameters
    ----------
    seq
        PyPulseq Sequence object
    system
        system limits
    n_pulses
        number of rf pulses in pulse train
    max_b1
        maximum B1 field amplitude in µT
    grad_strength_factor
        increase the spoiler gradient strength by this factor

    Returns
    -------
    seq
        PyPulseq Sequence object
    last_spoil_dur
        duration of the last spoiler gradient in s
    """
    # set system to default if not provided
    if system is None:
        system = sys_defaults

    # create new sequence if not provided
    if seq is None:
        seq = pp.Sequence(system=system)

    flip_angles = [115, 90, 125, 85, 176, 223]
    grad_dur = [5, 5.25, 4.5, 3.75, 3.5, 3, 4]
    grad_area_values = [153, 198, 136, 111, 72, 60, 119]
    grad_area = np.array(grad_area_values) * grad_strength_factor
    grad_ro = [-1, 1, -1, 0, 0, 0, 1]
    grad_pe = [0, 1, -1, 1, -1, 0, 0]
    grad_ss = [-1, 1, 0, -1, 0, 1, -1]

    dur_block = np.array(flip_angles) / (360 * system.gamma * 10e-7 * max_b1)
    dur_sinc = np.ceil(dur_block / 0.22570566672775255 * 1e5) * 1e-5  # factor equals mean of sinc shape
    dur_block = np.ceil(dur_block * 1e5) * 1e-5

    for i in range(n_pulses + 1):
        if grad_ro[i] != 0:
            gx = pp.make_trapezoid(
                channel='x', area=grad_ro[i] * grad_area[i], duration=grad_dur[i] / 1000, rise_time=5e-4
            )
        if grad_pe[i] != 0:
            gy = pp.make_trapezoid(
                channel='y', area=grad_pe[i] * grad_area[i], duration=grad_dur[i] / 1000, rise_time=5e-4
            )
        if grad_ss[i] != 0:
            gz = pp.make_trapezoid(
                channel='z', area=grad_ss[i] * grad_area[i], duration=grad_dur[i] / 1000, rise_time=5e-4
            )

        if grad_ro[i] != 0 and grad_pe[i] != 0 and grad_ss[i] != 0:
            seq.add_block(gx, gy, gz)
        elif grad_ro[i] != 0 and grad_pe[i] != 0:
            seq.add_block(gx, gy)
        elif grad_ro[i] != 0 and grad_ss[i] != 0:
            seq.add_block(gx, gz)
        elif grad_ro[i] != 0:
            seq.add_block(gx)
        elif grad_pe[i] != 0 and grad_ss[i] != 0:
            seq.add_block(gy, gz)
        elif grad_pe[i] != 0:
            seq.add_block(gy)
        elif grad_ss[i] != 0:
            seq.add_block(gz)

        if i < n_pulses:
            seq.add_block(
                pp.make_sinc_pulse(flip_angle=flip_angles[i] * np.pi / 180, duration=dur_sinc[i], system=system)
            )

    return seq, grad_dur[-1] * 1e-3
