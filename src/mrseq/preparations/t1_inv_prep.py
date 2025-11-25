"""Adiabatic T1 preparation block."""

import pypulseq as pp

from mrseq.utils import sys_defaults


def add_t1_inv_prep(
    seq: pp.Sequence | None = None,
    system: pp.Opts | None = None,
    rf_duration: float = 10.24e-3,
    add_spoiler: bool = True,
    spoiler_ramp_time: float = 6e-4,
    spoiler_flat_time: float = 8.4e-3,
) -> tuple[pp.Sequence, float, float]:
    """Add an adiabatic T1 preparation block to a sequence.

    The adiabatic inversion pulse is a hyperbolic secant pulse with default values similar to the one used by vendors.

    Parameters
    ----------
    seq
        PyPulseq Sequence object.
    system
        PyPulseq system limit object.
    rf_duration
        Duration of the adiabatic inversion pulse (in seconds).
    add_spoiler
        Toggles addition of spoiler gradients after the inversion pulse.
    spoiler_ramp_time
        Duration of gradient spoiler ramps (in seconds).
    spoiler_flat_time
        Duration of gradient spoiler plateau (in seconds).

    Returns
    -------
    seq
        PyPulseq Sequence object.
    block_duration
        Total duration of the T1 preparation block (in seconds).
    time_since_inversion
        Time passed since point of inversion (=middle of inversion pulse) (in seconds).
    """
    # set system to default if not provided
    if system is None:
        system = sys_defaults

    # create new sequence if not provided
    if seq is None:
        seq = pp.Sequence(system=system)

    # get current duration of sequence before adding T1 preparation block
    time_start = sum(seq.block_durations.values())

    # Add adiabatic inversion pulse
    rf = pp.make_adiabatic_pulse(
        pulse_type='hypsec',
        adiabaticity=6,
        beta=800,
        mu=4.9,
        delay=system.rf_dead_time,
        duration=rf_duration,
        system=system,
        use='inversion',
    )
    seq.add_block(rf)

    # Add spoiler gradient if requested
    if add_spoiler:
        gz_spoil = pp.make_trapezoid(
            channel='z',
            amplitude=0.4 * system.max_grad,
            flat_time=spoiler_flat_time,
            rise_time=spoiler_ramp_time,
            system=system,
        )
        seq.add_block(gz_spoil)

    # calculate total duration of T1prep block
    block_duration = sum(seq.block_durations.values()) - time_start

    # calculate time passed since point of inversion (=middle of inversion pulse)
    time_since_inversion = block_duration - system.rf_dead_time - rf_duration / 2

    return (seq, block_duration, time_since_inversion)
