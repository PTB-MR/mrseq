"""Helper functions for simulations with MRzero."""

from typing import Any

import numpy as np
import pypulseq as pp


def set_mr0_inversion_pulses(mr0_seq: Any, pp_seq: pp.Sequence, flip_angle: float = np.pi, phase: float = 0.0) -> int:
    """Set the flip angle of all inversion pulses of an imported MRzero sequence.

    MRzero >= 1.0 calculates the flip angle of a pulse as the magnitude of the integral of the complex RF shape,
    which is not meaningful for frequency-swept adiabatic pulses (see https://github.com/MRsources/MRzero-Core/issues/186).
    This function sets the flip angle and phase of all pulses with ``use='inversion'`` in the PyPulseq sequence to the
    given values. MRzero creates exactly one repetition per RF block, in the order of the blocks.

    Parameters
    ----------
    mr0_seq
        MRzero sequence obtained by ``MRzeroCore.Sequence.import_file`` of the written ``pp_seq``.
    pp_seq
        PyPulseq sequence the MRzero sequence was created from.
    flip_angle
        Flip angle to set for inversion pulses (in radians).
    phase
        Phase to set for inversion pulses (in radians).

    Returns
    -------
        Number of modified pulses.

    Raises
    ------
    ValueError
        If the number of RF blocks in ``pp_seq`` does not match the number of repetitions in ``mr0_seq``.
    """
    uses = [block.rf.use for block in (pp_seq.get_block(i) for i in pp_seq.block_events) if block.rf is not None]
    if len(uses) != len(mr0_seq):
        raise ValueError(
            f'Number of RF blocks in the PyPulseq sequence ({len(uses)}) does not match '
            f'the number of repetitions in the MRzero sequence ({len(mr0_seq)}).'
        )

    n_inversion_pulses = 0
    for rep, use in zip(mr0_seq, uses, strict=True):
        if use == 'inversion':
            rep.pulse.angle = rep.pulse.angle.new_tensor(flip_angle)
            rep.pulse.phase = rep.pulse.phase.new_tensor(phase)
            n_inversion_pulses += 1
    return n_inversion_pulses


def remove_mr0_residual_moments(mr0_seq: Any, max_residual: float = 1.0, axes: tuple[int, ...] = (0, 1)) -> int:
    """Remove small residual gradient moments from all repetitions of an imported MRzero sequence.

    MRzero >= 1.0 integrates arbitrary gradient waveforms as piecewise-linear functions (trapezoidal rule including
    the first/last sample), whereas waveforms such as a spiral readout followed by its rewinder are designed to have
    zero net moment on the gradient raster. The resulting small residual moment per repetition (~0.1 1/m) is
    physically negligible but prevents MRzero from merging the configuration states of balanced/FISP-type sequences,
    which changes the simulated signal.

    For every repetition and every axis in ``axes``, a residual with ``0 < abs(residual) < max_residual`` is
    subtracted from the last gradient event of the repetition, so the k-space trajectory during the ADC is unchanged.

    Parameters
    ----------
    mr0_seq
        MRzero sequence obtained by ``MRzeroCore.Sequence.import_file``.
    max_residual
        Maximum absolute residual moment (in 1/m) that is considered unintentional and removed. Larger net moments,
        e.g. phase-encoding steps or spoilers, are kept.
    axes
        Gradient axes to correct. Default is the in-plane axes x and y.

    Returns
    -------
        Number of modified repetitions.
    """
    n_modified = 0
    for rep in mr0_seq:
        residual = rep.gradm.sum(0)
        modified = False
        for axis in axes:
            if 0 < abs(float(residual[axis])) < max_residual:
                rep.gradm[-1, axis] -= residual[axis]
                modified = True
        n_modified += modified
    return n_modified
