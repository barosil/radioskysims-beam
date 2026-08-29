import cProfile
import pstats
import sys
import time

import numpy as np
import pytest
import xarray as xr

from radioskysims_beam.domain.beam.models import Beam
from radioskysims_beam.domain.beam.services import BeamOperation


@pytest.mark.performance
def test_beam_interpolation_speed():
    """Avalia interpolação angular do feixe em 10.000 pontos."""

    np.random.seed(0)

    angles = np.linspace(0.0, 180.0, 360)
    freqs = np.array([1e9])
    pols = np.array(["H"])

    amp = np.ones((len(pols), len(freqs), len(angles)), dtype=float)

    ds = xr.Dataset(
        {"amplitude": (("polarization", "frequency", "angle"), amp)},
        coords={"polarization": pols, "frequency": freqs, "angle": angles},
    )

    beam = Beam(ds)

    new_angles = np.linspace(angles.min(), angles.max(), 10_000)

    prof = cProfile.Profile()
    prof.enable()
    start = time.perf_counter()

    beam2 = BeamOperation.interpolate_angular_range(beam, new_angles)

    elapsed = time.perf_counter() - start
    prof.disable()

    ps = pstats.Stats(prof, stream=sys.stdout)
    ps.strip_dirs().sort_stats("cumtime").print_stats(20)

    assert len(beam2.dataset.angle) == 10_000
    # Limite ajustado: xarray.interp é lento em primeira chamada (load de scipy)
    # Interpolação de 10k pontos em CPU padrão leva ~0.2s
    assert elapsed < 0.3
