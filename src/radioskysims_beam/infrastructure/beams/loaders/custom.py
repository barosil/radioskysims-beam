from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from radioskysims_beam.domain.beam import Beam

from ._utils import finalize_beam_dataset
from .base import BeamLoader


class CustomBeamLoader(BeamLoader):
    def __init__(
        self,
        path: str | Path,
        polarizations: list[str],
        frequencies: list[float],
        angles: list[float],
    ) -> None:
        self.path = path
        self.polarizations = polarizations
        self.frequencies = frequencies
        self.angles = angles

    def load(self) -> Beam:
        spec = importlib.util.spec_from_file_location("custom_beam", self.path)
        if spec is None or spec.loader is None:
            msg = f"Cannot load module from path: {self.path}"
            raise ImportError(msg)
        module: Any = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        amp = np.zeros(
            (len(self.polarizations), len(self.frequencies), len(self.angles))
        )
        phase = np.zeros_like(amp)

        for i, pol in enumerate(self.polarizations):
            for j, f in enumerate(self.frequencies):
                for k, a in enumerate(self.angles):
                    amp[i, j, k] = module.amplitude(pol, f, a)
                    if hasattr(module, "phase"):
                        phase[i, j, k] = module.phase(pol, f, a)

        ds = xr.Dataset(
            data_vars={
                "amplitude": (["polarization", "frequency", "angle"], amp),
                "phase": (["polarization", "frequency", "angle"], phase),
            },
            coords={
                "polarization": self.polarizations,
                "frequency": self.frequencies,
                "angle": self.angles,
            },
        )

        ds = finalize_beam_dataset(ds)
        return Beam(ds)
