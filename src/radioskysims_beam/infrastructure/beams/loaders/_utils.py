from __future__ import annotations

import numpy as np
import xarray as xr


def finalize_beam_dataset(ds: xr.Dataset) -> xr.Dataset:
    """
    Garante invariantes estruturais do Beam.
    """

    required_dims = {"polarization", "frequency", "angle"}
    if not required_dims.issubset(ds.dims):
        raise ValueError(f"Dataset inválido, dims={ds.dims}")

    if "amplitude" not in ds:
        raise ValueError("Dataset não contém 'amplitude'")

    ds = ds.copy()

    ds["amplitude"] = ds["amplitude"].astype(np.float64)

    if "phase" in ds:
        ds["phase"] = ds["phase"].astype(np.float64)

    ds = ds.assign_coords(
        angle=ds.angle.astype(np.float64),
        frequency=ds.frequency.astype(np.float64),
    )

    return ds
