# tests/domain/beam/test_beam_properties.py
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from radioskysims_beam.domain.beam.properties import BeamProperties


def test_compute_properties_basic():
    angles = np.linspace(-90, 90, 181)
    freqs = np.array([1.4e9])
    pols = np.array(["H", "V"])

    amp = np.abs(np.cos(np.deg2rad(angles)))  # padrão coseno
    amp = np.tile(amp, (len(pols), len(freqs), 1)).transpose(2, 1, 0)

    ds = xr.Dataset(
        {"amplitude": (("angle", "frequency", "polarization"), amp)},
        coords={"angle": angles, "frequency": freqs, "polarization": pols},
    )

    result = BeamProperties.compute_properties(ds)

    assert "fwhm" in result
    assert "directivity_db" in result
    assert not np.isnan(result["fwhm"]).all()
    assert result["directivity_db"].values.shape == (len(pols), len(freqs))


def test_gaussian_fwhm_uses_beam_response_not_squared_amplitude():
    sigma_deg = 10.0
    angles = np.linspace(-90.0, 90.0, 18001)

    response = np.exp(-0.5 * (angles / sigma_deg) ** 2)

    ds = xr.Dataset(
        {
            "amplitude": (
                ("angle", "frequency", "polarization"),
                response[:, None, None],
            )
        },
        coords={
            "angle": angles,
            "frequency": [1.0e9],
            "polarization": ["H"],
        },
    )

    result = BeamProperties.compute_properties(ds)

    measured = float(
        result["fwhm"]
        .sel(
            polarization="H",
            frequency=1.0e9,
        )
        .values
    )

    expected = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_deg

    assert measured == pytest.approx(
        expected,
        abs=0.03,
    )
