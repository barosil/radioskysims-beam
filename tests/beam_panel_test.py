
from __future__ import annotations

import numpy as np
import xarray as xr

from radioskysims_beam.application.beam_panel import (
    BeamPanel,
    create_gaussian_beam,
    gaussian_beam_dataset,
)
from radioskysims_beam.domain.beam.models import Beam


def test_gaussian_beam_dataset_structure() -> None:
    dataset = gaussian_beam_dataset(
        frequency_hz=1.0e9,
        width_deg=10.0,
        amplitude=1.0,
        n_points=181,
    )

    assert isinstance(dataset, xr.Dataset)

    assert set(dataset.dims) == {
        "polarization",
        "frequency",
        "angle",
    }

    assert dataset.sizes["polarization"] == 1
    assert dataset.sizes["frequency"] == 1
    assert dataset.sizes["angle"] == 181

    assert "amplitude" in dataset
    assert "phase" in dataset

    assert dataset["frequency"].attrs["units"] == "Hz"
    assert dataset["angle"].attrs["units"] == "deg"


def test_create_gaussian_beam_returns_domain_beam() -> None:
    beam = create_gaussian_beam(
        frequency_hz=1.0e9,
        width_deg=10.0,
        amplitude=1.0,
    )

    assert isinstance(beam, Beam)
    assert isinstance(beam.dataset, xr.Dataset)

    assert beam.dataset.attrs["name"] == "Gaussian Beam"
    assert beam.dataset.attrs["model"] == "gaussian"


def test_gaussian_beam_has_peak_at_center() -> None:
    beam = create_gaussian_beam(
        frequency_hz=1.0e9,
        width_deg=10.0,
        amplitude=2.0,
    )

    amplitude = beam.dataset["amplitude"].isel(
        polarization=0,
        frequency=0,
    )

    center_index = amplitude.sizes["angle"] // 2

    assert np.isclose(
        float(amplitude.isel(angle=center_index)),
        2.0,
    )


def test_gaussian_beam_is_symmetric() -> None:
    beam = create_gaussian_beam(
        frequency_hz=1.0e9,
        width_deg=10.0,
        amplitude=1.0,
    )

    amplitude = beam.dataset["amplitude"].isel(
        polarization=0,
        frequency=0,
    ).values

    assert np.allclose(amplitude, amplitude[::-1])


def test_beam_panel_uses_domain_beam() -> None:
    panel = BeamPanel()

    assert isinstance(panel.beam, Beam)
    assert panel.beam.dataset.attrs["name"] == "Gaussian Beam"
    assert panel.layout is not None


def test_beam_panel_update_recreates_beam() -> None:
    panel = BeamPanel()

    original_beam = panel.beam

    panel.amplitude.value = 2.0
    panel.width.value = 15.0
    panel._update_beam()

    assert panel.beam is not original_beam
    assert isinstance(panel.beam, Beam)
    assert panel.beam.dataset.attrs["width_deg"] == 15.0

    maximum = float(panel.beam.dataset["amplitude"].max())

    assert np.isclose(maximum, 2.0)
