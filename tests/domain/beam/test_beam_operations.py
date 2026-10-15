# tests/domain/beam/test_beam_operations.py
from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from radioskysims_beam.domain.beam.models import Beam
from radioskysims_beam.domain.beam.services import BeamOperation


@pytest.fixture
def sample_beam():
    angles = np.linspace(-90, 90, 7)
    freqs = np.array([1.4e9])
    amp = np.abs(np.cos(np.deg2rad(angles)))  # padrão simples

    ds = xr.Dataset(
        {
            "amplitude": (("angle", "frequency"), np.expand_dims(amp, 1)),
        },
        coords={
            "angle": angles,
            "frequency": freqs,
        },
    )
    return Beam(ds)


@pytest.fixture
def sample_beam_two_freq() -> Beam:
    angles = np.linspace(-90, 90, 181)
    freqs = np.array([0.9e9, 1.1e9])
    amp_angle = np.exp(-0.5 * (angles / 10.0) ** 2)
    amp = np.stack([amp_angle * 0.9, amp_angle * 1.1], axis=1)[:, :, None]

    ds = xr.Dataset(
        {
            "amplitude": (("angle", "frequency", "polarization"), amp),
        },
        coords={
            "angle": angles,
            "frequency": freqs,
            "polarization": ["H"],
        },
    )
    ds["angle"].attrs["units"] = "deg"
    ds["frequency"].attrs["units"] = "Hz"

    return Beam(ds)


@pytest.fixture
def sample_beam_two_freq_dask() -> Beam:
    angles = np.linspace(-90, 90, 181)
    freqs = np.array([0.9e9, 1.1e9])
    amp_angle = np.exp(-0.5 * (angles / 10.0) ** 2)
    amp_np = np.stack([amp_angle * 0.9, amp_angle * 1.1], axis=1)[:, :, None]
    amp_da = da.from_array(amp_np, chunks=(60, 1, 1))

    ds = xr.Dataset(
        {
            "amplitude": (("angle", "frequency", "polarization"), amp_da),
        },
        coords={
            "angle": angles,
            "frequency": freqs,
            "polarization": ["H"],
        },
    )
    ds["angle"].attrs["units"] = "deg"
    ds["frequency"].attrs["units"] = "Hz"

    return Beam(ds)


def test_normalize(sample_beam):
    normalized = BeamOperation.normalize(sample_beam)
    assert np.isclose(normalized.dataset["amplitude"].max(), 1.0)


def test_slice_angular_range(sample_beam):
    sliced = BeamOperation.slice_angular_range(sample_beam, -45, 45)
    assert (sliced.dataset.angle.min() >= -45) and (sliced.dataset.angle.max() <= 45)


def test_interpolate_angular_range(sample_beam):
    new_angles = np.linspace(-90, 90, 13)
    interpolated = BeamOperation.interpolate_angular_range(sample_beam, new_angles)
    assert len(interpolated.dataset.angle) == len(new_angles)


def test_symmetrize(sample_beam):
    sym = BeamOperation.symmetrize(sample_beam)
    # simetria esperada
    amp = sym.dataset["amplitude"].values
    assert np.allclose(amp, amp[::-1], atol=1e-3)


def test_beam_sorts_interpolation_coordinates_and_preserves_evaluation():
    """Beam deve ordenar angle/frequency sem alterar a avaliação física."""

    angles = np.array([30.0, -30.0, 0.0])
    freqs = np.array([1400.0, 1000.0, 1200.0])

    # Função determinística B(angle, frequency).
    amplitude = angles[:, np.newaxis] + freqs[np.newaxis, :] / 1000.0

    ds_unsorted = xr.Dataset(
        {
            "amplitude": (
                ("angle", "frequency"),
                amplitude,
            ),
        },
        coords={
            "angle": angles,
            "frequency": freqs,
        },
    )

    ds_unsorted["angle"].attrs["units"] = "deg"
    ds_unsorted["frequency"].attrs["units"] = "MHz"

    # Referência explicitamente ordenada.
    ds_sorted = ds_unsorted.sortby(["angle", "frequency"])

    beam_unsorted = Beam(ds_unsorted)
    beam_reference = Beam(ds_sorted)

    # O próprio Beam deve garantir a pré-condição necessária para
    # interp(..., assume_sorted=True).
    assert np.all(np.diff(beam_unsorted.dataset["angle"].values) >= 0)
    assert np.all(np.diff(beam_unsorted.dataset["frequency"].values) >= 0)

    target_angle = xr.DataArray(
        np.array([-15.0, 15.0]),
        dims="time",
        attrs={"units": "deg"},
    )

    target_frequency = xr.DataArray(
        np.array([1100.0, 1300.0]),
        dims="frequency",
        attrs={"units": "MHz"},
    )

    result = beam_unsorted.evaluate(
        frequency=target_frequency,
        angle=target_angle,
    )

    expected = beam_reference.evaluate(
        frequency=target_frequency,
        angle=target_angle,
    )

    xr.testing.assert_allclose(
        result,
        expected,
    )


def test_evaluate_numpy_radians_input_keeps_numpy_output(sample_beam_two_freq: Beam) -> None:
    angle_rad = xr.DataArray(
        np.linspace(-0.2, 0.2, 11),
        dims="time",
        coords={"time": np.arange(11)},
        attrs={"units": "rad"},
    )
    frequency_hz = xr.DataArray(
        np.array([1.0e9]),
        dims="frequency",
        coords={"frequency": np.array([1.0e9])},
        attrs={"units": "Hz"},
    )

    result = sample_beam_two_freq.evaluate(
        frequency=frequency_hz,
        angle=angle_rad,
        polarization="H",
    )

    assert isinstance(result.data, np.ndarray)
    assert result.shape == (11, 1)


def test_evaluate_dask_radians_input_preserves_lazy_and_matches_numpy(
    sample_beam_two_freq: Beam,
) -> None:
    values_rad = np.linspace(-0.2, 0.2, 11)

    angle_np = xr.DataArray(
        values_rad,
        dims="time",
        coords={"time": np.arange(11)},
        attrs={"units": "rad"},
    )
    angle_da = xr.DataArray(
        da.from_array(values_rad, chunks=(4,)),
        dims="time",
        coords={"time": np.arange(11)},
        attrs={"units": "rad"},
    )

    frequency_np = xr.DataArray(
        np.array([1.0e9]),
        dims="frequency",
        coords={"frequency": np.array([1.0e9])},
        attrs={"units": "Hz"},
    )
    frequency_da = xr.DataArray(
        da.from_array(np.array([1.0e9]), chunks=(1,)),
        dims="frequency",
        coords={"frequency": np.array([1.0e9])},
        attrs={"units": "Hz"},
    )

    result_np = sample_beam_two_freq.evaluate(
        frequency=frequency_np,
        angle=angle_np,
        polarization="H",
    )
    result_da = sample_beam_two_freq.evaluate(
        frequency=frequency_da,
        angle=angle_da,
        polarization="H",
    )

    xr.testing.assert_allclose(
        result_np,
        result_da,
    )


def test_evaluate_dask_degrees_input_still_matches_numpy(
    sample_beam_two_freq: Beam,
) -> None:
    values_deg = np.linspace(-10.0, 10.0, 11)

    angle_np = xr.DataArray(
        values_deg,
        dims="time",
        coords={"time": np.arange(11)},
        attrs={"units": "deg"},
    )
    angle_da = xr.DataArray(
        da.from_array(values_deg, chunks=(4,)),
        dims="time",
        coords={"time": np.arange(11)},
        attrs={"units": "deg"},
    )

    frequency_np = xr.DataArray(
        np.array([1.0e9]),
        dims="frequency",
        coords={"frequency": np.array([1.0e9])},
        attrs={"units": "Hz"},
    )
    frequency_da = xr.DataArray(
        da.from_array(np.array([1.0e9]), chunks=(1,)),
        dims="frequency",
        coords={"frequency": np.array([1.0e9])},
        attrs={"units": "Hz"},
    )

    result_np = sample_beam_two_freq.evaluate(
        frequency=frequency_np,
        angle=angle_np,
        polarization="H",
    )
    result_da = sample_beam_two_freq.evaluate(
        frequency=frequency_da,
        angle=angle_da,
        polarization="H",
    )

    xr.testing.assert_allclose(
        result_np,
        result_da,
    )


def test_evaluate_dask_radians_input_keeps_dask_output_when_beam_is_dask_backed(
    sample_beam_two_freq_dask: Beam,
) -> None:
    angle_da = xr.DataArray(
        da.from_array(np.linspace(-0.2, 0.2, 11), chunks=(4,)),
        dims="time",
        coords={"time": np.arange(11)},
        attrs={"units": "rad"},
    )
    frequency_da = xr.DataArray(
        da.from_array(np.array([1.0e9]), chunks=(1,)),
        dims="frequency",
        coords={"frequency": np.array([1.0e9])},
        attrs={"units": "Hz"},
    )

    result_da = sample_beam_two_freq_dask.evaluate(
        frequency=frequency_da,
        angle=angle_da,
        polarization="H",
    )

    assert isinstance(result_da.data, da.Array)
    assert result_da.chunks is not None


def test_evaluate_dask_radians_input_numpy_vs_dask_backed_beam_equivalence(
    sample_beam_two_freq: Beam,
    sample_beam_two_freq_dask: Beam,
) -> None:
    values_rad = np.linspace(-0.2, 0.2, 11)
    angle_np = xr.DataArray(
        values_rad,
        dims="time",
        coords={"time": np.arange(11)},
        attrs={"units": "rad"},
    )
    angle_da = xr.DataArray(
        da.from_array(values_rad, chunks=(4,)),
        dims="time",
        coords={"time": np.arange(11)},
        attrs={"units": "rad"},
    )
    frequency_np = xr.DataArray(
        np.array([1.0e9]),
        dims="frequency",
        coords={"frequency": np.array([1.0e9])},
        attrs={"units": "Hz"},
    )
    frequency_da = xr.DataArray(
        da.from_array(np.array([1.0e9]), chunks=(1,)),
        dims="frequency",
        coords={"frequency": np.array([1.0e9])},
        attrs={"units": "Hz"},
    )

    expected_np = sample_beam_two_freq.evaluate(
        frequency=frequency_np,
        angle=angle_np,
        polarization="H",
    )
    result_da = sample_beam_two_freq_dask.evaluate(
        frequency=frequency_da,
        angle=angle_da,
        polarization="H",
    )

    xr.testing.assert_allclose(
        expected_np,
        result_da.compute(),
    )


def test_evaluate_single_frequency_beam_returns_finite_values() -> None:
    angle_grid = np.linspace(-2.0, 2.0, 801)
    frequency_grid = np.array([1.0e9])
    polarization_grid = np.array(["H"])

    sigma_deg = 0.6 / (2.0 * np.sqrt(np.log(2.0)))

    amplitude = np.exp(-0.5 * (angle_grid[:, None, None] / sigma_deg) ** 2)

    ds = xr.Dataset(
        data_vars={
            "amplitude": (
                ("angle", "frequency", "polarization"),
                amplitude,
            ),
        },
        coords={
            "angle": angle_grid,
            "frequency": frequency_grid,
            "polarization": polarization_grid,
        },
    )

    ds["angle"].attrs["units"] = "deg"
    ds["frequency"].attrs["units"] = "Hz"

    beam = Beam(ds)

    target_angle = xr.DataArray(
        np.array([0.0, 0.1, 0.3, 0.6]),
        dims="time",
        attrs={"units": "deg"},
    )

    target_frequency = xr.DataArray(
        np.array([1.0e9]),
        dims="frequency",
        coords={"frequency": np.array([1.0e9])},
        attrs={"units": "Hz"},
    )

    result = beam.evaluate(
        frequency=target_frequency,
        angle=target_angle,
        polarization="H",
    )

    assert result.dims == ("time", "frequency")
    assert result.shape == (4, 1)
    assert np.all(np.isfinite(result.values))

    np.testing.assert_allclose(
        result[:, 0].values,
        np.array(
            [
                1.0,
                np.exp(-0.5 * (0.1 / sigma_deg) ** 2),
                np.exp(-0.5 * (0.3 / sigma_deg) ** 2),
                np.exp(-0.5 * (0.6 / sigma_deg) ** 2),
            ]
        ),
        rtol=1e-6,
        atol=1e-12,
    )



def test_evaluate_single_frequency_beam_preserves_matching_frequency_in_mixed_request() -> None:
    angle_grid = np.linspace(-2.0, 2.0, 801)
    frequency_grid = np.array([1.0e9])
    polarization_grid = np.array(["H"])

    sigma_deg = 0.25

    amplitude = np.exp(
        -0.5
        * (
            angle_grid[:, None, None]
            / sigma_deg
        )
        ** 2
    )

    ds = xr.Dataset(
        data_vars={
            "amplitude": (
                (
                    "angle",
                    "frequency",
                    "polarization",
                ),
                amplitude,
            ),
        },
        coords={
            "angle": angle_grid,
            "frequency": frequency_grid,
            "polarization": polarization_grid,
        },
    )

    ds["angle"].attrs["units"] = "deg"
    ds["frequency"].attrs["units"] = "Hz"

    beam = Beam(ds)

    target_angle = xr.DataArray(
        np.array(
            [
                0.0,
                0.1,
            ]
        ),
        dims="time",
        attrs={
            "units": "deg",
        },
    )

    target_frequency = xr.DataArray(
        np.array(
            [
                1.0e9,
                1.1e9,
            ]
        ),
        dims="frequency",
        coords={
            "frequency": np.array(
                [
                    1.0e9,
                    1.1e9,
                ]
            ),
        },
        attrs={
            "units": "Hz",
        },
    )

    result = beam.evaluate(
        frequency=target_frequency,
        angle=target_angle,
        polarization="H",
    )

    assert result.dims == (
        "time",
        "frequency",
    )

    assert result.shape == (
        2,
        2,
    )

    expected_matching = np.array(
        [
            1.0,
            np.exp(
                -0.5
                * (
                    0.1
                    / sigma_deg
                )
                ** 2
            ),
        ]
    )

    np.testing.assert_allclose(
        result.isel(
            frequency=0
        ).values,
        expected_matching,
        rtol=1e-6,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        result.isel(
            frequency=1
        ).values,
        0.0,
        rtol=0.0,
        atol=0.0,
    )


def test_normalize_peak_zero_db_per_frequency_and_cross_pol_reference():
    angles = np.linspace(-90, 90, 181)
    main = np.exp(-0.5 * (angles / 10.0) ** 2)
    amp = np.zeros((181, 2, 2))
    amp[:, 0, 0] = 0.5 * main  # H_COPOL, f0
    amp[:, 1, 0] = 0.8 * main  # H_COPOL, f1
    amp[:, 0, 1] = 0.005 * main  # H_CROSS, f0
    amp[:, 1, 1] = 0.008 * main  # H_CROSS, f1
    ds = xr.Dataset(
        {"amplitude": (("angle", "frequency", "polarization"), amp)},
        coords={
            "angle": angles,
            "frequency": [1000.0, 1100.0],
            "polarization": ["H_COPOL", "H_CROSS"],
        },
    )
    out = BeamOperation.normalize(Beam(ds)).dataset["amplitude"]
    co = out.sel(polarization="H_COPOL")
    cross = out.sel(polarization="H_CROSS")
    np.testing.assert_allclose(co.max("angle"), 1.0)
    np.testing.assert_allclose(cross.max("angle"), 0.01)
