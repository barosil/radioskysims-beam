"""
Módulo: src/radioskysims/domain/beam/models.py

Descrição:
    Define o modelo de dados Beam (feixe) que representa o padrão de radiação
    do radiotelescópio no domínio angular e espectral. Contém a classe principal
    para avaliação e interpolação de feixes usando xarray.

Copyright BINGO Collaboration
Last modified: 2026-07-26
"""

from __future__ import annotations

from typing import Annotated

import numpy as np
import xarray as xr
from pydantic import BaseModel, BeforeValidator, ConfigDict


def _ensure_ndarray(v: object) -> np.ndarray:
    """Converte listas/tuplas para numpy array."""
    if isinstance(v, np.ndarray):
        return v
    return np.asarray(v, dtype=float)


NDArrayField = Annotated[np.ndarray, BeforeValidator(_ensure_ndarray)]


def _safe_beam_interp(
    ds: xr.Dataset,
    *,
    frequency: xr.DataArray,
    angle: xr.DataArray,
) -> xr.DataArray:
    """Interpola a amplitude do feixe tratando eixos de ponto único.

    O ``xarray.interp`` (via ``scipy.interp1d``) produz NaN quando um eixo
    tem um único ponto (``x_hi - x_lo = 0``). Nesse caso o padrão é constante
    ao longo do eixo: remove-se o eixo, interpola-se o restante e propaga-se
    o valor para a grade alvo por broadcast (``* ones_like``).

    ``frequency``/``angle`` devem já estar nas unidades do dataset (a conversão
    é feita antes, em ``Beam.evaluate``).
    """
    single_freq = "frequency" in ds.dims and ds.sizes["frequency"] == 1
    single_angle = "angle" in ds.dims and ds.sizes["angle"] == 1

    work = ds["amplitude"]
    if single_freq:
        work = work.isel(frequency=0, drop=True)
    if single_angle:
        work = work.isel(angle=0, drop=True)

    # Interpola os eixos restantes (frequência e/ou ângulo)
    _kwargs = {"fill_value": 0.0}
    if "frequency" in work.dims and "angle" in work.dims:
        result = work.interp(frequency=frequency, angle=angle, kwargs=_kwargs)
    elif "frequency" in work.dims:
        result = work.interp(frequency=frequency, kwargs=_kwargs)
    elif "angle" in work.dims:
        result = work.interp(angle=angle, kwargs=_kwargs)
    else:
        result = work

    # Propaga para os eixos removidos (padrão constante nesses eixos)
    if single_freq:
        result = result * xr.ones_like(frequency)
    if single_angle:
        result = result * xr.ones_like(angle)

    return result


class BeamParamsDomain(BaseModel):
    """Entidade de domínio que representa as propriedades físicas de um feixe."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    polarizations: list[str]
    frequencies: list[float]
    has_phase: bool
    fwhm: NDArrayField
    directivity: float
    main_lobe_directivity: float
    main_lobe_efficiency: float
    directivity_db: float


class Beam:
    """
    Feixe do radiotelescópio no domínio angular e espectral.
    """

    def __init__(self, dataset: xr.Dataset):
        # As coordenadas usadas por interp() são invariantes durante
        # a vida do Beam. Mantê-las ordenadas evita que xarray
        # reordene/indexe todo o dataset a cada avaliação.
        if "angle" in dataset.coords:
            dataset = dataset.sortby("angle")

        if "frequency" in dataset.coords:
            dataset = dataset.sortby("frequency")

        self.dataset = dataset

    def evaluate(
        self,
        *,
        frequency: xr.DataArray,
        angle: xr.DataArray,
        polarization: str | None = None,
    ) -> xr.DataArray:
        """
        Avalia o feixe B(freq, angle).

        Retorna:
            DataArray com dims (..., freq)
        """
        ds = self.dataset

        # If the beam is defined in degrees but angle is in radians, convert
        if "angle" in ds.coords:
            a_min = float(ds.angle.min())
            a_max = float(ds.angle.max())
            ds_angle_units = str(
                ds["angle"].attrs.get(
                    "units",
                    "",
                )
            ).lower()

            if "rad" in ds_angle_units:
                ds_is_degrees = False
            elif "deg" in ds_angle_units:
                ds_is_degrees = True
            else:
                ds_is_degrees = not (abs(a_min) <= 7.0 and abs(a_max) <= 7.0)

            arg_min = float(angle.min())
            arg_max = float(angle.max())

            arg_angle_units = str(
                angle.attrs.get(
                    "units",
                    "",
                )
            ).lower()

            # Unidades explícitas têm precedência sobre qualquer heurística.
            if "rad" in arg_angle_units:
                arg_is_degrees = False

            elif "deg" in arg_angle_units:
                arg_is_degrees = True

            else:
                # Compatibilidade retroativa para DataArrays sem metadados:
                # valores dentro de aproximadamente +/- 2*pi são tratados
                # como radianos; intervalos maiores, como graus.
                arg_looks_radians = abs(arg_min) <= 7.0 and abs(arg_max) <= 7.0
                arg_is_degrees = not arg_looks_radians

            if ds_is_degrees and not arg_is_degrees:
                factor = angle.dtype.type(180.0 / np.pi)
                angle = angle * factor
                angle = angle.assign_attrs(
                    {
                        **angle.attrs,
                        "units": "deg",
                    }
                )

            elif not ds_is_degrees and arg_is_degrees:
                factor = angle.dtype.type(np.pi / 180.0)
                angle = angle * factor
                angle = angle.assign_attrs(
                    {
                        **angle.attrs,
                        "units": "rad",
                    }
                )

        if polarization is not None and "polarization" in ds.dims:
            ds = ds.sel(polarization=polarization)

        ds_freq_units = str(
            ds["frequency"].attrs.get(
                "units",
                "",
            )
        ).lower()

        arg_freq_units = str(
            frequency.attrs.get(
                "units",
                "",
            )
        ).lower()

        try:
            ds_fmax = float(ds["frequency"].max())
        except Exception:
            ds_fmax = None

        try:
            arg_fmax = float(frequency.max())
        except Exception:
            arg_fmax = None

        if "mhz" in ds_freq_units and "hz" in arg_freq_units:
            frequency = frequency / 1e6

        elif "hz" in ds_freq_units and "mhz" in arg_freq_units:
            frequency = frequency * 1e6

        elif ds_fmax is not None and arg_fmax is not None:
            if ds_fmax < 1e6 and arg_fmax > 1e6:
                frequency = frequency / 1e6

            elif ds_fmax > 1e6 and arg_fmax < 1e6:
                frequency = frequency * 1e6

        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.filterwarnings("ignore", message="All-NaN slice")
            _warnings.filterwarnings("ignore", message="divide by zero")
            _warnings.filterwarnings("ignore", message="invalid value")
            result = _safe_beam_interp(
                ds,
                frequency=frequency,
                angle=angle,
            )

            if ds.sizes.get("frequency", 0) == 1:
                frequency_value = float(ds["frequency"].values[0])

                requested = np.asarray(
                    frequency.values,
                    dtype=float,
                )

                # Interpola apenas no eixo angular. Uma tabela com uma única
                # frequência não pode usar interpolação linear nesse eixo.
                angular_response = (
                    ds["amplitude"]
                    .isel(
                        frequency=0,
                        drop=True,
                    )
                    .interp(
                        angle=angle,
                        assume_sorted=True,
                        kwargs={"fill_value": 0.0},
                    )
                )

                result = angular_response.expand_dims(
                    frequency=frequency.values,
                    axis=-1,
                )

                # Preserva individualmente as frequências que coincidem com
                # a única frequência disponível e zera somente as que estão
                # fora da cobertura.
                frequency_in_coverage = xr.DataArray(
                    np.isclose(
                        requested,
                        frequency_value,
                        rtol=1e-12,
                        atol=0.0,
                    ),
                    dims=("frequency",),
                    coords={
                        "frequency": frequency.values,
                    },
                )

                result = result.where(
                    frequency_in_coverage,
                    0.0,
                )

            else:
                result = ds["amplitude"].interp(
                    frequency=frequency,
                    angle=angle,
                    assume_sorted=True,
                    kwargs={"fill_value": 0.0},
                )

        return result

    def params(self) -> BeamParamsDomain:
        """Calcula as propriedades físicas do feixe e retorna um BeamParamsDomain."""
        from radioskysims_beam.domain.beam.properties import BeamProperties

        ds_props = BeamProperties.compute_properties(self.dataset)

        return BeamParamsDomain(
            polarizations=self.dataset.polarization.values.tolist(),
            frequencies=self.dataset.frequency.values.tolist(),
            has_phase="phase" in self.dataset,
            fwhm=np.array(ds_props["fwhm"].values),
            directivity=float(np.nanmean(ds_props["directivity"].values)),
            main_lobe_directivity=float(np.nanmean(ds_props["main_lobe_directivity"].values)),
            directivity_db=float(np.nanmean(ds_props["directivity_db"].values)),
            main_lobe_efficiency=float(np.nanmean(ds_props["main_lobe_efficiency"].values)),
        )
