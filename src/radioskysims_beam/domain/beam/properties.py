"""
Módulo: src/radioskysims/domain/beam/properties.py

Descrição:
    Cálculo de propriedades físicas do feixe de radiação.
    Implementa métodos para calcular FWHM (Full Width at Half Maximum),
    diretividade e outras métricas importantes de antenas.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from .models import Beam


class BeamProperties:
    """
    Calcula propriedades físicas do feixe de radiação.

    Métodos para computar métricas importantes como FWHM (largura à meia altura),
    diretividade e diretividade em dB.
    """

    @staticmethod
    def compute_properties(ds: xr.Dataset) -> xr.Dataset:
        """
        Calcula propriedades do feixe por (polarization, frequency).
        Retorna um novo Dataset com as métricas calculadas.
        """
        try:
            len(ds.polarization.values)
            len(ds.frequency.values)

            angles = ds.angle.values
            dtheta = np.gradient(np.deg2rad(angles))

            fwhm = []
            directivity = []
            main_lobe_directivity = []
            directivity_db = []
            main_lobe_efficiency = []

            for pol in ds.polarization.values:
                f_fwhm, f_d, f_md, f_ddb, f_eff = [], [], [], [], []
                for f in ds.frequency.values:
                    response = ds.sel(
                        polarization=pol,
                        frequency=f,
                    )["amplitude"].values

                    p_max = response.max()

                    # --- FWHM (lóbulo principal, contíguo em torno do pico) ---
                    half_power = p_max / 2
                    k_peak = int(np.argmax(response))

                    k_left = k_peak
                    while k_left > 0 and response[k_left - 1] >= half_power:
                        k_left -= 1

                    k_right = k_peak
                    while k_right < len(response) - 1 and response[k_right + 1] >= half_power:
                        k_right += 1

                    if k_right > k_left:
                        width = angles[k_right] - angles[k_left]
                    else:
                        width = np.nan

                    f_fwhm.append(width)

                    # --- Diretividade ---
                    integral = np.sum(response * dtheta)
                    d = (2 * np.pi * p_max) / integral if integral > 0 else np.nan
                    f_d.append(d)
                    f_ddb.append(10 * np.log10(d) if d > 0 else np.nan)

                    # --- Main lobe ---
                    if not np.isnan(width):
                        theta_max = angles[np.argmax(response)]
                        mask = (angles >= theta_max - width / 2) & (angles <= theta_max + width / 2)
                        integral_main = np.sum(response[mask] * dtheta[mask])
                    else:
                        integral_main = np.nan

                    if integral_main and integral_main > 0:
                        d_main = (2 * np.pi * p_max) / integral_main
                        eff = integral_main / integral
                    else:
                        d_main, eff = np.nan, np.nan

                    f_md.append(d_main)
                    f_eff.append(eff)

                fwhm.append(f_fwhm)
                directivity.append(f_d)
                main_lobe_directivity.append(f_md)
                directivity_db.append(f_ddb)
                main_lobe_efficiency.append(f_eff)

            coords = {"polarization": ds.polarization.values, "frequency": ds.frequency.values}

            result = xr.Dataset(
                {
                    "fwhm": (coords.keys(), np.array(fwhm)),
                    "directivity": (coords.keys(), np.array(directivity)),
                    "main_lobe_directivity": (
                        coords.keys(),
                        np.array(main_lobe_directivity),
                    ),
                    "directivity_db": (coords.keys(), np.array(directivity_db)),
                    "main_lobe_efficiency": (
                        coords.keys(),
                        np.array(main_lobe_efficiency),
                    ),
                },
                coords=coords,
            )
            return result
        except Exception:
            raise

    @staticmethod
    def compute(beam: Beam) -> Beam:
        """
        Computa propriedades físicas do feixe (FWHM, diretividade).

        Args:
            beam (Beam): Feixe para calcular propriedades.

        Returns:
            Beam: Novo feixe com propriedades adicionadas ao dataset:
                - fwhm: Largura à meia altura (Full Width at Half Maximum)
                - directivity: Diretividade linear
                - directivity_db: Diretividade em decibéis

        Notes:
            A diretividade é calculada como D = (2π * Pmax) / ∫P(θ)dθ
        """
        ds = beam.dataset
        angles = np.deg2rad(ds.angle.values)
        dtheta = np.gradient(angles)

        def _compute_one(power):
            pmax = power.max()
            half = pmax / 2
            # FWHM do lóbulo principal (contíguo em torno do pico), sem somar
            # lóbulos laterais que estejam acima da meia-potência.
            k_peak = int(np.argmax(power))
            k_left = k_peak
            while k_left > 0 and power[k_left - 1] >= half:
                k_left -= 1
            k_right = k_peak
            while k_right < len(power) - 1 and power[k_right + 1] >= half:
                k_right += 1
            fwhm = ds.angle[k_right] - ds.angle[k_left] if k_right > k_left else np.nan
            integral = np.sum(power * dtheta)
            d = (2 * np.pi * pmax) / integral if integral > 0 else np.nan
            return fwhm, d, 10 * np.log10(d) if d > 0 else np.nan

        fwhm, directivity, directivity_db = xr.apply_ufunc(
            _compute_one,
            ds["amplitude"],
            input_core_dims=[["angle"]],
            output_core_dims=[[], [], []],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float, float, float],
        )
        ds_props = xr.Dataset(
            {
                "fwhm": fwhm,
                "directivity": directivity,
                "directivity_db": directivity_db,
            }
        )
        return Beam(ds.update(ds_props))
