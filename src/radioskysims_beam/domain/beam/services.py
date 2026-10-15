# domain/beam/services.py
from __future__ import annotations

import numpy as np
import xarray as xr
from radioskysims_core.logging_config import get_logger

from radioskysims_beam.domain.beam.models import Beam

logger = get_logger(__name__)


# =====================================================
# 🔧 Operações diretas sobre o Beam (normalização, corte etc.)
# =====================================================


class BeamOperation:
    """
    Serviços para manipulação do Beam (normalização, recorte, simetrização, etc.)
    """

    @staticmethod
    def _peak_reference(amp: xr.DataArray) -> xr.DataArray:
        """Pico (potência) de referência por frequência/polarização.

        O pico é buscado só ao longo de ``angle``. Canais ``*_CROSS`` usam o pico
        do co-pol correspondente (``H_CROSS`` -> ``H_COPOL``), preservando o
        nível relativo da polarização cruzada.
        """
        peak = amp.max("angle")
        if "polarization" not in amp.dims:
            return peak

        pols = [str(p) for p in amp["polarization"].values]
        ref = []
        for i, pol in enumerate(pols):
            upper = pol.upper()
            own = peak.isel(polarization=i, drop=True)
            if upper.endswith("CROSS"):
                co = upper[: -len("CROSS")] + "COPOL"
                match = [j for j, p in enumerate(pols) if p.upper() == co]
                if match:
                    own = peak.isel(polarization=match[0], drop=True)
            ref.append(own)
        return xr.concat(ref, dim=amp["polarization"])

    @staticmethod
    def normalize(beam: Beam) -> Beam:
        """Normaliza o padrão de potência para pico de 0 dB (Kraus).

        ``P_n = P / P_max`` por frequência e polarização; em dB equivale a uma
        translação, ``P_n[dB] = P[dB] - P_max[dB]``. A amplitude é potência.
        """
        try:
            logger.info("beam_normalize_start", beam_dims=dict(beam.dataset.sizes))
            ds = beam.dataset.copy()
            amp = ds["amplitude"]
            if "angle" in amp.dims:
                ref = BeamOperation._peak_reference(amp)
                ref = ref.where(ref > 0)
                ds["amplitude"] = (amp / ref).fillna(0.0).transpose(*amp.dims)
            else:
                ds["amplitude"] = amp / amp.max()
            logger.info("beam_normalize_success")
            return Beam(ds)
        except Exception as e:
            logger.exception("beam_normalize_failed", error=str(e))
            raise

    @staticmethod
    def slice_angular_range(beam: Beam, min_angle: float, max_angle: float) -> Beam:
        try:
            logger.info(
                "beam_slice_start",
                min_angle_deg=min_angle,
                max_angle_deg=max_angle,
            )
            ds = beam.dataset.sel(angle=slice(min_angle, max_angle))
            logger.info("beam_slice_success", output_points=len(ds.angle))
            return Beam(ds)
        except Exception as e:
            logger.exception(
                "beam_slice_failed",
                min_angle=min_angle,
                max_angle=max_angle,
                error=str(e),
            )
            raise

    @staticmethod
    def interpolate_angular_range(beam: Beam, new_angles: np.ndarray) -> Beam:
        try:
            logger.info(
                "beam_interpolate_start",
                original_points=len(beam.dataset.angle),
                new_points=len(new_angles),
            )
            ds = beam.dataset.interp(angle=new_angles)
            logger.info("beam_interpolate_success", new_points=len(ds.angle))
            return Beam(ds)
        except Exception as e:
            logger.exception(
                "beam_interpolate_failed",
                num_new_angles=len(new_angles),
                error=str(e),
            )
            raise

    @staticmethod
    def symmetrize(beam: Beam) -> Beam:
        ds = beam.dataset.copy()
        a = ds["amplitude"]
        ds["amplitude"] = (a + a.isel(angle=slice(None, None, -1))) / 2
        return Beam(ds)
